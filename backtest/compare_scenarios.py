"""Compare scenarios: leverage, trend shorts, combinations."""

import logging
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd

from config.settings import Config
from backtest.backtester import Backtester

logging.basicConfig(level=logging.WARNING)


def fetch_btc(days=1095, timeframe="1h"):
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    since = ex.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )
    all_candles = []
    while True:
        batch = ex.fetch_ohlcv("BTC/USDT", timeframe, since=since, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    return df


def run_scenario(df, label, market_type="spot", leverage=1,
                 trend_short=False, short_alloc=0.50):
    cfg = Config()
    cfg.exchange.market_type = market_type
    cfg.exchange.leverage = leverage
    cfg.strategy.trend_short_enabled = trend_short
    cfg.risk.trend_short_alloc_pct = short_alloc

    bt = Backtester(cfg)
    result = bt.run(df, initial_capital=100, symbol="BTC/USDT")

    trend_longs = [t for t in result.trades if t.module == "trend" and t.side == "long"]
    trend_shorts = [t for t in result.trades if t.module == "trend" and t.side == "short"]
    mr_trades = [t for t in result.trades if t.module != "trend"]

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Return     : {result.total_return_pct:+.1f}%")
    print(f"  Max DD     : {result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe     : {result.sharpe_ratio:.2f}")
    print(f"  Trades     : {result.total_trades} "
          f"(T-long:{len(trend_longs)} T-short:{len(trend_shorts)} MR:{len(mr_trades)})")
    print(f"  Win rate   : {result.win_rate:.0%}")
    print(f"  Exposure   : {result.market_exposure_pct:.1f}%")
    print(f"  Final      : ${result.equity_curve[-1]:.2f}")

    if trend_shorts:
        s_pnl = sum(t.pnl for t in trend_shorts)
        s_wins = sum(1 for t in trend_shorts if t.pnl > 0)
        print(f"  T-Short PnL: ${s_pnl:+.2f} ({s_wins}W/{len(trend_shorts)-s_wins}L)")
        for t in trend_shorts:
            entry_dt = df.index[min(t.entry_idx, len(df)-1)]
            exit_dt = df.index[min(t.exit_idx, len(df)-1)]
            hold_d = (t.exit_idx - t.entry_idx) / 24
            print(
                f"    {entry_dt.strftime('%Y-%m-%d')} -> {exit_dt.strftime('%Y-%m-%d')} "
                f"({hold_d:.0f}d)  ${t.entry_price:.0f}->${t.exit_price:.0f}  "
                f"PnL=${t.pnl:+.2f} ({t.pnl_pct:+.1%})  [{t.exit_reason}]"
            )

    return result


def main():
    print("Fetching BTC 3yr data...")
    df = fetch_btc()
    hold_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"BTC: {len(df)} candles, Buy & Hold: +{hold_pct:.1f}%\n")

    # 1. Baseline
    run_scenario(df, "SPOT 1x — Long only (baseline)")

    # 2. Spot + trend shorts (requires futures in practice but test the signal)
    run_scenario(df, "FUTURES 1x — Long + Trend Short (50% alloc)",
                 market_type="futures", leverage=1, trend_short=True, short_alloc=0.50)

    # 3. Futures 1x + trend shorts 95%
    run_scenario(df, "FUTURES 1x — Long + Trend Short (95% alloc)",
                 market_type="futures", leverage=1, trend_short=True, short_alloc=0.95)

    # 4. Futures 2x long only
    run_scenario(df, "FUTURES 2x — Long only",
                 market_type="futures", leverage=2)

    # 5. Futures 2x + trend shorts
    run_scenario(df, "FUTURES 2x — Long + Trend Short (50%)",
                 market_type="futures", leverage=2, trend_short=True, short_alloc=0.50)

    # 6. Futures 2x + trend shorts 95%
    run_scenario(df, "FUTURES 2x — Long + Trend Short (95%)",
                 market_type="futures", leverage=2, trend_short=True, short_alloc=0.95)

    # 7. Futures 3x + trend shorts
    run_scenario(df, "FUTURES 3x — Long + Trend Short (50%)",
                 market_type="futures", leverage=3, trend_short=True, short_alloc=0.50)

    print(f"\n{'='*65}")
    print(f"  Buy & Hold : +{hold_pct:.1f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
