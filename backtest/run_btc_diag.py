"""Final BTC diagnostic with default config."""

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


def main():
    print("Fetching BTC 3yr data...")
    df = fetch_btc()
    hold_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"BTC: {len(df)} candles ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"Price: ${df['close'].iloc[0]:.0f} -> ${df['close'].iloc[-1]:.0f}")
    print(f"Buy & Hold: +{hold_pct:.1f}%\n")

    config = Config()  # uses all defaults
    bt = Backtester(config)
    result = bt.run(df, initial_capital=100, symbol="BTC/USDT")

    trend_trades = [t for t in result.trades if t.module == "trend"]
    mr_trades = [t for t in result.trades if t.module != "trend"]

    print(f"{'='*80}")
    print(f"  FINAL BTC BACKTEST — Default Config")
    print(f"{'='*80}")
    print(f"  Initial capital : $100.00")
    print(f"  Final equity    : ${result.equity_curve[-1]:.2f}")
    print(f"  Total return    : {result.total_return_pct:+.2f}%")
    print(f"  Buy & Hold      : {hold_pct:+.1f}%")
    print(f"  vs Hold         : {result.total_return_pct - hold_pct:+.1f}%")
    print(f"  Max drawdown    : {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe ratio    : {result.sharpe_ratio:.2f}")
    print(f"  Market exposure : {result.market_exposure_pct:.1f}%")
    print(f"  Total trades    : {result.total_trades}")
    print(f"    Trend trades  : {len(trend_trades)}")
    print(f"    MR trades     : {len(mr_trades)}")

    print(f"\n  TREND TRADES:")
    for t in trend_trades:
        entry_dt = df.index[min(t.entry_idx, len(df)-1)]
        exit_dt = df.index[min(t.exit_idx, len(df)-1)]
        hold_h = t.exit_idx - t.entry_idx
        hold_d = hold_h / 24
        print(
            f"    {entry_dt.strftime('%Y-%m-%d')} -> {exit_dt.strftime('%Y-%m-%d')} "
            f"({hold_d:.0f}d)  ${t.entry_price:.0f}->${t.exit_price:.0f}  "
            f"size={t.size:.6f} BTC  PnL=${t.pnl:+.2f} ({t.pnl_pct:+.1%})  [{t.exit_reason}]"
        )

    if mr_trades:
        print(f"\n  MR TRADES:")
        for t in mr_trades:
            entry_dt = df.index[min(t.entry_idx, len(df)-1)]
            exit_dt = df.index[min(t.exit_idx, len(df)-1)]
            hold_h = t.exit_idx - t.entry_idx
            print(
                f"    {entry_dt.strftime('%Y-%m-%d')} -> {exit_dt.strftime('%Y-%m-%d')} "
                f"({hold_h:4d}h)  ${t.entry_price:.0f}->${t.exit_price:.0f}  "
                f"PnL=${t.pnl:+.2f} ({t.pnl_pct:+.1%})  [{t.exit_reason}]"
            )

    print(f"\n  CONFIG:")
    print(f"    Trend EMA:      {config.indicators.trend_ema_fast}/{config.indicators.trend_ema_slow} "
          f"(~{config.indicators.trend_ema_fast//24}d/{config.indicators.trend_ema_slow//24}d)")
    print(f"    Trend alloc:    {config.risk.trend_alloc_pct:.0%}")
    print(f"    Trend stop:     {config.risk.trend_sl_pct:.0%}")
    print(f"    MR RSI:         {config.strategy.rsi_oversold}/{config.strategy.rsi_overbought}")
    print(f"    MR SL:          {config.risk.mean_reversion_sl_atr}x ATR")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
