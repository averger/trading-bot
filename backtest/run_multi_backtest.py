"""Multi-pair backtest runner — tests all pairs over long periods.

Usage:
    python -m backtest.run_multi_backtest --days 730 --capital 100
    python -m backtest.run_multi_backtest --days 1095 --capital 100 --optimize
"""

import logging
import argparse
import time as _time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import ccxt
import pandas as pd
import numpy as np

from config.settings import Config, StrategyConfig, RiskConfig
from backtest.backtester import Backtester
from src.sentiment import SentimentAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multi_backtest")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "AVAX/USDT"]


def fetch_history(
    symbol: str, timeframe: str, days: int, exchange_name: str = "binance",
) -> pd.DataFrame | None:
    try:
        ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
        ex.load_markets()
    except Exception as e:
        log.error("Cannot connect to %s: %s", exchange_name, e)
        return None

    since = ex.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )
    all_candles: list = []

    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception as e:
            log.warning("Fetch error for %s: %s — retrying", symbol, e)
            _time.sleep(2)
            continue
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    if not all_candles:
        log.warning("No data for %s", symbol)
        return None

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    return df


def run_single(
    symbol: str, df: pd.DataFrame, config: Config, capital: float,
    fng_history: dict[str, int] | None = None,
) -> dict:
    bt = Backtester(config)
    result = bt.run(df, initial_capital=capital, symbol=symbol, fng_history=fng_history)
    final = result.equity_curve[-1] if result.equity_curve else capital
    return {
        "symbol": symbol,
        "candles": len(df),
        "period": f"{df.index[0].date()} -> {df.index[-1].date()}",
        "trades": result.total_trades,
        "win_rate": result.win_rate,
        "return_pct": result.total_return_pct,
        "max_dd_pct": result.max_drawdown_pct,
        "profit_factor": result.profit_factor,
        "sharpe": result.sharpe_ratio,
        "final_equity": final,
        "asset_return": (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100,
        "result": result,
    }


def optimize_params(
    datasets: dict[str, pd.DataFrame], capital: float,
) -> Config:
    """Grid search over key parameters, split 70/30 in-sample/out-of-sample."""
    log.info("=" * 60)
    log.info("  PARAMETER OPTIMIZATION (70/30 in-sample / out-of-sample)")
    log.info("=" * 60)

    # Parameter grid
    rsi_oversold_values = [30, 33, 35, 38]
    rsi_overbought_values = [65, 68, 70, 73]
    volume_spike_values = [1.0, 1.2, 1.5]
    adx_trend_values = [23, 25, 28]

    best_score = -999
    best_params = {}
    best_config = Config()
    total_combos = (
        len(rsi_oversold_values) * len(rsi_overbought_values)
        * len(volume_spike_values) * len(adx_trend_values)
    )
    log.info("Testing %d parameter combinations...", total_combos)

    # Split data: first 70% = in-sample, last 30% = out-of-sample
    in_sample: dict[str, pd.DataFrame] = {}
    out_sample: dict[str, pd.DataFrame] = {}
    for sym, df in datasets.items():
        split = int(len(df) * 0.7)
        in_sample[sym] = df.iloc[:split]
        out_sample[sym] = df.iloc[split:]
        log.info(
            "  %s: in-sample %d candles (%s->%s), out-of-sample %d (%s->%s)",
            sym, split, df.index[0].date(), df.index[split].date(),
            len(df) - split, df.index[split].date(), df.index[-1].date(),
        )

    combo_num = 0
    for rsi_os in rsi_oversold_values:
        for rsi_ob in rsi_overbought_values:
            for vol_sp in volume_spike_values:
                for adx_t in adx_trend_values:
                    combo_num += 1
                    config = Config()
                    config.strategy.rsi_oversold = float(rsi_os)
                    config.strategy.rsi_overbought = float(rsi_ob)
                    config.strategy.volume_spike_multiplier = float(vol_sp)
                    config.strategy.adx_trending_threshold = float(adx_t)

                    # Score across all symbols on in-sample data
                    total_return = 0
                    total_dd = 0
                    total_trades = 0

                    for sym, df in in_sample.items():
                        if len(df) < 200:
                            continue
                        bt = Backtester(config)
                        res = bt.run(df, initial_capital=capital, symbol=sym)
                        total_return += res.total_return_pct
                        total_dd += res.max_drawdown_pct
                        total_trades += res.total_trades

                    # Score = return - 2*drawdown (penalize risk)
                    score = total_return - 2 * total_dd
                    if total_trades < 5:
                        score -= 50  # penalize too few trades

                    if score > best_score:
                        best_score = score
                        best_params = {
                            "rsi_oversold": rsi_os,
                            "rsi_overbought": rsi_ob,
                            "volume_spike": vol_sp,
                            "adx_trending": adx_t,
                        }
                        best_config = config

                    if combo_num % 48 == 0:
                        log.info(
                            "  [%d/%d] best so far: score=%.2f %s",
                            combo_num, total_combos, best_score, best_params,
                        )

    log.info("")
    log.info("  BEST IN-SAMPLE PARAMS: %s (score=%.2f)", best_params, best_score)
    log.info("")

    # Validate on out-of-sample
    log.info("  OUT-OF-SAMPLE VALIDATION:")
    for sym, df in out_sample.items():
        if len(df) < 100:
            continue
        r = run_single(sym, df, best_config, capital)
        log.info(
            "    %s: return=%+.2f%%  max_dd=%.2f%%  trades=%d  WR=%.0f%%",
            sym, r["return_pct"], r["max_dd_pct"], r["trades"],
            r["win_rate"] * 100,
        )

    return best_config


def main():
    ap = argparse.ArgumentParser(description="Multi-pair backtest")
    ap.add_argument("--days", type=int, default=730, help="Days of history (default 730 = 2yr)")
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    args = ap.parse_args()

    # ── fetch all data ───────────────────────────────────────────
    log.info("Fetching %d days of data for %d pairs...", args.days, len(SYMBOLS))
    datasets: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        log.info("  %s...", sym)
        df = fetch_history(sym, args.timeframe, args.days, args.exchange)
        if df is not None and len(df) > 200:
            datasets[sym] = df
            log.info("    -> %d candles (%s -> %s)", len(df), df.index[0].date(), df.index[-1].date())
        else:
            log.warning("    -> skipped (insufficient data)")

    # ── optimization ─────────────────────────────────────────────
    if args.optimize:
        config = optimize_params(datasets, args.capital)
    else:
        config = Config()

    config.trading.timeframe = args.timeframe

    # ── fetch Fear & Greed history ────────────────────────────────
    fng_history: dict[str, int] = {}
    if config.sentiment.fear_greed_enabled:
        log.info("Fetching Fear & Greed Index history...")
        fng_history = SentimentAnalyzer.fetch_fear_greed_history(days=args.days + 60)
        log.info("  -> %d days of F&G data", len(fng_history))

    # ── run full backtests ───────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("  FULL BACKTEST — ALL PAIRS (%d days)", args.days)
    log.info("=" * 70)

    results = []
    for sym, df in datasets.items():
        r = run_single(sym, df, config, args.capital, fng_history=fng_history)
        results.append(r)

    # ── summary table ────────────────────────────────────────────
    print("\n")
    print("=" * 95)
    print(f"  MULTI-PAIR BACKTEST RESULTS — {args.days} days — ${args.capital:.0f} capital per pair")
    print("=" * 95)
    hdr = (
        f"{'Pair':<12} {'Period':<25} {'Trades':>6} {'WR':>6} "
        f"{'Return':>8} {'MaxDD':>7} {'PF':>6} {'Sharpe':>7} "
        f"{'Final$':>8} {'Asset%':>8}"
    )
    print(hdr)
    print("-" * 95)

    total_pnl = 0
    for r in results:
        pf_str = f"{r['profit_factor']:.2f}" if r["profit_factor"] < 100 else "inf"
        print(
            f"{r['symbol']:<12} {r['period']:<25} {r['trades']:>6} "
            f"{r['win_rate']:>5.0%} {r['return_pct']:>+7.2f}% "
            f"{r['max_dd_pct']:>6.2f}% {pf_str:>6} {r['sharpe']:>7.2f} "
            f"{r['final_equity']:>7.2f}$ {r['asset_return']:>+7.1f}%"
        )
        total_pnl += r["final_equity"] - args.capital

    print("-" * 95)
    total_invested = args.capital * len(results)
    total_final = total_invested + total_pnl
    print(
        f"{'PORTFOLIO':<12} {'':25} {'':>6} {'':>6} "
        f"{(total_pnl/total_invested)*100:>+7.2f}% "
        f"{'':>7} {'':>6} {'':>7} "
        f"{total_final:>7.2f}$"
    )
    print("=" * 95)

    # ── per-pair detail ──────────────────────────────────────────
    print("\n\nDETAILED TRADE LOG PER PAIR:")
    for r in results:
        res = r["result"]
        if not res.trades:
            continue
        longs = [t for t in res.trades if t.side == "long"]
        shorts = [t for t in res.trades if t.side == "short"]
        print(f"\n--- {r['symbol']} ({res.total_trades} trades: {len(longs)}L / {len(shorts)}S) ---")
        # Show win/loss breakdown by module
        for mod_name in ["mean_reversion", "momentum", "trend_follow", "breakout"]:
            mod_trades = [t for t in res.trades if t.module == mod_name]
            if not mod_trades:
                continue
            mod_longs = [t for t in mod_trades if t.side == "long"]
            mod_shorts = [t for t in mod_trades if t.side == "short"]
            wins = sum(1 for t in mod_trades if t.pnl > 0)
            pnl = sum(t.pnl for t in mod_trades)
            label = {"mean_reversion": "MeanRev", "momentum": "Momentum", "trend_follow": "TrendFol", "breakout": "Breakout"}[mod_name]
            print(
                f"  {label:>10}: {len(mod_trades):3d} trades "
                f"({len(mod_longs)}L/{len(mod_shorts)}S), "
                f"{wins}W/{len(mod_trades)-wins}L, "
                f"PnL=${pnl:.2f}"
            )

    # ── params used ──────────────────────────────────────────────
    print(f"\nParameters used:")
    print(f"  Leverage:         {config.exchange.leverage}x ({config.exchange.market_type})")
    print(f"  Risk/trade:       {config.risk.max_risk_per_trade:.0%}")
    print(f"  Max positions:    {config.risk.max_concurrent_positions}")
    print(f"  RSI oversold:     {config.strategy.rsi_oversold}")
    print(f"  RSI overbought:   {config.strategy.rsi_overbought}")
    print(f"  Volume spike:     {config.strategy.volume_spike_multiplier}")
    print(f"  ADX trending:     {config.strategy.adx_trending_threshold}")
    print(f"  SL ATR (MR/MOM/TF): {config.risk.mean_reversion_sl_atr} / {config.risk.momentum_sl_atr} / {config.risk.trend_follow_sl_atr}")
    print(f"  Cooldown:         {config.strategy.cooldown_candles} candles")
    print(f"  Partial TP:       {config.risk.partial_tp_enabled} ({config.risk.partial_tp_ratio}x, {config.risk.partial_tp_size:.0%})")
    print(f"  Trend-follow:     {config.strategy.trend_follow_enabled}")
    print(f"  Breakout:         {config.strategy.breakout_enabled}"
          f" (lookback={config.strategy.breakout_lookback}h,"
          f" vol>={config.strategy.breakout_volume_min},"
          f" range<{config.strategy.breakout_max_range_pct:.0%})")
    print(f"  Vol cooldown:     threshold={config.strategy.cooldown_vol_threshold:.0%}")
    print(f"  Mom short SL:     {config.risk.momentum_short_sl_atr}x ATR")
    print(f"  Sentiment F&G:    {config.sentiment.fear_greed_enabled}"
          f" (fear<={config.sentiment.extreme_fear_threshold} blocks shorts,"
          f" greed>={config.sentiment.extreme_greed_threshold} blocks longs)")
    if fng_history:
        print(f"  F&G data points:  {len(fng_history)} days")


if __name__ == "__main__":
    main()
