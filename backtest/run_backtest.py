"""CLI runner — fetch historical data and run a backtest."""

import sys
import logging
import argparse
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd

from config.settings import Config
from backtest.backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backtest")


def fetch_history(
    symbol: str,
    timeframe: str,
    days: int,
    exchange_name: str = "bybit",
) -> pd.DataFrame:
    """Pull OHLCV candles from a public (no-auth) exchange endpoint."""
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    ex.load_markets()

    since = ex.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )
    all_candles: list = []

    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        log.info("  fetched %d candles…", len(all_candles))

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    return df


def main():
    ap = argparse.ArgumentParser(description="Backtest the trading strategy")
    ap.add_argument("--symbol",   default="BTC/USDT")
    ap.add_argument("--days",     type=int, default=90)
    ap.add_argument("--capital",  type=float, default=100.0)
    ap.add_argument("--exchange", default="bybit")
    ap.add_argument("--timeframe", default="1h")
    args = ap.parse_args()

    log.info(
        "Fetching %d days of %s %s from %s…",
        args.days, args.symbol, args.timeframe, args.exchange,
    )
    df = fetch_history(args.symbol, args.timeframe, args.days, args.exchange)
    log.info(
        "Loaded %d candles  (%s → %s)",
        len(df), df.index[0], df.index[-1],
    )

    config = Config()
    config.trading.timeframe = args.timeframe

    bt = Backtester(config)
    result = bt.run(df, initial_capital=args.capital, symbol=args.symbol)

    print(result.summary())

    if result.trades:
        print("Trade log:")
        hdr = f"{'#':>4}  {'Side':<6} {'Module':<16} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'PnL%':>8}  Reason"
        print(hdr)
        print("-" * len(hdr))
        for n, t in enumerate(result.trades, 1):
            print(
                f"{n:4d}  {t.side:<6} {t.module:<16} {t.entry_price:10.2f} "
                f"{t.exit_price:10.2f} {t.pnl:10.4f} "
                f"{t.pnl_pct:7.2%}  {t.exit_reason}"
            )


if __name__ == "__main__":
    main()
