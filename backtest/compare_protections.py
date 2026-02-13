"""Compare loss protection strategies on 3yr BTC data.

Tests:
  1. Baseline (30% stop, no tightening)
  2. Progressive stop (tighten as profit grows)
  3. Volatility regime (tighten when vol spikes)
  4. Fear & Greed index (tighten on extreme greed/fear)
  5. Progressive + Vol combined
  6. All three combined
"""

import logging
from datetime import datetime, timezone, timedelta

import ccxt
import numpy as np
import pandas as pd

from config.settings import Config
from backtest.backtester import Backtester
from src.sentiment import SentimentAnalyzer

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


def fetch_fng_series(days=1200) -> pd.Series:
    """Fetch Fear & Greed history and return as daily Series."""
    fng_data = SentimentAnalyzer.fetch_fear_greed_history(days)
    if not fng_data:
        print("  [WARN] Could not fetch F&G data — skipping F&G tests")
        return pd.Series(dtype=float)
    s = pd.Series(fng_data, dtype=float)
    s.index = pd.to_datetime(s.index, utc=True)
    s = s.sort_index()
    return s


def add_fng_to_df(df: pd.DataFrame, fng_series: pd.Series) -> pd.DataFrame:
    """Merge daily F&G into hourly dataframe (forward fill)."""
    df = df.copy()
    df["date"] = df.index.normalize()
    fng_df = fng_series.to_frame("fng")
    fng_df.index = fng_df.index.normalize()
    df = df.merge(fng_df, left_on="date", right_index=True, how="left")
    df["fng"] = df["fng"].ffill()
    df = df.drop(columns=["date"])
    return df


class FngBacktester(Backtester):
    """Extended backtester that uses Fear & Greed for stop tightening."""

    def __init__(self, config, fng_extreme_greed=80, fng_tight_sl=0.15):
        super().__init__(config)
        self.fng_extreme_greed = fng_extreme_greed
        self.fng_tight_sl = fng_tight_sl
        self.fng_enabled = True

    def _update_trend_stop(self, pos, price, highest, lowest, cur, rc):
        """Override to add F&G-based stop tightening."""
        # Call parent logic (progressive + vol)
        super()._update_trend_stop(pos, price, highest, lowest, cur, rc)

        # F&G: tighten stop during extreme greed
        if self.fng_enabled:
            fng = cur.get("fng", np.nan)
            if not pd.isna(fng):
                if pos.side == "long" and fng >= self.fng_extreme_greed:
                    fng_stop = highest * (1 - self.fng_tight_sl)
                    if fng_stop > pos.stop_loss:
                        pos.stop_loss = fng_stop
                elif pos.side == "short" and fng <= (100 - self.fng_extreme_greed):
                    fng_stop = lowest * (1 + self.fng_tight_sl)
                    if fng_stop < pos.stop_loss:
                        pos.stop_loss = fng_stop


def run_scenario(df, label, cfg, backtester_cls=Backtester, **bt_kwargs):
    bt = backtester_cls(cfg, **bt_kwargs) if bt_kwargs else backtester_cls(cfg)
    result = bt.run(df, initial_capital=1000, symbol="BTC/USDT")

    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  Return     : {result.total_return_pct:+.1f}%")
    print(f"  Max DD     : {result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe     : {result.sharpe_ratio:.2f}")
    print(f"  Trades     : {result.total_trades}  (W:{result.wins} L:{result.losses})")
    print(f"  Final      : ${result.equity_curve[-1]:,.2f}")

    # Trade details
    for t in result.trades:
        entry_dt = df.index[min(t.entry_idx, len(df) - 1)]
        exit_dt = df.index[min(t.exit_idx, len(df) - 1)]
        hold_d = (t.exit_idx - t.entry_idx) / 24
        print(
            f"    {t.side:5s} {entry_dt.strftime('%Y-%m-%d')} -> "
            f"{exit_dt.strftime('%Y-%m-%d')} ({hold_d:.0f}d)  "
            f"${t.entry_price:.0f}->${t.exit_price:.0f}  "
            f"PnL=${t.pnl:+.2f} ({t.pnl_pct:+.1%})  [{t.exit_reason}]"
        )

    return result


def main():
    print("Fetching BTC 3yr data...")
    df = fetch_btc()
    hold_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"BTC: {len(df)} candles, Buy & Hold: +{hold_pct:.1f}%\n")

    print("Fetching Fear & Greed history...")
    fng_series = fetch_fng_series()
    has_fng = len(fng_series) > 0
    if has_fng:
        df_fng = add_fng_to_df(df, fng_series)
        print(f"F&G data: {len(fng_series)} days, merged into dataframe")
    else:
        df_fng = df.copy()
        df_fng["fng"] = np.nan

    # -- 1. Baseline ----------------------------------------------
    cfg = Config()
    cfg.exchange.market_type = "futures"
    cfg.strategy.trend_short_enabled = True
    cfg.risk.trend_short_alloc_pct = 0.95
    run_scenario(df, "1. BASELINE (30% stop)", cfg)

    # -- 2. Progressive stop only ---------------------------------
    cfg2 = Config()
    cfg2.exchange.market_type = "futures"
    cfg2.strategy.trend_short_enabled = True
    cfg2.risk.trend_short_alloc_pct = 0.95
    cfg2.risk.progressive_stop = True
    run_scenario(df, "2. PROGRESSIVE STOP (+20%->15% trail, +50%->10%)", cfg2)

    # -- 3. Volatility regime only --------------------------------
    cfg3 = Config()
    cfg3.exchange.market_type = "futures"
    cfg3.strategy.trend_short_enabled = True
    cfg3.risk.trend_short_alloc_pct = 0.95
    cfg3.risk.vol_stop_enabled = True
    run_scenario(df, "3. VOL REGIME (vol>1.5x avg -> 15% stop)", cfg3)

    # -- 4. Fear & Greed only ------------------------------------
    if has_fng:
        cfg4 = Config()
        cfg4.exchange.market_type = "futures"
        cfg4.strategy.trend_short_enabled = True
        cfg4.risk.trend_short_alloc_pct = 0.95
        run_scenario(
            df_fng, "4. FEAR & GREED (F&G>80 -> 15% stop)",
            cfg4, backtester_cls=FngBacktester,
            fng_extreme_greed=80, fng_tight_sl=0.15,
        )

    # -- 5. Progressive + Vol ------------------------------------
    cfg5 = Config()
    cfg5.exchange.market_type = "futures"
    cfg5.strategy.trend_short_enabled = True
    cfg5.risk.trend_short_alloc_pct = 0.95
    cfg5.risk.progressive_stop = True
    cfg5.risk.vol_stop_enabled = True
    run_scenario(df, "5. PROGRESSIVE + VOL", cfg5)

    # -- 6. All three --------------------------------------------
    if has_fng:
        cfg6 = Config()
        cfg6.exchange.market_type = "futures"
        cfg6.strategy.trend_short_enabled = True
        cfg6.risk.trend_short_alloc_pct = 0.95
        cfg6.risk.progressive_stop = True
        cfg6.risk.vol_stop_enabled = True
        run_scenario(
            df_fng, "6. ALL THREE (progressive + vol + F&G)",
            cfg6, backtester_cls=FngBacktester,
            fng_extreme_greed=80, fng_tight_sl=0.15,
        )

    # -- 7. Progressive with different thresholds -----------------
    cfg7 = Config()
    cfg7.exchange.market_type = "futures"
    cfg7.strategy.trend_short_enabled = True
    cfg7.risk.trend_short_alloc_pct = 0.95
    cfg7.risk.progressive_stop = True
    cfg7.risk.prog_stop_tier1_profit = 0.30   # more conservative
    cfg7.risk.prog_stop_tier1_trail = 0.20
    cfg7.risk.prog_stop_tier2_profit = 0.80
    cfg7.risk.prog_stop_tier2_trail = 0.15
    run_scenario(df, "7. PROGRESSIVE CONSERVATIVE (+30%->20%, +80%->15%)", cfg7)

    print(f"\n{'='*60}")
    print(f"  Buy & Hold : +{hold_pct:.1f}%  (${1000 * (1 + hold_pct/100):,.2f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
