"""Data manager — fetches candles and computes all indicators."""

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from config.settings import Config
from src.exchange import Exchange
from src.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_atr,
    compute_ema,
    compute_volume_ma,
)

log = logging.getLogger(__name__)


class DataManager:
    def __init__(self, exchange: Exchange, config: Config):
        self.exchange = exchange
        self.config = config
        self._cache: dict[str, pd.DataFrame] = {}
        self._history: dict[str, pd.DataFrame] = {}

    def warmup(self, symbol: str):
        """Fetch historical candles for EMA warmup (call once at startup).

        Trend EMAs need ~2400 candles (100 days). We fetch 2600 to be safe.
        Most exchanges limit to 1000 per call, so we paginate.
        """
        needed = self.config.indicators.trend_ema_fast + 200  # 2400+200 buffer
        log.info("Warming up %s: fetching %d candles...", symbol, needed)
        df = self._fetch_paginated(symbol, needed)
        self._history[symbol] = df
        log.info("Warmup %s: %d candles loaded (%s to %s)",
                 symbol, len(df), df.index[0], df.index[-1])

    def get_analysis(self, symbol: str) -> pd.DataFrame:
        """Fetch latest candles, merge with history, compute indicators."""
        # Fetch recent candles
        fresh = self.exchange.fetch_ohlcv(
            symbol,
            self.config.trading.timeframe,
            limit=500,
        )

        if symbol in self._history and len(self._history[symbol]) > 0:
            # Merge: append new candles to history, remove duplicates
            history = self._history[symbol]
            combined = pd.concat([history, fresh])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            # Keep only what we need (trim old data to save memory)
            max_keep = self.config.indicators.trend_ema_fast + 500
            if len(combined) > max_keep:
                combined = combined.iloc[-max_keep:]
            self._history[symbol] = combined
            df = combined
        else:
            df = fresh
            self._history[symbol] = df

        df = self._add_indicators(df)
        self._cache[symbol] = df
        return df

    def get_cached(self, symbol: str) -> pd.DataFrame | None:
        return self._cache.get(symbol)

    def _fetch_paginated(self, symbol: str, needed: int) -> pd.DataFrame:
        """Fetch `needed` candles by paginating backwards."""
        tf = self.config.trading.timeframe
        batch_size = 1000
        all_frames = []

        # Start from now, go back
        end_ts = None
        remaining = needed

        while remaining > 0:
            limit = min(batch_size, remaining)
            try:
                if end_ts is None:
                    df = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                else:
                    # Fetch candles before end_ts
                    df = self._fetch_before(symbol, tf, end_ts, limit)

                if df.empty:
                    break

                all_frames.insert(0, df)
                remaining -= len(df)
                end_ts = df.index[0]

                if len(df) < limit:
                    break  # no more data available

                log.info("  fetched %d candles (remaining: %d)", len(df), remaining)

            except Exception as exc:
                log.warning("Paginated fetch error: %s", exc)
                break

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames)
        result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()
        return result

    def _fetch_before(self, symbol: str, tf: str, before_ts, limit: int) -> pd.DataFrame:
        """Fetch candles ending before a given timestamp."""
        # Calculate how far back to go
        tf_ms = self._timeframe_ms(tf)
        since_ms = int(before_ts.timestamp() * 1000) - (limit * tf_ms)
        raw = self.exchange._ex.fetch_ohlcv(
            symbol, tf, since=since_ms, limit=limit,
        )
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        # Only keep candles before our boundary
        df = df[df.index < before_ts]
        return df

    @staticmethod
    def _timeframe_ms(tf: str) -> int:
        """Convert timeframe string to milliseconds."""
        units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        for suffix, ms in units.items():
            if tf.endswith(suffix):
                return int(tf[:-1]) * ms
        return 3_600_000  # default 1h

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ic = self.config.indicators

        df["rsi"] = compute_rsi(df["close"], ic.rsi_period)

        df["bb_upper"], df["bb_middle"], df["bb_lower"] = (
            compute_bollinger_bands(df["close"], ic.bb_period, ic.bb_std)
        )

        df["atr"] = compute_atr(
            df["high"], df["low"], df["close"], ic.atr_period,
        )

        df["ema_short_filter"] = compute_ema(df["close"], ic.ema_short_filter)
        df["ema_long_filter"] = compute_ema(df["close"], ic.ema_long_filter)

        df["volume_ma"] = compute_volume_ma(df["volume"], ic.volume_ma_period)
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Trend EMAs (slow — need many candles for convergence)
        df["ema_trend_fast"] = compute_ema(df["close"], ic.trend_ema_fast)
        df["ema_trend_slow"] = compute_ema(df["close"], ic.trend_ema_slow)

        return df
