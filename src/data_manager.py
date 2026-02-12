"""Data manager — fetches candles and computes all indicators."""

import logging
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

    def get_analysis(self, symbol: str) -> pd.DataFrame:
        """Fetch latest candles, compute indicators, cache result."""
        df = self.exchange.fetch_ohlcv(
            symbol,
            self.config.trading.timeframe,
            self.config.trading.candle_limit,
        )
        df = self._add_indicators(df)
        self._cache[symbol] = df
        return df

    def get_cached(self, symbol: str) -> pd.DataFrame | None:
        return self._cache.get(symbol)

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

        return df
