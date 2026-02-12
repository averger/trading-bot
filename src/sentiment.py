"""Sentiment analysis — Fear & Greed Index + Funding Rate."""

import logging
import time

import requests
import pandas as pd

from config.settings import Config

log = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Fetches and caches sentiment data for strategy filtering."""

    FNG_URL = "https://api.alternative.me/fng/"

    def __init__(self, config: Config):
        self.config = config
        self._fng_cache: int = 50
        self._fng_last_fetch: float = 0

    # -- Fear & Greed (live) --------------------------------------------

    def fetch_fear_greed(self) -> int:
        """Fetch current Fear & Greed Index. Cached for 1 hour."""
        if time.time() - self._fng_last_fetch < 3600:
            return self._fng_cache
        try:
            resp = requests.get(
                self.FNG_URL, params={"limit": 1}, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._fng_cache = int(data["data"][0]["value"])
            self._fng_last_fetch = time.time()
            log.info("Fear & Greed Index: %d", self._fng_cache)
        except Exception as e:
            log.warning("Failed to fetch F&G index: %s", e)
        return self._fng_cache

    # -- Fear & Greed (historical, for backtesting) ---------------------

    @staticmethod
    def fetch_fear_greed_history(days: int = 1200) -> dict[str, int]:
        """Fetch historical F&G data. Returns {YYYY-MM-DD: value}."""
        try:
            resp = requests.get(
                SentimentAnalyzer.FNG_URL,
                params={"limit": days, "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            result: dict[str, int] = {}
            for entry in data["data"]:
                ts = int(entry["timestamp"])
                date_str = pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")
                result[date_str] = int(entry["value"])
            log.info("Fetched %d days of F&G history", len(result))
            return result
        except Exception as e:
            log.warning("Failed to fetch F&G history: %s", e)
            return {}

    # -- Funding Rate (live, per-symbol) --------------------------------

    @staticmethod
    def fetch_funding_rate(exchange, symbol: str) -> float:
        """Fetch current funding rate from exchange. Returns rate as float."""
        try:
            info = exchange.fetch_funding_rate(symbol)
            rate = float(info.get("fundingRate", 0))
            return rate
        except Exception as e:
            log.debug("Failed to fetch funding rate for %s: %s", symbol, e)
            return 0.0

    # -- Filtering logic ------------------------------------------------

    def should_block_long(self, fear_greed: int) -> bool:
        """Block new longs during extreme greed (market overextended)."""
        sc = self.config.sentiment
        if not sc.fear_greed_enabled:
            return False
        return fear_greed >= sc.extreme_greed_threshold

    def should_block_short(self, fear_greed: int) -> bool:
        """Block new shorts during extreme fear (market may be bottoming)."""
        sc = self.config.sentiment
        if not sc.fear_greed_enabled:
            return False
        return fear_greed <= sc.extreme_fear_threshold

    def funding_favors_short(self, funding_rate: float) -> bool:
        """High positive funding = too many longs = favor shorts."""
        sc = self.config.sentiment
        if not sc.funding_rate_enabled:
            return False
        return funding_rate >= sc.funding_long_crowded

    def funding_favors_long(self, funding_rate: float) -> bool:
        """High negative funding = too many shorts = favor longs."""
        sc = self.config.sentiment
        if not sc.funding_rate_enabled:
            return False
        return funding_rate <= sc.funding_short_crowded
