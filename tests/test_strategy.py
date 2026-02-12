"""Unit tests for strategy signal generation."""

import numpy as np
import pandas as pd
import pytest

from config.settings import Config
from src.strategy import Strategy, Signal, Regime
from src.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_atr,
    compute_adx,
    compute_ema,
    compute_volume_ma,
    compute_macd,
)


def _df_with_indicators(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    volume = rng.uniform(100, 1000, n)
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": high, "low": low, "close": close, "volume": volume,
    })
    df["rsi"] = compute_rsi(df["close"], 14)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = (
        compute_bollinger_bands(df["close"], 20, 2.0)
    )
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["adx"] = compute_adx(df["high"], df["low"], df["close"], 14)
    df["ema_fast"] = compute_ema(df["close"], 9)
    df["ema_slow"] = compute_ema(df["close"], 21)
    df["ema_long"] = compute_ema(df["close"], 50)
    df["volume_ma"] = compute_volume_ma(df["volume"], 20)
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    # MACD + crossover columns
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"], 12, 26, 9)
    df["prev_ema_fast"] = df["ema_fast"].shift(1)
    df["prev_ema_slow"] = df["ema_slow"].shift(1)
    return df


class TestRegimeDetection:
    def test_returns_valid_enum(self):
        s = Strategy(Config())
        regime = s.detect_regime(_df_with_indicators())
        assert isinstance(regime, Regime)

    def test_nan_adx_gives_transition(self):
        s = Strategy(Config())
        df = _df_with_indicators(n=5)  # too short for ADX
        assert s.detect_regime(df) == Regime.TRANSITION


class TestSignalGeneration:
    def test_returns_trade_signal(self):
        s = Strategy(Config())
        sig = s.generate_signal(_df_with_indicators(), has_position=False)
        assert sig.signal in (
            Signal.NO_SIGNAL, Signal.LONG_ENTRY, Signal.LONG_EXIT,
            Signal.SHORT_ENTRY, Signal.SHORT_EXIT,
        )

    def test_forced_mean_reversion_entry(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        # force oversold + ranging + volume spike
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0
        bb_low = df.iloc[-1]["bb_lower"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_low - 1
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.module == "mean_reversion"

    def test_forced_mean_reversion_short_entry(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        # force overbought + ranging + volume spike
        df.iloc[-1, df.columns.get_loc("rsi")] = 75.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0
        bb_up = df.iloc[-1]["bb_upper"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_up + 1
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0
        # Make sure high is >= close to avoid NaN issues
        df.iloc[-1, df.columns.get_loc("high")] = bb_up + 3
        # Macro trend must be bearish: ema_long above price
        df.iloc[-1, df.columns.get_loc("ema_long")] = bb_up + 10

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.SHORT_ENTRY
        assert sig.module == "mean_reversion"

    def test_overbought_exit(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 75.0

        sig = s.generate_signal(df, has_position=True)
        assert sig.signal == Signal.LONG_EXIT

    def test_oversold_short_exit(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0

        sig = s.generate_signal(
            df, has_position=True,
            position_module="mean_reversion", position_side="short",
        )
        assert sig.signal == Signal.SHORT_EXIT

    def test_no_entry_during_transition(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("adx")] = 22.0  # transition zone
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
        # Disable trend-follow to test transition blocking
        s.config.strategy.trend_follow_enabled = False

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.NO_SIGNAL

    def test_trend_follow_long_on_golden_cross(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        # Force EMA golden cross + MACD positive + ADX strong
        df.iloc[-1, df.columns.get_loc("ema_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_slow")] = 105.0
        df.iloc[-1, df.columns.get_loc("prev_ema_fast")] = 104.0  # was below
        df.iloc[-1, df.columns.get_loc("prev_ema_slow")] = 105.0
        df.iloc[-1, df.columns.get_loc("macd_hist")] = 0.5
        df.iloc[-1, df.columns.get_loc("adx")] = 25.0
        # Clear MR/MOM conditions to ensure trend-follow fires
        df.iloc[-1, df.columns.get_loc("rsi")] = 50.0
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 0.5

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.module == "trend_follow"

    def test_trend_follow_short_on_death_cross(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        close_val = df.iloc[-1]["close"]
        # Force EMA death cross + MACD negative + ADX strong
        df.iloc[-1, df.columns.get_loc("ema_fast")] = 95.0
        df.iloc[-1, df.columns.get_loc("ema_slow")] = 100.0
        df.iloc[-1, df.columns.get_loc("prev_ema_fast")] = 101.0  # was above
        df.iloc[-1, df.columns.get_loc("prev_ema_slow")] = 100.0
        df.iloc[-1, df.columns.get_loc("macd_hist")] = -0.5
        df.iloc[-1, df.columns.get_loc("macd")] = -1.0  # MACD below zero
        df.iloc[-1, df.columns.get_loc("adx")] = 25.0
        # Macro trend bearish: ema_long above price
        df.iloc[-1, df.columns.get_loc("ema_long")] = close_val + 10
        # Clear MR/MOM conditions
        df.iloc[-1, df.columns.get_loc("rsi")] = 50.0
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 0.5

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.SHORT_ENTRY
        assert sig.module == "trend_follow"


class TestSentimentFilter:
    def test_extreme_greed_blocks_long(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        # force MR long conditions
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0
        bb_low = df.iloc[-1]["bb_lower"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_low - 1
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0

        # Without sentiment -> long fires
        sig = s.generate_signal(df, has_position=False, fear_greed=-1)
        assert sig.signal == Signal.LONG_ENTRY

        # Extreme greed (85) -> long blocked
        sig = s.generate_signal(df, has_position=False, fear_greed=85)
        assert sig.signal == Signal.NO_SIGNAL

    def test_extreme_fear_blocks_short(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        # force MR short conditions
        df.iloc[-1, df.columns.get_loc("rsi")] = 75.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0
        bb_up = df.iloc[-1]["bb_upper"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_up + 1
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0
        df.iloc[-1, df.columns.get_loc("high")] = bb_up + 3
        df.iloc[-1, df.columns.get_loc("ema_long")] = bb_up + 10

        # Without sentiment -> short fires
        sig = s.generate_signal(df, has_position=False, fear_greed=-1)
        assert sig.signal == Signal.SHORT_ENTRY

        # Extreme fear (20) -> short blocked
        sig = s.generate_signal(df, has_position=False, fear_greed=20)
        assert sig.signal == Signal.NO_SIGNAL

    def test_neutral_sentiment_allows_all(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0
        bb_low = df.iloc[-1]["bb_lower"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_low - 1
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0

        # Neutral F&G (50) -> long allowed
        sig = s.generate_signal(df, has_position=False, fear_greed=50)
        assert sig.signal == Signal.LONG_ENTRY

    def test_sentiment_never_blocks_exits(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 75.0

        # Even during extreme greed, exit still fires
        sig = s.generate_signal(df, has_position=True, fear_greed=80)
        assert sig.signal == Signal.LONG_EXIT
