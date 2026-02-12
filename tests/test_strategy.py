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
        s.config.strategy.trend_follow_enabled = True
        df = _df_with_indicators()
        # Add TF-specific EMA columns
        df["tf_ema_fast"] = df["ema_fast"]
        df["tf_ema_slow"] = df["ema_slow"]
        df["prev_tf_ema_fast"] = df["tf_ema_fast"].shift(1)
        df["prev_tf_ema_slow"] = df["tf_ema_slow"].shift(1)
        # Force TF EMA golden cross + MACD positive + ADX strong
        df.iloc[-1, df.columns.get_loc("tf_ema_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("tf_ema_slow")] = 105.0
        df.iloc[-1, df.columns.get_loc("prev_tf_ema_fast")] = 104.0  # was below
        df.iloc[-1, df.columns.get_loc("prev_tf_ema_slow")] = 105.0
        df.iloc[-1, df.columns.get_loc("macd_hist")] = 0.5
        df.iloc[-1, df.columns.get_loc("adx")] = 36.0
        # Clear MR/MOM conditions to ensure trend-follow fires
        df.iloc[-1, df.columns.get_loc("rsi")] = 50.0
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 0.5

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.module == "trend_follow"

    def test_trend_follow_short_on_death_cross(self):
        s = Strategy(Config())
        s.config.strategy.trend_follow_enabled = True
        df = _df_with_indicators()
        # Add TF-specific EMA columns
        df["tf_ema_fast"] = df["ema_fast"]
        df["tf_ema_slow"] = df["ema_slow"]
        df["prev_tf_ema_fast"] = df["tf_ema_fast"].shift(1)
        df["prev_tf_ema_slow"] = df["tf_ema_slow"].shift(1)
        close_val = df.iloc[-1]["close"]
        # Force TF EMA death cross + MACD negative + ADX strong
        df.iloc[-1, df.columns.get_loc("tf_ema_fast")] = 95.0
        df.iloc[-1, df.columns.get_loc("tf_ema_slow")] = 100.0
        df.iloc[-1, df.columns.get_loc("prev_tf_ema_fast")] = 101.0  # was above
        df.iloc[-1, df.columns.get_loc("prev_tf_ema_slow")] = 100.0
        df.iloc[-1, df.columns.get_loc("macd_hist")] = -0.5
        df.iloc[-1, df.columns.get_loc("macd")] = -1.0  # MACD below zero
        df.iloc[-1, df.columns.get_loc("adx")] = 36.0
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


class TestHTFFilter:
    def _mr_long_df(self):
        """DataFrame with forced MR long conditions."""
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0
        bb_low = df.iloc[-1]["bb_lower"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_low - 1
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0
        return df

    def _mr_short_df(self):
        """DataFrame with forced MR short conditions."""
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 75.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0
        bb_up = df.iloc[-1]["bb_upper"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_up + 1
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0
        df.iloc[-1, df.columns.get_loc("high")] = bb_up + 3
        df.iloc[-1, df.columns.get_loc("ema_long")] = bb_up + 10
        return df

    def test_bearish_htf_allows_long(self):
        s = Strategy(Config())
        df = self._mr_long_df()
        # htf_bias=-1 (bearish) does NOT block longs (asymmetric filter)
        sig = s.generate_signal(df, has_position=False, htf_bias=-1)
        assert sig.signal == Signal.LONG_ENTRY

    def test_bullish_htf_blocks_short(self):
        s = Strategy(Config())
        df = self._mr_short_df()
        # htf_bias=1 (bullish) should block short entry
        sig = s.generate_signal(df, has_position=False, htf_bias=1)
        assert sig.signal == Signal.NO_SIGNAL

    def test_neutral_htf_allows_all(self):
        s = Strategy(Config())
        df = self._mr_long_df()
        # htf_bias=0 (neutral) allows long
        sig = s.generate_signal(df, has_position=False, htf_bias=0)
        assert sig.signal == Signal.LONG_ENTRY

    def test_neutral_htf_allows_short(self):
        s = Strategy(Config())
        df = self._mr_short_df()
        # htf_bias=0 (neutral) allows short (only bullish blocks shorts)
        sig = s.generate_signal(df, has_position=False, htf_bias=0)
        assert sig.signal == Signal.SHORT_ENTRY

    def test_bullish_htf_allows_long(self):
        s = Strategy(Config())
        df = self._mr_long_df()
        # htf_bias=1 (bullish) allows long
        sig = s.generate_signal(df, has_position=False, htf_bias=1)
        assert sig.signal == Signal.LONG_ENTRY

    def test_bearish_htf_allows_short(self):
        s = Strategy(Config())
        df = self._mr_short_df()
        # htf_bias=-1 (bearish) allows short
        sig = s.generate_signal(df, has_position=False, htf_bias=-1)
        assert sig.signal == Signal.SHORT_ENTRY

    def test_htf_never_blocks_exits(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 75.0
        # Even bearish HTF doesn't block long exit
        sig = s.generate_signal(df, has_position=True, htf_bias=-1)
        assert sig.signal == Signal.LONG_EXIT

    def test_htf_disabled_allows_all(self):
        s = Strategy(Config())
        s.config.strategy.htf_enabled = False
        df = self._mr_long_df()
        # HTF disabled -> bearish bias doesn't block
        sig = s.generate_signal(df, has_position=False, htf_bias=-1)
        assert sig.signal == Signal.LONG_ENTRY


class TestVolatilityAdaptiveCooldown:
    def test_normal_volatility_uses_base_cooldown(self):
        s = Strategy(Config())
        s.set_candle_index(0)
        s.record_trade("BTC/USDT")
        # ATR/price = 1% < threshold (3%), so base cooldown=24
        s.set_candle_index(24)
        assert s._cooldown_ok("BTC/USDT", atr_pct=0.01)

    def test_high_volatility_extends_cooldown(self):
        s = Strategy(Config())
        s.set_candle_index(0)
        s.record_trade("AVAX/USDT")
        # ATR/price = 6% > threshold 3%, multiplier=2x, effective=48
        s.set_candle_index(24)
        assert not s._cooldown_ok("AVAX/USDT", atr_pct=0.06)  # 24 < 48

    def test_high_volatility_allows_after_extended(self):
        s = Strategy(Config())
        s.set_candle_index(0)
        s.record_trade("AVAX/USDT")
        # ATR/price = 6%, multiplier=2x, effective=48
        s.set_candle_index(48)
        assert s._cooldown_ok("AVAX/USDT", atr_pct=0.06)

    def test_no_previous_trade_always_ok(self):
        s = Strategy(Config())
        s.set_candle_index(0)
        assert s._cooldown_ok("NEW/USDT", atr_pct=0.10)


class TestMomentumShortWiderStop:
    def test_momentum_short_uses_wider_stop(self):
        s = Strategy(Config())
        df = _df_with_indicators()
        last = df.iloc[-1]
        price = last["close"]
        atr = last["atr"]
        # Force momentum short conditions
        df.iloc[-1, df.columns.get_loc("rsi")] = 40.0
        df.iloc[-1, df.columns.get_loc("adx")] = 30.0
        bb_lower = last["bb_lower"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_lower - 5
        df.iloc[-1, df.columns.get_loc("ema_fast")] = bb_lower - 3
        df.iloc[-1, df.columns.get_loc("ema_slow")] = bb_lower - 1
        df.iloc[-1, df.columns.get_loc("macd")] = -1.0
        df.iloc[-1, df.columns.get_loc("macd_hist")] = -0.5
        df.iloc[-1, df.columns.get_loc("macd_signal")] = -0.5
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 1.5
        df.iloc[-1, df.columns.get_loc("ema_long")] = bb_lower + 10

        sig = s.generate_signal(df, has_position=False)
        if sig.signal == Signal.SHORT_ENTRY:
            # Stop should use momentum_short_sl_atr (2.0) not momentum_sl_atr (1.5)
            expected_stop = sig.price + s.config.risk.momentum_short_sl_atr * sig.atr
            assert sig.stop_loss == pytest.approx(expected_stop, rel=0.01)


class TestBreakoutModule:
    def _breakout_df(self, n: int = 200):
        """Create a longer DataFrame for breakout testing."""
        return _df_with_indicators(n=n)

    def test_breakout_long_above_range(self):
        s = Strategy(Config())
        df = self._breakout_df(n=250)
        s.config.strategy.breakout_enabled = True
        s.config.strategy.breakout_lookback = 50
        # Force price above 50-candle range high
        window = df.iloc[-51:-1]
        range_high = window["high"].max()
        df.iloc[-1, df.columns.get_loc("close")] = range_high + 5
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 1.5
        # Disable other modules to isolate breakout
        s.config.strategy.trend_follow_enabled = False
        df.iloc[-1, df.columns.get_loc("rsi")] = 50.0  # neutral RSI

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.module == "breakout"

    def test_breakout_short_below_range(self):
        s = Strategy(Config())
        df = self._breakout_df(n=250)
        s.config.strategy.breakout_enabled = True
        s.config.strategy.breakout_lookback = 50
        s.config.strategy.breakout_max_range_pct = 0.50  # relax for test data
        # Force price below 50-candle range low
        window = df.iloc[-51:-1]
        range_low = window["low"].min()
        df.iloc[-1, df.columns.get_loc("close")] = range_low - 5
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 1.5
        # Macro trend bearish: ema_long above price
        df.iloc[-1, df.columns.get_loc("ema_long")] = range_low + 50
        # Disable other modules
        s.config.strategy.trend_follow_enabled = False
        df.iloc[-1, df.columns.get_loc("rsi")] = 50.0

        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.SHORT_ENTRY
        assert sig.module == "breakout"

    def test_breakout_disabled_no_signal(self):
        s = Strategy(Config())
        s.config.strategy.breakout_enabled = False
        df = self._breakout_df(n=250)
        s.config.strategy.breakout_lookback = 50
        window = df.iloc[-51:-1]
        range_high = window["high"].max()
        df.iloc[-1, df.columns.get_loc("close")] = range_high + 5
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 1.5
        s.config.strategy.trend_follow_enabled = False
        df.iloc[-1, df.columns.get_loc("rsi")] = 50.0

        sig = s.generate_signal(df, has_position=False)
        # Breakout disabled, so shouldn't get breakout signal
        assert sig.module != "breakout"

    def test_breakout_low_volume_no_signal(self):
        s = Strategy(Config())
        df = self._breakout_df(n=250)
        s.config.strategy.breakout_lookback = 50
        window = df.iloc[-51:-1]
        range_high = window["high"].max()
        df.iloc[-1, df.columns.get_loc("close")] = range_high + 5
        df.iloc[-1, df.columns.get_loc("volume_ratio")] = 0.5  # below min
        s.config.strategy.trend_follow_enabled = False
        df.iloc[-1, df.columns.get_loc("rsi")] = 50.0
        df.iloc[-1, df.columns.get_loc("adx")] = 15.0  # ranging, no trend-follow

        sig = s.generate_signal(df, has_position=False)
        assert sig.module != "breakout"
