"""Unit tests for trend following + mean-reversion strategy."""

import numpy as np
import pandas as pd
import pytest

from config.settings import Config
from src.strategy import Strategy, Signal
from src.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_atr,
    compute_ema,
    compute_volume_ma,
)


def _df_with_indicators(n: int = 250, seed: int = 42) -> pd.DataFrame:
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
    df["ema_short_filter"] = compute_ema(df["close"], 50)
    df["ema_long_filter"] = compute_ema(df["close"], 200)
    df["volume_ma"] = compute_volume_ma(df["volume"], 20)
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    df["ema_trend_fast"] = np.nan
    df["ema_trend_slow"] = np.nan
    return df


def _force_mr_long(df: pd.DataFrame) -> pd.DataFrame:
    df.iloc[-1, df.columns.get_loc("rsi")] = 25.0
    bb_low = df.iloc[-1]["bb_lower"]
    df.iloc[-1, df.columns.get_loc("close")] = bb_low - 1
    df.iloc[-1, df.columns.get_loc("volume_ratio")] = 2.0
    # Ensure downtrend so MR is allowed when trend is enabled
    df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 90.0
    df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
    return df


class TestSignalGeneration:
    def test_returns_trade_signal(self):
        s = Strategy(Config())
        sig = s.generate_signal(_df_with_indicators(), has_position=False)
        assert sig.signal in (
            Signal.NO_SIGNAL, Signal.LONG_ENTRY, Signal.LONG_EXIT,
            Signal.SHORT_ENTRY, Signal.SHORT_EXIT,
        )

    def test_forced_mean_reversion_long(self):
        s = Strategy(Config())
        df = _force_mr_long(_df_with_indicators())
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.module == "mean_reversion"

    def test_nan_indicators_no_signal(self):
        s = Strategy(Config())
        df = _df_with_indicators(n=5)
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.NO_SIGNAL


class TestMRExits:
    def test_rsi_overbought_exit(self):
        """MR position: RSI > 65 triggers exit."""
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 70.0
        sig = s.generate_signal(
            df, has_position=True, position_module="mean_reversion",
        )
        assert sig.signal == Signal.LONG_EXIT

    def test_bb_middle_exit(self):
        """MR position: price >= BB middle + RSI > 50 triggers exit."""
        s = Strategy(Config())
        df = _df_with_indicators()
        bb_mid = df.iloc[-1]["bb_middle"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_mid + 1
        df.iloc[-1, df.columns.get_loc("rsi")] = 55.0
        sig = s.generate_signal(
            df, has_position=True, position_module="mean_reversion",
        )
        assert sig.signal == Signal.LONG_EXIT

    def test_mr_moderate_rsi_no_exit(self):
        """MR position: RSI 55 with price below BB middle doesn't exit."""
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 55.0
        bb_mid = df.iloc[-1]["bb_middle"]
        df.iloc[-1, df.columns.get_loc("close")] = bb_mid - 10
        sig = s.generate_signal(
            df, has_position=True, position_module="mean_reversion",
        )
        assert sig.signal == Signal.NO_SIGNAL


class TestTrendFollowing:
    def _cfg_instant_confirm(self):
        """Config with trend_confirm_candles=1 for basic trend tests."""
        cfg = Config()
        cfg.strategy.trend_confirm_candles = 1
        return cfg

    def test_trend_entry_in_uptrend(self):
        """Uptrend (EMA fast > slow) triggers trend entry."""
        cfg = self._cfg_instant_confirm()
        s = Strategy(cfg)
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.module == "trend"

    def test_no_long_entry_in_downtrend(self):
        """Downtrend (EMA fast < slow) does NOT trigger long trend entry."""
        cfg = self._cfg_instant_confirm()
        s = Strategy(cfg)
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 90.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal != Signal.LONG_ENTRY

    def test_trend_short_entry_in_downtrend(self):
        """Downtrend triggers short trend entry when enabled."""
        cfg = self._cfg_instant_confirm()
        cfg.strategy.trend_short_enabled = True
        s = Strategy(cfg)
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 90.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.SHORT_ENTRY
        assert sig.module == "trend"

    def test_trend_short_exit_on_golden_cross(self):
        """Trend short exits on golden cross."""
        cfg = Config()
        cfg.strategy.trend_short_enabled = True
        s = Strategy(cfg)
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(
            df, has_position=True, position_module="trend",
            position_side="short",
        )
        assert sig.signal == Signal.SHORT_EXIT

    def test_trend_exit_on_death_cross(self):
        """Trend position exits on death cross."""
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 90.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(
            df, has_position=True, position_module="trend",
        )
        assert sig.signal == Signal.LONG_EXIT

    def test_trend_no_exit_on_rsi(self):
        """Trend position: RSI 75 does NOT trigger exit."""
        s = Strategy(Config())
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("rsi")] = 75.0
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(
            df, has_position=True, position_module="trend",
        )
        assert sig.signal == Signal.NO_SIGNAL

    def test_mr_only_in_downtrend(self):
        """MR entry only fires when trend EMAs show downtrend."""
        cfg = self._cfg_instant_confirm()
        s = Strategy(cfg)
        df = _force_mr_long(_df_with_indicators())
        # Set uptrend -> MR should NOT fire (trend entry instead)
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(df, has_position=False)
        assert sig.module == "trend"

    def test_trend_stop_loss(self):
        """Trend entry sets stop at trend_sl_pct below price."""
        cfg = self._cfg_instant_confirm()
        s = Strategy(cfg)
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        expected_stop = sig.price * (1 - cfg.risk.trend_sl_pct)
        assert sig.stop_loss == pytest.approx(expected_stop, rel=0.01)


class TestCrossoverConfirmation:
    def test_no_entry_before_confirmation(self):
        """Golden cross doesn't trigger entry before confirm period."""
        cfg = Config()
        cfg.strategy.trend_confirm_candles = 5
        s = Strategy(cfg)
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        # Call 4 times — not enough for 5-candle confirmation
        for _ in range(4):
            sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.NO_SIGNAL

    def test_entry_after_confirmation(self):
        """Golden cross triggers entry after confirm period."""
        cfg = Config()
        cfg.strategy.trend_confirm_candles = 5
        s = Strategy(cfg)
        df = _df_with_indicators()
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        # Call 5 times — should trigger on the 5th
        for _ in range(5):
            sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.module == "trend"

    def test_counter_resets_on_death_cross(self):
        """Counter resets when EMA fast drops below slow."""
        cfg = Config()
        cfg.strategy.trend_confirm_candles = 5
        s = Strategy(cfg)
        df = _df_with_indicators()

        # 3 candles of golden cross
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        for _ in range(3):
            s.generate_signal(df, has_position=False)
        assert s._trend_cross_count == 3

        # Death cross resets counter
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 90.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        s.generate_signal(df, has_position=False)
        assert s._trend_cross_count == 0

        # Need full 5 candles again
        df.iloc[-1, df.columns.get_loc("ema_trend_fast")] = 110.0
        df.iloc[-1, df.columns.get_loc("ema_trend_slow")] = 100.0
        for _ in range(4):
            sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.NO_SIGNAL  # only 4, need 5


class TestCooldown:
    def test_cooldown_12_candles(self):
        s = Strategy(Config())
        s.set_candle_index(0)
        s.record_trade("BTC/USDT")
        s.set_candle_index(11)
        assert not s._cooldown_ok("BTC/USDT")
        s.set_candle_index(12)
        assert s._cooldown_ok("BTC/USDT")

    def test_cooldown_blocks_entry(self):
        s = Strategy(Config())
        df = _force_mr_long(_df_with_indicators())
        s.set_candle_index(0)
        s.record_trade("BTC/USDT")
        s.set_candle_index(5)
        sig = s.generate_signal(df, has_position=False, symbol="BTC/USDT")
        assert sig.signal == Signal.NO_SIGNAL

    def test_cooldown_allows_after_period(self):
        s = Strategy(Config())
        df = _force_mr_long(_df_with_indicators())
        s.set_candle_index(0)
        s.record_trade("BTC/USDT")
        s.set_candle_index(12)
        sig = s.generate_signal(df, has_position=False, symbol="BTC/USDT")
        assert sig.signal == Signal.LONG_ENTRY


class TestATRVolatilityFilter:
    def test_high_volatility_blocks_entry(self):
        s = Strategy(Config())
        df = _force_mr_long(_df_with_indicators())
        price = df.iloc[-1]["close"]
        df.iloc[-1, df.columns.get_loc("atr")] = price * 0.10
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.NO_SIGNAL


class TestStopLoss:
    def test_long_stop_below_entry(self):
        s = Strategy(Config())
        df = _force_mr_long(_df_with_indicators())
        sig = s.generate_signal(df, has_position=False)
        assert sig.signal == Signal.LONG_ENTRY
        assert sig.stop_loss < sig.price
        expected = sig.price - s.config.risk.mean_reversion_sl_atr * sig.atr
        assert sig.stop_loss == pytest.approx(expected, rel=0.01)
