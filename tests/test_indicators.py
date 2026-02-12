"""Unit tests for technical indicators."""

import pandas as pd
import numpy as np
import pytest

from src.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_atr,
    compute_adx,
    compute_ema,
    compute_volume_ma,
    compute_true_range,
    compute_macd,
)


def _random_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": close + rng.normal(0, 0.5, n),
         "high": high, "low": low, "close": close, "volume": volume}
    )


class TestRSI:
    def test_bounded_0_100(self):
        rsi = compute_rsi(_random_ohlcv()["close"], 14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_oversold_on_decline(self):
        prices = pd.Series([100 - i * 0.5 for i in range(50)])
        assert compute_rsi(prices, 14).iloc[-1] < 30

    def test_overbought_on_rally(self):
        prices = pd.Series([100 + i * 0.5 for i in range(50)])
        assert compute_rsi(prices, 14).iloc[-1] > 70


class TestBollingerBands:
    def test_band_ordering(self):
        upper, mid, lower = compute_bollinger_bands(
            _random_ohlcv()["close"], 20, 2.0,
        )
        idx = upper.dropna().index
        assert (upper[idx] >= mid[idx]).all()
        assert (mid[idx] >= lower[idx]).all()

    def test_middle_equals_sma(self):
        close = _random_ohlcv()["close"]
        _, mid, _ = compute_bollinger_bands(close, 20, 2.0)
        sma = close.rolling(20).mean()
        pd.testing.assert_series_equal(mid, sma, check_names=False)


class TestATR:
    def test_always_positive(self):
        df = _random_ohlcv()
        atr = compute_atr(df["high"], df["low"], df["close"], 14).dropna()
        assert (atr > 0).all()

    def test_tr_gte_high_minus_low(self):
        df = _random_ohlcv()
        tr = compute_true_range(df["high"], df["low"], df["close"]).iloc[1:]
        hl = (df["high"] - df["low"]).iloc[1:]
        assert (tr >= hl - 1e-10).all()


class TestADX:
    def test_bounded(self):
        df = _random_ohlcv()
        adx = compute_adx(df["high"], df["low"], df["close"], 14).dropna()
        assert (adx >= 0).all() and (adx <= 100).all()


class TestEMA:
    def test_lags_in_uptrend(self):
        prices = pd.Series([float(i) for i in range(100)])
        ema = compute_ema(prices, 10)
        assert ema.iloc[-1] < prices.iloc[-1]


class TestVolumeMA:
    def test_smooths_spikes(self):
        vol = pd.Series([100.0] * 20 + [1000.0] + [100.0] * 20)
        vma = compute_volume_ma(vol, 20)
        assert vma.iloc[-1] < 200


class TestMACD:
    def test_histogram_is_macd_minus_signal(self):
        close = _random_ohlcv()["close"]
        macd, signal, hist = compute_macd(close, 12, 26, 9)
        idx = hist.dropna().index
        diff = (macd - signal)[idx]
        pd.testing.assert_series_equal(
            hist[idx], diff, check_names=False,
        )

    def test_macd_crosses_zero(self):
        # uptrend then downtrend should produce positive then negative MACD
        prices = pd.Series(
            [100 + i for i in range(60)] + [160 - i for i in range(60)]
        )
        macd, _, _ = compute_macd(prices, 12, 26, 9)
        assert macd.iloc[50] > 0   # during uptrend
        assert macd.iloc[110] < 0  # during downtrend

    def test_returns_three_series(self):
        close = _random_ohlcv()["close"]
        result = compute_macd(close)
        assert len(result) == 3
        for s in result:
            assert isinstance(s, pd.Series)
            assert len(s) == len(close)
