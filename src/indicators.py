"""Technical indicators — pure functions, no side effects."""

import pandas as pd
import numpy as np


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series,
) -> pd.Series:
    """True Range — max of (H-L, |H-prevC|, |L-prevC|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> pd.Series:
    """Average True Range with Wilder's smoothing."""
    tr = compute_true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = compute_true_range(high, low, close)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_smooth

    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


def compute_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return volume.rolling(window=period).mean()


def compute_realized_vol(
    close: pd.Series, window: int = 720, annualize: int = 8760,
) -> pd.Series:
    """Rolling realized volatility (annualized).

    Default window=720 (~30 days on 1h candles).
    annualize=8760 (hours in a year) for 1h timeframe.
    """
    returns = close.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(annualize)


def compute_realized_vol_ratio(
    close: pd.Series, fast_window: int = 720, slow_window: int = 2160,
) -> pd.Series:
    """Ratio of short-term vol to long-term vol.

    > 1.5 means vol is spiking relative to recent history.
    """
    vol_fast = close.pct_change().rolling(fast_window).std()
    vol_slow = close.pct_change().rolling(slow_window).std()
    return vol_fast / vol_slow


def compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
