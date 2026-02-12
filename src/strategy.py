"""Strategy — trend following + mean-reversion (long only)."""

import logging
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import numpy as np

from config.settings import Config

log = logging.getLogger(__name__)


# -- enums & data -----------------------------------------------------

class Signal(Enum):
    NO_SIGNAL = "no_signal"
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"


@dataclass
class TradeSignal:
    signal: Signal
    module: str        # "trend" | "mean_reversion" | "exit" | "none"
    price: float
    atr: float
    stop_loss: float
    reason: str


# -- strategy engine ---------------------------------------------------

class Strategy:
    def __init__(self, config: Config):
        self.config = config
        self._last_trade_candle: dict[str, int] = {}  # symbol -> candle index
        self._candle_counter: int = 0

    def set_candle_index(self, idx: int):
        self._candle_counter = idx

    def record_trade(self, symbol: str):
        self._last_trade_candle[symbol] = self._candle_counter

    def _cooldown_ok(self, symbol: str) -> bool:
        last = self._last_trade_candle.get(symbol, -999)
        return (self._candle_counter - last) >= self.config.strategy.cooldown_candles

    # -- master signal router -------------------------------------------

    def generate_signal(
        self, df: pd.DataFrame, has_position: bool, symbol: str = "",
        position_module: str = "", position_side: str = "long",
    ) -> TradeSignal:
        last = df.iloc[-1]
        price = last["close"]
        atr = last["atr"]

        if pd.isna(atr) or pd.isna(last["rsi"]):
            return self._no_signal(price, atr)

        # exits first — always checked
        if has_position:
            exit_sig = self._check_exit(last, price, atr, position_side, position_module)
            if exit_sig is not None:
                return exit_sig

        # entries
        if not has_position:
            atr_pct = atr / price if price > 0 else 0
            if symbol and not self._cooldown_ok(symbol):
                return self._no_signal(price, atr)
            if atr_pct > self.config.strategy.max_atr_pct:
                return self._no_signal(price, atr)

            # trend entry first (captures uptrends with large positions)
            if self.config.strategy.trend_enabled:
                sig = self._trend_entry(last, price, atr)
                if sig.signal != Signal.NO_SIGNAL:
                    return sig

            # mean reversion (only in downtrend or when trend disabled)
            if self._mr_allowed(last):
                sig = self._mean_reversion_entry(last, price, atr)
                if sig.signal != Signal.NO_SIGNAL:
                    return sig

        return self._no_signal(price, atr)

    # -- trend following module ----------------------------------------

    def _trend_entry(
        self, last, price: float, atr: float,
    ) -> TradeSignal:
        ema_fast = last.get("ema_trend_fast", np.nan)
        ema_slow = last.get("ema_trend_slow", np.nan)
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return self._no_signal(price, atr)
        if ema_fast > ema_slow:
            stop = price * (1 - self.config.risk.trend_sl_pct)
            return TradeSignal(
                signal=Signal.LONG_ENTRY,
                module="trend",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=f"TREND LONG: EMA_fast > EMA_slow",
            )
        return self._no_signal(price, atr)

    def _check_trend_exit(
        self, last, price: float, atr: float,
    ) -> TradeSignal | None:
        ema_fast = last.get("ema_trend_fast", np.nan)
        ema_slow = last.get("ema_trend_slow", np.nan)
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return None
        if ema_fast < ema_slow:
            return TradeSignal(
                signal=Signal.LONG_EXIT,
                module="exit",
                price=price,
                atr=atr,
                stop_loss=0,
                reason="TREND EXIT: death cross",
            )
        return None

    def _mr_allowed(self, last) -> bool:
        """MR entries only when trend is disabled or in downtrend."""
        if not self.config.strategy.trend_enabled:
            return True
        ema_fast = last.get("ema_trend_fast", np.nan)
        ema_slow = last.get("ema_trend_slow", np.nan)
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return True  # trend EMAs not available yet, allow MR
        return ema_fast <= ema_slow  # downtrend -> MR allowed

    # -- mean-reversion module (long) -----------------------------------

    def _long_trend_ok(self, last, price: float) -> bool:
        """No trend filter for longs — let BB + RSI do the work."""
        return True

    def _mean_reversion_entry(
        self, last, price: float, atr: float,
    ) -> TradeSignal:
        sc = self.config.strategy
        rc = self.config.risk

        rsi = last["rsi"]
        bb_lower = last["bb_lower"]
        vol_ratio = last["volume_ratio"]

        if (
            rsi < sc.rsi_oversold
            and price <= bb_lower
            and vol_ratio > sc.volume_spike_multiplier
            and self._long_trend_ok(last, price)
        ):
            stop = price - rc.mean_reversion_sl_atr * atr
            return TradeSignal(
                signal=Signal.LONG_ENTRY,
                module="mean_reversion",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=(
                    f"MR LONG: RSI={rsi:.1f} price<=BB_lower "
                    f"vol_ratio={vol_ratio:.1f}"
                ),
            )
        return self._no_signal(price, atr)

    # -- short trend filter ---------------------------------------------

    def _short_trend_ok(self, last, price: float) -> bool:
        """Only allow shorts when EMA(50) < EMA(200) (death cross = downtrend)."""
        ema_short = last.get("ema_short_filter", np.nan)
        ema_long = last.get("ema_long_filter", np.nan)
        if pd.isna(ema_short) or pd.isna(ema_long):
            return False
        return ema_short < ema_long

    # -- mean-reversion module (short) ----------------------------------

    def _mean_reversion_short_entry(
        self, last, price: float, atr: float,
    ) -> TradeSignal:
        sc = self.config.strategy
        rc = self.config.risk

        rsi = last["rsi"]
        bb_upper = last["bb_upper"]
        vol_ratio = last["volume_ratio"]

        if (
            rsi > sc.rsi_overbought
            and price >= bb_upper
            and vol_ratio > sc.volume_spike_multiplier
            and self._short_trend_ok(last, price)
        ):
            stop = price + rc.mean_reversion_sl_atr * atr
            return TradeSignal(
                signal=Signal.SHORT_ENTRY,
                module="mean_reversion",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=(
                    f"MR SHORT: RSI={rsi:.1f} price>=BB_upper "
                    f"vol_ratio={vol_ratio:.1f}"
                ),
            )
        return self._no_signal(price, atr)

    # -- exit checks (side aware) --------------------------------------

    def _check_exit(
        self, last, price: float, atr: float,
        position_side: str = "long",
        position_module: str = "",
    ) -> TradeSignal | None:
        if position_side == "short":
            return self._check_short_exit(last, price, atr)

        # Trend position: exit only on death cross
        if position_module == "trend":
            return self._check_trend_exit(last, price, atr)

        # MR position: RSI overbought or BB middle target
        sc = self.config.strategy

        if last["rsi"] > sc.rsi_overbought:
            return TradeSignal(
                signal=Signal.LONG_EXIT,
                module="exit",
                price=price,
                atr=atr,
                stop_loss=0,
                reason=f"RSI overbought: {last['rsi']:.1f}",
            )

        if price >= last["bb_middle"] and last["rsi"] > sc.rsi_exit:
            return TradeSignal(
                signal=Signal.LONG_EXIT,
                module="exit",
                price=price,
                atr=atr,
                stop_loss=0,
                reason="MR target: price >= BB_middle & RSI > exit",
            )

        return None

    def _check_short_exit(
        self, last, price: float, atr: float,
    ) -> TradeSignal | None:
        sc = self.config.strategy

        # RSI oversold exit
        if last["rsi"] < sc.rsi_oversold:
            return TradeSignal(
                signal=Signal.SHORT_EXIT,
                module="exit",
                price=price,
                atr=atr,
                stop_loss=0,
                reason=f"RSI oversold: {last['rsi']:.1f}",
            )

        # BB middle target exit
        if price <= last["bb_middle"] and last["rsi"] < sc.rsi_exit:
            return TradeSignal(
                signal=Signal.SHORT_EXIT,
                module="exit",
                price=price,
                atr=atr,
                stop_loss=0,
                reason="MR SHORT target: price <= BB_middle & RSI < 50",
            )

        return None

    # -- helpers --------------------------------------------------------

    def _no_signal(self, price: float, atr: float) -> TradeSignal:
        return TradeSignal(
            signal=Signal.NO_SIGNAL,
            module="none",
            price=price,
            atr=atr if not pd.isna(atr) else 0,
            stop_loss=0,
            reason="",
        )
