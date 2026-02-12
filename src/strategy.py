"""Strategy — regime detection + signal generation (long + short)."""

import logging
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import numpy as np

from config.settings import Config

log = logging.getLogger(__name__)


# -- enums & data -----------------------------------------------------

class Regime(Enum):
    RANGING = "ranging"
    TRENDING = "trending"
    TRANSITION = "transition"


class Signal(Enum):
    NO_SIGNAL = "no_signal"
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"


@dataclass
class TradeSignal:
    signal: Signal
    regime: Regime
    module: str        # "mean_reversion" | "momentum" | "trend_follow" | "exit" | "none"
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

    # -- regime detection -----------------------------------------------

    def detect_regime(self, df: pd.DataFrame) -> Regime:
        adx = df["adx"].iloc[-1]
        sc = self.config.strategy
        if pd.isna(adx):
            return Regime.TRANSITION
        if adx < sc.adx_ranging_threshold:
            return Regime.RANGING
        if adx > sc.adx_trending_threshold:
            return Regime.TRENDING
        return Regime.TRANSITION

    # -- sentiment filter ------------------------------------------------

    def _sentiment_blocks(self, signal: Signal, fear_greed: int) -> bool:
        """Check if Fear & Greed sentiment blocks this signal type."""
        sc = self.config.sentiment
        if not sc.fear_greed_enabled or fear_greed < 0:
            return False
        # Extreme Greed -> block new longs (market overextended)
        if signal in (Signal.LONG_ENTRY,) and fear_greed >= sc.extreme_greed_threshold:
            return True
        # Extreme Fear -> block new shorts (market may be bottoming)
        if signal in (Signal.SHORT_ENTRY,) and fear_greed <= sc.extreme_fear_threshold:
            return True
        return False

    # -- master signal router -------------------------------------------

    def generate_signal(
        self, df: pd.DataFrame, has_position: bool, symbol: str = "",
        position_module: str = "", position_side: str = "long",
        fear_greed: int = -1,
    ) -> TradeSignal:
        regime = self.detect_regime(df)
        last = df.iloc[-1]
        price = last["close"]
        atr = last["atr"]

        if pd.isna(atr) or pd.isna(last["rsi"]):
            return self._no_signal(regime, price, atr)

        # exits first (module-aware + side-aware) — sentiment never blocks exits
        if has_position:
            exit_sig = self._check_exit(
                last, regime, price, atr, position_module, position_side,
            )
            if exit_sig is not None:
                return exit_sig

        # entries (skip during transition, cooldown, or extreme volatility)
        if not has_position:
            if regime == Regime.TRANSITION and not self.config.strategy.trend_follow_enabled:
                return self._no_signal(regime, price, atr)
            if symbol and not self._cooldown_ok(symbol):
                return self._no_signal(regime, price, atr)
            atr_pct = atr / price if price > 0 else 0
            if atr_pct > self.config.strategy.max_atr_pct:
                return self._no_signal(regime, price, atr)

            # mean reversion (RANGING)
            if regime == Regime.RANGING:
                sig = self._mean_reversion_entry(last, regime, price, atr)
                if sig.signal != Signal.NO_SIGNAL:
                    if not self._sentiment_blocks(sig.signal, fear_greed):
                        return sig
                sig = self._mean_reversion_short_entry(last, regime, price, atr)
                if sig.signal != Signal.NO_SIGNAL:
                    if not self._sentiment_blocks(sig.signal, fear_greed):
                        return sig

            # momentum (TRENDING)
            if regime == Regime.TRENDING:
                allowed = self.config.strategy.momentum_symbols
                if not allowed or symbol in allowed:
                    sig = self._momentum_entry(last, regime, price, atr)
                    if sig.signal != Signal.NO_SIGNAL:
                        if not self._sentiment_blocks(sig.signal, fear_greed):
                            return sig
                    sig = self._momentum_short_entry(last, regime, price, atr)
                    if sig.signal != Signal.NO_SIGNAL:
                        if not self._sentiment_blocks(sig.signal, fear_greed):
                            return sig

            # trend-follow (any regime, uses its own ADX filter)
            if self.config.strategy.trend_follow_enabled:
                sig = self._trend_follow_entry(last, regime, price, atr)
                if sig.signal != Signal.NO_SIGNAL:
                    if not self._sentiment_blocks(sig.signal, fear_greed):
                        return sig

        return self._no_signal(regime, price, atr)

    # -- mean-reversion module (long) -----------------------------------

    def _mean_reversion_entry(
        self, last, regime: Regime, price: float, atr: float,
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
        ):
            stop = price - rc.mean_reversion_sl_atr * atr
            return TradeSignal(
                signal=Signal.LONG_ENTRY,
                regime=regime,
                module="mean_reversion",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=(
                    f"MR LONG: RSI={rsi:.1f} price<=BB_lower "
                    f"vol_ratio={vol_ratio:.1f}"
                ),
            )
        return self._no_signal(regime, price, atr)

    # -- macro trend filter for shorts ----------------------------------

    def _short_trend_ok(self, last, price: float) -> bool:
        """Only allow shorts when macro trend is bearish (price < EMA long)."""
        ema_long = last.get("ema_long", np.nan)
        if pd.isna(ema_long):
            return False
        return price < ema_long

    # -- mean-reversion module (short) ----------------------------------

    def _mean_reversion_short_entry(
        self, last, regime: Regime, price: float, atr: float,
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
            stop = price + rc.mean_reversion_sl_atr * atr  # stop ABOVE entry
            return TradeSignal(
                signal=Signal.SHORT_ENTRY,
                regime=regime,
                module="mean_reversion",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=(
                    f"MR SHORT: RSI={rsi:.1f} price>=BB_upper "
                    f"vol_ratio={vol_ratio:.1f}"
                ),
            )
        return self._no_signal(regime, price, atr)

    # -- momentum module (long) -----------------------------------------

    def _momentum_entry(
        self, last, regime: Regime, price: float, atr: float,
    ) -> TradeSignal:
        sc = self.config.strategy
        rc = self.config.risk

        rsi = last["rsi"]
        bb_upper = last["bb_upper"]
        ema_fast = last["ema_fast"]
        ema_slow = last["ema_slow"]

        if (
            price > bb_upper
            and sc.rsi_momentum_min < rsi < sc.rsi_momentum_max
            and ema_fast > ema_slow
            and price > ema_slow
        ):
            stop = price - rc.momentum_sl_atr * atr
            return TradeSignal(
                signal=Signal.LONG_ENTRY,
                regime=regime,
                module="momentum",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=(
                    f"MOM LONG: RSI={rsi:.1f} price>BB_upper EMA aligned"
                ),
            )
        return self._no_signal(regime, price, atr)

    # -- momentum module (short) ----------------------------------------

    def _momentum_short_entry(
        self, last, regime: Regime, price: float, atr: float,
    ) -> TradeSignal:
        sc = self.config.strategy
        rc = self.config.risk

        rsi = last["rsi"]
        bb_lower = last["bb_lower"]
        ema_fast = last["ema_fast"]
        ema_slow = last["ema_slow"]
        macd = last.get("macd", np.nan)
        macd_hist = last.get("macd_hist", np.nan)
        macd_sig = last.get("macd_signal", np.nan)
        vol_ratio = last["volume_ratio"]

        if (
            price < bb_lower
            and rsi < (100 - sc.rsi_momentum_min)  # mirror: RSI < 45
            and ema_fast < ema_slow
            and price < ema_slow
            and not pd.isna(macd) and macd < 0  # MACD line below zero
            and not pd.isna(macd_hist) and macd_hist < 0  # histogram confirms
            and not pd.isna(macd_sig) and macd_sig < 0  # signal line below zero = established downtrend
            and vol_ratio > sc.momentum_volume_min
            and self._short_trend_ok(last, price)  # price below EMA(50)
        ):
            stop = price + rc.momentum_sl_atr * atr
            return TradeSignal(
                signal=Signal.SHORT_ENTRY,
                regime=regime,
                module="momentum",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=(
                    f"MOM SHORT: RSI={rsi:.1f} price<BB_lower MACD bearish"
                ),
            )
        return self._no_signal(regime, price, atr)

    # -- trend-following module -----------------------------------------

    def _trend_follow_entry(
        self, last, regime: Regime, price: float, atr: float,
    ) -> TradeSignal:
        sc = self.config.strategy
        rc = self.config.risk

        ema_fast = last["ema_fast"]
        ema_slow = last["ema_slow"]
        prev_ema_fast = last.get("prev_ema_fast", np.nan)
        prev_ema_slow = last.get("prev_ema_slow", np.nan)
        macd_hist = last.get("macd_hist", np.nan)
        adx = last["adx"]

        if pd.isna(prev_ema_fast) or pd.isna(prev_ema_slow) or pd.isna(macd_hist):
            return self._no_signal(regime, price, atr)

        if pd.isna(adx) or adx < sc.trend_follow_adx_min:
            return self._no_signal(regime, price, atr)

        # LONG: EMA golden cross + MACD histogram > 0
        if (
            prev_ema_fast <= prev_ema_slow
            and ema_fast > ema_slow
            and macd_hist > 0
        ):
            stop = price - rc.trend_follow_sl_atr * atr
            return TradeSignal(
                signal=Signal.LONG_ENTRY,
                regime=regime,
                module="trend_follow",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=f"TF LONG: EMA golden cross MACD={macd_hist:.4f}",
            )

        # SHORT: EMA death cross + MACD histogram < 0 + price confirms bearish
        macd = last.get("macd", np.nan)
        if (
            prev_ema_fast >= prev_ema_slow
            and ema_fast < ema_slow
            and macd_hist < 0
            and price < ema_slow  # price must confirm bearish trend
            and not pd.isna(macd) and macd < 0  # macro MACD below zero
            and self._short_trend_ok(last, price)  # price below EMA(200)
        ):
            stop = price + rc.trend_follow_sl_atr * atr
            return TradeSignal(
                signal=Signal.SHORT_ENTRY,
                regime=regime,
                module="trend_follow",
                price=price,
                atr=atr,
                stop_loss=stop,
                reason=f"TF SHORT: EMA death cross MACD={macd_hist:.4f}",
            )

        return self._no_signal(regime, price, atr)

    # -- exit checks (module + side aware) ------------------------------

    def _check_exit(
        self, last, regime: Regime, price: float, atr: float,
        position_module: str = "", position_side: str = "long",
    ) -> TradeSignal | None:
        sc = self.config.strategy

        if position_side == "short":
            return self._check_short_exit(last, regime, price, atr, position_module)

        # --- LONG exits ---

        if position_module == "trend_follow":
            if last["rsi"] > sc.rsi_momentum_max:
                return TradeSignal(
                    signal=Signal.LONG_EXIT,
                    regime=regime,
                    module="exit",
                    price=price,
                    atr=atr,
                    stop_loss=0,
                    reason=f"TF extreme RSI: {last['rsi']:.1f}",
                )
            return None  # trailing handles the rest

        if position_module == "momentum":
            if last["rsi"] > sc.rsi_momentum_max:
                return TradeSignal(
                    signal=Signal.LONG_EXIT,
                    regime=regime,
                    module="exit",
                    price=price,
                    atr=atr,
                    stop_loss=0,
                    reason=f"MOM extreme RSI: {last['rsi']:.1f}",
                )
            return None

        # Mean reversion long exits
        if last["rsi"] > sc.rsi_overbought:
            return TradeSignal(
                signal=Signal.LONG_EXIT,
                regime=regime,
                module="exit",
                price=price,
                atr=atr,
                stop_loss=0,
                reason=f"RSI overbought: {last['rsi']:.1f}",
            )

        if regime == Regime.RANGING:
            if price >= last["bb_middle"] and last["rsi"] > sc.rsi_exit:
                return TradeSignal(
                    signal=Signal.LONG_EXIT,
                    regime=regime,
                    module="exit",
                    price=price,
                    atr=atr,
                    stop_loss=0,
                    reason="MR target: price >= BB_middle & RSI > 50",
                )

        return None

    def _check_short_exit(
        self, last, regime: Regime, price: float, atr: float,
        position_module: str = "",
    ) -> TradeSignal | None:
        sc = self.config.strategy

        if position_module == "trend_follow":
            if last["rsi"] < (100 - sc.rsi_momentum_max):  # RSI < 20
                return TradeSignal(
                    signal=Signal.SHORT_EXIT,
                    regime=regime,
                    module="exit",
                    price=price,
                    atr=atr,
                    stop_loss=0,
                    reason=f"TF extreme oversold RSI: {last['rsi']:.1f}",
                )
            return None

        if position_module == "momentum":
            if last["rsi"] < (100 - sc.rsi_momentum_max):  # RSI < 20
                return TradeSignal(
                    signal=Signal.SHORT_EXIT,
                    regime=regime,
                    module="exit",
                    price=price,
                    atr=atr,
                    stop_loss=0,
                    reason=f"MOM extreme oversold RSI: {last['rsi']:.1f}",
                )
            return None

        # Mean reversion short exits
        if last["rsi"] < sc.rsi_oversold:
            return TradeSignal(
                signal=Signal.SHORT_EXIT,
                regime=regime,
                module="exit",
                price=price,
                atr=atr,
                stop_loss=0,
                reason=f"RSI oversold: {last['rsi']:.1f}",
            )

        if regime == Regime.RANGING:
            if price <= last["bb_middle"] and last["rsi"] < sc.rsi_exit:
                return TradeSignal(
                    signal=Signal.SHORT_EXIT,
                    regime=regime,
                    module="exit",
                    price=price,
                    atr=atr,
                    stop_loss=0,
                    reason="MR SHORT target: price <= BB_middle & RSI < 50",
                )

        return None

    # -- helpers --------------------------------------------------------

    def _no_signal(
        self, regime: Regime, price: float, atr: float,
    ) -> TradeSignal:
        return TradeSignal(
            signal=Signal.NO_SIGNAL,
            regime=regime,
            module="none",
            price=price,
            atr=atr if not pd.isna(atr) else 0,
            stop_loss=0,
            reason="",
        )
