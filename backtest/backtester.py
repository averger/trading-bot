"""Backtester — walk-forward simulation reusing the live strategy."""

import logging
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from config.settings import Config
from src.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_atr,
    compute_adx,
    compute_ema,
    compute_volume_ma,
    compute_macd,
)
from src.strategy import Strategy, Signal
from src.sentiment import SentimentAnalyzer

log = logging.getLogger(__name__)


# -- trade record ------------------------------------------------------

@dataclass
class BTTrade:
    entry_idx: int
    entry_price: float
    size: float
    stop_loss: float
    module: str
    regime: str
    side: str = "long"
    exit_idx: int = 0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    partial_taken: bool = False
    original_size: float = 0.0
    initial_stop: float = 0.0


# -- result ------------------------------------------------------------

@dataclass
class BacktestResult:
    trades: list[BTTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 100.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades else 0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_return_pct(self) -> float:
        if not self.equity_curve:
            return 0
        return (self.equity_curve[-1] / self.initial_capital - 1) * 100

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.initial_capital
        dd = 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = max(dd, (peak - eq) / peak)
        return dd * 100

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_win / gross_loss if gross_loss else float("inf")

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0
        rets = pd.Series(self.equity_curve).pct_change().dropna()
        if rets.std() == 0:
            return 0
        return float((rets.mean() / rets.std()) * np.sqrt(8760))

    def summary(self) -> str:
        final = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        longs = [t for t in self.trades if t.side == "long"]
        shorts = [t for t in self.trades if t.side == "short"]
        return (
            f"\n{'='*50}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*50}\n"
            f"  Initial capital : ${self.initial_capital:.2f}\n"
            f"  Final equity    : ${final:.2f}\n"
            f"  Total return    : {self.total_return_pct:+.2f}%\n"
            f"  Max drawdown    : {self.max_drawdown_pct:.2f}%\n"
            f"  Sharpe ratio    : {self.sharpe_ratio:.2f}\n"
            f"  Profit factor   : {self.profit_factor:.2f}\n"
            f"  Trades          : {self.total_trades}"
            f"  (L:{len(longs)} S:{len(shorts)})\n"
            f"  Win rate        : {self.win_rate:.1%}"
            f"  ({self.wins}W / {self.losses}L)\n"
            f"  Total PnL       : ${self.total_pnl:.2f}\n"
            f"{'='*50}\n"
        )


# -- engine ------------------------------------------------------------

class Backtester:
    def __init__(self, config: Config):
        self.config = config
        self.strategy = Strategy(config)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ic = self.config.indicators
        df = df.copy()
        df["rsi"] = compute_rsi(df["close"], ic.rsi_period)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = (
            compute_bollinger_bands(df["close"], ic.bb_period, ic.bb_std)
        )
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], ic.atr_period)
        df["adx"] = compute_adx(df["high"], df["low"], df["close"], ic.adx_period)
        df["ema_fast"] = compute_ema(df["close"], ic.ema_fast)
        df["ema_slow"] = compute_ema(df["close"], ic.ema_slow)
        df["ema_long"] = compute_ema(df["close"], ic.ema_long)
        df["volume_ma"] = compute_volume_ma(df["volume"], ic.volume_ma_period)
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
            df["close"], ic.macd_fast, ic.macd_slow, ic.macd_signal,
        )
        # Previous candle EMAs for crossover detection
        df["prev_ema_fast"] = df["ema_fast"].shift(1)
        df["prev_ema_slow"] = df["ema_slow"].shift(1)
        # Trend-follow specific EMAs (longer periods for fewer, stronger crosses)
        df["tf_ema_fast"] = compute_ema(df["close"], ic.tf_ema_fast)
        df["tf_ema_slow"] = compute_ema(df["close"], ic.tf_ema_slow)
        df["prev_tf_ema_fast"] = df["tf_ema_fast"].shift(1)
        df["prev_tf_ema_slow"] = df["tf_ema_slow"].shift(1)
        return df

    def _compute_htf_bias(self, df: pd.DataFrame) -> pd.Series:
        """Resample 1h data to 4h and compute trend bias per 1h candle.

        Returns a Series aligned to df.index with values:
          1 = bullish (4h EMA fast > slow, close > slow)
         -1 = bearish (4h EMA fast < slow, close < slow)
          0 = neutral
        Uses the *previous completed* 4h bar to avoid look-ahead.
        """
        sc = self.config.strategy
        tf = sc.htf_timeframe  # e.g. "4h"

        df_htf = df.resample(tf).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        if len(df_htf) < sc.htf_ema_slow + 2:
            return pd.Series(0, index=df.index, dtype=int)

        ema_f = compute_ema(df_htf["close"], sc.htf_ema_fast)
        ema_s = compute_ema(df_htf["close"], sc.htf_ema_slow)

        bias_htf = pd.Series(0, index=df_htf.index, dtype=int)
        bias_htf[(ema_f > ema_s) & (df_htf["close"] > ema_s)] = 1
        bias_htf[(ema_f < ema_s) & (df_htf["close"] < ema_s)] = -1

        # Shift by 1 to use previous completed bar (no look-ahead)
        bias_htf = bias_htf.shift(1).fillna(0).astype(int)

        # Map back to 1h via forward-fill
        bias_1h = bias_htf.reindex(df.index, method="ffill").fillna(0).astype(int)
        return bias_1h

    def run(
        self, df: pd.DataFrame, initial_capital: float = 100.0,
        symbol: str = "", fng_history: dict[str, int] | None = None,
    ) -> BacktestResult:
        df = self.add_indicators(df)
        rc = self.config.risk
        result = BacktestResult(initial_capital=initial_capital)

        # Build date -> F&G mapping from candle index
        _fng = fng_history or {}

        # HTF bias (resample 1h -> 4h)
        htf_bias_series = pd.Series(0, index=df.index, dtype=int)
        if self.config.strategy.htf_enabled:
            htf_bias_series = self._compute_htf_bias(df)

        balance = initial_capital
        peak = initial_capital
        pos: BTTrade | None = None
        highest = 0.0
        lowest = float("inf")
        trailing = False
        candles_held = 0
        partial_pnl_accum = 0.0  # track partial TP PnL separately

        warmup = (
            max(
                self.config.indicators.bb_period,
                self.config.indicators.adx_period * 2,
                self.config.indicators.ema_slow,
                self.config.indicators.ema_long,
                self.config.indicators.macd_slow,
                self.config.indicators.tf_ema_slow,
                self.config.strategy.breakout_lookback if self.config.strategy.breakout_enabled else 0,
            )
            + 5
        )

        for i in range(warmup, len(df)):
            cur = df.iloc[i]
            price = cur["close"]
            atr = cur["atr"]

            if pd.isna(atr) or pd.isna(cur["rsi"]):
                result.equity_curve.append(
                    balance + self._unrealized(pos, price),
                )
                continue

            # -- manage open position -----------------------------------
            if pos is not None:
                candles_held += 1
                leverage = self.config.exchange.leverage

                # Funding cost every 8h (= every 8 candles on 1h timeframe)
                if leverage > 1 and candles_held % 8 == 0:
                    funding_cost = pos.size * price * rc.funding_rate_8h
                    partial_pnl_accum -= funding_cost
                    balance -= funding_cost

                # Liquidation check: if unrealized loss exceeds margin
                if leverage > 1:
                    notional = pos.size * pos.entry_price
                    margin = notional / leverage
                    unreal = self._unrealized(pos, price)
                    if unreal < -margin * 0.9:  # 90% of margin gone = liquidation
                        reason = f"LIQUIDATED (loss>${margin:.2f} margin)"
                        if pos.side == "long":
                            self._close_long(pos, i, price, reason, rc)
                        else:
                            self._close_short(pos, i, price, reason, rc)
                        pos.pnl += partial_pnl_accum
                        balance += pos.pnl
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        result.equity_curve.append(balance)
                        continue

                if pos.side == "long":
                    if price > highest:
                        highest = price

                    # stop-loss (use candle low for realism)
                    if cur["low"] <= pos.stop_loss:
                        self._close_long(pos, i, pos.stop_loss, "stop_loss", rc)
                        pos.pnl += partial_pnl_accum
                        balance += pos.pnl
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        result.equity_curve.append(balance)
                        continue

                    # partial take-profit
                    if rc.partial_tp_enabled and not pos.partial_taken and pos.initial_stop > 0:
                        risk_dist = abs(pos.entry_price - pos.initial_stop)
                        tp_price = pos.entry_price + rc.partial_tp_ratio * risk_dist
                        if price >= tp_price and risk_dist > 0:
                            close_size = pos.original_size * rc.partial_tp_size
                            ptp = (price - pos.entry_price) * close_size
                            fees = (pos.entry_price * close_size + price * close_size) * rc.fee_pct
                            partial_pnl_accum += ptp - fees
                            balance += ptp - fees
                            pos.size -= close_size
                            pos.partial_taken = True

                    # trailing activation
                    pct = (price - pos.entry_price) / pos.entry_price
                    if pct >= rc.trailing_activation_pct:
                        trailing = True
                    if trailing and atr > 0:
                        ns = highest - rc.trailing_atr_multiplier * atr
                        if ns > pos.stop_loss:
                            pos.stop_loss = ns

                else:  # SHORT
                    if price < lowest:
                        lowest = price

                    # stop-loss for short (use candle high)
                    if cur["high"] >= pos.stop_loss:
                        self._close_short(pos, i, pos.stop_loss, "stop_loss", rc)
                        pos.pnl += partial_pnl_accum
                        balance += pos.pnl
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        result.equity_curve.append(balance)
                        continue

                    # partial take-profit for short
                    if rc.partial_tp_enabled and not pos.partial_taken and pos.initial_stop > 0:
                        risk_dist = abs(pos.initial_stop - pos.entry_price)
                        tp_price = pos.entry_price - rc.partial_tp_ratio * risk_dist
                        if price <= tp_price and risk_dist > 0:
                            close_size = pos.original_size * rc.partial_tp_size
                            ptp = (pos.entry_price - price) * close_size
                            fees = (pos.entry_price * close_size + price * close_size) * rc.fee_pct
                            partial_pnl_accum += ptp - fees
                            balance += ptp - fees
                            pos.size -= close_size
                            pos.partial_taken = True

                    # trailing for short
                    pct = (pos.entry_price - price) / pos.entry_price
                    if pct >= rc.trailing_activation_pct:
                        trailing = True
                    if trailing and atr > 0:
                        ns = lowest + rc.trailing_atr_multiplier * atr
                        if ns < pos.stop_loss:
                            pos.stop_loss = ns

                # time stop (both sides)
                if candles_held >= rc.time_stop_candles:
                    move = abs(price - pos.entry_price) / pos.entry_price
                    if move < rc.time_stop_min_move:
                        if pos.side == "long":
                            self._close_long(pos, i, price, "time_stop", rc)
                        else:
                            self._close_short(pos, i, price, "time_stop", rc)
                        pos.pnl += partial_pnl_accum
                        balance += pos.pnl
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        result.equity_curve.append(balance)
                        continue

            # -- signals ------------------------------------------------
            self.strategy.set_candle_index(i)
            window = df.iloc[: i + 1]
            pos_module = pos.module if pos is not None else ""
            pos_side = pos.side if pos is not None else "long"

            # Look up Fear & Greed for this candle's date
            fng_val = -1
            if _fng and hasattr(cur, "name") and cur.name is not None:
                date_str = str(cur.name)[:10]  # YYYY-MM-DD
                fng_val = _fng.get(date_str, -1)
            elif _fng and "timestamp" in df.columns:
                ts = cur.get("timestamp", None)
                if ts is not None:
                    date_str = pd.Timestamp(ts, unit="ms").strftime("%Y-%m-%d")
                    fng_val = _fng.get(date_str, -1)

            htf_val = int(htf_bias_series.iloc[i])

            sig = self.strategy.generate_signal(
                window, pos is not None, symbol,
                position_module=pos_module, position_side=pos_side,
                fear_greed=fng_val, htf_bias=htf_val,
            )

            # LONG EXIT
            if sig.signal == Signal.LONG_EXIT and pos is not None and pos.side == "long":
                self._close_long(pos, i, price, sig.reason, rc)
                pos.pnl += partial_pnl_accum
                balance += pos.pnl
                result.trades.append(pos)
                self.strategy.record_trade(symbol)
                pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0

            # SHORT EXIT
            elif sig.signal == Signal.SHORT_EXIT and pos is not None and pos.side == "short":
                self._close_short(pos, i, price, sig.reason, rc)
                pos.pnl += partial_pnl_accum
                balance += pos.pnl
                result.trades.append(pos)
                self.strategy.record_trade(symbol)
                pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0

            # LONG ENTRY
            elif sig.signal == Signal.LONG_ENTRY and pos is None:
                pos = self._open_position(
                    sig, i, price, atr, balance, peak, rc, side="long",
                )
                if pos is not None:
                    highest = pos.entry_price
                    lowest = float("inf")
                    trailing = False
                    candles_held = 0
                    partial_pnl_accum = 0.0
                    self.strategy.record_trade(symbol)

            # SHORT ENTRY
            elif sig.signal == Signal.SHORT_ENTRY and pos is None:
                pos = self._open_position(
                    sig, i, price, atr, balance, peak, rc, side="short",
                )
                if pos is not None:
                    lowest = pos.entry_price
                    highest = 0.0
                    trailing = False
                    candles_held = 0
                    partial_pnl_accum = 0.0
                    self.strategy.record_trade(symbol)

            if balance > peak:
                peak = balance
            result.equity_curve.append(
                balance + self._unrealized(pos, price),
            )

        # close anything still open
        if pos is not None:
            if pos.side == "long":
                self._close_long(pos, len(df) - 1, df.iloc[-1]["close"], "backtest_end", rc)
            else:
                self._close_short(pos, len(df) - 1, df.iloc[-1]["close"], "backtest_end", rc)
            pos.pnl += partial_pnl_accum
            balance += pos.pnl
            result.trades.append(pos)
            if result.equity_curve:
                result.equity_curve[-1] = balance

        return result

    # -- helpers --------------------------------------------------------

    def _open_position(self, sig, i, price, atr, balance, peak, rc, side):
        leverage = self.config.exchange.leverage
        risk_amt = balance * rc.max_risk_per_trade
        if peak > 0 and (1 - balance / peak) >= rc.max_drawdown_reduce:
            risk_amt *= 0.5
        sd = abs(price - sig.stop_loss)
        if sd <= 0:
            return None
        size = risk_amt / sd
        notional = size * price
        margin = notional / leverage  # leverage reduces margin required
        fee = notional * rc.fee_pct
        if side == "long":
            entry = price * (1 + rc.slippage_pct)
        else:
            entry = price * (1 - rc.slippage_pct)
        if margin + fee <= balance:
            return BTTrade(
                entry_idx=i,
                entry_price=entry,
                size=size,
                stop_loss=sig.stop_loss,
                module=sig.module,
                regime=sig.regime.value,
                side=side,
                original_size=size,
                initial_stop=sig.stop_loss,
            )
        return None

    @staticmethod
    def _unrealized(pos: BTTrade | None, price: float) -> float:
        if pos is None:
            return 0.0
        if pos.side == "short":
            return (pos.entry_price - price) * pos.size
        return (price - pos.entry_price) * pos.size

    def _close_long(self, pos: BTTrade, idx: int, price: float, reason: str, rc):
        slip = price * (1 - rc.slippage_pct)
        fees = (pos.entry_price * pos.size + slip * pos.size) * rc.fee_pct
        pos.exit_idx = idx
        pos.exit_price = slip
        pos.pnl = (slip - pos.entry_price) * pos.size - fees
        pos.pnl_pct = pos.pnl / (pos.entry_price * pos.original_size) if pos.entry_price else 0
        pos.exit_reason = reason

    def _close_short(self, pos: BTTrade, idx: int, price: float, reason: str, rc):
        slip = price * (1 + rc.slippage_pct)  # worse fill for short exit
        fees = (pos.entry_price * pos.size + slip * pos.size) * rc.fee_pct
        pos.exit_idx = idx
        pos.exit_price = slip
        pos.pnl = (pos.entry_price - slip) * pos.size - fees
        pos.pnl_pct = pos.pnl / (pos.entry_price * pos.original_size) if pos.entry_price else 0
        pos.exit_reason = reason
