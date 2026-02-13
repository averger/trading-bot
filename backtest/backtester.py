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
    compute_ema,
    compute_volume_ma,
)
from src.strategy import Strategy, Signal

log = logging.getLogger(__name__)


# -- trade record ------------------------------------------------------

@dataclass
class BTTrade:
    entry_idx: int
    entry_price: float
    size: float
    stop_loss: float
    module: str
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

    @property
    def market_exposure_pct(self) -> float:
        if not self.trades:
            return 0
        candles_in = sum(t.exit_idx - t.entry_idx for t in self.trades)
        total = len(self.equity_curve) if self.equity_curve else 1
        return (candles_in / total) * 100

    @property
    def avg_hold_candles(self) -> float:
        if not self.trades:
            return 0
        return sum(t.exit_idx - t.entry_idx for t in self.trades) / len(self.trades)

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
            f"  Market exposure : {self.market_exposure_pct:.1f}%\n"
            f"  Avg hold        : {self.avg_hold_candles:.0f} candles\n"
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
        df["ema_short_filter"] = compute_ema(df["close"], ic.ema_short_filter)
        df["ema_long_filter"] = compute_ema(df["close"], ic.ema_long_filter)
        df["volume_ma"] = compute_volume_ma(df["volume"], ic.volume_ma_period)
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        df["ema_trend_fast"] = compute_ema(df["close"], ic.trend_ema_fast)
        df["ema_trend_slow"] = compute_ema(df["close"], ic.trend_ema_slow)
        return df

    def run(
        self, df: pd.DataFrame, initial_capital: float = 100.0,
        symbol: str = "",
    ) -> BacktestResult:
        df = self.add_indicators(df)
        rc = self.config.risk
        result = BacktestResult(initial_capital=initial_capital)

        balance = initial_capital
        peak = initial_capital
        pos: BTTrade | None = None
        highest = 0.0
        lowest = float("inf")
        trailing = False
        candles_held = 0
        partial_pnl_accum = 0.0
        # DCA state
        dca_count = 0
        dca_last_candle = 0
        # RSI hedging state
        trend_reduced = False

        warmup_vals = [
            self.config.indicators.bb_period,
            self.config.indicators.atr_period,
            self.config.indicators.ema_long_filter,
            self.config.indicators.volume_ma_period,
        ]
        if self.config.strategy.trend_enabled:
            # Use fast EMA for warmup — slow EMA converges enough for crossover
            warmup_vals.append(self.config.indicators.trend_ema_fast)
        warmup = max(warmup_vals) + 5

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
                    if unreal < -margin * 0.9:
                        reason = f"LIQUIDATED (loss>${margin:.2f} margin)"
                        if pos.side == "long":
                            self._close_long(pos, i, price, reason, rc)
                        else:
                            self._close_short(pos, i, price, reason, rc)
                        balance += pos.pnl  # partials already in balance
                        pos.pnl += partial_pnl_accum  # include in trade record
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        dca_count, trend_reduced = 0, False
                        result.equity_curve.append(balance)
                        continue

                if pos.side == "long":
                    if price > highest:
                        highest = price

                    # stop-loss (use candle low for realism)
                    if cur["low"] <= pos.stop_loss:
                        self._close_long(pos, i, pos.stop_loss, "stop_loss", rc)
                        balance += pos.pnl  # partials already in balance
                        pos.pnl += partial_pnl_accum  # include in trade record
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        dca_count, trend_reduced = 0, False
                        result.equity_curve.append(balance)
                        continue

                    # -- RSI hedging for trend positions ---------
                    if pos.module == "trend":
                        sc = self.config.strategy
                        rsi = cur["rsi"]
                        # Sell portion when RSI > threshold
                        if not trend_reduced and rsi > sc.trend_rsi_reduce:
                            sell_size = pos.size * sc.trend_reduce_pct
                            ptp = (price - pos.entry_price) * sell_size
                            fees = (pos.entry_price * sell_size + price * sell_size) * rc.fee_pct
                            partial_pnl_accum += ptp - fees
                            balance += ptp - fees
                            pos.size -= sell_size
                            trend_reduced = True
                        # Re-buy when RSI cools down and still in uptrend
                        elif trend_reduced and rsi < sc.trend_rsi_rebuy:
                            ema_f = cur.get("ema_trend_fast", np.nan)
                            ema_s = cur.get("ema_trend_slow", np.nan)
                            if not pd.isna(ema_f) and not pd.isna(ema_s) and ema_f > ema_s:
                                rebuy_alloc = balance * sc.trend_reduce_pct * rc.trend_alloc_pct
                                add_size = rebuy_alloc / price
                                if add_size * price <= balance:
                                    total = pos.size + add_size
                                    pos.entry_price = (pos.entry_price * pos.size + price * add_size) / total
                                    pos.size = total
                                    pos.original_size = total
                                    trend_reduced = False

                    if pos.module != "trend":
                        # partial take-profit (MR only)
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

                        # trailing activation (MR only)
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
                        balance += pos.pnl  # partials already in balance
                        pos.pnl += partial_pnl_accum  # include in trade record
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        dca_count, trend_reduced = 0, False
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

                # time stop (MR only — trend positions hold until death cross)
                if pos.module != "trend" and candles_held >= rc.time_stop_candles:
                    move = abs(price - pos.entry_price) / pos.entry_price
                    if move < rc.time_stop_min_move:
                        if pos.side == "long":
                            self._close_long(pos, i, price, "time_stop", rc)
                        else:
                            self._close_short(pos, i, price, "time_stop", rc)
                        balance += pos.pnl  # partials already in balance
                        pos.pnl += partial_pnl_accum  # include in trade record
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        dca_count, trend_reduced = 0, False
                        result.equity_curve.append(balance)
                        continue

            # -- signals ------------------------------------------------
            self.strategy.set_candle_index(i)
            window = df.iloc[: i + 1]
            pos_module = pos.module if pos is not None else ""
            pos_side = pos.side if pos is not None else "long"

            sig = self.strategy.generate_signal(
                window, pos is not None, symbol,
                position_module=pos_module, position_side=pos_side,
            )

            # LONG EXIT
            if sig.signal == Signal.LONG_EXIT and pos is not None and pos.side == "long":
                self._close_long(pos, i, price, sig.reason, rc)
                balance += pos.pnl  # partials already in balance
                pos.pnl += partial_pnl_accum  # include in trade record
                result.trades.append(pos)
                self.strategy.record_trade(symbol)
                pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                dca_count, trend_reduced = 0, False

            # SHORT EXIT
            elif sig.signal == Signal.SHORT_EXIT and pos is not None and pos.side == "short":
                self._close_short(pos, i, price, sig.reason, rc)
                balance += pos.pnl  # partials already in balance
                pos.pnl += partial_pnl_accum  # include in trade record
                result.trades.append(pos)
                self.strategy.record_trade(symbol)
                pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                dca_count, trend_reduced = 0, False

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
                    trend_reduced = False
                    self.strategy.record_trade(symbol)
                    if sig.module == "trend":
                        dca_count = 1
                        dca_last_candle = i

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

            # -- DCA: add to trend position at intervals -------
            if (
                pos is not None
                and pos.module == "trend"
                and rc.dca_tranches > 1
                and dca_count < rc.dca_tranches
                and (i - dca_last_candle) >= rc.dca_interval_candles
            ):
                ema_f = cur.get("ema_trend_fast", np.nan)
                ema_s = cur.get("ema_trend_slow", np.nan)
                if not pd.isna(ema_f) and not pd.isna(ema_s) and ema_f > ema_s:
                    tranche_alloc = balance * (rc.trend_alloc_pct / rc.dca_tranches)
                    add_size = tranche_alloc / price
                    margin_needed = add_size * price
                    if margin_needed <= balance * 0.95:  # keep 5% cash buffer
                        total = pos.size + add_size
                        pos.entry_price = (pos.entry_price * pos.size + price * add_size) / total
                        pos.size = total
                        pos.original_size = total
                        dca_count += 1
                        dca_last_candle = i

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
            balance += pos.pnl  # partials already in balance
            pos.pnl += partial_pnl_accum  # include in trade record
            result.trades.append(pos)
            if result.equity_curve:
                result.equity_curve[-1] = balance

        return result

    # -- helpers --------------------------------------------------------

    def _open_position(self, sig, i, price, atr, balance, peak, rc, side):
        leverage = self.config.exchange.leverage

        # Trend positions: capital allocation sizing
        if sig.module == "trend" and getattr(rc, 'trend_alloc_pct', 0) > 0:
            if side == "short":
                alloc = balance * getattr(rc, 'trend_short_alloc_pct', rc.trend_alloc_pct)
            else:
                alloc = balance * rc.trend_alloc_pct
            if side == "long" and rc.dca_tranches > 1:
                alloc /= rc.dca_tranches
            if peak > 0 and (1 - balance / peak) >= rc.max_drawdown_reduce:
                alloc *= 0.5
            size = (alloc / price) * leverage
            notional = size * price
            margin = notional / leverage
        elif getattr(rc, 'capital_alloc_pct', 0) > 0:
            # Legacy capital allocation mode
            alloc = balance * rc.capital_alloc_pct
            if peak > 0 and (1 - balance / peak) >= rc.max_drawdown_reduce:
                alloc *= 0.5
            size = (alloc / price) * leverage
            notional = size * price
            margin = notional / leverage
        else:
            # Risk-based sizing (MR positions)
            risk_amt = balance * rc.max_risk_per_trade
            if peak > 0 and (1 - balance / peak) >= rc.max_drawdown_reduce:
                risk_amt *= 0.5
            sd = abs(price - sig.stop_loss)
            if sd <= 0:
                return None
            size = risk_amt / sd
            notional = size * price
            margin = notional / leverage

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
        slip = price * (1 + rc.slippage_pct)
        fees = (pos.entry_price * pos.size + slip * pos.size) * rc.fee_pct
        pos.exit_idx = idx
        pos.exit_price = slip
        pos.pnl = (pos.entry_price - slip) * pos.size - fees
        pos.pnl_pct = pos.pnl / (pos.entry_price * pos.original_size) if pos.entry_price else 0
        pos.exit_reason = reason
