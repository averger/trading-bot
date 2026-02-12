"""Order manager — tracks positions, executes orders, manages stops."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from config.settings import Config
from src.exchange import Exchange
from src.logger import TradeLogger
from src.risk_manager import RiskManager
from src.strategy import TradeSignal, Signal

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    module: str
    regime: str = ""
    trade_id: int
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    trailing_active: bool = False
    candles_held: int = 0
    initial_stop_loss: float = 0.0
    partial_taken: bool = False
    original_size: float = 0.0


class OrderManager:
    def __init__(
        self,
        exchange: Exchange,
        config: Config,
        risk_manager: RiskManager,
        trade_logger: TradeLogger,
    ):
        self.exchange = exchange
        self.config = config
        self.risk_manager = risk_manager
        self.trade_logger = trade_logger
        self.positions: dict[str, Position] = {}
        self._restore_positions()

    # -- persistence ----------------------------------------------------

    def _restore_positions(self):
        """Reload open positions from DB (survives bot restart)."""
        for t in self.trade_logger.get_open_trades():
            self.positions[t["symbol"]] = Position(
                symbol=t["symbol"],
                side=t["side"],
                entry_price=t["entry_price"],
                size=t["size"],
                entry_time=datetime.fromisoformat(t["entry_time"]),
                stop_loss=0,  # will be recalculated on next tick
                module=t["module"],
                regime=t["regime"],
                trade_id=t["id"],
                highest_price=t["entry_price"],
                lowest_price=t["entry_price"],
                original_size=t["size"],
            )
            log.info(
                "Restored position: %s %s %.8f @ %.2f",
                t["side"], t["symbol"], t["size"], t["entry_price"],
            )

    # -- signal processing ----------------------------------------------

    def process_signal(
        self, signal: TradeSignal, symbol: str, balance: float,
    ) -> bool:
        """Process a trade signal. Returns True if order was executed."""
        if signal.signal == Signal.LONG_ENTRY:
            return self._open_position(signal, symbol, balance, "long")
        if signal.signal == Signal.SHORT_ENTRY:
            return self._open_position(signal, symbol, balance, "short")
        if signal.signal in (Signal.LONG_EXIT, Signal.SHORT_EXIT):
            return self._close_position(symbol, signal.price, signal.reason)
        return False

    # -- open -----------------------------------------------------------

    def _open_position(
        self, signal: TradeSignal, symbol: str, balance: float, side: str,
    ) -> bool:
        if symbol in self.positions:
            return False

        if not self.risk_manager.can_open_position(len(self.positions)):
            log.info("Max concurrent positions reached -- skipping")
            return False

        size = self.risk_manager.calculate_position_size(
            balance, signal.price, signal.stop_loss, symbol,
        )
        if size <= 0:
            return False

        # exchange minimums
        min_amt = self.exchange.get_min_order_amount(symbol)
        min_cost = self.exchange.get_min_order_cost(symbol)
        cost = size * signal.price
        if size < min_amt or cost < min_cost:
            log.warning(
                "Order too small: size=%.8f (min=%.8f) cost=%.2f (min=%.2f)",
                size, min_amt, cost, min_cost,
            )
            return False

        precision = self.exchange.get_amount_precision(symbol)
        size = round(size, precision)

        try:
            if side == "long":
                order = self.exchange.market_buy(symbol, size)
            else:
                order = self.exchange.market_short_open(symbol, size)
        except Exception as exc:
            log.error("%s order failed: %s", side.upper(), exc)
            return False

        fill_price = (
            order.get("average") or order.get("price") or signal.price
        )
        now = datetime.now(timezone.utc)

        trade_id = self.trade_logger.log_entry(
            symbol=symbol,
            side=side,
            module=signal.module,
            regime="",
            entry_price=fill_price,
            size=size,
            entry_time=now,
        )

        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            size=size,
            entry_time=now,
            stop_loss=signal.stop_loss,
            module=signal.module,
            regime="",
            trade_id=trade_id,
            highest_price=fill_price,
            lowest_price=fill_price,
            initial_stop_loss=signal.stop_loss,
            original_size=size,
        )

        log.info(
            "OPENED %s %s @ %.2f | size=%.8f | SL=%.2f | %s",
            symbol, side.upper(), fill_price, size, signal.stop_loss, signal.module,
        )
        return True

    # -- close ----------------------------------------------------------

    def _close_position(
        self, symbol: str, price: float, reason: str,
    ) -> bool:
        pos = self.positions.get(symbol)
        if pos is None:
            return False

        try:
            if pos.side == "long":
                order = self.exchange.market_sell(symbol, pos.size)
            else:
                order = self.exchange.market_short_close(symbol, pos.size)
        except Exception as exc:
            log.error("Close order failed: %s", exc)
            return False

        fill = order.get("average") or order.get("price") or price
        now = datetime.now(timezone.utc)

        fees = (
            (pos.entry_price * pos.size + fill * pos.size)
            * self.config.risk.fee_pct
        )
        if pos.side == "long":
            pnl = (fill - pos.entry_price) * pos.size - fees
        else:
            pnl = (pos.entry_price - fill) * pos.size - fees
        pnl_pct = pnl / (pos.entry_price * pos.original_size) if pos.original_size else 0

        self.trade_logger.log_exit(
            trade_id=pos.trade_id,
            exit_price=fill,
            exit_time=now,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )

        log.info(
            "CLOSED %s %s @ %.2f | PnL $%.4f (%.2f%%) | %s",
            symbol, pos.side.upper(), fill, pnl, pnl_pct * 100, reason,
        )
        del self.positions[symbol]
        return True

    # -- stop & trail management (called every tick) --------------------

    def manage_positions(
        self,
        current_prices: dict[str, float],
        atrs: dict[str, float],
        new_candle: bool = False,
    ):
        rc = self.config.risk

        for symbol, pos in list(self.positions.items()):
            price = current_prices.get(symbol)
            atr = atrs.get(symbol, 0)
            if price is None:
                continue

            if pos.side == "long":
                self._manage_long(symbol, pos, price, atr, rc, new_candle)
            else:
                self._manage_short(symbol, pos, price, atr, rc, new_candle)

    def _manage_long(self, symbol, pos, price, atr, rc, new_candle):
        # update high-water mark
        if price > pos.highest_price:
            pos.highest_price = price

        # hard stop-loss
        if price <= pos.stop_loss and pos.stop_loss > 0:
            self._close_position(
                symbol, price, f"stop_loss @ {pos.stop_loss:.2f}",
            )
            return

        # trailing activation
        profit_pct = (price - pos.entry_price) / pos.entry_price
        if profit_pct >= rc.trailing_activation_pct:
            if not pos.trailing_active:
                pos.trailing_active = True
                log.info(
                    "Trailing activated for %s (%.2f%% profit)",
                    symbol, profit_pct * 100,
                )

        # trailing stop update
        if pos.trailing_active and atr > 0:
            new_stop = pos.highest_price - rc.trailing_atr_multiplier * atr
            if new_stop > pos.stop_loss:
                pos.stop_loss = new_stop

        # time stop (only count actual candles, not ticks)
        if new_candle:
            pos.candles_held += 1
        if pos.candles_held >= rc.time_stop_candles:
            move = abs(price - pos.entry_price) / pos.entry_price
            if move < rc.time_stop_min_move:
                self._close_position(
                    symbol, price,
                    f"time_stop: {pos.candles_held} candles, {move:.2%} move",
                )

    def _manage_short(self, symbol, pos, price, atr, rc, new_candle):
        # update low-water mark
        if price < pos.lowest_price:
            pos.lowest_price = price

        # hard stop-loss (short: price goes UP)
        if price >= pos.stop_loss and pos.stop_loss > 0:
            self._close_position(
                symbol, price, f"stop_loss @ {pos.stop_loss:.2f}",
            )
            return

        # trailing activation
        profit_pct = (pos.entry_price - price) / pos.entry_price
        if profit_pct >= rc.trailing_activation_pct:
            if not pos.trailing_active:
                pos.trailing_active = True
                log.info(
                    "Trailing activated for %s SHORT (%.2f%% profit)",
                    symbol, profit_pct * 100,
                )

        # trailing stop update (short: stop moves DOWN)
        if pos.trailing_active and atr > 0:
            new_stop = pos.lowest_price + rc.trailing_atr_multiplier * atr
            if new_stop < pos.stop_loss:
                pos.stop_loss = new_stop

        # time stop
        if new_candle:
            pos.candles_held += 1
        if pos.candles_held >= rc.time_stop_candles:
            move = abs(price - pos.entry_price) / pos.entry_price
            if move < rc.time_stop_min_move:
                self._close_position(
                    symbol, price,
                    f"time_stop: {pos.candles_held} candles, {move:.2%} move",
                )
