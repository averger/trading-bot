"""Risk manager — position sizing, circuit breakers, drawdown control."""

import logging
from datetime import datetime, timezone

from config.settings import Config
from src.logger import TradeLogger

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: Config, trade_logger: TradeLogger):
        self.config = config
        self.trade_logger = trade_logger
        self.peak_balance: float = 0.0
        self.halted: bool = False
        self.halt_reason: str = ""

    def update_peak(self, balance: float):
        if balance > self.peak_balance:
            self.peak_balance = balance

    # ── circuit breakers ─────────────────────────────────────────

    def check_circuit_breakers(
        self, current_balance: float, initial_capital: float,
    ) -> bool:
        """Return True if trading is allowed, False if halted."""
        rc = self.config.risk

        # max drawdown from peak → full stop
        if self.peak_balance > 0:
            dd = 1.0 - current_balance / self.peak_balance
            if dd >= rc.max_drawdown_stop:
                self.halted = True
                self.halt_reason = (
                    f"MAX DRAWDOWN STOP: {dd:.1%} from peak "
                    f"${self.peak_balance:.2f}"
                )
                log.critical(self.halt_reason)
                return False

        # daily loss limit → pause 24 h
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = self.trade_logger.get_daily_pnl(today)
        limit = initial_capital * rc.max_daily_loss
        if daily_pnl < 0 and abs(daily_pnl) >= limit:
            self.halted = True
            self.halt_reason = (
                f"DAILY LOSS LIMIT: ${daily_pnl:.2f} "
                f"(limit: -${limit:.2f})"
            )
            log.warning(self.halt_reason)
            return False

        self.halted = False
        self.halt_reason = ""
        return True

    # ── position sizing ──────────────────────────────────────────

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        symbol: str,
    ) -> float:
        """Return position size in base currency, risking max_risk_per_trade."""
        rc = self.config.risk
        risk_amount = balance * rc.max_risk_per_trade

        # halve risk if in drawdown zone
        if self.peak_balance > 0:
            dd = 1.0 - balance / self.peak_balance
            if dd >= rc.max_drawdown_reduce:
                risk_amount *= 0.5
                log.info(
                    "Drawdown %.1f%% — halving position size", dd * 100,
                )

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            log.warning("Zero stop distance — skipping")
            return 0.0

        size = risk_amount / stop_distance

        # ensure fees don't eat the entire risk budget
        fee_cost = entry_price * size * rc.fee_pct * 2  # entry + exit
        if fee_cost >= risk_amount:
            log.warning(
                "Fees ($%.4f) >= risk budget ($%.4f) — skipping",
                fee_cost, risk_amount,
            )
            return 0.0

        return size

    def can_open_position(self, num_open: int) -> bool:
        return num_open < self.config.risk.max_concurrent_positions
