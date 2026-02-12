"""Unit tests for risk manager."""

import os
import tempfile
import pytest

from config.settings import Config
from src.risk_manager import RiskManager
from src.logger import TradeLogger


@pytest.fixture()
def risk_setup():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    logger = TradeLogger(path)
    rm = RiskManager(Config(), logger)
    rm.peak_balance = 100.0
    yield rm, logger, path
    logger.close()
    os.unlink(path)


class TestPositionSizing:
    def test_basic(self, risk_setup):
        rm, _, _ = risk_setup
        # 3% of $100 = $3 risk.  SL dist = $1,000.  size = 0.003
        size = rm.calculate_position_size(100, 50_000, 49_000, "BTC/USDT")
        assert 0.002 < size < 0.004

    def test_zero_stop_returns_zero(self, risk_setup):
        rm, _, _ = risk_setup
        assert rm.calculate_position_size(100, 50_000, 50_000, "X") == 0

    def test_drawdown_halves_size(self, risk_setup):
        rm, _, _ = risk_setup
        # peak = 100, balance = 80 -> 20% DD > 15% threshold
        normal = rm.calculate_position_size(100, 50_000, 49_000, "X")
        dd = rm.calculate_position_size(80, 50_000, 49_000, "X")
        assert dd < normal


class TestCircuitBreakers:
    def test_normal(self, risk_setup):
        rm, _, _ = risk_setup
        assert rm.check_circuit_breakers(95, 100) is True

    def test_max_dd_halt(self, risk_setup):
        rm, _, _ = risk_setup
        assert rm.check_circuit_breakers(70, 100) is False
        assert rm.halted

    def test_position_limit(self, risk_setup):
        rm, _, _ = risk_setup
        assert rm.can_open_position(0) is True
        assert rm.can_open_position(3) is True   # max is now 4
        assert rm.can_open_position(4) is False
