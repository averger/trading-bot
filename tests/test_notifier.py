"""Tests for Telegram notifier."""

from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest

from config.settings import Config
from src.notifier import Notifier


class TestNotifierDisabled:
    def test_disabled_without_credentials(self):
        cfg = Config()
        n = Notifier(cfg)
        assert not n.enabled

    def test_send_returns_false_when_disabled(self):
        cfg = Config()
        n = Notifier(cfg)
        assert not n._send("test")


class TestNotifierEnabled:
    def _cfg_with_telegram(self):
        cfg = Config()
        cfg.notifier.telegram_token = "123:ABC"
        cfg.notifier.telegram_chat_id = "456"
        return cfg

    @patch("src.notifier.urllib.request.urlopen")
    def test_send_success(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        n = Notifier(self._cfg_with_telegram())
        assert n.enabled
        assert n._send("hello")
        mock_urlopen.assert_called_once()

    @patch("src.notifier.urllib.request.urlopen")
    def test_send_failure_returns_false(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("timeout")

        n = Notifier(self._cfg_with_telegram())
        assert not n._send("hello")

    @patch("src.notifier.urllib.request.urlopen")
    def test_notify_entry(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        n = Notifier(self._cfg_with_telegram())
        n.notify_entry("BTC/USDT", "long", "trend", 50000, 0.019, 35000, 1000)
        mock_urlopen.assert_called_once()

    @patch("src.notifier.urllib.request.urlopen")
    def test_notify_exit(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        n = Notifier(self._cfg_with_telegram())
        n.notify_exit("BTC/USDT", "long", 55000, 50.0, 0.05, "death cross", 1050)
        mock_urlopen.assert_called_once()

    @patch("src.notifier.urllib.request.urlopen")
    def test_notify_daily_summary(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        n = Notifier(self._cfg_with_telegram())
        n.notify_daily_summary(1050, 1000, {}, 50.0, 1100)
        mock_urlopen.assert_called_once()

    @patch("src.notifier.urllib.request.urlopen")
    def test_notify_drawdown_warning(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        n = Notifier(self._cfg_with_telegram())
        n.notify_drawdown_warning(15.0, 850, 1000)
        mock_urlopen.assert_called_once()

    @patch("src.notifier.urllib.request.urlopen")
    def test_notify_circuit_breaker(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        n = Notifier(self._cfg_with_telegram())
        n.notify_circuit_breaker("MAX DRAWDOWN STOP", 750)
        mock_urlopen.assert_called_once()

    @patch("src.notifier.urllib.request.urlopen")
    def test_message_content_contains_symbol(self, mock_urlopen):
        """Verify the sent payload contains the symbol."""
        import json
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        n = Notifier(self._cfg_with_telegram())
        n.notify_entry("BTC/USDT", "long", "trend", 50000, 0.019, 35000, 1000)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        assert "BTC/USDT" in body["text"]
        assert body["chat_id"] == "456"
