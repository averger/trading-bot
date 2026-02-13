"""Telegram notifier — sends trade alerts and daily summaries."""

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone

from config.settings import Config

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Send alerts via Telegram Bot API (no extra dependencies)."""

    def __init__(self, config: Config):
        self.token = config.notifier.telegram_token
        self.chat_id = config.notifier.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            log.warning(
                "Telegram notifier disabled — "
                "set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

    # -- low-level send ---------------------------------------------------

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        url = TELEGRAM_API.format(token=self.token)
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
                log.warning("Telegram API returned %d", resp.status)
                return False
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Telegram send failed: %s", exc)
            return False

    # -- trade alerts -----------------------------------------------------

    def notify_entry(
        self, symbol: str, side: str, module: str,
        price: float, size: float, stop_loss: float,
        balance: float,
    ):
        emoji = "\U0001f7e2" if side == "long" else "\U0001f534"
        sl_pct = abs(price - stop_loss) / price * 100 if price else 0
        cost = price * size
        text = (
            f"{emoji} <b>{side.upper()} {symbol}</b>\n"
            f"Module: {module}\n"
            f"Price: <code>${price:,.2f}</code>\n"
            f"Size: <code>{size:.6f}</code> (${cost:,.2f})\n"
            f"Stop: <code>${stop_loss:,.2f}</code> ({sl_pct:.1f}%)\n"
            f"Balance: <code>${balance:,.2f}</code>"
        )
        self._send(text)

    def notify_exit(
        self, symbol: str, side: str, price: float,
        pnl: float, pnl_pct: float, reason: str,
        balance: float,
    ):
        emoji = "\U00002705" if pnl >= 0 else "\U0000274c"
        text = (
            f"{emoji} <b>CLOSE {side.upper()} {symbol}</b>\n"
            f"Price: <code>${price:,.2f}</code>\n"
            f"PnL: <code>${pnl:+,.2f}</code> ({pnl_pct:+.2%})\n"
            f"Reason: {reason}\n"
            f"Balance: <code>${balance:,.2f}</code>"
        )
        self._send(text)

    # -- daily summary ----------------------------------------------------

    def notify_daily_summary(
        self, balance: float, initial_capital: float,
        positions: dict, daily_pnl: float,
        peak_balance: float,
    ):
        total_pct = (balance / initial_capital - 1) * 100 if initial_capital else 0
        dd = (1 - balance / peak_balance) * 100 if peak_balance > 0 else 0

        pos_lines = []
        for sym, pos in positions.items():
            pos_lines.append(
                f"  {pos.side.upper()} {sym} @ ${pos.entry_price:,.2f}"
            )
        pos_text = "\n".join(pos_lines) if pos_lines else "  No open positions"

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        text = (
            f"\U0001f4ca <b>Daily Summary</b> — {now}\n\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Total return: <code>{total_pct:+.1f}%</code>\n"
            f"Today PnL: <code>${daily_pnl:+,.2f}</code>\n"
            f"Drawdown: <code>{dd:.1f}%</code>\n"
            f"Peak: <code>${peak_balance:,.2f}</code>\n\n"
            f"<b>Positions:</b>\n{pos_text}"
        )
        self._send(text)

    # -- alerts -----------------------------------------------------------

    def notify_drawdown_warning(self, dd_pct: float, balance: float, peak: float):
        text = (
            f"\U000026a0 <b>DRAWDOWN WARNING</b>\n\n"
            f"Drawdown: <code>{dd_pct:.1f}%</code>\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Peak: <code>${peak:,.2f}</code>"
        )
        self._send(text)

    def notify_circuit_breaker(self, reason: str, balance: float):
        text = (
            f"\U0001f6a8 <b>CIRCUIT BREAKER TRIGGERED</b>\n\n"
            f"{reason}\n"
            f"Balance: <code>${balance:,.2f}</code>\n\n"
            f"Trading halted. Manual review required."
        )
        self._send(text)

    def notify_startup(self, balance: float, symbols: list[str], testnet: bool):
        mode = "TESTNET" if testnet else "LIVE"
        text = (
            f"\U0001f680 <b>Bot Started</b> ({mode})\n\n"
            f"Balance: <code>${balance:,.2f}</code>\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Strategy: Trend EMA 100d/200d + MR"
        )
        self._send(text)

    def notify_shutdown(self, balance: float):
        text = (
            f"\U0001f6d1 <b>Bot Stopped</b>\n\n"
            f"Final balance: <code>${balance:,.2f}</code>"
        )
        self._send(text)

    def notify_error(self, error: str):
        text = (
            f"\U00002757 <b>Error</b>\n\n"
            f"<code>{error[:500]}</code>"
        )
        self._send(text)
