"""Quick test to verify Telegram bot is working.

Usage:
    python scripts/test_telegram.py

Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_config
from src.notifier import Notifier


def main():
    config = load_config()
    n = Notifier(config)

    if not n.enabled:
        print("Telegram not configured!")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        print()
        print("Steps:")
        print("  1. Open Telegram, search @BotFather")
        print("  2. Send /newbot, follow prompts, copy the token")
        print("  3. Search @userinfobot, send /start, copy your chat ID")
        print("  4. Add to .env:")
        print("     TELEGRAM_BOT_TOKEN=123456:ABC-DEF...")
        print("     TELEGRAM_CHAT_ID=987654321")
        sys.exit(1)

    print(f"Token: {config.notifier.telegram_token[:10]}...")
    print(f"Chat ID: {config.notifier.telegram_chat_id}")
    print()

    ok = n._send("Trading Bot - test message. If you see this, Telegram is working!")
    if ok:
        print("Message sent! Check your Telegram.")
    else:
        print("Failed to send. Check token and chat ID.")
        sys.exit(1)

    # Test a trade alert
    n.notify_startup(1000.0, ["BTC/USDT"], testnet=True)
    print("Startup notification sent!")

    n.notify_entry("BTC/USDT", "long", "trend", 97000, 0.0098, 67900, 1000)
    print("Trade entry notification sent!")

    n.notify_daily_summary(1000, 1000, {}, 0.0, 1000)
    print("Daily summary sent!")

    print("\nAll notifications sent successfully!")


if __name__ == "__main__":
    main()
