"""Main entry point — runs the trading loop."""

import sys
import time
import signal as os_signal
import logging
from datetime import datetime, timezone

from config.settings import load_config
from src.exchange import Exchange
from src.data_manager import DataManager
from src.strategy import Strategy, Signal
from src.risk_manager import RiskManager
from src.order_manager import OrderManager
from src.logger import TradeLogger
from src.sentiment import SentimentAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log"),
    ],
)
log = logging.getLogger("main")

RUNNING = True


def _shutdown(signum, _frame):
    global RUNNING
    log.info("Shutdown signal received — stopping gracefully…")
    RUNNING = False


def main():
    global RUNNING
    os_signal.signal(os_signal.SIGINT, _shutdown)
    os_signal.signal(os_signal.SIGTERM, _shutdown)

    log.info("=" * 60)
    log.info("  TRADING BOT — starting up")
    log.info("=" * 60)

    # ── bootstrap ────────────────────────────────────────────────
    config = load_config()

    if not config.exchange.api_key or not config.exchange.api_secret:
        log.error("API_KEY / API_SECRET missing in .env — aborting")
        sys.exit(1)

    trade_logger = TradeLogger()
    exchange = Exchange(config.exchange)
    data_mgr = DataManager(exchange, config)
    strategy = Strategy(config)
    risk_mgr = RiskManager(config, trade_logger)
    order_mgr = OrderManager(exchange, config, risk_mgr, trade_logger)
    sentiment = SentimentAnalyzer(config)

    initial_capital = exchange.get_balance("USDT")
    risk_mgr.update_peak(initial_capital)
    trade_logger.set_state("initial_capital", str(initial_capital))

    log.info("Balance: $%.2f USDT", initial_capital)
    if initial_capital < 10:
        log.error("Balance too low ($%.2f). Need >= $10 USDT.", initial_capital)
        sys.exit(1)

    log.info(
        "Symbols: %s | Timeframe: %s | Testnet: %s",
        config.trading.symbols, config.trading.timeframe,
        config.exchange.testnet,
    )
    log.info("-" * 60)

    last_candle_hour = -1
    tick = 30  # seconds between position checks

    # ── main loop ────────────────────────────────────────────────
    while RUNNING:
        try:
            now = datetime.now(timezone.utc)
            new_candle = (
                now.hour != last_candle_hour and now.minute >= 1
            )

            if new_candle:
                last_candle_hour = now.hour
                balance = exchange.get_balance("USDT")
                risk_mgr.update_peak(balance)

                log.info(
                    "── candle %s | balance $%.2f ──",
                    now.strftime("%Y-%m-%d %H:00"), balance,
                )

                init_cap = float(
                    trade_logger.get_state(
                        "initial_capital", str(initial_capital),
                    )
                )
                if not risk_mgr.check_circuit_breakers(balance, init_cap):
                    log.warning("HALTED: %s", risk_mgr.halt_reason)
                    time.sleep(tick)
                    continue

                # fetch sentiment data once per candle
                fng = sentiment.fetch_fear_greed()
                log.info("Sentiment: Fear & Greed = %d", fng)

                # analyse every symbol
                for symbol in config.trading.symbols:
                    try:
                        df = data_mgr.get_analysis(symbol)
                        htf = data_mgr.get_htf_bias(symbol)
                        has_pos = symbol in order_mgr.positions
                        pos_mod = order_mgr.positions[symbol].module if has_pos else ""
                        pos_side = order_mgr.positions[symbol].side if has_pos else "long"
                        sig = strategy.generate_signal(
                            df, has_pos, symbol,
                            position_module=pos_mod, position_side=pos_side,
                            fear_greed=fng, htf_bias=htf,
                        )

                        if sig.signal != Signal.NO_SIGNAL:
                            log.info(
                                "[%s] %s — %s",
                                symbol, sig.signal.value, sig.reason,
                            )
                            order_mgr.process_signal(sig, symbol, balance)

                    except Exception:
                        log.exception("Error analysing %s", symbol)

            # position management every tick
            if order_mgr.positions:
                prices: dict[str, float] = {}
                atrs: dict[str, float] = {}
                for sym in list(order_mgr.positions):
                    try:
                        prices[sym] = exchange.get_ticker(sym)["last"]
                        cached = data_mgr.get_cached(sym)
                        if cached is not None and "atr" in cached.columns:
                            atrs[sym] = cached["atr"].iloc[-1]
                    except Exception:
                        log.exception("Tick error for %s", sym)
                order_mgr.manage_positions(prices, atrs, new_candle=new_candle)

            time.sleep(tick)

        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("Unexpected error in main loop")
            time.sleep(60)

    # ── cleanup ──────────────────────────────────────────────────
    log.info("Shutting down…")
    trade_logger.close()
    log.info("Goodbye.")


if __name__ == "__main__":
    main()
