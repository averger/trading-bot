"""Exchange wrapper — unified interface over ccxt with retry logic."""

import time
import logging
import functools

import ccxt
import pandas as pd

from config.settings import ExchangeConfig

log = logging.getLogger(__name__)


def retry(max_attempts: int = 3, base_delay: float = 2.0):
    """Exponential-backoff retry for transient network / exchange errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable,
                        ccxt.RequestTimeout) as exc:
                    last_exc = exc
                    delay = base_delay * (2 ** attempt)
                    log.warning(
                        "%s attempt %d/%d failed (%s) — retry in %.1fs",
                        func.__name__, attempt + 1, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
                except ccxt.ExchangeError:
                    raise  # business errors — don't retry
            raise last_exc
        return wrapper
    return decorator


class Exchange:
    """Thin wrapper around a ccxt exchange instance."""

    def __init__(self, config: ExchangeConfig):
        exchange_cls = getattr(ccxt, config.name)
        default_type = "swap" if config.market_type == "futures" else "spot"
        opts: dict = {
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        }
        self._ex = exchange_cls(opts)

        if config.testnet:
            self._ex.set_sandbox_mode(True)
            log.info("TESTNET mode enabled for %s", config.name)

        self._ex.load_markets()
        self.name = config.name
        self._market_type = config.market_type
        self._leverage = config.leverage
        log.info(
            "Exchange connected: %s (%d markets, type=%s)",
            config.name, len(self._ex.markets), default_type,
        )

    # -- market data ----------------------------------------------------

    @retry()
    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100,
    ) -> pd.DataFrame:
        raw = self._ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df

    @retry()
    def get_ticker(self, symbol: str) -> dict:
        return self._ex.fetch_ticker(symbol)

    # -- account --------------------------------------------------------

    @retry()
    def get_balance(self, currency: str = "USDT") -> float:
        bal = self._ex.fetch_balance()
        return float(bal.get("free", {}).get(currency, 0))

    # -- orders ---------------------------------------------------------

    @retry()
    def market_buy(self, symbol: str, amount: float) -> dict:
        log.info("MARKET BUY  %s  %.8f %s", symbol, amount, symbol.split("/")[0])
        order = self._ex.create_market_buy_order(symbol, amount)
        log.info(
            "  filled id=%s avg=%.2f cost=%.2f",
            order["id"], order.get("average", 0), order.get("cost", 0),
        )
        return order

    @retry()
    def market_sell(self, symbol: str, amount: float) -> dict:
        log.info("MARKET SELL %s  %.8f %s", symbol, amount, symbol.split("/")[0])
        order = self._ex.create_market_sell_order(symbol, amount)
        log.info(
            "  filled id=%s avg=%.2f cost=%.2f",
            order["id"], order.get("average", 0), order.get("cost", 0),
        )
        return order

    # -- futures / short support ----------------------------------------

    @retry()
    def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for a futures symbol."""
        if self._market_type != "futures":
            return
        self._ex.set_leverage(leverage, symbol)
        log.info("Set leverage %dx for %s", leverage, symbol)

    @retry()
    def market_short_open(self, symbol: str, amount: float) -> dict:
        """Open short position (sell to open on futures)."""
        if self._market_type != "futures":
            raise ValueError("Short selling requires futures market type")
        log.info("MARKET SHORT OPEN %s  %.8f", symbol, amount)
        order = self._ex.create_market_sell_order(symbol, amount)
        log.info(
            "  filled id=%s avg=%.2f cost=%.2f",
            order["id"], order.get("average", 0), order.get("cost", 0),
        )
        return order

    @retry()
    def market_short_close(self, symbol: str, amount: float) -> dict:
        """Close short position (buy to close on futures)."""
        log.info("MARKET SHORT CLOSE %s  %.8f", symbol, amount)
        order = self._ex.create_market_buy_order(symbol, amount)
        log.info(
            "  filled id=%s avg=%.2f cost=%.2f",
            order["id"], order.get("average", 0), order.get("cost", 0),
        )
        return order

    # -- market info helpers --------------------------------------------

    def get_min_order_amount(self, symbol: str) -> float:
        mkt = self._ex.market(symbol)
        return float(mkt.get("limits", {}).get("amount", {}).get("min", 0))

    def get_min_order_cost(self, symbol: str) -> float:
        mkt = self._ex.market(symbol)
        return float(mkt.get("limits", {}).get("cost", {}).get("min", 0))

    def get_amount_precision(self, symbol: str) -> int:
        mkt = self._ex.market(symbol)
        return int(mkt.get("precision", {}).get("amount", 6))
