"""Bot configuration — all tunable parameters in one place."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class ExchangeConfig:
    name: str = "bybit"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    market_type: str = "futures"    # "spot" or "futures"
    leverage: int = 1               # 1x (no leverage)


@dataclass
class TradingConfig:
    symbols: list = field(default_factory=lambda: [
        "BTC/USDT",
        "SOL/USDT",
    ])
    timeframe: str = "1h"
    candle_limit: int = 100  # candles to fetch for indicator warmup


@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ema_short_filter: int = 50      # medium-term trend filter for shorts
    ema_long_filter: int = 200      # long-term trend filter for longs
    volume_ma_period: int = 20
    trend_ema_fast: int = 2400      # ~100 days on 1h
    trend_ema_slow: int = 4800      # ~200 days on 1h


@dataclass
class StrategyConfig:
    # Mean reversion thresholds
    rsi_oversold: float = 33.0
    rsi_overbought: float = 65.0
    rsi_exit: float = 50.0
    volume_spike_multiplier: float = 1.0
    # Anti-noise
    cooldown_candles: int = 12
    max_atr_pct: float = 0.06
    # Trend following
    trend_enabled: bool = True
    trend_confirm_candles: int = 48     # golden cross must hold 48h (2 days)
    trend_short_enabled: bool = True    # short on death cross (requires futures)
    # Trend hedging (partial sell/rebuy on RSI) — disabled by default
    trend_rsi_reduce: float = 95.0      # sell 50% when RSI > 95 (effectively off)
    trend_rsi_rebuy: float = 50.0       # re-buy when RSI drops < 50
    trend_reduce_pct: float = 0.50      # sell 50% of position


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.03        # 3% of equity
    max_concurrent_positions: int = 4
    max_daily_loss: float = 0.05             # 5% -> halt 24h
    max_drawdown_reduce: float = 0.15        # 15% -> halve position size
    max_drawdown_stop: float = 0.25          # 25% -> full shutdown
    mean_reversion_sl_atr: float = 1.5       # stop = 1.5 x ATR
    trailing_activation_pct: float = 0.03    # activate after +3%
    trailing_atr_multiplier: float = 2.0     # trail at 2 x ATR
    time_stop_candles: int = 168             # exit if flat after 1 week
    time_stop_min_move: float = 0.02         # minimum 2% move
    slippage_pct: float = 0.0005             # 0.05% assumed slippage
    fee_pct: float = 0.001                   # 0.10% per side
    funding_rate_8h: float = 0.0001          # 0.01% per 8h (futures only)
    # Partial take-profit
    partial_tp_enabled: bool = True          # take profit at 1.5x risk
    partial_tp_ratio: float = 1.5            # TP at 1.5x risk distance
    partial_tp_size: float = 0.5             # close 50% at TP
    capital_alloc_pct: float = 0.0            # 0 = use per-module sizing
    # Trend following
    trend_alloc_pct: float = 0.95            # 95% allocation for trend trades
    trend_short_alloc_pct: float = 0.95      # 95% allocation for trend shorts
    trend_sl_pct: float = 0.30               # 30% stop for trend (crash protection only)
    # DCA (gradual entry into trend positions) — disabled by default
    dca_tranches: int = 1                    # 1 = disabled (enter all at once)
    dca_interval_candles: int = 336          # 2 weeks between entries (if enabled)


@dataclass
class SentimentConfig:
    fear_greed_enabled: bool = False
    extreme_fear_threshold: int = 25
    extreme_greed_threshold: int = 75
    funding_rate_enabled: bool = True
    funding_long_crowded: float = 0.01
    funding_short_crowded: float = -0.01


@dataclass
class Config:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)


def load_config() -> Config:
    """Load configuration from .env + defaults."""
    load_dotenv()

    symbols_str = os.getenv("SYMBOLS", "BTC/USDT")
    symbols = [s.strip() for s in symbols_str.split(",")]

    return Config(
        exchange=ExchangeConfig(
            name=os.getenv("EXCHANGE_NAME", "bybit"),
            api_key=os.getenv("API_KEY", ""),
            api_secret=os.getenv("API_SECRET", ""),
            testnet=os.getenv("TESTNET", "true").lower() == "true",
            market_type=os.getenv("MARKET_TYPE", "spot"),
            leverage=int(os.getenv("LEVERAGE", "1")),
        ),
        trading=TradingConfig(
            symbols=symbols,
            timeframe=os.getenv("TIMEFRAME", "1h"),
        ),
    )
