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
    leverage: int = 2               # 2x leverage on futures


@dataclass
class TradingConfig:
    symbols: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "AVAX/USDT",
    ])
    timeframe: str = "1h"
    candle_limit: int = 100  # candles to fetch for indicator warmup


@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    atr_period: int = 14
    ema_long: int = 720         # macro trend filter (720h ~ 30 days)
    volume_ma_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    tf_ema_fast: int = 50          # trend-follow uses long EMAs
    tf_ema_slow: int = 200         # EMA(50)/EMA(200) = only major trend changes


@dataclass
class StrategyConfig:
    # Regime detection (ADX thresholds)
    adx_ranging_threshold: float = 20.0
    adx_trending_threshold: float = 28.0
    # Mean reversion
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_exit: float = 50.0
    volume_spike_multiplier: float = 1.0
    # Momentum breakout
    rsi_momentum_min: float = 60.0
    rsi_momentum_max: float = 75.0
    # Anti-noise filters
    cooldown_candles: int = 24           # min 24h between trades per symbol
    max_atr_pct: float = 0.06           # skip when ATR/price > 6%
    momentum_volume_min: float = 1.2    # momentum needs above-average volume
    # Momentum module (empty = all allowed, ["DISABLED"] = none)
    momentum_enabled: bool = False
    momentum_symbols: list = field(default_factory=list)
    # Trend-following module
    trend_follow_adx_min: float = 35.0  # very strong trends only
    trend_follow_enabled: bool = False
    # Volatility-adaptive cooldown
    cooldown_vol_threshold: float = 0.03  # when ATR/price > 3%, scale cooldown up
    # Higher-timeframe filter
    htf_enabled: bool = True
    htf_timeframe: str = "4h"           # higher timeframe for trend confirmation
    htf_ema_fast: int = 9               # ~1.5 days on 4h
    htf_ema_slow: int = 21              # ~3.5 days on 4h
    # Range breakout module
    breakout_enabled: bool = False
    breakout_lookback: int = 336        # 336h = 14 days
    breakout_volume_min: float = 1.0    # average volume enough (BTC has steady volume)
    breakout_sl_atr: float = 2.5        # wider stop for breakout trades
    breakout_max_range_pct: float = 0.20  # range must be < 20% of price (consolidation)


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.04        # 4% of equity
    max_concurrent_positions: int = 4
    max_daily_loss: float = 0.05             # 5% -> halt 24h
    max_drawdown_reduce: float = 0.15        # 15% -> halve position size
    max_drawdown_stop: float = 0.25          # 25% -> full shutdown
    mean_reversion_sl_atr: float = 1.5       # stop = 1.5 x ATR
    momentum_sl_atr: float = 1.5             # stop = 1.5 x ATR (long)
    momentum_short_sl_atr: float = 1.5       # same as long side (wider hurts globally)
    trend_follow_sl_atr: float = 2.0         # wider stop for trend trades
    trailing_activation_pct: float = 0.008   # activate after +0.8%
    trailing_atr_multiplier: float = 1.0     # trail at 1 x ATR
    time_stop_candles: int = 12              # exit if flat after 12 candles
    time_stop_min_move: float = 0.005        # minimum 0.5% move
    slippage_pct: float = 0.0005             # 0.05% assumed slippage
    fee_pct: float = 0.001                   # 0.10% per side
    funding_rate_8h: float = 0.0001          # 0.01% per 8h (futures only)
    # Partial take-profit
    partial_tp_enabled: bool = True
    partial_tp_ratio: float = 2.0            # TP at 2x risk distance
    partial_tp_size: float = 0.5             # close 50% at TP


@dataclass
class SentimentConfig:
    fear_greed_enabled: bool = True
    extreme_fear_threshold: int = 25   # below = block shorts (market bottoming)
    extreme_greed_threshold: int = 75  # above = block longs (euphoria)
    funding_rate_enabled: bool = True
    funding_long_crowded: float = 0.01   # >0.01% = too many longs
    funding_short_crowded: float = -0.01 # <-0.01% = too many shorts


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

    symbols_str = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT")
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
