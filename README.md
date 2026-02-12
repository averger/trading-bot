# Crypto Trading Bot

Automated multi-pair cryptocurrency trading bot with mean reversion, momentum breakout, and trend-following strategies. Supports long and short positions with sentiment-based filtering.

## Backtest Results (3 years, $100/pair)

| Pair | Trades | Win Rate | Return | Max DD | Sharpe | Final |
|------|--------|----------|--------|--------|--------|-------|
| BTC/USDT | 1 | 0% | -0.10% | 1.81% | -0.02 | $99.90 |
| ETH/USDT | 10 | 20% | -12.81% | 14.22% | -1.13 | $87.19 |
| SOL/USDT | 26 | 65% | +22.29% | 14.74% | 0.59 | $122.29 |
| XRP/USDT | 37 | 43% | -6.86% | 23.04% | -0.18 | $93.14 |
| DOGE/USDT | 29 | 66% | +106.72% | 10.61% | 1.43 | $206.72 |
| AVAX/USDT | 132 | 42% | -13.10% | 35.30% | -0.19 | $86.90 |
| **Portfolio** | **235** | | **+16.02%** | | | **$696.15** |

Period: Feb 2023 - Feb 2026 | Capital: $600 total ($100/pair x 6 pairs)

## Architecture

```
config/
  settings.py          # All tunable parameters (strategy, risk, sentiment)
src/
  strategy.py          # Signal generation (3 modules + sentiment filter)
  indicators.py        # RSI, Bollinger Bands, ATR, ADX, EMA, MACD
  sentiment.py         # Fear & Greed Index + Funding Rate
  risk_manager.py      # Position sizing, circuit breakers, drawdown limits
  order_manager.py     # Order execution, trailing stops, partial TP
  exchange.py          # Exchange abstraction (Bybit via CCXT)
  data_manager.py      # Candle fetching + indicator computation
  logger.py            # Trade logging to SQLite
  main.py              # Live trading loop
backtest/
  backtester.py        # Walk-forward backtester with slippage + fees
  run_backtest.py      # Single-pair backtest runner
  run_multi_backtest.py # Multi-pair backtest with optimization
tests/
  test_indicators.py   # 16 indicator tests
  test_strategy.py     # 15 strategy + sentiment tests
  test_risk_manager.py # 8 risk management tests
  test_sentiment.py    # 7 sentiment filter tests
```

## Strategy Modules

### 1. Mean Reversion (Ranging markets, ADX < 20)
- **Long**: RSI oversold + price below Bollinger lower band + volume spike
- **Short**: RSI overbought + price above Bollinger upper band + volume spike + bearish macro trend
- Exit: RSI returns to neutral (50) or opposite extreme

### 2. Momentum Breakout (Trending markets, ADX > 28)
- **Long**: Price above Bollinger upper + RSI 55-80 + EMA fast > EMA slow
- **Short**: Price below Bollinger lower + RSI < 45 + MACD negative + bearish macro + volume confirmation
- Exit: RSI extreme or trailing stop

### 3. Trend Following (Any regime, ADX > 25)
- **Long**: EMA golden cross (fast crosses above slow) + MACD histogram positive
- **Short**: EMA death cross + MACD histogram negative + bearish macro trend
- Exit: Trailing stop (2x ATR)

### Sentiment Filter (Fear & Greed Index)
- Extreme Greed (>= 75): blocks new long entries
- Extreme Fear (<= 25): blocks new short entries
- Never blocks exits
- Historical F&G data used in backtesting

## Risk Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| Risk per trade | 4% | Max capital risked per position |
| Max positions | 4 | Concurrent open positions |
| Stop-loss (MR/MOM) | 1.5x ATR | Mean reversion & momentum |
| Stop-loss (TF) | 2.0x ATR | Trend-follow (wider) |
| Trailing stop | 1x ATR | Activates after +0.8% profit |
| Partial TP | 50% at 2x risk | Lock in profits, let rest run |
| Time stop | 12 candles | Exit if flat after 12h |
| Daily loss halt | 5% | Stop trading for 24h |
| Drawdown halve | 15% | Reduce position size by 50% |
| Drawdown stop | 25% | Full shutdown |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Bybit API key (TRADE permission only, no withdrawal)

# Run tests
python -m pytest tests/ -v

# Backtest (2 years, all pairs)
python -m backtest.run_multi_backtest --days 730 --capital 100

# Backtest with optimization
python -m backtest.run_multi_backtest --days 730 --capital 100 --optimize

# Live trading (start with testnet!)
python -m src.main
```

## Key Parameters

```python
# config/settings.py
ema_long = 720              # 30-day macro trend filter
cooldown_candles = 6        # Min candles between trades/symbol
max_atr_pct = 0.06          # Skip when volatility > 6%
momentum_volume_min = 1.2   # Momentum shorts need above-avg volume
fear_greed_threshold = 75/25 # Sentiment entry gates
```

## Dependencies

- Python 3.11+
- ccxt (exchange connectivity)
- pandas / numpy (data processing)
- python-dotenv (configuration)
