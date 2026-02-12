# Crypto Trading Bot

BTC-specialized trading bot combining **trend following** (EMA 100d/200d crossover) with **mean reversion** (RSI + Bollinger Bands) during downtrends. Spot only, long only, 1x leverage.

## Backtest Results (3 years, BTC/USDT, $100 capital)

| Metric | Bot | Buy & Hold |
|--------|-----|------------|
| **Return** | **+223%** | +203% |
| Max Drawdown | 34.4% | ~40%+ |
| Sharpe Ratio | 1.21 | ~0.8 |
| Market Exposure | 92.4% | 100% |
| Trades | 2 | 1 |

Period: Feb 2023 - Feb 2026

### How it beats buy & hold

The bot enters on the EMA golden cross (100d > 200d) with 95% of capital, rides the full uptrend, and **exits on the death cross before major crashes**. During downtrends, mean reversion captures short-term bounces.

Key trade: entered May 2023 at $26,273, exited Nov 2025 at $91,155 (+246%) via death cross — avoiding the subsequent decline to $66k.

## Architecture

```
config/
  settings.py          # All tunable parameters (dataclasses)
src/
  strategy.py          # Signal generation (trend + mean reversion)
  indicators.py        # RSI, Bollinger Bands, ATR, EMA, MACD, ADX
  sentiment.py         # Fear & Greed Index + Funding Rate filters
  risk_manager.py      # Position sizing, circuit breakers, drawdown limits
  order_manager.py     # Order execution, trailing stops, partial TP
  exchange.py          # Exchange abstraction (Bybit via CCXT)
  data_manager.py      # Candle fetching + indicator computation
  logger.py            # Trade logging to SQLite
  main.py              # Live trading loop
backtest/
  backtester.py        # Walk-forward backtester with slippage + fees
  run_multi_backtest.py # Multi-pair backtest with optimizer
  run_btc_diag.py      # BTC parameter sweep + trade diagnostics
tests/                 # 43 tests (pytest)
```

## Strategy

### Trend Following (primary)

- **Entry**: EMA(100d) crosses above EMA(200d) — golden cross
- **Exit**: EMA(100d) crosses below EMA(200d) — death cross
- **Sizing**: 95% of capital
- **Stop**: 30% below entry (crash protection only)
- No trailing stop, no time stop, no partial TP — holds for months/years

### Mean Reversion (supplement, downtrends only)

- **Entry**: RSI < 33 + price below Bollinger lower band + volume spike
- **Exit**: RSI > 65, or price reaches BB middle + RSI > 50
- **Sizing**: Risk-based (3% of equity per trade)
- **Stop**: 1.5x ATR, trailing at 2x ATR after +3%
- Partial TP: 50% at 1.5x risk distance
- Only fires when trend EMAs show downtrend (EMA fast < slow)

## Risk Management

| Parameter | Trend | Mean Reversion |
|-----------|-------|----------------|
| Position size | 95% of capital | 3% risk-based |
| Stop-loss | 30% fixed | 1.5x ATR |
| Trailing stop | None | 2x ATR after +3% |
| Partial TP | None | 50% at 1.5x risk |
| Time stop | None | 168h (1 week) |
| Drawdown halve | 15% | 15% |
| Drawdown stop | 25% | 25% |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Bybit API key (TRADE permission only, no withdrawal)

# Run tests
python -m pytest tests/ -v

# Backtest (3 years, all pairs)
python -m backtest.run_multi_backtest --days 1095 --capital 100

# BTC diagnostic + parameter sweep
python -m backtest.run_btc_diag

# Live trading (start with testnet!)
python -m src.main
```

## Key Parameters

```python
# config/settings.py — trend following
trend_ema_fast = 2400       # ~100 days on 1h
trend_ema_slow = 4800       # ~200 days on 1h
trend_alloc_pct = 0.95      # 95% capital allocation
trend_sl_pct = 0.30         # 30% crash stop

# config/settings.py — mean reversion
rsi_oversold = 33           # entry threshold
rsi_overbought = 65         # exit threshold
mean_reversion_sl_atr = 1.5 # stop = 1.5x ATR
cooldown_candles = 12       # min 12h between trades
```

## Dependencies

- Python 3.11+
- ccxt (exchange connectivity)
- pandas / numpy (data processing)
- python-dotenv (configuration)
