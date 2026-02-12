# STEP 2 — Architecture & Deployment Guide

## Project Structure

```
trading-bot/
├── .env.example            # API key template (copy to .env)
├── .gitignore              # Excludes secrets, caches, DBs
├── requirements.txt        # Python dependencies
│
├── config/
│   └── settings.py         # All tunable parameters (dataclasses)
│
├── src/
│   ├── main.py             # Entry point — main trading loop
│   ├── exchange.py         # ccxt wrapper + retry logic
│   ├── data_manager.py     # Candle fetching + indicator pipeline
│   ├── indicators.py       # RSI, BB, ADX, EMA, ATR (pure functions)
│   ├── strategy.py         # Regime detection + signal generation
│   ├── risk_manager.py     # Position sizing + circuit breakers
│   ├── order_manager.py    # Order execution + stop management
│   └── logger.py           # SQLite trade persistence
│
├── backtest/
│   ├── backtester.py       # Walk-forward simulation engine
│   └── run_backtest.py     # CLI runner for backtests
│
├── tests/
│   ├── test_indicators.py  # 10 indicator tests
│   ├── test_strategy.py    # 6 strategy tests
│   └── test_risk_manager.py # 6 risk management tests
│
└── docs/
    ├── STEP1_MARKET_ANALYSIS.md
    └── STEP2_ARCHITECTURE.md
```

## Data Flow

```
Exchange API (Bybit)
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Exchange    │────▶│ DataManager  │────▶│  Indicators  │
│  (ccxt)      │     │ (fetch+cache)│     │  (pure math) │
└─────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Strategy    │──── Regime Detection (ADX)
                    │  (signals)   │──── Mean Reversion (RSI+BB)
                    └──────────────┘──── Momentum (BB+EMA)
                            │
                            ▼
                    ┌──────────────┐     ┌──────────────┐
                    │ RiskManager  │────▶│ OrderManager │
                    │ (sizing/halt)│     │ (exec/stops) │
                    └──────────────┘     └──────────────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │ TradeLogger  │
                                        │  (SQLite)    │
                                        └──────────────┘
```

## Main Loop (30-second tick)

```
every 30 seconds:
  ├── if new hourly candle:
  │     ├── fetch balance
  │     ├── check circuit breakers (daily loss, max drawdown)
  │     ├── for each symbol:
  │     │     ├── fetch 100 candles → compute all indicators
  │     │     ├── detect regime (ranging/trending/transition)
  │     │     ├── generate entry/exit signal
  │     │     └── execute order if signal + risk check pass
  │     └── done
  └── manage open positions:
        ├── fetch current prices
        ├── check hard stop-losses
        ├── update trailing stops
        └── check time stops
```

## Prerequisites

### 1. Python Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Exchange Account (Bybit)

1. Create a Bybit account at bybit.com
2. Go to API Management
3. Create a new API key with these permissions:
   - **Read**: YES
   - **Trade**: YES (Spot only)
   - **Withdraw**: NO ← critical for security
4. Set IP whitelist to your server's IP
5. Save the API key and secret

### 3. Configuration

```bash
copy .env.example .env
# Edit .env with your API key and secret
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Paper Trading (Testnet)

```bash
# Make sure TESTNET=true in .env (default)
python -m src.main
```

### 6. Backtest First!

```bash
# Run a 90-day backtest before going live
python -m backtest.run_backtest --symbol BTC/USDT --days 90 --capital 100

# Try different pairs
python -m backtest.run_backtest --symbol ETH/USDT --days 90 --capital 100
```

### 7. Go Live (when ready)

```bash
# Change TESTNET=false in .env
# Deposit $100 USDT to Bybit spot wallet
python -m src.main
```

## Security Checklist

- [x] API keys in `.env` (never committed)
- [x] `.gitignore` excludes `.env`, `*.db`, `*.log`
- [x] API key has NO withdrawal permission
- [x] IP whitelist on exchange
- [x] No hardcoded secrets anywhere
- [x] Spot-only trading (no leverage/liquidation)
- [x] Circuit breakers halt trading on excessive loss
- [x] All input validated at exchange boundary

## Tunable Parameters (config/settings.py)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_risk_per_trade` | 2% | Max capital risked per trade |
| `max_concurrent_positions` | 2 | Max open positions |
| `max_daily_loss` | 5% | Daily loss → halt 24h |
| `max_drawdown_stop` | 25% | Drawdown → full shutdown |
| `rsi_oversold` | 30 | Mean reversion entry threshold |
| `rsi_overbought` | 70 | Exit threshold |
| `adx_ranging_threshold` | 20 | Below = ranging market |
| `adx_trending_threshold` | 25 | Above = trending market |
| `trailing_activation_pct` | 1% | Profit needed to activate trail |
| `time_stop_candles` | 6 | Max candles before time stop |
