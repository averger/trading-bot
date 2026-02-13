# Crypto Trading Bot

BTC trend-following bot that goes **long on golden cross** and **short on death cross** using EMA 100d/200d crossovers. Always positioned on the right side of the trend.

## Backtest Results (3 years, BTC/USDT, $100 capital)

| Metric | Bot | Buy & Hold |
|--------|-----|------------|
| **Return** | **+285%** | +212% |
| Max Drawdown | 36% | ~40%+ |
| Sharpe Ratio | 1.32 | ~0.8 |
| Market Exposure | 99.5% | 100% |
| Trades | 6 | 1 |
| Win Rate | 83% | - |

Period: Feb 2023 - Feb 2026. Futures 1x (no leverage), long + short.

## How it works

The core idea is simple: **detect the trend direction with EMA crossovers, and be positioned accordingly**.

```
Price
  ^
  |    LONG 95%              LONG 95%
  |   ┌────────────┐        ┌──────
  |  /              \      /
  | /    golden      \    / golden
  |/     cross        \  /  cross
  |                    \/
  |               death cross
  |              SHORT 95%
  +──────────────────────────────────> Time
       uptrend    downtrend   uptrend
```

### Signal flow

```
Every 1h candle:
  ├─ Compute indicators (EMA 100d, EMA 200d, RSI, BB, ATR)
  ├─ If has position → check exit signals
  │   ├─ Trend long  → exit on death cross (EMA fast < slow)
  │   ├─ Trend short → exit on golden cross (EMA fast > slow)
  │   └─ MR position → exit on RSI/BB target or stop-loss
  └─ If no position → check entry signals
      ├─ EMA fast > slow for 48h → LONG ENTRY (trend)
      ├─ EMA fast < slow for 48h → SHORT ENTRY (trend)
      └─ RSI < 33 + BB lower     → LONG ENTRY (mean reversion, downtrends only)
```

### Entry logic in detail

The strategy uses two EMA (Exponential Moving Average) crossovers computed on 1h candles:

- **EMA fast** = 2400 periods (~100 days)
- **EMA slow** = 4800 periods (~200 days)

When EMA fast crosses above EMA slow (**golden cross**), it signals a bullish trend. When it crosses below (**death cross**), bearish. These are slow-moving signals — on BTC, they trigger 1-2 times per year.

**Confirmation delay**: the crossover must persist for 48 consecutive candles (2 days) before triggering an entry. This filters out ephemeral crossovers that reverse within hours.

```python
# strategy.py — _trend_entry()
if ema_fast > ema_slow:
    self._trend_cross_count += 1       # count consecutive golden cross candles
    self._trend_short_count = 0        # reset death cross counter
    if self._trend_cross_count >= 48:  # confirmed golden cross
        → LONG ENTRY
else:
    self._trend_short_count += 1       # count consecutive death cross candles
    self._trend_cross_count = 0        # reset golden cross counter
    if self._trend_short_count >= 48:  # confirmed death cross
        → SHORT ENTRY
```

### Position sizing

Trend positions use **capital allocation** (not risk-based sizing):

```
Trend long:  size = (balance * 0.95) / price   → 95% of capital
Trend short: size = (balance * 0.95) / price   → 95% of capital
MR position: size = (balance * 0.03) / |entry - stop|  → risk-based, ~3% at risk
```

Trend positions are large because the signal has high conviction (100d/200d crossover filters out noise). MR positions are small because they're counter-trend bets.

### Exit logic

**Trend long**: exits only on death cross. No trailing stop, no time stop, no partial take-profit. The trade holds for months/years — the May 2023 trade held for 916 days (+240%).

**Trend short**: exits on golden cross OR trailing stop. Shorts get a trailing stop (2x ATR after +3% gain) because downtrends are choppier than uptrends — this locks in profits during bounces.

**Stop-loss**: 30% from entry for both long and short. This is crash protection only — it should almost never trigger on a confirmed crossover.

### Mean reversion (supplement)

During downtrends (when the bot is short or waiting for confirmation), mean reversion picks up short-term bounces:

- **Entry**: RSI < 33 AND price <= Bollinger lower band AND volume spike
- **Exit**: RSI > 65, or price reaches BB middle + RSI > 50
- **Stop**: 1.5x ATR, trailing at 2x ATR after +3%

MR only fires when EMA fast < EMA slow (downtrend). It's disabled during uptrends to avoid interfering with the trend position.

## Architecture

```
config/
  settings.py          # All tunable parameters (dataclasses)
src/
  strategy.py          # Signal generation (trend long/short + MR)
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
  run_btc_diag.py      # BTC diagnostic + trade log
  run_multi_backtest.py # Multi-pair grid search + IS/OOS validation
  compare_scenarios.py  # Leverage/short scenario comparison
tests/                 # 46 tests (pytest)
```

### Backtester details

The backtester (`backtester.py`) does a walk-forward simulation:

1. Computes all indicators on the full dataframe
2. Iterates candle by candle from warmup onwards
3. For each candle: manage open position (stops, trailing, exits) → check signals → open new position
4. Tracks `balance` (realized P&L) + `_unrealized()` (floating P&L) for equity curve
5. Applies slippage (0.05%), fees (0.10% per side), and funding rate (0.01% per 8h for futures)

**Partial P&L accounting**: when a partial sell/rebuy happens mid-trade (RSI hedging, partial TP), the realized profit is added to `balance` immediately. At trade close, only the remaining position's P&L is added to `balance` to avoid double-counting. The trade record includes total P&L (remaining + partials) for reporting.

## Key Parameters

```python
# config/settings.py
# -- Exchange --
market_type = "futures"       # futures required for shorts
leverage = 1                  # no leverage

# -- Trend following --
trend_ema_fast = 2400         # ~100 days on 1h candles
trend_ema_slow = 4800         # ~200 days on 1h candles
trend_confirm_candles = 48    # crossover must hold 48h (2 days)
trend_short_enabled = True    # short on death cross
trend_alloc_pct = 0.95        # 95% capital for trend longs
trend_short_alloc_pct = 0.95  # 95% capital for trend shorts
trend_sl_pct = 0.30           # 30% stop (crash protection)

# -- Mean reversion --
rsi_oversold = 33             # entry threshold
rsi_overbought = 65           # exit threshold
mean_reversion_sl_atr = 1.5   # stop = 1.5x ATR
cooldown_candles = 12         # min 12h between MR trades
```

## Scenario comparison (3yr BTC backtest)

| Scenario | Return | vs B&H | Max DD | Sharpe |
|----------|--------|--------|--------|--------|
| Spot 1x long only | +217% | +5% | 34% | 1.19 |
| **Futures 1x long+short 95%** | **+285%** | **+73%** | **36%** | **1.32** |
| Futures 2x long only | +269% | +57% | 48% | 1.06 |
| Futures 2x long+short 50% | +382% | +170% | 48% | 1.20 |
| Futures 3x long+short 50% | +567% | +355% | 57% | 1.27 |

The 1x long+short config is the default — best risk-adjusted return (highest Sharpe).

## What was tested and rejected

- **Fast EMAs (20d/50d)**: too many whipsaws, 28 stop-outs, -8% on BTC
- **DCA progressive entry**: averaging into position at higher prices reduces returns
- **RSI hedging (sell 50% at RSI>80, rebuy at RSI<50)**: sells too early in strong rallies
- **Mean reversion only**: max 3-15% exposure, can't beat buy & hold
- **Leverage >1x**: amplifies returns but max DD jumps to 47-57%
- **Altcoins (SOL, ETH, etc.)**: consistently lose with this strategy

## Setup

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: API_KEY, API_SECRET, MARKET_TYPE=futures

# Tests
python -m pytest tests/ -v

# Backtest
python -m backtest.run_btc_diag

# Compare scenarios (leverage, shorts)
python -m backtest.compare_scenarios

# Live trading (start with testnet!)
python -m src.main
```

## Dependencies

- Python 3.11+
- ccxt (exchange connectivity)
- pandas / numpy (data processing)
- python-dotenv (configuration)
