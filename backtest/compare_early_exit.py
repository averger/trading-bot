"""Compare early exit signals to reduce drawdown on trend positions.

The main loss comes from the EMA death cross lag — by the time it fires,
price has already dropped 15-20% from peak. Can we detect the reversal earlier?

Tests:
  0. Baseline (exit on death cross only)
  1. EMA gap convergence (exit when gap < X%)
  2. Price below EMA fast for N candles
  3. MACD histogram divergence
  4. Volume divergence (price up, vol down)
  5. RSI momentum shift (RSI < 45 for N candles)
  6. Combined: best signals together
"""

import logging
from datetime import datetime, timezone, timedelta

import ccxt
import numpy as np
import pandas as pd

from config.settings import Config
from backtest.backtester import Backtester, BTTrade, BacktestResult
from src.indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_atr,
    compute_ema,
    compute_volume_ma,
    compute_realized_vol_ratio,
    compute_macd,
)
from src.strategy import Strategy, Signal

logging.basicConfig(level=logging.WARNING)


def fetch_btc(days=1095, timeframe="1h"):
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    since = ex.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )
    all_candles = []
    while True:
        batch = ex.fetch_ohlcv("BTC/USDT", timeframe, since=since, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    return df


class EarlyExitBacktester(Backtester):
    """Backtester with configurable early exit signals for trend positions."""

    def __init__(self, config, early_exit_mode="none", **params):
        super().__init__(config)
        self.early_exit_mode = early_exit_mode
        self.params = params

    def add_indicators(self, df):
        df = super().add_indicators(df)
        # Add MACD for divergence detection
        df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
        # Volume MA for divergence
        df["vol_ma_long"] = df["volume"].rolling(480).mean()  # 20-day vol avg
        # Price relative to EMA fast
        df["price_vs_ema_fast"] = df["close"] / df["ema_trend_fast"] - 1
        return df

    def run(self, df, initial_capital=100.0, symbol=""):
        df = self.add_indicators(df)
        rc = self.config.risk
        result = BacktestResult(initial_capital=initial_capital)

        balance = initial_capital
        peak = initial_capital
        pos = None
        highest = 0.0
        lowest = float("inf")
        trailing = False
        candles_held = 0
        partial_pnl_accum = 0.0
        dca_count = 0
        dca_last_candle = 0
        trend_reduced = False

        # Early exit tracking
        early_exit_counter = 0

        warmup_vals = [
            self.config.indicators.bb_period,
            self.config.indicators.atr_period,
            self.config.indicators.ema_long_filter,
            self.config.indicators.volume_ma_period,
        ]
        if self.config.strategy.trend_enabled:
            warmup_vals.append(self.config.indicators.trend_ema_fast)
        warmup = max(warmup_vals) + 5

        for i in range(warmup, len(df)):
            cur = df.iloc[i]
            price = cur["close"]
            atr = cur["atr"]

            if pd.isna(atr) or pd.isna(cur["rsi"]):
                result.equity_curve.append(
                    balance + self._unrealized(pos, price),
                )
                continue

            # -- manage open position -----------------------------------
            if pos is not None:
                candles_held += 1
                leverage = self.config.exchange.leverage

                # Funding cost
                if leverage > 1 and candles_held % 8 == 0:
                    funding_cost = pos.size * price * rc.funding_rate_8h
                    partial_pnl_accum -= funding_cost
                    balance -= funding_cost

                if pos.side == "long":
                    if price > highest:
                        highest = price

                    # stop-loss
                    if cur["low"] <= pos.stop_loss:
                        self._close_long(pos, i, pos.stop_loss, "stop_loss", rc)
                        balance += pos.pnl
                        pos.pnl += partial_pnl_accum
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        early_exit_counter = 0
                        result.equity_curve.append(balance)
                        continue

                    # -- EARLY EXIT CHECK for trend longs --
                    if pos.module == "trend":
                        should_exit, reason = self._check_early_exit(
                            df, i, pos, highest, early_exit_counter,
                        )
                        if should_exit:
                            self._close_long(pos, i, price, reason, rc)
                            balance += pos.pnl
                            pos.pnl += partial_pnl_accum
                            result.trades.append(pos)
                            self.strategy.record_trade(symbol)
                            pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                            early_exit_counter = 0
                            result.equity_curve.append(balance)
                            continue
                        early_exit_counter = self._update_early_counter(
                            df, i, pos, highest, early_exit_counter,
                        )

                else:  # SHORT
                    if price < lowest:
                        lowest = price

                    if cur["high"] >= pos.stop_loss:
                        self._close_short(pos, i, pos.stop_loss, "stop_loss", rc)
                        balance += pos.pnl
                        pos.pnl += partial_pnl_accum
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        early_exit_counter = 0
                        result.equity_curve.append(balance)
                        continue

                    # Early exit for trend shorts
                    if pos.module == "trend":
                        should_exit, reason = self._check_early_exit_short(
                            df, i, pos, lowest, early_exit_counter,
                        )
                        if should_exit:
                            self._close_short(pos, i, price, reason, rc)
                            balance += pos.pnl
                            pos.pnl += partial_pnl_accum
                            result.trades.append(pos)
                            self.strategy.record_trade(symbol)
                            pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                            early_exit_counter = 0
                            result.equity_curve.append(balance)
                            continue
                        early_exit_counter = self._update_early_counter_short(
                            df, i, pos, lowest, early_exit_counter,
                        )

                    # trailing for short (MR only)
                    if pos.module != "trend":
                        pct = (pos.entry_price - price) / pos.entry_price
                        if pct >= rc.trailing_activation_pct:
                            trailing = True
                        if trailing and atr > 0:
                            ns = lowest + rc.trailing_atr_multiplier * atr
                            if ns < pos.stop_loss:
                                pos.stop_loss = ns

                # time stop (MR only)
                if pos is not None and pos.module != "trend" and candles_held >= rc.time_stop_candles:
                    move = abs(price - pos.entry_price) / pos.entry_price
                    if move < rc.time_stop_min_move:
                        if pos.side == "long":
                            self._close_long(pos, i, price, "time_stop", rc)
                        else:
                            self._close_short(pos, i, price, "time_stop", rc)
                        balance += pos.pnl
                        pos.pnl += partial_pnl_accum
                        result.trades.append(pos)
                        pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                        early_exit_counter = 0
                        result.equity_curve.append(balance)
                        continue

            # -- signals ------------------------------------------------
            self.strategy.set_candle_index(i)
            window = df.iloc[: i + 1]
            pos_module = pos.module if pos is not None else ""
            pos_side = pos.side if pos is not None else "long"

            sig = self.strategy.generate_signal(
                window, pos is not None, symbol,
                position_module=pos_module, position_side=pos_side,
            )

            # LONG EXIT
            if sig.signal == Signal.LONG_EXIT and pos is not None and pos.side == "long":
                self._close_long(pos, i, price, sig.reason, rc)
                balance += pos.pnl
                pos.pnl += partial_pnl_accum
                result.trades.append(pos)
                self.strategy.record_trade(symbol)
                pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                early_exit_counter = 0

            # SHORT EXIT
            elif sig.signal == Signal.SHORT_EXIT and pos is not None and pos.side == "short":
                self._close_short(pos, i, price, sig.reason, rc)
                balance += pos.pnl
                pos.pnl += partial_pnl_accum
                result.trades.append(pos)
                self.strategy.record_trade(symbol)
                pos, trailing, candles_held, partial_pnl_accum = None, False, 0, 0.0
                early_exit_counter = 0

            # LONG ENTRY
            elif sig.signal == Signal.LONG_ENTRY and pos is None:
                pos = self._open_position(
                    sig, i, price, atr, balance, peak, rc, side="long",
                )
                if pos is not None:
                    highest = pos.entry_price
                    lowest = float("inf")
                    trailing = False
                    candles_held = 0
                    partial_pnl_accum = 0.0
                    early_exit_counter = 0
                    self.strategy.record_trade(symbol)
                    if sig.module == "trend":
                        dca_count = 1
                        dca_last_candle = i

            # SHORT ENTRY
            elif sig.signal == Signal.SHORT_ENTRY and pos is None:
                pos = self._open_position(
                    sig, i, price, atr, balance, peak, rc, side="short",
                )
                if pos is not None:
                    lowest = pos.entry_price
                    highest = 0.0
                    trailing = False
                    candles_held = 0
                    partial_pnl_accum = 0.0
                    early_exit_counter = 0
                    self.strategy.record_trade(symbol)

            if balance > peak:
                peak = balance
            result.equity_curve.append(
                balance + self._unrealized(pos, price),
            )

        # close anything still open
        if pos is not None:
            if pos.side == "long":
                self._close_long(pos, len(df) - 1, df.iloc[-1]["close"], "backtest_end", rc)
            else:
                self._close_short(pos, len(df) - 1, df.iloc[-1]["close"], "backtest_end", rc)
            balance += pos.pnl
            pos.pnl += partial_pnl_accum
            result.trades.append(pos)
            if result.equity_curve:
                result.equity_curve[-1] = balance

        return result

    # -- Early exit logic -----------------------------------------------

    def _check_early_exit(self, df, i, pos, highest, counter):
        """Check if trend long should exit early. Returns (bool, reason)."""
        cur = df.iloc[i]
        mode = self.early_exit_mode

        if mode == "ema_gap":
            # Exit when EMA gap narrows to < threshold %
            threshold = self.params.get("gap_pct", 0.01)
            confirm = self.params.get("confirm", 24)
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if pd.isna(ema_f) or pd.isna(ema_s):
                return False, ""
            gap = (ema_f - ema_s) / ema_s
            if gap < threshold and counter >= confirm:
                return True, f"EARLY EXIT: EMA gap {gap:.3%} < {threshold:.1%} ({counter}h)"
            return False, ""

        elif mode == "price_below_ema":
            # Exit when price stays below EMA fast for N candles
            confirm = self.params.get("confirm", 72)
            ema_f = cur.get("ema_trend_fast", np.nan)
            if pd.isna(ema_f):
                return False, ""
            if cur["close"] < ema_f and counter >= confirm:
                return True, f"EARLY EXIT: price < EMA fast for {counter}h"
            return False, ""

        elif mode == "macd_divergence":
            # Exit when MACD histogram has been declining for N candles
            confirm = self.params.get("confirm", 72)
            if counter >= confirm:
                return True, f"EARLY EXIT: MACD declining {counter}h"
            return False, ""

        elif mode == "volume_divergence":
            # Exit when volume is declining while price is elevated
            confirm = self.params.get("confirm", 168)  # 1 week
            if counter >= confirm:
                return True, f"EARLY EXIT: vol divergence {counter}h"
            return False, ""

        elif mode == "rsi_momentum":
            # Exit when RSI stays below 45 for N candles (bulls losing)
            confirm = self.params.get("confirm", 48)
            rsi_threshold = self.params.get("rsi_threshold", 45)
            if cur["rsi"] < rsi_threshold and counter >= confirm:
                return True, f"EARLY EXIT: RSI<{rsi_threshold} for {counter}h"
            return False, ""

        elif mode == "combined":
            # Multi-signal scoring
            score = 0
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if not pd.isna(ema_f) and not pd.isna(ema_s):
                gap = (ema_f - ema_s) / ema_s
                if gap < 0.02:
                    score += 1
                if gap < 0.01:
                    score += 1
            if not pd.isna(ema_f) and cur["close"] < ema_f:
                score += 1
            macd_h = cur.get("macd_hist", np.nan)
            if not pd.isna(macd_h) and macd_h < 0:
                score += 1
            if cur["rsi"] < 45:
                score += 1
            min_score = self.params.get("min_score", 3)
            confirm = self.params.get("confirm", 48)
            if score >= min_score and counter >= confirm:
                return True, f"EARLY EXIT: score {score}/{5} for {counter}h"
            return False, ""

        return False, ""

    def _update_early_counter(self, df, i, pos, highest, counter):
        """Update the early exit counter for longs."""
        cur = df.iloc[i]
        mode = self.early_exit_mode

        if mode == "ema_gap":
            threshold = self.params.get("gap_pct", 0.01)
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if pd.isna(ema_f) or pd.isna(ema_s):
                return 0
            gap = (ema_f - ema_s) / ema_s
            return counter + 1 if gap < threshold else 0

        elif mode == "price_below_ema":
            ema_f = cur.get("ema_trend_fast", np.nan)
            if pd.isna(ema_f):
                return 0
            return counter + 1 if cur["close"] < ema_f else 0

        elif mode == "macd_divergence":
            macd_h = cur.get("macd_hist", np.nan)
            if pd.isna(macd_h):
                return 0
            # Check if histogram is declining (current < previous)
            if i > 0:
                prev_h = df.iloc[i - 1].get("macd_hist", np.nan)
                if not pd.isna(prev_h) and macd_h < prev_h and macd_h < 0:
                    return counter + 1
            return 0

        elif mode == "volume_divergence":
            vol = cur["volume"]
            vol_ma = cur.get("vol_ma_long", np.nan)
            if pd.isna(vol_ma) or vol_ma == 0:
                return 0
            # Volume below average while in profit
            profit = (cur["close"] - pos.entry_price) / pos.entry_price
            if profit > 0.05 and vol < vol_ma * 0.7:
                return counter + 1
            return 0

        elif mode == "rsi_momentum":
            rsi_threshold = self.params.get("rsi_threshold", 45)
            return counter + 1 if cur["rsi"] < rsi_threshold else 0

        elif mode == "combined":
            score = 0
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if not pd.isna(ema_f) and not pd.isna(ema_s):
                gap = (ema_f - ema_s) / ema_s
                if gap < 0.02:
                    score += 1
            if not pd.isna(ema_f) and cur["close"] < ema_f:
                score += 1
            macd_h = cur.get("macd_hist", np.nan)
            if not pd.isna(macd_h) and macd_h < 0:
                score += 1
            if cur["rsi"] < 45:
                score += 1
            min_score = self.params.get("min_score", 3)
            return counter + 1 if score >= min_score else 0

        return 0

    def _check_early_exit_short(self, df, i, pos, lowest, counter):
        """Check if trend short should exit early (mirror of long)."""
        cur = df.iloc[i]
        mode = self.early_exit_mode

        if mode == "ema_gap":
            threshold = self.params.get("gap_pct", 0.01)
            confirm = self.params.get("confirm", 24)
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if pd.isna(ema_f) or pd.isna(ema_s):
                return False, ""
            gap = (ema_s - ema_f) / ema_f  # inverted for short
            if gap < threshold and counter >= confirm:
                return True, f"EARLY EXIT SHORT: EMA gap {gap:.3%} ({counter}h)"
            return False, ""

        elif mode == "price_below_ema":
            confirm = self.params.get("confirm", 72)
            ema_f = cur.get("ema_trend_fast", np.nan)
            if pd.isna(ema_f):
                return False, ""
            if cur["close"] > ema_f and counter >= confirm:
                return True, f"EARLY EXIT SHORT: price > EMA fast ({counter}h)"
            return False, ""

        elif mode == "rsi_momentum":
            confirm = self.params.get("confirm", 48)
            rsi_threshold = self.params.get("rsi_threshold", 45)
            if cur["rsi"] > (100 - rsi_threshold) and counter >= confirm:
                return True, f"EARLY EXIT SHORT: RSI>{100-rsi_threshold} ({counter}h)"
            return False, ""

        elif mode == "combined":
            score = 0
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if not pd.isna(ema_f) and not pd.isna(ema_s):
                gap = (ema_s - ema_f) / ema_f
                if gap < 0.02:
                    score += 1
            if not pd.isna(ema_f) and cur["close"] > ema_f:
                score += 1
            macd_h = cur.get("macd_hist", np.nan)
            if not pd.isna(macd_h) and macd_h > 0:
                score += 1
            if cur["rsi"] > 55:
                score += 1
            min_score = self.params.get("min_score", 3)
            confirm = self.params.get("confirm", 48)
            if score >= min_score and counter >= confirm:
                return True, f"EARLY EXIT SHORT: score {score} ({counter}h)"
            return False, ""

        return False, ""

    def _update_early_counter_short(self, df, i, pos, lowest, counter):
        """Update the early exit counter for shorts."""
        cur = df.iloc[i]
        mode = self.early_exit_mode

        if mode == "ema_gap":
            threshold = self.params.get("gap_pct", 0.01)
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if pd.isna(ema_f) or pd.isna(ema_s):
                return 0
            gap = (ema_s - ema_f) / ema_f
            return counter + 1 if gap < threshold else 0

        elif mode == "price_below_ema":
            ema_f = cur.get("ema_trend_fast", np.nan)
            if pd.isna(ema_f):
                return 0
            return counter + 1 if cur["close"] > ema_f else 0

        elif mode == "rsi_momentum":
            rsi_threshold = self.params.get("rsi_threshold", 45)
            return counter + 1 if cur["rsi"] > (100 - rsi_threshold) else 0

        elif mode == "combined":
            score = 0
            ema_f = cur.get("ema_trend_fast", np.nan)
            ema_s = cur.get("ema_trend_slow", np.nan)
            if not pd.isna(ema_f) and not pd.isna(ema_s):
                gap = (ema_s - ema_f) / ema_f
                if gap < 0.02:
                    score += 1
            if not pd.isna(ema_f) and cur["close"] > ema_f:
                score += 1
            macd_h = cur.get("macd_hist", np.nan)
            if not pd.isna(macd_h) and macd_h > 0:
                score += 1
            if cur["rsi"] > 55:
                score += 1
            min_score = self.params.get("min_score", 3)
            return counter + 1 if score >= min_score else 0

        return 0


def run_scenario(df, label, cfg, early_exit_mode="none", **params):
    bt = EarlyExitBacktester(cfg, early_exit_mode=early_exit_mode, **params)
    result = bt.run(df, initial_capital=1000, symbol="BTC/USDT")

    # Find the biggest losing trade's drawdown
    max_trade_loss = 0
    for t in result.trades:
        if t.pnl_pct < max_trade_loss:
            max_trade_loss = t.pnl_pct

    print(f"\n{'-'*65}")
    print(f"  {label}")
    print(f"{'-'*65}")
    print(f"  Return     : {result.total_return_pct:+.1f}%")
    print(f"  Max DD     : {result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe     : {result.sharpe_ratio:.2f}")
    print(f"  Trades     : {result.total_trades}  (W:{result.wins} L:{result.losses})")
    print(f"  Worst trade: {max_trade_loss:+.1%}")
    print(f"  Final      : ${result.equity_curve[-1]:,.2f}")

    for t in result.trades:
        entry_dt = df.index[min(t.entry_idx, len(df) - 1)]
        exit_dt = df.index[min(t.exit_idx, len(df) - 1)]
        hold_d = (t.exit_idx - t.entry_idx) / 24
        print(
            f"    {t.side:5s} {entry_dt.strftime('%Y-%m-%d')} -> "
            f"{exit_dt.strftime('%Y-%m-%d')} ({hold_d:.0f}d)  "
            f"${t.entry_price:.0f}->${t.exit_price:.0f}  "
            f"PnL=${t.pnl:+.2f} ({t.pnl_pct:+.1%})  [{t.exit_reason}]"
        )

    return result


def main():
    print("Fetching BTC 3yr data...")
    df = fetch_btc()
    hold_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"BTC: {len(df)} candles, Buy & Hold: +{hold_pct:.1f}%\n")

    base_cfg = Config()
    base_cfg.exchange.market_type = "futures"
    base_cfg.strategy.trend_short_enabled = True
    base_cfg.risk.trend_short_alloc_pct = 0.95

    # 0. Baseline
    run_scenario(df, "0. BASELINE (death cross exit only)", base_cfg)

    # 1a. EMA gap < 1%, confirm 24h
    run_scenario(df, "1a. EMA GAP < 1% (24h confirm)", base_cfg,
                 early_exit_mode="ema_gap", gap_pct=0.01, confirm=24)

    # 1b. EMA gap < 2%, confirm 48h
    run_scenario(df, "1b. EMA GAP < 2% (48h confirm)", base_cfg,
                 early_exit_mode="ema_gap", gap_pct=0.02, confirm=48)

    # 1c. EMA gap < 0.5%, confirm 12h
    run_scenario(df, "1c. EMA GAP < 0.5% (12h confirm)", base_cfg,
                 early_exit_mode="ema_gap", gap_pct=0.005, confirm=12)

    # 2a. Price below EMA fast for 72h (3 days)
    run_scenario(df, "2a. PRICE < EMA FAST (72h confirm)", base_cfg,
                 early_exit_mode="price_below_ema", confirm=72)

    # 2b. Price below EMA fast for 168h (1 week)
    run_scenario(df, "2b. PRICE < EMA FAST (168h confirm)", base_cfg,
                 early_exit_mode="price_below_ema", confirm=168)

    # 3. MACD histogram declining 72h
    run_scenario(df, "3. MACD DIVERGENCE (72h decline)", base_cfg,
                 early_exit_mode="macd_divergence", confirm=72)

    # 4. Volume divergence 1 week
    run_scenario(df, "4. VOLUME DIVERGENCE (168h)", base_cfg,
                 early_exit_mode="volume_divergence", confirm=168)

    # 5a. RSI < 45 for 48h
    run_scenario(df, "5a. RSI MOMENTUM < 45 (48h)", base_cfg,
                 early_exit_mode="rsi_momentum", rsi_threshold=45, confirm=48)

    # 5b. RSI < 40 for 72h
    run_scenario(df, "5b. RSI MOMENTUM < 40 (72h)", base_cfg,
                 early_exit_mode="rsi_momentum", rsi_threshold=40, confirm=72)

    # 6a. Combined (score >= 3, confirm 48h)
    run_scenario(df, "6a. COMBINED (score>=3/5, 48h)", base_cfg,
                 early_exit_mode="combined", min_score=3, confirm=48)

    # 6b. Combined (score >= 4, confirm 24h)
    run_scenario(df, "6b. COMBINED (score>=4/5, 24h)", base_cfg,
                 early_exit_mode="combined", min_score=4, confirm=24)

    print(f"\n{'='*65}")
    print(f"  Buy & Hold : +{hold_pct:.1f}%  (${1000 * (1 + hold_pct/100):,.2f})")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
