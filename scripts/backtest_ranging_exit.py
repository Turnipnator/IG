#!/usr/bin/env python3
"""
Ranging-exit fix backtest.

Live behavior (src/strategy.py:should_close_position) closes a non-MACD-exit
position the moment ADX drops below adx_threshold-10 for a single candle.
Journal data shows this fires too quickly and bled -£99 across 8 trades
(6 of them Gold). Compare three exit variants per non-MACD market:

  current        : 1 candle below threshold (matches live)
  confirmed_3    : 3 consecutive candles below threshold
  confirmed_3_dec: 3 consecutive candles below + ADX declining each candle
  off            : disable the exit entirely (let stop/limit/RSI extreme work)

Markets tested are exactly the live non-MACD strategies (Gold, EUR/USD,
GBP/USD, USD/JPY). Each runs with that market's actual live profile.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    DISABLE_MACD_EXIT,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    TICKER_MAP,
    Backtester,
)

logging.basicConfig(level=logging.WARNING)


GOLD = {
    "ema_fast": 3, "ema_medium": 8, "ema_slow": 21,
    "rsi_period": 7, "rsi_overbought": 85, "rsi_oversold": 15,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 3.0,
}

FOREX = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 30, "atr_period": 14,
    "stop_atr_multiplier": 1.0, "reward_risk_ratio": 2.0,
}

USDJPY = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 80, "rsi_oversold": 20,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}


# (market name, yfinance ticker, profile, candle interval, min_stop in price units)
MARKETS = [
    ("Gold",    "GC=F",       GOLD,   "5m",  25.0),
    ("EUR/USD", "EURUSD=X",   FOREX,  "5m",  0.005),
    ("GBP/USD", "GBPUSD=X",   FOREX,  "5m",  0.005),
    ("USD/JPY", "JPY=X",      USDJPY, "15m", 0.10),
]

# (label, consecutive, require_declining)
VARIANTS = [
    ("current (1 candle)",       1, False),
    ("confirmed 3 candles",      3, False),
    ("confirmed 3 + declining",  3, True),
    ("off (no ranging exit)",    0, False),
]


def patch_market(name: str, ticker: str, min_stop: float, profile: dict):
    TICKER_MAP[name] = ticker
    MIN_STOP_DISTANCE_MAP[name] = min_stop
    MIN_CONFIDENCE_MAP[name] = 0.55
    REWARD_RISK_MAP[name] = profile["reward_risk_ratio"]
    # The ranging exit only fires when MACD exit is disabled — same as live.
    DISABLE_MACD_EXIT.add(name)


def run_one(market: str, profile: dict, interval: str,
            consecutive: int, require_declining: bool, days: int):
    params = DEFAULT_PARAMS.copy()
    params.update(profile)
    params["ranging_exit_consecutive"] = consecutive
    params["ranging_exit_require_declining"] = require_declining
    bt = Backtester(params=params)
    htf = "1h" if interval in ("5m", "15m", "30m") else "1d"
    return bt.run(market, days=days, interval=interval,
                  htf_interval=htf, require_htf_alignment=True)


def count_by_reason(trades):
    counts = {}
    pnls = {}
    for t in trades:
        r = t.exit_reason or "open"
        counts[r] = counts.get(r, 0) + 1
        pnls[r] = pnls.get(r, 0.0) + (t.pnl_percent or 0.0)
    return counts, pnls


def main():
    days = 60

    overall = {}  # variant_label -> [(market, result)]

    for market, ticker, profile, interval, min_stop in MARKETS:
        patch_market(market, ticker, min_stop, profile)

        print()
        print("=" * 110)
        print(f"{market} — {interval} entries, {days}d")
        print("=" * 110)
        print(f"{'Variant':<28} {'Trades':>7} {'WinRate':>8} {'P&L%':>8} "
              f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'Ranging-exits':>14}")
        print("-" * 110)

        for v_label, consecutive, declining in VARIANTS:
            try:
                r = run_one(market, profile, interval, consecutive, declining, days)
                counts, pnls = count_by_reason(r.trades) if hasattr(r, "trades") else ({}, {})
                ranging_n = counts.get("Market turned ranging", 0)
                ranging_pnl = pnls.get("Market turned ranging", 0.0)
                ranging_str = (
                    f"{ranging_n}t {ranging_pnl:+.2f}%"
                    if ranging_n else "—"
                )
                if r.total_trades == 0:
                    print(f"{v_label:<28} {'0':>7} {'—':>8} {'—':>8} {'—':>6} "
                          f"{'—':>8} {'—':>9} {ranging_str:>14}")
                else:
                    print(f"{v_label:<28} {r.total_trades:>7} {r.win_rate:>7.1%} "
                          f"{r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                          f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% {ranging_str:>14}")
                overall.setdefault(v_label, []).append((market, r))
            except Exception as e:
                print(f"{v_label:<28} ERROR: {e}")

    # Aggregate per variant across all markets
    print()
    print("=" * 110)
    print("AGGREGATE — sum of P&L% across all 4 markets (with ≥1 trade)")
    print("=" * 110)
    print(f"{'Variant':<28} {'Trades':>7} {'TotalP&L%':>11} {'AvgPF':>7}")
    print("-" * 110)
    for v_label, _, _ in VARIANTS:
        results = overall.get(v_label, [])
        total_trades = sum(r.total_trades for _, r in results)
        total_pnl = sum(r.total_pnl for _, r in results)
        nonzero = [r for _, r in results if r.total_trades > 0]
        avg_pf = (
            sum(r.profit_factor for r in nonzero) / len(nonzero) if nonzero else 0.0
        )
        print(f"{v_label:<28} {total_trades:>7} {total_pnl:>+10.2f}% {avg_pf:>7.2f}")


if __name__ == "__main__":
    main()
