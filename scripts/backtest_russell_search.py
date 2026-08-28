#!/usr/bin/env python3
"""
Russell 2000 deeper search — find a strategy that actually works.

Live: 13 trades, 38% WR, -£11.50. 12 BUY signals, 1 SELL (lost).
Backtest at 5m showed every config was negative. The user wants
per-instrument strategy if that's what it takes.

Search across:
  - Timeframes: 5m / 15m / 1h
  - Long-only variants (Russell live was 12/13 BUYs anyway)
  - Slower EMAs (9/21/50)
  - Higher ADX (40)

1h entries can use 180d of yfinance data (vs 60d cap for sub-hourly).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)


MARKET = "US Russell 2000"
TICKER = "^RUT"


# Base params (matches indices_adx35 profile we just deployed)
BASE = {
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}

VARIANTS = [
    ("indices_adx35 (current)",  {}),
    ("indices_adx35 long-only",  {"long_only": True}),
    ("ADX 40",                    {"adx_threshold": 40}),
    ("ADX 40 long-only",          {"adx_threshold": 40, "long_only": True}),
    ("Slow EMAs 9/21/50",         {"ema_fast": 9, "ema_medium": 21, "ema_slow": 50}),
    ("Slow EMAs long-only",       {"ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "long_only": True}),
    ("Wide stops 2.5×, R:R 3",    {"stop_atr_multiplier": 2.5, "reward_risk_ratio": 3.0}),
]

# Timeframe configs: (label, entry_interval, htf_interval, days)
TIMEFRAMES = [
    ("5m / 1h HTF",  "5m",  "1h", 60),
    ("15m / 1h HTF", "15m", "1h", 60),
    ("30m / 1h HTF", "30m", "1h", 60),
    ("1h / 1d HTF",  "1h",  "1d", 730),
]


def patch():
    bt_module.TICKER_MAP[MARKET] = TICKER
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = 8.0
    MIN_CONFIDENCE_MAP[MARKET] = 0.55
    REWARD_RISK_MAP[MARKET] = 2.0


def run_one(interval: str, htf_interval: str, days: int, overrides: dict):
    params = DEFAULT_PARAMS.copy()
    params.update(BASE)
    params.update(overrides)
    if "reward_risk_ratio" in overrides:
        REWARD_RISK_MAP[MARKET] = overrides["reward_risk_ratio"]
    else:
        REWARD_RISK_MAP[MARKET] = BASE["reward_risk_ratio"]
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, interval=interval,
                  htf_interval=htf_interval, require_htf_alignment=True)


def main():
    patch()

    all_results = {}  # (timeframe_label, variant_label) -> result

    for tf_label, interval, htf_interval, days in TIMEFRAMES:
        print()
        print("=" * 105)
        print(f"Russell 2000 — {tf_label} ({days}d, HTF aligned)")
        print("=" * 105)
        print(f"{'Variant':<32} {'Trades':>7} {'WinRate':>8} {'P&L%':>8} "
              f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>7}")
        print("-" * 105)

        for v_label, overrides in VARIANTS:
            try:
                r = run_one(interval, htf_interval, days, overrides)
                all_results[(tf_label, v_label)] = r
                if r.total_trades == 0:
                    print(f"{v_label:<32} {'0':>7} {'—':>8} {'—':>8} {'—':>6} "
                          f"{'—':>8} {'—':>9} {'—':>7}")
                else:
                    print(f"{v_label:<32} {r.total_trades:>7} {r.win_rate:>7.1%} "
                          f"{r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                          f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% {r.max_drawdown:>6.1%}")
            except Exception as e:
                print(f"{v_label:<32} ERROR: {e}")

    # Top 5 across all (tf, variant) combinations by P&L
    ranked = [
        (tf, v, r) for (tf, v), r in all_results.items()
        if r.total_trades >= 5
    ]
    ranked.sort(key=lambda x: x[2].total_pnl, reverse=True)

    print()
    print("=" * 105)
    print("TOP 5 (≥5 trades)")
    print("=" * 105)
    for tf, v, r in ranked[:5]:
        print(f"  {tf:<14} | {v:<28} | P&L {r.total_pnl:+6.2f}%  PF {r.profit_factor:5.2f}  "
              f"WR {r.win_rate:5.1%}  trades {r.total_trades}")

    print()
    print("BOTTOM 5 (≥5 trades)")
    for tf, v, r in ranked[-5:]:
        print(f"  {tf:<14} | {v:<28} | P&L {r.total_pnl:+6.2f}%  PF {r.profit_factor:5.2f}  "
              f"WR {r.win_rate:5.1%}  trades {r.total_trades}")


if __name__ == "__main__":
    main()
