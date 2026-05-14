#!/usr/bin/env python3
"""
Extended Germany 40 sweep — explore tweaks beyond the original 9 configs.

Current 5m baseline shows PF 1.04 / +0.05% over 60d — basically zero edge.
Live result: -£46.90 over 7 trades (last on 2026-04-14, none since).

This sweep tests:
  - 1h candles (less 5m intraday noise)
  - Higher confidence threshold (filter more rigorously)
  - RSI bands tighter (avoid chasing extended moves)
  - Combination wins from 5m backtest

yfinance only — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)

MARKET = "Germany 40"
TICKER = "^GDAXI"
MIN_STOP_PRICE = 25.0


def patch_market(min_conf=0.55):
    bt_module.TICKER_MAP[MARKET] = TICKER
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = MIN_STOP_PRICE
    MIN_CONFIDENCE_MAP[MARKET] = min_conf
    REWARD_RISK_MAP[MARKET] = 2.0


INDICES = {
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 30, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}

CONFIGS = [
    # Baseline at 5m
    ("CURRENT 5m: indices ADX30 1.5x",        {**INDICES}, "5m", 0.55),
    # Higher confidence filter (live trades were 75-78% conf)
    ("5m ADX30 1.5x conf>=0.70",              {**INDICES}, "5m", 0.70),
    ("5m ADX35 1.5x conf>=0.70",              {**INDICES, "adx_threshold": 35}, "5m", 0.70),
    # Tighter RSI buy window
    ("5m RSI buy<=50, sell>=50",              {**INDICES, "rsi_buy_max": 50, "rsi_sell_min": 50}, "5m", 0.55),
    # Higher R:R
    ("5m R:R 3.0",                            {**INDICES, "reward_risk_ratio": 3.0}, "5m", 0.55),
    # Move to 1h candles
    ("1h indices ADX30 1.5x",                 {**INDICES}, "1h", 0.55),
    ("1h ADX30 2.0x stop",                    {**INDICES, "stop_atr_multiplier": 2.0}, "1h", 0.55),
    ("1h ADX35 2.0x stop R:R 3.0",            {**INDICES, "adx_threshold": 35,
                                               "stop_atr_multiplier": 2.0,
                                               "reward_risk_ratio": 3.0}, "1h", 0.55),
    ("1h Slow EMAs 9/21/50",                  {**INDICES, "ema_fast": 9, "ema_medium": 21,
                                               "ema_slow": 50}, "1h", 0.55),
    ("1h Slow EMAs conf>=0.70",               {**INDICES, "ema_fast": 9, "ema_medium": 21,
                                               "ema_slow": 50}, "1h", 0.70),
]


def run_one(days, overrides, interval, min_conf, htf=True):
    patch_market(min_conf=min_conf)
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, interval=interval, require_htf_alignment=htf)


def main():
    days_5m = 59
    days_1h = 365  # full year on 1h data

    print("=" * 105)
    print(f"Germany 40 ({TICKER}) — extended sweep (5m=59d, 1h=365d)")
    print("=" * 105)
    print(f"{'Config':<42} {'Bars':>4} {'Days':>5} {'Trades':>7} "
          f"{'WR':>7} {'P&L%':>8} {'PF':>6} {'MaxDD%':>7}")
    print("-" * 105)

    results = {}
    for label, overrides, interval, min_conf in CONFIGS:
        days = days_5m if interval == "5m" else days_1h
        try:
            r = run_one(days, overrides, interval, min_conf)
            results[label] = r
            if r.total_trades == 0:
                print(f"{label:<42} {interval:>4} {days:>5} {'0':>7} "
                      f"{'-':>7} {'-':>8} {'-':>6} {'-':>7}")
            else:
                print(f"{label:<42} {interval:>4} {days:>5} {r.total_trades:>7} "
                      f"{r.win_rate:>6.1%} {r.total_pnl:>+7.2f}% "
                      f"{r.profit_factor:>6.2f} {r.max_drawdown:>6.1%}")
        except Exception as e:
            print(f"{label:<42} ERROR: {e}")

    print()
    ranked = sorted(
        [(l, r) for l, r in results.items() if r.total_trades > 0],
        key=lambda x: x[1].profit_factor,
        reverse=True,
    )
    print("Ranked by Profit Factor:")
    for label, r in ranked:
        print(f"  {label:<44}  PF {r.profit_factor:5.2f}  P&L {r.total_pnl:+6.2f}%  "
              f"WR {r.win_rate:5.1%}  trades {r.total_trades}")


if __name__ == "__main__":
    main()
