#!/usr/bin/env python3
"""
Gold extended backtest — validate post-tightening profile vs alternatives.

Live journal by month:
  2026-03  17 trades  4W/12L  -£108.62  (pre-tightening, default profile)
  2026-04  13 trades  8W/4L   +£84.76   (post-2026-04-07 tightening)
  2026-05   6 trades  1W/3L   -£29.88   (recent slump, tiny sample)
  All-time: 36 trades  13W/19L -£53.74

2026-04-07 tightening (commit 1b18cc2):
  - ADX 30 → 35  (filters 17 low-quality ranging entries)
  - stop_atr_mult 2.5 → 1.5  (tighter stops, smaller losses)
  - reward_risk 2.0 → 3.0  (bigger winners compensate)
  - EMAs stayed at 3/8/21 (gold-specific fast)
  - RSI 85/15 (wide — gold trends push RSI high)

This sweep tests:
  - Current vs other intervals (5m, 15m, 1h)
  - Even tighter ADX (40)
  - Different EMA spans
  - R:R variants (2.0, 3.0, 4.0)
  - Higher confidence threshold

yfinance (GC=F) — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)

MARKET = "Gold"
TICKER = "GC=F"
MIN_STOP_PRICE = 25.0  # ~0.5% of $4500/oz


def patch_market(min_conf=0.55, rr=3.0):
    bt_module.TICKER_MAP[MARKET] = TICKER
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = MIN_STOP_PRICE
    MIN_CONFIDENCE_MAP[MARKET] = min_conf
    REWARD_RISK_MAP[MARKET] = rr


# Current live gold profile (post 2026-04-07 tightening)
GOLD = {
    "ema_fast": 3, "ema_medium": 8, "ema_slow": 21,
    "rsi_period": 7, "rsi_overbought": 85, "rsi_oversold": 15,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 3.0,
}

# Pre-tightening config (for control)
GOLD_OLD = {**GOLD, "adx_threshold": 30,
            "stop_atr_multiplier": 2.5, "reward_risk_ratio": 2.0}

# Standard EMA spans
EMA_STANDARD = {**GOLD, "ema_fast": 9, "ema_medium": 21, "ema_slow": 50}

# Indices-style EMAs
EMA_INDICES = {**GOLD, "ema_fast": 5, "ema_medium": 12, "ema_slow": 26}

CONFIGS = [
    ("CURRENT 5m gold (ADX35 1.5x R:R3)",     {**GOLD}, "5m", 0.55, 3.0),
    ("Pre-tightening 5m (ADX30 2.5x R:R2)",   {**GOLD_OLD}, "5m", 0.55, 2.0),
    ("5m ADX 40 (stricter)",                   {**GOLD, "adx_threshold": 40}, "5m", 0.55, 3.0),
    ("5m R:R 2.0",                             {**GOLD, "reward_risk_ratio": 2.0}, "5m", 0.55, 2.0),
    ("5m R:R 4.0",                             {**GOLD, "reward_risk_ratio": 4.0}, "5m", 0.55, 4.0),
    ("5m wider stop 2.0x R:R 3.0",             {**GOLD, "stop_atr_multiplier": 2.0}, "5m", 0.55, 3.0),
    ("5m conf >= 0.70",                        {**GOLD}, "5m", 0.70, 3.0),
    ("5m standard EMAs 9/21/50",               {**EMA_STANDARD}, "5m", 0.55, 3.0),
    ("5m indices EMAs 5/12/26",                {**EMA_INDICES}, "5m", 0.55, 3.0),
    ("15m gold ADX 35",                        {**GOLD}, "15m", 0.55, 3.0),
    ("15m gold ADX 40",                        {**GOLD, "adx_threshold": 40}, "15m", 0.55, 3.0),
    ("1h gold ADX 30",                         {**GOLD, "adx_threshold": 30}, "1h", 0.55, 3.0),
    ("1h gold ADX 35",                         {**GOLD}, "1h", 0.55, 3.0),
    ("1h gold standard EMAs",                  {**EMA_STANDARD}, "1h", 0.55, 3.0),
    ("1h R:R 4.0",                             {**GOLD, "reward_risk_ratio": 4.0}, "1h", 0.55, 4.0),
]


def run_one(days, overrides, interval, min_conf, rr, htf=True):
    patch_market(min_conf=min_conf, rr=rr)
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, interval=interval, require_htf_alignment=htf)


def main():
    days_intraday = 59
    days_1h = 365

    print("=" * 105)
    print(f"Gold ({TICKER}) — extended sweep (intraday=59d, 1h=365d)")
    print("=" * 105)
    print(f"{'Config':<42} {'Bars':>4} {'Days':>5} {'Trades':>7} "
          f"{'WR':>7} {'P&L%':>8} {'PF':>6} {'MaxDD%':>7}")
    print("-" * 105)

    results = {}
    for label, overrides, interval, min_conf, rr in CONFIGS:
        days = days_1h if interval == "1h" else days_intraday
        try:
            r = run_one(days, overrides, interval, min_conf, rr)
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
