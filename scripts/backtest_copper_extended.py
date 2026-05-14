#!/usr/bin/env python3
"""
Copper extended backtest — diagnose 2026-05-11 -£41.40 BUY loss.

Live journal (all-time, 2 trades):
  2026-03-13 SELL +£7.50 (RSI exit)
  2026-05-11 BUY  -£41.40 (stop hit at 82% confidence)

Single bad trade isn't statistically meaningful, but worth validating
that the "default" profile (EMA 9/21/50, ADX 30, stop 1.8x) holds
its edge or whether a different interval/profile suits Copper better.

Current: 5m candles, default strategy, conf 0.55, trading 13-20 UTC.

yfinance (HG=F) — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)

MARKET = "Copper"
TICKER = "HG=F"
MIN_STOP_PRICE = 0.03  # COMEX copper ~$4.50/lb, 0.5% = $0.02-0.03


def patch_market(min_conf=0.55):
    bt_module.TICKER_MAP[MARKET] = TICKER
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = MIN_STOP_PRICE
    MIN_CONFIDENCE_MAP[MARKET] = min_conf
    REWARD_RISK_MAP[MARKET] = 2.0


DEFAULT = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 30, "atr_period": 14,
    "stop_atr_multiplier": 1.8, "reward_risk_ratio": 2.0,
}

# Fast indices-style EMAs for comparison
FAST = {**DEFAULT, "ema_fast": 5, "ema_medium": 12, "ema_slow": 26}

CONFIGS = [
    ("CURRENT 5m default ADX30 1.8x",         {**DEFAULT}, "5m", 0.55),
    ("5m wider stop 2.5x",                    {**DEFAULT, "stop_atr_multiplier": 2.5}, "5m", 0.55),
    ("5m ADX 35",                             {**DEFAULT, "adx_threshold": 35}, "5m", 0.55),
    ("5m ADX 40",                             {**DEFAULT, "adx_threshold": 40}, "5m", 0.55),
    ("5m R:R 3.0",                            {**DEFAULT, "reward_risk_ratio": 3.0}, "5m", 0.55),
    ("5m conf >= 0.70",                       {**DEFAULT}, "5m", 0.70),
    ("5m fast EMAs 5/12/26",                  {**FAST}, "5m", 0.55),
    ("1h default ADX30 1.8x",                 {**DEFAULT}, "1h", 0.55),
    ("1h wider stop 2.5x R:R 3.0",            {**DEFAULT, "stop_atr_multiplier": 2.5,
                                               "reward_risk_ratio": 3.0}, "1h", 0.55),
    ("1h ADX 35 R:R 3.0",                     {**DEFAULT, "adx_threshold": 35,
                                               "reward_risk_ratio": 3.0}, "1h", 0.55),
    ("1h fast EMAs 5/12/26",                  {**FAST}, "1h", 0.55),
    ("15m default",                           {**DEFAULT}, "15m", 0.55),
    ("15m ADX 35",                            {**DEFAULT, "adx_threshold": 35}, "15m", 0.55),
]


def run_one(days, overrides, interval, min_conf, htf=True):
    patch_market(min_conf=min_conf)
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, interval=interval, require_htf_alignment=htf)


def main():
    days_intraday = 59
    days_1h = 365

    print("=" * 105)
    print(f"Copper ({TICKER}) — extended sweep (intraday=59d, 1h=365d)")
    print("=" * 105)
    print(f"{'Config':<42} {'Bars':>4} {'Days':>5} {'Trades':>7} "
          f"{'WR':>7} {'P&L%':>8} {'PF':>6} {'MaxDD%':>7}")
    print("-" * 105)

    results = {}
    for label, overrides, interval, min_conf in CONFIGS:
        days = days_1h if interval == "1h" else days_intraday
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
