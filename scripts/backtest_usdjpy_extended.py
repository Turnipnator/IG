#!/usr/bin/env python3
"""
USD/JPY extended backtest — verify already-tuned profile vs alternatives.

Live journal (all-time, 4 trades):
  2026-03-16 SELL +£9.25 (RSI exit)
  2026-04-29 BUY  +£5.52 (stop hit, near BE)
  2026-05-01 SELL -£16.45 (stop hit)
  2026-05-04 SELL  -£9.95 (HTF reversed)

Net -£11.63. Three of four are SELLs (two losses, one win).
Recent two SELLs lost ~£26 — but sample too small to action.

Profile is already heavily tuned:
  EMA 9/21/50, ADX 35 (raised from 30), RSI 80/20 (wider than other forex),
  stop 1.5x (widened from 1.0), BE 90%, conf 0.55, 15m candles.

This sweep verifies the existing profile still beats alternatives
and explores: 5m vs 15m vs 1h, ADX 30/40, conf 0.65, MACD exit on.

yfinance (JPY=X) — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)

MARKET = "USD/JPY"
TICKER = "JPY=X"
MIN_STOP_PRICE = 0.20  # USDJPY ~155, 0.5% = 0.78 — using 0.20 (~20 pips)


def patch_market(min_conf=0.55):
    bt_module.TICKER_MAP[MARKET] = TICKER
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = MIN_STOP_PRICE
    MIN_CONFIDENCE_MAP[MARKET] = min_conf
    REWARD_RISK_MAP[MARKET] = 2.0


# Current live profile
USDJPY = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 80, "rsi_oversold": 20,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}

# Default forex profile (looser RSI, tighter stop)
FOREX = {
    **USDJPY,
    "rsi_overbought": 70, "rsi_oversold": 30,
    "adx_threshold": 30, "stop_atr_multiplier": 1.0,
}

CONFIGS = [
    ("CURRENT 15m usdjpy profile",            {**USDJPY}, "15m", 0.55),
    ("15m ADX 30 (looser)",                   {**USDJPY, "adx_threshold": 30}, "15m", 0.55),
    ("15m ADX 40 (stricter)",                 {**USDJPY, "adx_threshold": 40}, "15m", 0.55),
    ("15m RSI 70/30 (tighter exits)",         {**USDJPY, "rsi_overbought": 70, "rsi_oversold": 30}, "15m", 0.55),
    ("15m conf >= 0.65",                      {**USDJPY}, "15m", 0.65),
    ("15m conf >= 0.70",                      {**USDJPY}, "15m", 0.70),
    ("15m R:R 3.0",                           {**USDJPY, "reward_risk_ratio": 3.0}, "15m", 0.55),
    ("15m stop 2.0x",                         {**USDJPY, "stop_atr_multiplier": 2.0}, "15m", 0.55),
    ("15m stop 1.0x (default forex)",         {**USDJPY, "stop_atr_multiplier": 1.0}, "15m", 0.55),
    ("5m usdjpy profile",                     {**USDJPY}, "5m", 0.55),
    ("1h usdjpy profile",                     {**USDJPY}, "1h", 0.55),
    ("1h R:R 3.0",                            {**USDJPY, "reward_risk_ratio": 3.0}, "1h", 0.55),
    ("Default forex (RSI 70/30, stop 1.0x)",  {**FOREX}, "15m", 0.55),
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
    print(f"USD/JPY ({TICKER}) — extended sweep (intraday=59d, 1h=365d)")
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
