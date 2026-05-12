#!/usr/bin/env python3
"""
60-day Japan 225 backtest — tune the freshly-added Asian-session market.

Added to live config 2026-05-07. Initial 30d sweep (config.py:506):
  3 trades, 66.7% WR, PF 6.24, +£93.94 — promising but tiny sample.

Tokyo cash session is only ~6 hours (00:00–06:00 UTC), so each day yields
~5–10 5m candles in-window. 60d gives a few hundred candles total — small
but enough to compare parameter variants on the same data.

Tests current indices profile vs wider stops, higher ADX, slower EMAs.
Yahoo data (^N225) — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)

MARKET = "Japan 225"
TICKER = "^N225"
MIN_STOP_PRICE = 40.0  # ~0.1% of 40000 = 40 pts (IG min stop is 20)


def patch_market():
    """Add Japan 225 to the backtester's lookup tables."""
    bt_module.TICKER_MAP[MARKET] = TICKER
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = MIN_STOP_PRICE
    MIN_CONFIDENCE_MAP[MARKET] = 0.55
    REWARD_RISK_MAP[MARKET] = 2.0


# Indices-profile defaults (matches config.py "indices" StrategyConfig)
INDICES = {
    "ema_fast": 5,
    "ema_medium": 12,
    "ema_slow": 26,
    "rsi_period": 7,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_buy_max": 55,
    "rsi_sell_min": 45,
    "adx_threshold": 30,
    "atr_period": 14,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
}

# Slower default-profile EMAs (matches 9/21/50)
DEFAULT_EMAS = {
    **INDICES,
    "ema_fast": 9,
    "ema_medium": 21,
    "ema_slow": 50,
}


CONFIGS = [
    ("Current: indices ADX30 1.5×",      {**INDICES}),
    ("ADX 25, 1.5× (looser)",            {**INDICES, "adx_threshold": 25}),
    ("ADX 35, 1.5×",                     {**INDICES, "adx_threshold": 35}),
    ("ADX 40, 1.5×",                     {**INDICES, "adx_threshold": 40}),
    ("ADX 30, 2.0× stop",                {**INDICES, "stop_atr_multiplier": 2.0}),
    ("ADX 30, 2.5× stop",                {**INDICES, "stop_atr_multiplier": 2.5}),
    ("ADX 35, 2.0× stop",                {**INDICES, "adx_threshold": 35,
                                          "stop_atr_multiplier": 2.0}),
    ("ADX 35, 2.5× stop",                {**INDICES, "adx_threshold": 35,
                                          "stop_atr_multiplier": 2.5}),
    ("ADX 30, R:R 3.0",                  {**INDICES, "reward_risk_ratio": 3.0}),
    ("Slow EMAs 9/21/50, ADX30 1.5×",   {**DEFAULT_EMAS}),
    ("Slow EMAs, ADX 35, 2.0×",          {**DEFAULT_EMAS, "adx_threshold": 35,
                                          "stop_atr_multiplier": 2.0}),
]


def run_one(days: int, overrides: dict, htf: bool):
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, require_htf_alignment=htf)


def main():
    patch_market()
    days = 60  # max yfinance window for 5m

    print("=" * 100)
    print(f"Japan 225 ({TICKER}) — {days}d backtest, 5m candles, HTF aligned")
    print("=" * 100)
    print(f"{'Config':<38} {'Trades':>7} {'WinRate':>8} {'P&L%':>8} "
          f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>7}")
    print("-" * 100)

    results = {}
    for label, overrides in CONFIGS:
        try:
            r = run_one(days, overrides, htf=True)
            results[label] = r
            if r.total_trades == 0:
                print(f"{label:<38} {'0':>7} {'—':>8} {'—':>8} {'—':>6} "
                      f"{'—':>8} {'—':>9} {'—':>7}")
            else:
                print(f"{label:<38} {r.total_trades:>7} {r.win_rate:>7.1%} "
                      f"{r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                      f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% {r.max_drawdown:>6.1%}")
        except Exception as e:
            print(f"{label:<38} ERROR: {e}")

    # Best by P&L
    print()
    ranked = sorted(
        [(l, r) for l, r in results.items() if r.total_trades > 0],
        key=lambda x: x[1].total_pnl,
        reverse=True,
    )
    print("Ranked by total P&L:")
    for label, r in ranked:
        print(f"  {label:<40}  P&L {r.total_pnl:+6.2f}%  PF {r.profit_factor:5.2f}  "
              f"WR {r.win_rate:5.1%}  trades {r.total_trades}")


if __name__ == "__main__":
    main()
