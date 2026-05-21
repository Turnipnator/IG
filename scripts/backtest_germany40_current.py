#!/usr/bin/env python3
"""
Germany 40 backtest on the CURRENT live config (1h candles + daily HTF).

Triggered by the 2026-05-21 ATR-guard investigation: the corrupted-indicator
guard (max_sane_atr = min_stop*50 = 100) was false-firing on the DAX's real 1h
ATR (~90-130). After fixing the guard (price-relative ceiling), Germany 40's
signals are re-enabled — so we need to know whether it's actually profitable on
the config it now runs (indices_adx35: EMA 5/12/26, ADX 35, 1.5x stop, R:R 2.0,
MACD exit, HTF-aligned), not the old 5m config the journal P&L was logged under.

Yahoo data (^GDAXI) — NO IG API cost. Tests multiple windows for era-sensitivity
plus nearby ADX/stop variants at the 365d window.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.ERROR)

MARKET = "Germany 40"
TICKER = "^GDAXI"


def patch_market():
    bt_module.TICKER_MAP[MARKET] = TICKER
    # Live min_stop is 2.0 IG pts; never binds (1.5xATR ~135 always dominates).
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = 2.0
    MIN_CONFIDENCE_MAP[MARKET] = 0.55   # indices_adx35 min_confidence
    REWARD_RISK_MAP[MARKET] = 2.0       # indices_adx35 reward_risk


# Exact indices_adx35 profile (config.py STRATEGY_PROFILES["indices_adx35"])
ADX35 = {
    "ema_fast": 5,
    "ema_medium": 12,
    "ema_slow": 26,
    "rsi_period": 7,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_buy_max": 55,
    "rsi_sell_min": 45,
    "adx_threshold": 35,
    "atr_period": 14,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
}

# Neighbour variants (tested at the longest window) to see if ADX35/1.5x is near-optimal
VARIANTS = [
    ("CURRENT: ADX35 1.5x (indices_adx35)", {**ADX35}),
    ("ADX30 1.5x (plain indices)",          {**ADX35, "adx_threshold": 30}),
    ("ADX40 1.5x (indices_selective)",      {**ADX35, "adx_threshold": 40}),
    ("ADX35 2.0x stop",                     {**ADX35, "stop_atr_multiplier": 2.0}),
    ("ADX35 1.0x stop",                     {**ADX35, "stop_atr_multiplier": 1.0}),
    ("ADX35 1.5x R:R 3.0",                  {**ADX35, "reward_risk_ratio": 3.0}),
]

HEADER = (f"{'Config':<40} {'Trades':>7} {'WinRate':>8} {'P&L%':>9} "
          f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>7}")


def fmt(label, r):
    if r.total_trades == 0:
        return f"{label:<40} {'0':>7} {'—':>8} {'—':>9} {'—':>6} {'—':>8} {'—':>9} {'—':>7}"
    return (f"{label:<40} {r.total_trades:>7} {r.win_rate:>7.1%} "
            f"{r.total_pnl:>+8.2f}% {r.profit_factor:>6.2f} "
            f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% {r.max_drawdown:>6.1%}")


def run_one(days, overrides):
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, interval="1h", htf_interval="1d",
                  require_htf_alignment=True)


def main():
    patch_market()

    print("=" * 100)
    print(f"Germany 40 ({TICKER}) — CURRENT config (1h candles, daily HTF, indices_adx35)")
    print("REWARD_RISK=2.0  MIN_CONF=0.55  MACD-exit ON  HTF-aligned  (Yahoo data, no IG cost)")
    print("=" * 100)

    # 1) Era-sensitivity: current config across windows
    print("\n--- Current config (ADX35 1.5x) across time windows ---")
    print(HEADER)
    print("-" * 100)
    for days in (30, 90, 180, 365, 720):
        try:
            r = run_one(days, ADX35)
            print(fmt(f"{days}d window", r))
        except Exception as e:
            print(f"{days}d window — ERROR: {e}")

    # 2) Variant sweep at 365d
    print("\n--- Variant sweep @ 365d window ---")
    print(HEADER)
    print("-" * 100)
    ranked = []
    for label, ov in VARIANTS:
        try:
            r = run_one(365, ov)
            print(fmt(label, r))
            if r.total_trades > 0:
                ranked.append((label, r))
        except Exception as e:
            print(f"{label:<40} ERROR: {e}")

    print("\nRanked by P&L @365d:")
    for label, r in sorted(ranked, key=lambda x: x[1].total_pnl, reverse=True):
        print(f"  {label:<40} P&L {r.total_pnl:+7.2f}%  PF {r.profit_factor:5.2f}  "
              f"WR {r.win_rate:5.1%}  trades {r.total_trades}")


if __name__ == "__main__":
    main()
