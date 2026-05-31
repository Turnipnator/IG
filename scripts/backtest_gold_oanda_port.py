#!/usr/bin/env python3
"""
Gold: compare IG-live config vs Oanda EMA-Trend port (2026-05-28).

IG Gold has bled -£53.74 over 36 live trades. Oanda_Gold runs an almost-identical
EMA Trend strategy (3/8/21 EMAs, RSI 85/15, 1.5x ATR stop, 0.3% pullback, HTF
required) but is profitable. Three differences:
  - Timeframe:     IG 5m / HOUR HTF      vs Oanda 1h / H4 HTF
  - R:R:           IG 3.0                vs Oanda 2.0
  - BE trigger:    IG 70% of TP          vs Oanda 30% of TP   (NOT testable here)

This script covers what the backtester *can* model — timeframe + R:R. The BE
trigger is a separate live-only behaviour (backtester models fixed SL/TP only,
no BE-then-trail), so its effect can't be isolated here.

Yahoo data only — zero IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    Backtester,
)
from config import MARKETS, STRATEGY_PROFILES

logging.basicConfig(level=logging.ERROR)

GOLD_TICKER = "GC=F"
MARKET = "Gold"

# Cache Yahoo fetches so we don't repull the same data across configs.
_orig_fetch = Backtester.fetch_data
_cache: dict = {}


def cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _cache:
        _cache[key] = _orig_fetch(self, market, days, interval)
    df = _cache[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = cached_fetch


def params_from_profile(p) -> dict:
    base = DEFAULT_PARAMS.copy()
    base.update({
        "ema_fast": p.ema_fast,
        "ema_medium": p.ema_medium,
        "ema_slow": p.ema_slow,
        "rsi_period": p.rsi_period,
        "rsi_overbought": p.rsi_overbought,
        "rsi_oversold": p.rsi_oversold,
        "rsi_buy_max": p.rsi_buy_max,
        "rsi_sell_min": p.rsi_sell_min,
        "adx_threshold": p.adx_threshold,
        "atr_period": 14,
        "stop_atr_multiplier": p.stop_atr_mult,
        "reward_risk_ratio": p.reward_risk,
    })
    return base


def run_one(label, params, rr, interval, htf_interval, days):
    # Override the per-market R:R map (which the backtester reads inside run()).
    REWARD_RISK_MAP[MARKET] = rr
    p = params.copy()
    p["reward_risk_ratio"] = rr
    bt = Backtester(params=p)
    r = bt.run(MARKET, days=days, interval=interval,
               htf_interval=htf_interval, require_htf_alignment=True)
    return r


def fmt(r):
    if not r or r.total_trades == 0:
        return "0 trades"
    return (f"{r.total_trades:>3}t {r.win_rate:>5.0%} "
            f"{r.total_pnl:>+7.2f}% PF{r.profit_factor:>5.2f} "
            f"DD{r.max_drawdown:>4.0%}")


def main():
    # Resolve the live IG Gold market + profile so the script tracks any
    # future config tweaks automatically.
    gold_mc = next(m for m in MARKETS if m.name == MARKET)
    gold_profile = STRATEGY_PROFILES[gold_mc.strategy]
    base_params = params_from_profile(gold_profile)
    bt_module.TICKER_MAP[MARKET] = GOLD_TICKER
    MIN_STOP_DISTANCE_MAP[MARKET] = 0.0  # Yahoo prices don't need IG min-stop
    MIN_CONFIDENCE_MAP[MARKET] = gold_mc.min_confidence

    print("=" * 100)
    print(f"Gold: IG-live config vs Oanda EMA-Trend port")
    print(f"Profile: ema={gold_profile.ema_fast}/{gold_profile.ema_medium}/{gold_profile.ema_slow}, "
          f"RSI {gold_profile.rsi_oversold}/{gold_profile.rsi_overbought}, "
          f"ADX{gold_profile.adx_threshold}, stop={gold_profile.stop_atr_mult}x ATR, "
          f"pullback={gold_profile.pullback_pct}%")
    print(f"Live IG R:R = {gold_profile.reward_risk}, min_conf = {gold_mc.min_confidence}")
    print(f"Backtester models fixed SL/TP only — BE-trigger difference (IG 70% vs Oanda 30%) is NOT tested here")
    print("=" * 100)

    # Cell layout:
    #   columns: R:R = 3.0 (live IG)      |  R:R = 2.0 (Oanda-style)
    #   rows:    5m/1h HTF (live IG)
    #            1h/1d HTF (Oanda-style)
    #
    # Windows:
    #   - 5m: yfinance allows ~60d intraday history -> use 55d
    #   - 1h: yfinance allows ~730d -> test both 180d and 365d for stability
    test_grid = [
        ("5m / 1h HTF",  "5m",  "1h", [55]),
        ("1h / 1d HTF",  "1h",  "1d", [180, 365]),
    ]

    for label, primary, htf, windows in test_grid:
        print(f"\n--- {label} ---")
        print(f"  {'window':>7}  {'R:R 3.0 (IG live)':>32}    {'R:R 2.0 (Oanda)':>32}    delta")
        print("  " + "-" * 90)
        for d in windows:
            r_high = run_one(f"{label}, R:R 3.0, {d}d", base_params, 3.0, primary, htf, d)
            r_low = run_one(f"{label}, R:R 2.0, {d}d", base_params, 2.0, primary, htf, d)
            delta = ""
            if r_high and r_low and r_high.total_trades and r_low.total_trades:
                dp = r_low.total_pnl - r_high.total_pnl
                dpf = r_low.profit_factor - r_high.profit_factor
                delta = f"{dp:+.2f}% / PF{dpf:+.2f}"
            print(f"  {d:>6}d  {fmt(r_high):>32}    {fmt(r_low):>32}    {delta}")

    # --- summary --------------------------------------------------------
    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print("""
  - Compare row to row to see the timeframe effect (5m vs 1h).
  - Compare column to column to see the R:R effect (3.0 vs 2.0).
  - The bottom-right cell (1h / 1d HTF, R:R 2.0) is the closest backtestable
    proxy for the Oanda config.
  - The live BE-trigger difference (IG 70% vs Oanda 30%) lets winners sit
    longer in profit before locking — its impact on real results can only
    be tested by deploying the change and journalling.
""")


if __name__ == "__main__":
    main()
