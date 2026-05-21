#!/usr/bin/env python3
"""
ADX sweep for the 1h-candle markets (2026-05-21 follow-up to the Germany 40 find).

Germany 40 showed that on 1h candles a LOWER ADX beats the 5m-era threshold
(1h pre-filters chop). This checks whether the other markets recently switched
to 1h candles are also carrying a sub-optimal ADX.

For each market it loads the EXACT current profile from config.py and varies
ONLY adx_threshold {25,30,35,40}, at 365d and 720d windows. Stops are pure
ATR-based (min_stop=0) to isolate the ADX effect; reward_risk / EMAs / RSI all
come straight from the live profile.

Yahoo data — NO IG API cost. Caveat: the backtester models fixed stop/limit +
MACD exit only (no live breakeven / ATR-trailing), so trust the RELATIVE
ranking across ADX values, not the absolute %.
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

# (config market name, Yahoo ticker)
SWEEP = [
    ("Copper", "HG=F"),
    ("NY Cocoa", "CC=F"),
    ("EUR/USD", "EURUSD=X"),
    ("US 10-Year T-Note", "ZN=F"),
    ("Natural Gas", "NG=F"),  # already ADX25 — included for completeness
]

ADX_VALUES = [25, 30, 35, 40]
WINDOWS = [365, 720]

# --- Cache Yahoo fetches so each (market, days, interval) downloads once ---
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


def run(market, base_params, adx, days):
    params = base_params.copy()
    params["adx_threshold"] = adx
    bt = Backtester(params=params)
    return bt.run(market, days=days, interval="1h", htf_interval="1d",
                  require_htf_alignment=True)


def main():
    for name, ticker in SWEEP:
        mc = next((m for m in MARKETS if m.name == name), None)
        if mc is None:
            print(f"\n{name}: not in config, skipping")
            continue
        prof = STRATEGY_PROFILES[mc.strategy]
        base = params_from_profile(prof)
        cur_adx = prof.adx_threshold

        # Patch backtester lookup tables for this market
        bt_module.TICKER_MAP[name] = ticker
        MIN_STOP_DISTANCE_MAP[name] = 0.0          # pure ATR stop
        MIN_CONFIDENCE_MAP[name] = mc.min_confidence
        REWARD_RISK_MAP[name] = prof.reward_risk

        print("\n" + "=" * 92)
        print(f"{name} ({ticker}) — profile '{mc.strategy}'  "
              f"stop={prof.stop_atr_mult}x R:R={prof.reward_risk} "
              f"EMA={prof.ema_fast}/{prof.ema_medium}/{prof.ema_slow}  "
              f"current ADX={cur_adx}")
        print("=" * 92)
        for days in WINDOWS:
            print(f"\n  {days}d window")
            print(f"    {'ADX':>4} {'Trades':>7} {'WinRate':>8} {'P&L%':>9} "
                  f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>7}")
            print("    " + "-" * 70)
            for adx in ADX_VALUES:
                try:
                    r = run(name, base, adx, days)
                    mark = "  <- current" if adx == cur_adx else ""
                    if r.total_trades == 0:
                        print(f"    {adx:>4} {'0':>7} {'—':>8} {'—':>9} {'—':>6} "
                              f"{'—':>8} {'—':>9} {'—':>7}{mark}")
                    else:
                        print(f"    {adx:>4} {r.total_trades:>7} {r.win_rate:>7.1%} "
                              f"{r.total_pnl:>+8.2f}% {r.profit_factor:>6.2f} "
                              f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% "
                              f"{r.max_drawdown:>6.1%}{mark}")
                except Exception as e:
                    print(f"    {adx:>4}  ERROR: {e}")


if __name__ == "__main__":
    main()
