#!/usr/bin/env python3
"""ADX-ceiling effect split by DIRECTION (long vs short) on the 4 deployed markets.

Answers: does capping high-ADX entries help BUYs as much as SELLs? Tonight's
all-EPIC sweep pooled both directions; this isolates each. For every market we
run long-only and short-only, baseline (no ceiling) vs capped at the live
ceiling, and compare. Yahoo, zero IG cost.

Caveat up front: splitting by direction halves already-small samples (esp.
NASDAQ ~15t total), so read the big-sample commodities (Crude/Copper/Cocoa)
as the stronger evidence and treat NASDAQ's split as indicative only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import logging

import pandas as pd

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    TICKER_MAP,
    Backtester,
)

logging.basicConfig(level=logging.ERROR)

_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": None}


def _patched_get_rp(regime):
    p = _orig_get_rp(regime)
    if _FORCED_STOP["v"] is not None:
        return dataclasses.replace(p, stop_atr_multiplier=_FORCED_STOP["v"])
    return p


bt_module.get_regime_params = _patched_get_rp

# combined direction-restrict + ADX-ceiling gate
_orig_check = Backtester.check_entry_signal
_CEIL = {"v": None}
_DIR = {"v": None}   # "BUY" / "SELL" / None=both


def _patched_check(self, row, htf_trend, require_htf_alignment=False):
    direction, conf, reason = _orig_check(self, row, htf_trend, require_htf_alignment)
    if direction is None:
        return direction, conf, reason
    if _DIR["v"] is not None and direction != _DIR["v"]:
        return None, 0, f"direction != {_DIR['v']}"
    adx = row["adx"]
    if _CEIL["v"] is not None and not pd.isna(adx) and adx > _CEIL["v"]:
        return None, 0, f"ADX too high ({adx:.1f} > {_CEIL['v']})"
    return direction, conf, reason


Backtester.check_entry_signal = _patched_check

_CACHE: dict = {}
_orig_fetch = Backtester.fetch_data


def _cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _CACHE:
        _CACHE[key] = _orig_fetch(self, market, days, interval)
    df = _CACHE[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = _cached_fetch

INDICATOR_PARAMS = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40, "atr_period": 14,
}

# name, ticker, yf_interval, days, htf, adx_floor, stop, rr, conf, ceiling
MARKETS = [
    ("NASDAQ 100", "^NDX", "5m",  59,  "1h", 30, 2.0, 2.0, 0.55, 55),
    ("Crude Oil",  "CL=F", "15m", 59,  "1h", 30, 1.0, 2.0, 0.55, 50),
    ("Copper",     "HG=F", "1h",  700, "1d", 30, 1.8, 2.0, 0.55, 55),
    ("NY Cocoa",   "CC=F", "1h",  700, "1d", 30, 1.8, 2.0, 0.55, 55),
]


def run(spec, direction, ceiling):
    name, ticker, yf_int, days, htf, floor, stop, rr, conf, _ = spec
    _DIR["v"] = direction
    _CEIL["v"] = ceiling
    _FORCED_STOP["v"] = stop
    params = DEFAULT_PARAMS.copy()
    params.update(INDICATOR_PARAMS)
    params["adx_threshold"] = floor
    params["stop_atr_multiplier"] = stop
    bt = Backtester(params=params)
    return bt.run(name, days=days, interval=yf_int, htf_interval=htf,
                  require_htf_alignment=True)


def main():
    for spec in MARKETS:
        name, ticker, yf_int, days, htf, floor, stop, rr, conf, ceil = spec
        TICKER_MAP[name] = ticker
        MIN_CONFIDENCE_MAP[name] = conf
        REWARD_RISK_MAP[name] = rr
        # realistic min_stop
        bt0 = Backtester(params={**DEFAULT_PARAMS, **INDICATOR_PARAMS})
        df0 = bt0.fetch_data(name, days, yf_int)
        if df0 is None or len(df0) < 100:
            print(f"\n{name}: thin Yahoo data, skipped")
            continue
        MIN_STOP_DISTANCE_MAP[name] = float(df0["close"].median()) * 0.005

        print(f"\n{'='*84}")
        print(f"{name} ({ticker}, {yf_int}/{days}d) — ceiling {ceil}")
        print(f"{'='*84}")
        print(f"{'dir':>6}{'base_t':>8}{'baseWR':>8}{'baseP&L':>9}{'basePF':>8}"
              f"{'cap_t':>7}{'capP&L':>9}{'capPF':>8}{'rm':>4}{'ΔP&L':>8}  verdict")
        print("-" * 84)
        for d in ("BUY", "SELL"):
            base = run(spec, d, None)
            cap = run(spec, d, ceil)
            rm = base.total_trades - cap.total_trades
            dp = cap.total_pnl - base.total_pnl
            if base.total_trades == 0:
                print(f"{d:>6}{0:>8}  (no trades this direction)")
                continue
            if rm == 0:
                verdict = "no high-ADX entries"
            elif dp > 0.01:
                verdict = "HELPS (tail = loser)"
            elif dp < -0.01:
                verdict = "HURTS (tail = winner)"
            else:
                verdict = "neutral"
            print(f"{d:>6}{base.total_trades:>8}{base.win_rate:>7.0%}{base.total_pnl:>+8.2f}%"
                  f"{base.profit_factor:>8.2f}{cap.total_trades:>7}{cap.total_pnl:>+8.2f}%"
                  f"{cap.profit_factor:>8.2f}{rm:>4}{dp:>+7.2f}  {verdict}")
    print("\nHELPS on a direction = that side's high-ADX tail is a net loser (the")
    print("exhaustion-climax cap is justified there). HURTS = capping removes winners")
    print("on that side. Weigh commodities (big samples) over NASDAQ (thin once split).")


if __name__ == "__main__":
    main()
