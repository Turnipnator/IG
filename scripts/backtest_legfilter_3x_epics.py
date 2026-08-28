#!/usr/bin/env python3
"""~3×ATR leg-filter test across the index EPICs (2026-06-25).

Follow-up to backtest_wallst_adx_longs.py, which found the losing Wall St longs
weren't a high-ADX problem (high ADX wins) but an EXTENDED-LEG problem: entries
taken after price had already run >3×ATR over the prior ~12 candles lost on both
5m and 1h. NASDAQ already carries an OBSERVATIONAL leg-filter but at 5.0×ATR —
too loose to catch the ~4×ATR chase that stopped Wall St #207. This sweeps the
leg-filter threshold around 3× for every index EPIC, on its REAL live profile, to
see whether ~3× cuts losers without clipping the winners (the risk of any filter).

For each EPIC: run the live profile (correct EMA 5/12/26, its ADX floor, its stop
multiplier, R:R, and direction restriction — FTSE/S&P long-only) with NO filter
(baseline) then with leg_filter_lookback=12 at thresholds {2.5, 3.0, 3.5, 5.0}.
Report trades / WR / PF / P&L and the delta vs baseline. A good filter: PF up (or
flat) with P&L not materially down — i.e. it removes net-negative chases.

Both timeframes: 5m/59d (live) + 1h/700d (sample power; thin books otherwise).
Yahoo only (^FTSE/^NDX/^DJI/^GSPC) — zero IG API cost. CAVEAT: Yahoo cash != IG
DFB; small post-filter samples; relative read, not a deploy trigger.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import logging

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

# Force the regime stop override to each EPIC's live profile stop.
_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": None}


def _patched_get_rp(regime):
    p = _orig_get_rp(regime)
    if _FORCED_STOP["v"] is not None:
        return dataclasses.replace(p, stop_atr_multiplier=_FORCED_STOP["v"])
    return p


bt_module.get_regime_params = _patched_get_rp

_CACHE: dict = {}
_orig_fetch = Backtester.fetch_data


def _cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _CACHE:
        _CACHE[key] = _orig_fetch(self, market, days, interval)
    df = _CACHE[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = _cached_fetch

# Live index profiles (verified from config 2026-06-25). All EMA 5/12/26, RSI7
# band 30-55(buy)/45-70(sell). dir: FTSE/S&P long-only. min_stop kept modest so
# the ATR×mult stop drives (Yahoo price scale).
EPICS = {
    "FTSE 100":    dict(ticker="^FTSE", min_stop=3.0,  adx=30, stop=2.0, long_only=True),
    "NASDAQ 100":  dict(ticker="^NDX",  min_stop=10.0, adx=30, stop=2.0, long_only=False),
    "Wall Street": dict(ticker="^DJI",  min_stop=8.0,  adx=30, stop=1.5, long_only=False),
    "S&P 500":     dict(ticker="^GSPC", min_stop=4.0,  adx=25, stop=1.5, long_only=True),
}

BASE = {
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "atr_period": 14, "reward_risk_ratio": 2.0,
}

THRESHOLDS = [2.5, 3.0, 3.5, 5.0]   # 5.0 = NASDAQ's current setting (reference)
TFS = [("5m", 59, 12), ("1h", 700, 6)]   # (interval, days, leg_lookback)


def _params(cfg, leg_lb, leg_thr):
    p = DEFAULT_PARAMS.copy()
    p.update(BASE)
    p["adx_threshold"] = cfg["adx"]
    p["stop_atr_multiplier"] = cfg["stop"]
    p["long_only"] = cfg["long_only"]
    p["leg_filter_lookback"] = leg_lb
    p["leg_filter_threshold"] = leg_thr
    return p


def run(name, cfg, interval, days, leg_lb, leg_thr):
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[name] = 0.55
    REWARD_RISK_MAP[name] = 2.0
    _FORCED_STOP["v"] = cfg["stop"]
    bt = Backtester(params=_params(cfg, leg_lb, leg_thr))
    return bt.run(name, days=days, interval=interval, require_htf_alignment=True)


def row(label, r, base):
    if r.total_trades == 0:
        print(f"  {label:<22} {'0':>6}")
        return
    dp = r.total_pnl - base.total_pnl
    dt = r.total_trades - base.total_trades
    if r.total_pnl >= base.total_pnl - 0.01 and r.profit_factor > base.profit_factor + 0.01:
        note = "↑ PF, P&L kept"
    elif r.total_pnl > base.total_pnl and r.profit_factor > base.profit_factor:
        note = "↑ better"
    elif r.total_pnl < base.total_pnl - 0.05:
        note = "↓ worse"
    else:
        note = "≈ flat"
    star = "  <-- ~3x" if label.startswith("leg 3.0") else ""
    print(f"  {label:<22} {r.total_trades:>6} {r.win_rate:>6.1%} {r.total_pnl:>+7.2f}% "
          f"{r.profit_factor:>6.2f} {dp:>+7.2f}% {dt:>+5d}  {note}{star}")


def run_epic(name):
    cfg = EPICS[name]
    dirn = "LONG-only" if cfg["long_only"] else "both dirs"
    for interval, days, leg_lb in TFS:
        print("=" * 92)
        print(f"{name} ({cfg['ticker']}) — {interval}/{days}d, {dirn}, ADX>={cfg['adx']}, "
              f"stop {cfg['stop']}x, leg-lookback {leg_lb}")
        print(f"  {'config':<22} {'Trd':>6} {'WR':>6} {'P&L':>8} {'PF':>6} {'ΔP&L':>8} {'ΔTrd':>5}")
        print("-" * 92)
        base = run(name, cfg, interval, days, 0, 0.0)
        row("BASELINE (no filter)", base, base)
        for thr in THRESHOLDS:
            r = run(name, cfg, interval, days, leg_lb, thr)
            row(f"leg {thr:.1f}x", r, base)
        print()


def main():
    print("~3×ATR leg-filter test across index EPICs — does blocking extended-leg")
    print("chases cut losers without clipping winners? (live profiles, HTF-aligned)\n")
    for name in EPICS:
        run_epic(name)
    print("=" * 92)
    print("Read: prefer the threshold that lifts PF while keeping P&L ≈ baseline (removes")
    print("net-negative chases). If 3.0× only drops trades + P&L → the chase band is +EV")
    print("there, leave it. 5m is the live read; 1h adds sample. Small post-filter n — review evidence, not a deploy.")


if __name__ == "__main__":
    main()
