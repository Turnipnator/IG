#!/usr/bin/env python3
"""Re-entry cooldown sweep across the index EPICs (2026-06-25).

Idea (user, after Wall St #207 re-entered the same EPIC ~55min past a winning
long and chased it into a stop): instead of filtering on what the entry LOOKED
like (leg-extension — refuted at 3×), just STAY OUT of an EPIC longer after we've
just traded it. i.e. lengthen the post-close re-entry cooldown.

Current live: a general 6-candle cooldown after ANY close (= 30min on 5m, 6h on
1h) PLUS an extra 1h after a loss. So the lever is the post-close (incl. post-WIN)
cooldown. This sweeps it on each EPIC's REAL live profile and asks: does waiting
longer after exiting lift PF / P&L, or just drop trades we'd have won?

reentry_cooldown_mins added to src/backtest.py run() (default 0 = engine's old
loss-only behaviour; loss stays a 60-min floor). 5m sweep brackets the live 30min;
1h sweep brackets the live 360min. Report trades / WR / PF / P&L vs the ≈live row.

Both TFs: 5m/59d (live) + 1h/700d (sample). Yahoo only — zero IG API cost.
CAVEAT: thin books; Yahoo cash != IG; relative read, not a deploy trigger.
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

EPICS = {
    "FTSE 100":    dict(ticker="^FTSE", adx=30, stop=2.0, long_only=True),
    "NASDAQ 100":  dict(ticker="^NDX",  adx=30, stop=2.0, long_only=False),
    "Wall Street": dict(ticker="^DJI",  adx=30, stop=1.5, long_only=False),
    "S&P 500":     dict(ticker="^GSPC", adx=25, stop=1.5, long_only=True),
}
MIN_STOP = {"FTSE 100": 3.0, "NASDAQ 100": 10.0, "Wall Street": 8.0, "S&P 500": 4.0}

BASE = {
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "atr_period": 14, "reward_risk_ratio": 2.0,
}

# (interval, days, [cooldowns], live_baseline_mins)
TFS = [
    ("5m", 59, [0, 30, 60, 120, 180, 240], 30),
    ("1h", 700, [0, 360, 480, 720, 1440], 360),
]


def run(name, cfg, interval, days, cd):
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = MIN_STOP[name]
    MIN_CONFIDENCE_MAP[name] = 0.55
    REWARD_RISK_MAP[name] = 2.0
    _FORCED_STOP["v"] = cfg["stop"]
    p = DEFAULT_PARAMS.copy()
    p.update(BASE)
    p["adx_threshold"] = cfg["adx"]
    p["stop_atr_multiplier"] = cfg["stop"]
    p["long_only"] = cfg["long_only"]
    return bt_module.Backtester(params=p).run(
        name, days=days, interval=interval, require_htf_alignment=True,
        reentry_cooldown_mins=cd)


def row(label, r, base):
    if r.total_trades == 0:
        print(f"  {label:<22} {'0':>6}")
        return
    dp = r.total_pnl - base.total_pnl
    dt = r.total_trades - base.total_trades
    if r.total_pnl >= base.total_pnl - 0.02 and r.profit_factor > base.profit_factor + 0.02:
        note = "↑ PF, P&L kept"
    elif r.total_pnl > base.total_pnl + 0.02:
        note = "↑ better P&L"
    elif r.total_pnl < base.total_pnl - 0.05:
        note = "↓ worse"
    else:
        note = "≈ flat"
    print(f"  {label:<22} {r.total_trades:>6} {r.win_rate:>6.1%} {r.total_pnl:>+7.2f}% "
          f"{r.profit_factor:>6.2f} {dp:>+7.2f}% {dt:>+5d}  {note}")


def run_epic(name):
    cfg = EPICS[name]
    dirn = "LONG-only" if cfg["long_only"] else "both dirs"
    for interval, days, cds, live in TFS:
        print("=" * 92)
        print(f"{name} ({cfg['ticker']}) — {interval}/{days}d, {dirn}, ADX>={cfg['adx']}, "
              f"stop {cfg['stop']}x  (≈live cooldown = {live}m)")
        print(f"  {'reentry cooldown':<22} {'Trd':>6} {'WR':>6} {'P&L':>8} {'PF':>6} {'ΔP&L':>8} {'ΔTrd':>5}")
        print("-" * 92)
        # baseline = the ≈live cooldown, so deltas read as "vs current live"
        base = run(name, cfg, interval, days, live)
        for cd in cds:
            tag = f"{cd}m" + ("  (≈LIVE)" if cd == live else (" (engine off)" if cd == 0 else ""))
            row(tag, run(name, cfg, interval, days, cd), base)
        print()


def main():
    print("Re-entry cooldown sweep — does staying out of an EPIC longer after a close")
    print("lift PF/P&L, or just drop winners? Δ measured vs the ≈live cooldown row.\n")
    for name in EPICS:
        run_epic(name)
    print("=" * 92)
    print("Read: a LONGER cooldown helps only if PF/P&L rise as trades fall (it's pruning")
    print("net-losing re-entries). If P&L falls with trades → those re-entries were +EV, leave it.")


if __name__ == "__main__":
    main()
