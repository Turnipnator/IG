#!/usr/bin/env python3
"""ADX-ceiling sweep — does skipping HIGH-ADX entries remove losers? (S&P, NASDAQ)

Motivation (2026-06-09): the live S&P 500 SELL at 18:00 entered with ADX=57.2 —
a big outlier vs the book-wide ~38 mean — right at the bottom of a down-leg, on
the bounce, and lost. Thesis: a momentum system's signals are STRONGEST at
exhaustion, so an extreme ADX may mark a climax about to mean-revert rather than
a healthy trend to ride. An EARLIER finding (leg-filter work) was "ADX doesn't
separate winners from losers (38 vs 38 mean)" — but that was MID-range ADX. This
sweep asks specifically about the high tail: cap new entries at ADX N and see if
P&L/PF improve (i.e. the capped band was a net loser) without gutting trade count.

Each ceiling vs the no-cap baseline isolates one ADX band:
  removed = baseline.trades - ceiling.trades ; if P&L rises when removed, the
  band [ceiling .. max] was net-negative. So the sweep IS the outcome-linked
  band analysis (the Trade object carries no entry ADX to bucket on directly).

Live profiles mirrored exactly:
  S&P 500     indices_selective  ADX floor 40, stop 1.5x, R:R 2.0, conf 0.55, 5m
  NASDAQ 100  indices_wide       ADX floor 30, stop 2.0x, R:R 2.0, conf 0.55, 5m

Yahoo only (^GSPC / ^NDX 5m, ~59d) — ZERO IG API cost. One cached fetch/market.
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

logging.basicConfig(level=logging.WARNING)

# --- Neutralise the backtester's regime stop-override (live uses the profile
# stop directly; without this the swept stop is inert). Same pattern as
# backtest_stop_cap_indices.py / backtest_stop_width.py. ---
_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": None}


def _patched_get_rp(regime):
    p = _orig_get_rp(regime)
    if _FORCED_STOP["v"] is not None:
        return dataclasses.replace(p, stop_atr_multiplier=_FORCED_STOP["v"])
    return p


bt_module.get_regime_params = _patched_get_rp

# --- ADX ceiling: reject entries whose ADX exceeds the cap. Wraps the real
# check_entry_signal so all other gating (floor, EMA, RSI, HTF, confidence) is
# untouched — we only ADD an upper bound. ---
_orig_check = Backtester.check_entry_signal
_CEIL = {"v": None}


def _patched_check(self, row, htf_trend, require_htf_alignment=False):
    adx = row["adx"]
    if _CEIL["v"] is not None and not pd.isna(adx) and adx > _CEIL["v"]:
        return None, 0, f"ADX too high ({adx:.1f} > {_CEIL['v']})"
    return _orig_check(self, row, htf_trend, require_htf_alignment)


Backtester.check_entry_signal = _patched_check

# --- One cached Yahoo fetch per (market, days, interval). ---
_CACHE: dict = {}
_orig_fetch = Backtester.fetch_data


def _cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _CACHE:
        _CACHE[key] = _orig_fetch(self, market, days, interval)
    df = _CACHE[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = _cached_fetch

MARKETS = {
    "S&P 500":    dict(ticker="^GSPC", min_stop=30.0,  conf=0.55, adx_floor=40, stop=1.5, rr=2.0),
    "NASDAQ 100": dict(ticker="^NDX",  min_stop=100.0, conf=0.55, adx_floor=30, stop=2.0, rr=2.0),
}

BASE_INDICES = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "atr_period": 14,
}

DAYS = 59
INTERVAL = "5m"
# None = baseline (no ceiling). The losing live S&P trade was ADX 57.
CEILINGS = [None, 65, 60, 55, 50, 45, 40]


def patch(name):
    cfg = MARKETS[name]
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[name] = cfg["conf"]
    REWARD_RISK_MAP[name] = cfg["rr"]


def run_one(name, ceiling):
    cfg = MARKETS[name]
    _CEIL["v"] = ceiling
    _FORCED_STOP["v"] = cfg["stop"]
    params = DEFAULT_PARAMS.copy()
    params.update(BASE_INDICES)
    params["adx_threshold"] = cfg["adx_floor"]
    params["stop_atr_multiplier"] = cfg["stop"]
    params["reward_risk_ratio"] = cfg["rr"]
    bt = Backtester(params=params)
    return bt.run(name, days=DAYS, interval=INTERVAL, require_htf_alignment=True)


def run_market(name):
    patch(name)
    cfg = MARKETS[name]
    print("=" * 96)
    print(f"{name} ({cfg['ticker']}) — {INTERVAL}, {DAYS}d. LIVE: ADX floor {cfg['adx_floor']}, "
          f"stop {cfg['stop']}x, R:R {cfg['rr']}, conf {cfg['conf']}")
    print("=" * 96)
    print(f"{'ceiling':>8}{'Trades':>8}{'WR':>8}{'P&L%':>9}{'PF':>7}{'AvgWin':>8}{'AvgLoss':>9}"
          f"{'removed':>9}{'ΔP&L%':>8}  note")
    print("-" * 96)
    baseline = None
    for ceil in CEILINGS:
        r = run_one(name, ceil)
        label = "none" if ceil is None else str(ceil)
        if ceil is None:
            baseline = r
            removed = ""
            dpnl = ""
            note = "<= BASELINE"
        else:
            rem = baseline.total_trades - r.total_trades
            dp = r.total_pnl - baseline.total_pnl
            removed = str(rem)
            dpnl = f"{dp:+.2f}"
            # If P&L rises when we remove the >ceiling band, that band was a net loser.
            note = "band was NET LOSER" if dp > 0.01 else ("band was net winner" if dp < -0.01 else "")
        if r.total_trades == 0:
            print(f"{label:>8}{0:>8}{'-':>8}")
            continue
        print(f"{label:>8}{r.total_trades:>8}{r.win_rate:>7.1%}{r.total_pnl:>+8.2f}%"
              f"{r.profit_factor:>7.2f}{r.avg_win:>+7.2f}%{r.avg_loss:>+8.2f}%"
              f"{removed:>9}{dpnl:>8}  {note}")
    print()


def main():
    for name in MARKETS:
        run_market(name)
    print("Reading: each ceiling row removes the entries with ADX above it. A positive")
    print("ΔP&L% means removing that high-ADX band IMPROVED results (the band was a net")
    print("loser) — evidence for an exhaustion cap. Watch trade-count attrition too: a")
    print("cap that helps P&L but halves the trades may just be overfitting the tail.")


if __name__ == "__main__":
    main()
