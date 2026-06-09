#!/usr/bin/env python3
"""ADX-ceiling exhaustion sweep across ALL live EPICs (Yahoo, zero IG API cost).

Generalises scripts/backtest_adx_ceiling.py (S&P/NASDAQ) to the whole live
MARKETS list, per the 2026-06-09 standing decision to run entry-quality sweeps
per-EPIC rather than reactively (see memory feedback_per_epic_entry_quality_sweeps).

Each market mirrors its LIVE profile: interval, ADX floor, stop mult, R:R, conf.
min_stop is derived as 0.5% of the market's median Yahoo close (the backtester's
own convention) so stop scale is realistic per instrument. The backtester's
regime stop-override is neutralised so the swept profile stop actually binds.

A ceiling at ADX N rejects entries whose ADX exceeds N (the exhaustion-climax
thesis). Each ceiling vs the no-cap baseline isolates that ADX band's realised
P&L. Verdict per EPIC: does capping the high-ADX tail help, hurt, or nothing —
and is the sample even big enough to say.

SCOPE (this pass = "start proven"): ADX-ceiling only. The full battery
(stop-width + ADX-floor + ceiling per EPIC) is the follow-up — see the TODO note
in memory. AI Index has no Yahoo equivalent and is skipped.
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

logging.basicConfig(level=logging.ERROR)  # quiet yahoo fetch noise

# --- regime stop-override neutralisation (live uses the profile stop) ---
_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": None}


def _patched_get_rp(regime):
    p = _orig_get_rp(regime)
    if _FORCED_STOP["v"] is not None:
        return dataclasses.replace(p, stop_atr_multiplier=_FORCED_STOP["v"])
    return p


bt_module.get_regime_params = _patched_get_rp

# --- ADX ceiling gate ---
_orig_check = Backtester.check_entry_signal
_CEIL = {"v": None}


def _patched_check(self, row, htf_trend, require_htf_alignment=False):
    adx = row["adx"]
    if _CEIL["v"] is not None and not pd.isna(adx) and adx > _CEIL["v"]:
        return None, 0, f"ADX too high ({adx:.1f} > {_CEIL['v']})"
    return _orig_check(self, row, htf_trend, require_htf_alignment)


Backtester.check_entry_signal = _patched_check

# --- cached Yahoo fetch ---
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
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "atr_period": 14,
}

# Live market -> Yahoo mirror. interval/days/htf chosen from the live
# candle_interval (5/15 -> intraday 59d + 1h HTF; 60 -> 1h 700d + daily HTF).
# adx_floor/stop/rr/conf taken from each market's live profile (dumped from
# config.py). AI Index omitted — IG-proprietary, no Yahoo series.
def _intraday(i):  # yf interval string + days + htf for a live candle_interval
    if i == 5:
        return "5m", 59, "1h"
    if i == 15:
        return "15m", 59, "1h"
    if i == 60:
        return "1h", 700, "1d"
    raise ValueError(i)


MARKETS = [
    # name, ticker, candle_interval, adx_floor, stop, rr, conf
    ("S&P 500",          "^GSPC",  5,  40, 1.5, 2.0, 0.55),
    ("NASDAQ 100",       "^NDX",   5,  30, 2.0, 2.0, 0.55),
    ("Wall Street",      "^DJI",   5,  30, 1.5, 2.0, 0.55),
    ("FTSE 100",         "^FTSE",  5,  30, 2.0, 2.0, 0.55),
    ("Japan 225",        "^N225",  5,  30, 1.5, 2.0, 0.55),
    ("Hong Kong HS50",   "^HSI",   5,  30, 1.5, 2.0, 0.55),
    ("Crude Oil",        "CL=F",   15, 30, 1.0, 2.0, 0.55),
    ("Gold",             "GC=F",   5,  35, 1.5, 3.0, 0.60),
    ("Copper",           "HG=F",   60, 30, 1.8, 2.0, 0.55),
    ("NY Cocoa",         "CC=F",   60, 30, 1.8, 2.0, 0.55),
    ("NY Cotton",        "CT=F",   15, 30, 1.8, 2.0, 0.55),
    ("EUR/USD",          "EURUSD=X", 60, 30, 1.0, 2.0, 0.65),
    ("GBP/USD",          "GBPUSD=X", 60, 30, 1.0, 2.0, 0.55),
    ("USD/JPY",          "USDJPY=X", 15, 35, 1.5, 2.0, 0.55),
    ("US 2-Year T-Note", "ZT=F",   15, 30, 1.8, 2.0, 0.55),
    ("US 10-Year T-Note","ZN=F",   60, 30, 1.8, 2.0, 0.55),
    # ("AI Index", no Yahoo equivalent — skipped)
]

CEILINGS = [None, 70, 65, 60, 55, 50, 45]   # all > the max ADX floor (40)
MIN_TRADES_FOR_VERDICT = 8


def prep(name, ticker, conf, rr):
    TICKER_MAP[name] = ticker
    MIN_CONFIDENCE_MAP[name] = conf
    REWARD_RISK_MAP[name] = rr


def run_one(name, ci, adx_floor, stop, ceiling):
    yf_int, days, htf = _intraday(ci)
    _CEIL["v"] = ceiling
    _FORCED_STOP["v"] = stop
    params = DEFAULT_PARAMS.copy()
    params.update(INDICATOR_PARAMS)
    params["adx_threshold"] = adx_floor
    params["stop_atr_multiplier"] = stop
    bt = Backtester(params=params)
    return bt.run(name, days=days, interval=yf_int, htf_interval=htf,
                  require_htf_alignment=True)


def run_market(spec, summary):
    name, ticker, ci, adx_floor, stop, rr, conf = spec
    prep(name, ticker, conf, rr)
    yf_int, days, htf = _intraday(ci)
    # derive realistic min_stop = 0.5% of median close (one cached fetch)
    bt0 = Backtester(params={**DEFAULT_PARAMS, **INDICATOR_PARAMS})
    df0 = bt0.fetch_data(name, days, yf_int)
    if df0 is None or len(df0) < 100:
        n = 0 if df0 is None else len(df0)
        print(f"\n{name:18s} ({ticker}) — NO/THIN Yahoo data ({n} candles, {yf_int}). SKIPPED.")
        summary.append((name, "THIN DATA", None, None, None, None))
        return
    MIN_STOP_DISTANCE_MAP[name] = float(df0["close"].median()) * 0.005

    base = run_one(name, ci, adx_floor, stop, None)
    print(f"\n{name:18s} ({ticker}, {yf_int}/{days}d, floor {adx_floor}, stop {stop}x) "
          f"— baseline {base.total_trades}t WR {base.win_rate:.0%} "
          f"P&L {base.total_pnl:+.2f}% PF {base.profit_factor:.2f}")
    if base.total_trades < MIN_TRADES_FOR_VERDICT:
        print(f"   (baseline {base.total_trades}t < {MIN_TRADES_FOR_VERDICT} — too thin for a verdict)")
        summary.append((name, "THIN SAMPLE", base.total_trades, base.profit_factor, None, None))
        return

    print(f"   {'ceil':>5}{'t':>5}{'WR':>6}{'P&L%':>8}{'PF':>6}{'rm':>4}{'ΔP&L':>7}")
    best = None  # (ceiling, dp, removed, r)
    for c in CEILINGS:
        if c is None:
            continue
        r = run_one(name, ci, adx_floor, stop, c)
        rm = base.total_trades - r.total_trades
        dp = r.total_pnl - base.total_pnl
        flag = "*" if dp > 0.01 else (" " if abs(dp) <= 0.01 else "x")
        print(f"   {c:>5}{r.total_trades:>5}{r.win_rate:>5.0%}{r.total_pnl:>+7.2f}%"
              f"{r.profit_factor:>6.2f}{rm:>4}{dp:>+6.2f} {flag}")
        # best = largest positive ΔP&L that removes >0 and <50% of trades
        if dp > 0.01 and rm > 0 and rm < base.total_trades * 0.5:
            if best is None or dp > best[1]:
                best = (c, dp, rm, r)
    if best:
        c, dp, rm, r = best
        verdict = "HELPS"
        print(f"   -> HELPS: ceiling {c} removes {rm}/{base.total_trades}t, "
              f"P&L {base.total_pnl:+.2f}->{r.total_pnl:+.2f}%, PF {base.profit_factor:.2f}->{r.profit_factor:.2f}")
        summary.append((name, verdict, base.total_trades, base.profit_factor, c, dp))
    else:
        # any ceiling that removed trades and hurt? otherwise neutral
        print("   -> NEUTRAL/HURT: no ceiling improves P&L on a meaningful cut")
        summary.append((name, "NEUTRAL", base.total_trades, base.profit_factor, None, None))


def main():
    summary = []
    for spec in MARKETS:
        try:
            run_market(spec, summary)
        except Exception as e:
            print(f"\n{spec[0]:18s} — ERROR: {e}")
            summary.append((spec[0], "ERROR", None, None, None, None))

    print("\n" + "=" * 78)
    print("SUMMARY — ADX-ceiling per EPIC (Yahoo, zero IG cost)")
    print("=" * 78)
    print(f"{'Market':18s}{'Verdict':14s}{'base_t':>7}{'base_PF':>8}{'ceil':>6}{'ΔP&L%':>8}")
    print("-" * 78)
    order = {"HELPS": 0, "NEUTRAL": 1, "THIN SAMPLE": 2, "THIN DATA": 3, "ERROR": 4}
    for name, verdict, bt_, pf, ceil, dp in sorted(summary, key=lambda x: order.get(x[1], 9)):
        ceils = "" if ceil is None else str(ceil)
        dps = "" if dp is None else f"{dp:+.2f}"
        pfs = "" if pf is None else f"{pf:.2f}"
        bts = "" if bt_ is None else str(bt_)
        print(f"{name:18s}{verdict:14s}{bts:>7}{pfs:>8}{ceils:>6}{dps:>8}")
    print("-" * 78)
    print("HELPS = a high-ADX ceiling cut net-loser trades without gutting the count.")
    print("Treat HELPS as candidates for an OBSERVATIONAL ceiling (log-only, enforce")
    print("later) — same playbook as the live NASDAQ-55. THIN = needs the proper")
    print("full-battery / longer-history treatment (the follow-up pass).")


if __name__ == "__main__":
    main()
