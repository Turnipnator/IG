#!/usr/bin/env python3
"""Cull review (2026-06-11): do the bare-`default` no-edge candidates have ANY
tradeable edge under their LIVE profile? Yahoo, zero IG API cost.

Candidates (per-EPIC profitability audit): US 2-Year / 10-Year T-Notes, NY
Cotton, Copper — all running the generic `default` strategy with no per-EPIC
validation and 0-2 live trades each. NY Cocoa added as a bonus check (same
bare-default class). This reuses the all-EPIC harness so the profile stop is
forced and the regime stop-override is neutralised (the trap that overstates
live). For each market: baseline (both sides, exactly the live config), then
long-only and short-only — so a market with a one-sided edge isn't culled by a
muddy two-sided total.

Verdict rule: cull if no side clears PF ~1.2 on a non-trivial sample; keep/watch
if a side shows a real edge worth a dedicated profile.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_adx_ceiling_all as H
from src.backtest import Backtester

# Cull candidates — names must match the harness MARKETS specs.
CANDIDATES = ["Copper", "NY Cocoa", "NY Cotton", "US 2-Year T-Note", "US 10-Year T-Note"]
SPECS = {m[0]: m for m in H.MARKETS}

# --- direction gate, composed on top of the harness's check_entry_signal ---
_side_patched = Backtester.check_entry_signal  # already harness-patched (ceiling=None)
_SIDE = {"v": None}  # None=both, "BUY"=long-only, "SELL"=short-only


def _side_check(self, row, htf_trend, require_htf_alignment=False):
    sig, conf, reason = _side_patched(self, row, htf_trend, require_htf_alignment)
    if sig is not None and _SIDE["v"] is not None and sig != _SIDE["v"]:
        return None, 0, f"side-filtered ({sig} != {_SIDE['v']})"
    return sig, conf, reason


Backtester.check_entry_signal = _side_check


def run_side(spec, side):
    name, ticker, ci, adx_floor, stop, rr, conf = spec
    _SIDE["v"] = side
    return H.run_one(name, ci, adx_floor, stop, None)  # ceiling None = baseline


def main():
    print("=" * 84)
    print("CULL REVIEW — does each bare-default candidate have ANY edge? (live profile, Yahoo)")
    print("=" * 84)
    rows = []
    for name in CANDIDATES:
        spec = SPECS.get(name)
        if not spec:
            print(f"\n{name}: not in harness MARKETS — skipped")
            continue
        # realistic min_stop = 0.5% median close (one cached fetch), as the harness does
        yf_int, days, htf = H._intraday(spec[2])
        H.prep(name, spec[1], spec[6], spec[5])
        bt0 = Backtester(params={**H.DEFAULT_PARAMS, **H.INDICATOR_PARAMS})
        df0 = bt0.fetch_data(name, days, yf_int)
        if df0 is None or len(df0) < 100:
            n = 0 if df0 is None else len(df0)
            print(f"\n{name:18s} ({spec[1]}) — NO/THIN Yahoo data ({n} candles). SKIPPED.")
            rows.append((name, "THIN DATA", 0, 0, 0))
            continue
        H.MIN_STOP_DISTANCE_MAP[name] = float(df0["close"].median()) * 0.005

        print(f"\n{name:18s} ({spec[1]}, {yf_int}/{days}d, floor {spec[3]}, stop {spec[4]}x, R:R {spec[5]})")
        side_results = {}
        for label, side in (("both", None), ("long", "BUY"), ("short", "SELL")):
            r = run_side(spec, side)
            side_results[label] = r
            print(f"   {label:5s}: {r.total_trades:3d}t  WR {r.win_rate:4.0%}  "
                  f"P&L {r.total_pnl:+6.2f}%  PF {r.profit_factor:.2f}")
        b = side_results["both"]
        # best side by PF with a non-trivial sample
        edge = None
        for label in ("both", "long", "short"):
            r = side_results[label]
            if r.total_trades >= 8 and r.profit_factor >= 1.2 and r.total_pnl > 0:
                if edge is None or r.profit_factor > side_results[edge].profit_factor:
                    edge = label
        verdict = f"KEEP/RESTRICT ({edge})" if edge else "CULL"
        print(f"   -> {verdict}")
        rows.append((name, verdict, b.total_trades, b.total_pnl, b.profit_factor))

    print("\n" + "=" * 84)
    print(f"{'Market':18s}{'Verdict':22s}{'both_t':>8}{'both_P&L%':>11}{'both_PF':>9}")
    print("-" * 84)
    for name, verdict, t, pnl, pf in rows:
        print(f"{name:18s}{verdict:22s}{t:>8}{pnl:>+11.2f}{pf:>9.2f}")
    print("-" * 84)
    print("CULL = no side clears PF 1.2 on >=8 trades. Cross-check vs live journal")
    print("(all 0-2 live trades, all net-negative) before disabling in config.py.")


if __name__ == "__main__":
    main()
