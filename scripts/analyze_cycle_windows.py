#!/usr/bin/env python3
"""Split existing realised trades and breakout counterfactuals by PFO cycle state.

Phase 1 of the cycle work: MEASURE, change nothing. Reads only data we already
hold -- the journal's `trades` (realised momentum P&L) and `benched_outcomes`
(resolved breakout counterfactuals from the always-on shadow observer) -- and
tags each by the cycle state of its entry day.

⚠️ READ THE BASE RATE FIRST. With +/-1 day tolerance a single cross marks THREE
days, so a month with 9 crosses can be >50% "in window". If the in-window share
is high, an in-window concentration of trades means nothing -- the comparison
only carries information when the two buckets are of comparable size and the
base rate is well under half. The coverage table is printed before any P&L for
exactly that reason.

⚠️ Only the period the sheets COVER is comparable. Trades outside it are excluded
rather than counted as out-of-window, or months with no sheet would masquerade as
quiet periods and bias the split.

Usage:
  python scripts/analyze_cycle_windows.py
  python scripts/analyze_cycle_windows.py --tolerance 0     # sweep the tolerance
  python scripts/analyze_cycle_windows.py --epic IX.D.SPTRD.DAILY.IP
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cycles import (INSTRUMENT_EPICS, covered_range,  # noqa: E402
                        cycle_state, load_cycles)

DB = Path("/app/data/trade_journal.db") if os.path.exists("/app") \
    else Path("data/trade_journal.db")


def coverage(epic, lo, hi, tol):
    """How much of the covered period each state occupies -- the base rate."""
    counts = defaultdict(int)
    d, n = lo, 0
    while d <= hi:
        counts[cycle_state(epic, d, tol).label] += 1
        n += 1
        d += timedelta(days=1)
    return counts, n


def bucket(rows, epic_key, time_key, val_key, tol, lo, hi):
    """Group rows by cycle label, skipping anything outside sheet coverage."""
    out = defaultdict(list)
    skipped = 0
    for r in rows:
        try:
            d = date.fromisoformat(str(r[time_key])[:10])
        except (ValueError, TypeError):
            continue
        if not (lo <= d <= hi):
            skipped += 1
            continue
        v = r[val_key]
        if v is None:
            continue
        out[cycle_state(r[epic_key], d, tol).label].append(float(v))
    return out, skipped


def show(buckets, unit, indent="    "):
    order = ["strong", "cross", "week", "volatility", "none"]
    tot_n = sum(len(v) for v in buckets.values())
    if not tot_n:
        print(f"{indent}(no rows in the covered period)")
        return
    for k in order:
        v = buckets.get(k)
        if not v:
            continue
        w = sum(1 for x in v if x > 0)
        l = sum(1 for x in v if x < 0)
        avg = sum(v) / len(v)
        print(f"{indent}{k:11s} n={len(v):3d}  {w:2d}W/{l:2d}L  "
              f"tot={sum(v):+8.2f}{unit}  avg={avg:+6.3f}{unit}")
    inw = [x for k, v in buckets.items() if k != "none" for x in v]
    outw = buckets.get("none", [])
    if inw and outw:
        print(f"{indent}{'-'*54}")
        print(f"{indent}{'IN-window':11s} n={len(inw):3d}  "
              f"tot={sum(inw):+8.2f}{unit}  avg={sum(inw)/len(inw):+6.3f}{unit}")
        print(f"{indent}{'OUT':11s} n={len(outw):3d}  "
              f"tot={sum(outw):+8.2f}{unit}  avg={sum(outw)/len(outw):+6.3f}{unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=int, default=1,
                    help="+/- days around a cross (report says 'day before or after')")
    ap.add_argument("--epic", help="restrict to one EPIC")
    args = ap.parse_args()

    rng = covered_range()
    if rng is None:
        sys.exit("no cycle sheets in cycles/")
    lo, hi = rng
    cyc = load_cycles(cross_tolerance_days=args.tolerance)
    epics = [args.epic] if args.epic else sorted(
        {e for v in INSTRUMENT_EPICS.values() for e in v} & set(cyc))

    print(f"cycle sheets cover {lo} -> {hi}   cross tolerance = +/-{args.tolerance}d")
    print("\n=== BASE RATE (share of covered days in each state) ===")
    print("  a high in-window share makes any concentration meaningless")
    for e in epics:
        c, n = coverage(e, lo, hi, args.tolerance)
        inw = n - c["none"]
        print(f"  {e:26s} in-window {inw:3d}/{n} ({100*inw/n:4.1f}%)  "
              + " ".join(f"{k}={c[k]}" for k in
                         ("strong", "cross", "week", "volatility") if c[k]))

    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row

    print("\n=== REALISED MOMENTUM TRADES (journal, real P&L) ===")
    tr = db.execute("SELECT epic, entry_time, pnl FROM trades "
                    "WHERE pnl IS NOT NULL").fetchall()
    for e in epics:
        rows = [r for r in tr if r["epic"] == e]
        if not rows:
            continue
        b, skip = bucket(rows, "epic", "entry_time", "pnl", args.tolerance, lo, hi)
        if sum(len(v) for v in b.values()):
            print(f"  {e}   ({skip} trades outside sheet coverage, excluded)")
            show(b, "")

    print("\n=== BREAKOUT COUNTERFACTUALS (benched_outcomes, R) ===")
    bo = db.execute("SELECT epic, benched_at, r_multiple, bench_type "
                    "FROM benched_outcomes WHERE r_multiple IS NOT NULL "
                    "AND bench_type LIKE 'breakout%'").fetchall()
    for e in epics:
        rows = [r for r in bo if r["epic"] == e]
        if not rows:
            continue
        b, skip = bucket(rows, "epic", "benched_at", "r_multiple",
                         args.tolerance, lo, hi)
        if sum(len(v) for v in b.values()):
            print(f"  {e}   ({skip} episodes outside coverage, excluded)")
            show(b, "R")

    print("\n=== POOLED across all cycle-covered EPICs ===")
    for label, rows, ek, tk, vk, unit in (
            ("realised momentum (GBP)", tr, "epic", "entry_time", "pnl", ""),
            ("breakout counterfactual (R)", bo, "epic", "benched_at", "r_multiple", "R")):
        rows = [r for r in rows if r["epic"] in set(epics)]
        b, _ = bucket(rows, ek, tk, vk, args.tolerance, lo, hi)
        print(f"  {label}:")
        show(b, unit, indent="    ")


if __name__ == "__main__":
    main()
