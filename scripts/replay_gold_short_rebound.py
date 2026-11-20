#!/usr/bin/env python3
"""Sell the REBOUND after a bearish Gold break instead of the break itself.

Pre-registered in research_notes.md ("Sell the REBOUND after a bearish break, Gold
shorts", 2026-11-20). Read that before changing anything.

    venv/bin/python scripts/replay_gold_short_rebound.py --count      # outcome-blind
    venv/bin/python scripts/replay_gold_short_rebound.py --outcomes   # the verdict

Comparator = replay_gold_short_trail.run(M=27) (the live break entry). Offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_breakout_atr_exit as rae  # noqa: E402
import replay_breakout_news as rbn  # noqa: E402
import replay_gold_short_trail as gst  # noqa: E402
from config import MARKETS  # noqa: E402
from src.breakout import BREAKOUT_CONFIGS  # noqa: E402
from src.indicators import calculate_atr  # noqa: E402

GOLD, START, HALVES = gst.GOLD, gst.START, gst.HALVES
ARMS, PRIMARY, W = (0.5, 1.0, 1.5), 1.0, 24
OUT = rbn.DATA / "gold_short_rebound_outcomes.json"


def run(df: pd.DataFrame, x: float, diffs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (trades, signal episodes). An episode = one resting order from the bar it
    was (re)armed to its fill or expiry."""
    cfg = BREAKOUT_CONFIGS[GOLD]
    mk = next(m for m in MARKETS if m.epic == GOLD)
    N, M, k = cfg.n, cfg.m, cfg.stop_atr_mult
    h_lo, h_hi = (mk.trading_start + 1) % 24, (mk.trading_end + 1) % 24
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = calculate_atr(df["high"], df["low"], df["close"], 14).to_numpy(float)
    htf = df["htf"].to_numpy(object)
    dates = df["date"].to_numpy("datetime64[ns]")
    hours = df["date"].dt.hour.to_numpy()
    cost = rbn.SPREAD[GOLD]
    trades, episodes = [], []
    armed = None                      # (signal bar, limit, stop distance, last valid bar)
    i, n = N + 1, len(df)
    while i < n:
        if armed is not None and i > armed[3]:
            episodes.append(dict(date=pd.Timestamp(dates[armed[0]]), filled=False))
            armed = None
        if armed is not None and hi[i] >= armed[1]:
            sig, L, sd, _ = armed
            armed = None
            entry = max(o[i], L)
            stop = entry + sd
            j, xp, why = i, None, None
            if hi[i] >= stop:
                xp, why = stop, "stop"
            else:
                j = i + 1
                while j < n:
                    if hi[j] >= stop:
                        xp, why = max(o[j], stop), "stop"; break
                    lvl = hi[j - M:j].max()
                    if hi[j] >= lvl:
                        xp, why = max(o[j], lvl), "trail"; break
                    j += 1
                if xp is None:
                    xp, why, j = cl[-1], "open-at-end", n - 1
            g = entry - xp
            nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
            fin = nights * rbn.financing_rate(GOLD, "SELL", pd.Timestamp(dates[i]), diffs) / 365.0 * entry
            trades.append(dict(date=pd.Timestamp(dates[i]), signal=pd.Timestamp(dates[sig]), why=why, sd=sd,
                               r=(g - cost - fin) / sd, r_gross=g / sd, bars=j - i, wait=i - sig))
            episodes.append(dict(date=pd.Timestamp(dates[sig]), filled=True))
            i = j + 1
            continue
        a = atr[i]
        hh = hours[i]
        in_hours = not ((hh < h_lo or hh >= h_hi) if h_lo < h_hi else (h_hi <= hh < h_lo))
        if np.isfinite(a) and a > 0 and in_hours and htf[i] == "BEARISH" and lo[i] <= lo[i - N:i].min() \
                and not hi[i] >= hi[i - N:i].max():       # the live rule checks the BUY side first
            if armed is not None:
                episodes.append(dict(date=pd.Timestamp(dates[armed[0]]), filled=False, replaced=True))
            armed = (i, cl[i] + x * a, max(a * k, mk.min_stop_distance), i + W)
        i += 1
    t, e = pd.DataFrame(trades), pd.DataFrame(episodes)
    return t[t["date"] >= START].reset_index(drop=True), e[e["date"] >= START].reset_index(drop=True)


def summ(t: pd.DataFrame) -> dict:
    se = t["r"].std(ddof=1) / np.sqrt(len(t))
    return {"n": len(t), "net_R": round(float(t["r"].sum()), 2), "mean_R": round(float(t["r"].mean()), 4),
            "t": round(float(t["r"].mean() / se), 2), "win_rate": round(float((t["r"] > 0).mean()), 3),
            "max_win_R": round(float(t["r"].max()), 2), "ended_at_initial_stop": round(float((t["why"] == "stop").mean()), 3),
            "by_half": {f"{a[:4]}-{b[:4]}": round(float(t[(t.date >= a) & (t.date <= b)]["r"].sum()), 2) for a, b in HALVES}}


def count() -> None:
    diffs = rbn.rate_diffs()
    gst.fidelity(diffs)
    for x in ARMS:
        t, e = run(gst.frame(), x, diffs)
        print(f"x={x}: trades n = {len(t)} | order episodes {len(e)} | filled {e['filled'].mean():.1%} | median wait {int(t['wait'].median())} bars")
    n = len(run(gst.frame(), PRIMARY, diffs)[0])
    print(f"power gate (primary n >= 100): {'PASS' if n >= 100 else 'UNDERPOWERED'}")


def outcomes() -> None:
    diffs = rbn.rate_diffs()
    gst.fidelity(diffs)
    df = gst.frame()
    idx = pd.period_range(START, df["date"].max(), freq="Q")
    base = gst.run(df, gst.BASE, diffs)
    qb = rae.quarterly(base, idx)
    years = np.array([p.year for p in idx])
    res = {"break_entry_baseline": gst.summ(base), "arms": {}}
    q = {}
    for x in ARMS:
        t, e = run(df, x, diffs)
        q[x] = rae.quarterly(t, idx)
        dq = q[x] - qb
        res["arms"][f"x{x}"] = {**summ(t), "delta_R": round(float(dq.sum()), 2), "ci95": rae.boot_ci(dq),
                                "episodes": len(e), "fill_rate": round(float(e["filled"].mean()), 3),
                                "median_wait_bars": int(t["wait"].median())}
        if x == PRIMARY:
            # Baseline break-entry trades whose signal bar's rebound order never produced a fill
            # carrying that signal date: the runaways the rebound entry gives up.
            hit = set(t["signal"])
            miss = base[~base["date"].isin(hit)]
            res["exploratory_missed_break_trades"] = {
                "n": len(miss), "net_R_at_break_entry": round(float(miss["r"].sum()), 2),
                "mean_R": round(float(miss["r"].mean()), 3),
                "taken_n": int(base["date"].isin(hit).sum()),
                "taken_net_R_at_break_entry": round(float(base[base["date"].isin(hit)]["r"].sum()), 2)}
    p = res["arms"][f"x{PRIMARY}"]
    dq = q[PRIMARY] - qb
    ci = rae.boot_ci(dq)
    drop3 = float(np.sort(q[PRIMARY])[:-3].sum())
    others = [res["arms"][f"x{x}"] for x in ARMS if x != PRIMARY]
    crit = {"1_primary_mean_gt0_t_ge_2": bool(p["mean_R"] > 0 and p["t"] >= 2.0),
            "2_delta_vs_break_gt0_ci_excludes0": bool(dq.sum() > 0 and ci[0] > 0),
            "3_primary_positive_both_halves": bool(all(v > 0 for v in p["by_half"].values())),
            "4_primary_positive_without_top3_quarters": bool(drop3 > 0),
            "5_neighbours_positive_and_beat_break": bool(all(a["net_R"] > 0 and a["delta_R"] > 0 for a in others))}
    res["verdict"] = {"delta_vs_break": round(float(dq.sum()), 2), "ci95": ci,
                      "primary_without_top3_quarters": round(drop3, 2), "criteria": crit,
                      "passed": int(sum(crit.values())), "result": "PASS" if all(crit.values()) else "FAIL"}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--outcomes", action="store_true")
    a = ap.parse_args()
    count() if a.count else outcomes()
