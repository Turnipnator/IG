#!/usr/bin/env python3
"""Do Gold breakout SHORTS only pay in a long-term downtrend (daily close < SMA_L)?

Pre-registered in research_notes.md ("Long-term REGIME filter for Gold breakout SHORTS",
2026-11-20). Read that before changing anything.

    venv/bin/python scripts/replay_gold_short_regime.py --count      # outcome-blind
    venv/bin/python scripts/replay_gold_short_regime.py --outcomes   # the verdict

Engine = replay_gold_short_trail.run (the live rule) with the HTF label forced to NEUTRAL
on bars outside the regime, so a SELL break there is refused. Offline.
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

START, HALVES = gst.START, gst.HALVES
ARMS, PRIMARY = (100, 200, 300), 200
SPLIT = pd.Timestamp("2016-01-01")
OUT = rbn.DATA / "gold_short_regime_outcomes.json"


def regime(df: pd.DataFrame, L: int) -> np.ndarray:
    """Per-1h-bar flag: last daily bar dated strictly before the bar's date closed < SMA_L."""
    g = df.groupby(df["date"].dt.normalize())["close"].agg(["last", "size"])
    d = g[g["size"] >= 12]["last"]
    below = (d < d.rolling(L).mean()) & d.rolling(L).mean().notna()
    tab = pd.DataFrame({"day": d.index, "flag": below.to_numpy()})
    q = pd.DataFrame({"D": df["date"].dt.normalize().to_numpy(), "pos": np.arange(len(df))}).sort_values("D")
    m = pd.merge_asof(q, tab, left_on="D", right_on="day", direction="backward", allow_exact_matches=False)
    return m.sort_values("pos")["flag"].fillna(False).to_numpy(bool)


def filtered(df: pd.DataFrame, flag: np.ndarray, diffs) -> pd.DataFrame:
    d2 = df.copy()
    d2["htf"] = np.where(flag, df["htf"].to_numpy(object), "NEUTRAL")
    return gst.run(d2, gst.BASE, diffs)


def count() -> None:
    diffs = rbn.rate_diffs()
    gst.fidelity(diffs)
    df = gst.frame()
    ok = True
    for L in ARMS:
        f = regime(df, L)
        t = filtered(df, f, diffs)
        late = int((t["date"] >= SPLIT).sum())
        print(f"L{L}: bars in regime {f[df['date'].to_numpy() >= np.datetime64(START)].mean():.1%} | trades n = {len(t)} | dated 2016+ = {late}")
        if L == PRIMARY:
            ok = len(t) >= 100 and late >= 30
    print(f"power gate (primary n >= 100 and >= 30 trades 2016+): {'PASS' if ok else 'UNDERPOWERED'}")


def kept_minus_dropped(r: np.ndarray, keep: np.ndarray) -> float:
    if keep.sum() == 0 or (~keep).sum() == 0:
        return np.nan
    return float(r[keep].mean() - r[~keep].mean())


def outcomes() -> None:
    diffs = rbn.rate_diffs()
    gst.fidelity(diffs)
    df = gst.frame()
    idx = pd.period_range(START, df["date"].max(), freq="Q")
    base = gst.run(df, gst.BASE, diffs)
    qb = rae.quarterly(base, idx)
    res = {"unfiltered_baseline": gst.summ(base), "arms": {}}
    q, sets, flags = {}, {}, {}
    for L in ARMS:
        flags[L] = regime(df, L)
        t = sets[L] = filtered(df, flags[L], diffs)
        q[L] = rae.quarterly(t, idx)
        dq = q[L] - qb
        res["arms"][f"L{L}"] = {**gst.summ(t), "delta_R": round(float(dq.sum()), 2), "ci95": rae.boot_ci(dq),
                                "bars_in_regime": round(float(flags[L].mean()), 3)}
    t, p = sets[PRIMARY], res["arms"][f"L{PRIMARY}"]
    by_year = t.groupby(t["date"].dt.year)["r"].agg(["count", "sum"]).round(2)
    res["primary_by_year"] = {int(y): {"n": int(c), "R": float(s)} for y, c, s in by_year.itertuples()}
    best_year = by_year["sum"].idxmax()
    no_best_year = float(t[t["date"].dt.year != best_year]["r"].sum())
    drop3 = float(np.sort(q[PRIMARY])[:-3].sum())
    # Rotation null on the unfiltered trades, by entry bar.
    pos = np.flatnonzero(df["date"].isin(base["date"]).to_numpy())
    assert len(pos) == len(base)
    r, f = base["r"].to_numpy(), flags[PRIMARY]
    obs = kept_minus_dropped(r, f[pos])
    rng = np.random.default_rng(0)
    yr = 24 * 260
    null = np.array([kept_minus_dropped(r, np.roll(f, int(s))[pos]) for s in rng.integers(yr, len(f) - yr, 2000)])
    null = null[np.isfinite(null)]
    pval = float((null >= obs).mean())
    res["rotation_null"] = {"kept_n": int(f[pos].sum()), "kept_net_R": round(float(r[f[pos]].sum()), 2),
                            "dropped_net_R": round(float(r[~f[pos]].sum()), 2),
                            "kept_minus_dropped_mean_R": round(obs, 4), "p_one_sided": round(pval, 4),
                            "null_95th": round(float(np.percentile(null, 95)), 4)}
    dq = q[PRIMARY] - qb
    ci = rae.boot_ci(dq)
    others = [res["arms"][f"L{L}"] for L in ARMS if L != PRIMARY]
    crit = {"1_primary_mean_gt0_t_ge_2": bool(p["mean_R"] > 0 and p["t"] >= 2.0),
            "2_delta_vs_unfiltered_gt0_ci_excludes0": bool(dq.sum() > 0 and ci[0] > 0),
            "3_primary_positive_both_halves": bool(all(v > 0 for v in p["by_half"].values())),
            "4_positive_without_best_year_and_without_top3_quarters": bool(no_best_year > 0 and drop3 > 0),
            "5_neighbours_positive_and_beat_unfiltered": bool(all(a["net_R"] > 0 and a["delta_R"] > 0 for a in others)),
            "6_rotation_null_p_lt_0.05": bool(pval < 0.05)}
    res["verdict"] = {"delta_vs_unfiltered": round(float(dq.sum()), 2), "ci95": ci, "best_year": int(best_year),
                      "primary_without_best_year": round(no_best_year, 2),
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
