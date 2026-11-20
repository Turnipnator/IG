#!/usr/bin/env python3
"""Does a faster Donchian trail suit Gold breakout SHORTS?

Pre-registered in research_notes.md ("Faster Donchian trail for Gold breakout SHORTS",
2026-11-20). Read that before changing anything.

    venv/bin/python scripts/replay_gold_short_trail.py --count      # outcome-blind
    venv/bin/python scripts/replay_gold_short_trail.py --outcomes   # the verdict

Engine = replay_breakout_atr_exit.run's baseline path with the exit-channel lookback M
as a parameter. M=27 must reproduce that script's baseline Gold SELL trades exactly.
Offline; touches no bot state.
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
from config import MARKETS  # noqa: E402
from src.breakout import BREAKOUT_CONFIGS  # noqa: E402
from src.indicators import calculate_atr  # noqa: E402

GOLD = "CS.D.USCGC.TODAY.IP"
START = rae.START
ARMS = (27, 20, 13, 8)
BASE, PRIMARY = 27, 13
HALVES = rae.HALVES
OUT = rbn.DATA / "gold_short_trail_outcomes.json"


def run(df: pd.DataFrame, M: int, diffs, direction: str = "SELL", k: float | None = None) -> pd.DataFrame:
    cfg = BREAKOUT_CONFIGS[GOLD]
    mk = next(m for m in MARKETS if m.epic == GOLD)
    N, k0 = cfg.n, cfg.stop_atr_mult
    k = k0 if k is None else k
    h_lo, h_hi = (mk.trading_start + 1) % 24, (mk.trading_end + 1) % 24
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = calculate_atr(df["high"], df["low"], df["close"], 14).to_numpy(float)
    htf = df["htf"].to_numpy(object)
    dates = df["date"].to_numpy("datetime64[ns]")
    hours = df["date"].dt.hour.to_numpy()
    cost = rbn.SPREAD[GOLD]
    out = []
    i, n = N + 1, len(df)
    while i < n:
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1; continue
        upper, lower = hi[i - N:i].max(), lo[i - N:i].min()
        d = "BUY" if hi[i] >= upper else "SELL" if lo[i] <= lower else None
        if d != direction:
            i += 1; continue
        hh = hours[i]
        if (hh < h_lo or hh >= h_hi) if h_lo < h_hi else (h_hi <= hh < h_lo):
            i += 1; continue
        if htf[i] != ("BULLISH" if d == "BUY" else "BEARISH"):
            i += 1; continue
        s = 1.0 if d == "BUY" else -1.0
        entry = cl[i]
        sd = max(a * k, mk.min_stop_distance)
        sd0 = max(a * k0, mk.min_stop_distance)     # the live 2xATR stop, for fixed-size reporting
        stop = entry - s * sd
        j, xp, why = i + 1, None, None
        while j < n:
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * stop:
                xp, why = (min(o[j], stop) if d == "BUY" else max(o[j], stop)), "stop"; break
            lvl = lo[j - M:j].min() if d == "BUY" else hi[j - M:j].max()
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * lvl:
                xp, why = (min(o[j], lvl) if d == "BUY" else max(o[j], lvl)), "trail"; break
            j += 1
        if xp is None:
            xp, why, j = cl[-1], "open-at-end", n - 1
        g = s * (xp - entry)
        mfe = max(s * ((hi[i + 1:j + 1].max() if d == "BUY" else lo[i + 1:j + 1].min()) - entry), 0.0) if j > i else 0.0
        nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
        fin = nights * rbn.financing_rate(GOLD, d, pd.Timestamp(dates[i]), diffs) / 365.0 * entry
        # After a stop-out: did price trade a further 1R BEYOND the entry (in the trade's
        # direction) within the next 120 bars (~5 sessions)? Exploratory only.
        cont = np.nan
        if why == "stop":
            w = slice(j + 1, min(j + 121, n))
            if w.stop > w.start:
                cont = float(s * ((hi[w].max() if d == "BUY" else lo[w].min()) - entry) >= sd)
        out.append(dict(date=pd.Timestamp(dates[i]), d=d, why=why, sd=sd, r=(g - cost - fin) / sd,
                        r_gross=g / sd, mfe_r=mfe / sd, bars=j - i, stop_then_1R=cont,
                        r_fixed=(g - cost - fin) / sd0))
        i = j + 1
    t = pd.DataFrame(out)
    return t[t["date"] >= START].reset_index(drop=True)


_DF = None


def frame() -> pd.DataFrame:
    global _DF
    if _DF is None:
        _DF = rbn.load_market(GOLD)
        _DF["htf"] = rbn.htf_day_fast(_DF)
    return _DF


def fidelity(diffs) -> None:
    ref = rae.run(frame(), GOLD, diffs, directions=("SELL",))
    ref = ref[ref["date"] >= START].reset_index(drop=True)
    mine = run(frame(), BASE, diffs)
    assert len(ref) == len(mine) and (ref["date"].values == mine["date"].values).all() \
        and float((ref["r"] - mine["r"]).abs().max()) < 1e-9, "M27 does not reproduce the 10-10 engine"
    print(f"fidelity: {len(mine)} Gold SELL baseline trades reproduced exactly")


def count() -> None:
    diffs = rbn.rate_diffs()
    fidelity(diffs)
    df = frame()
    print(f"data {df['date'].min()} -> {df['date'].max()}")
    for M in ARMS:
        print(f"M{M}: n = {len(run(df, M, diffs))}")
    n = len(run(df, BASE, diffs))
    print(f"power gate (base n >= 100): {'PASS' if n >= 100 else 'UNDERPOWERED'}")


def summ(t: pd.DataFrame) -> dict:
    se = t["r"].std(ddof=1) / np.sqrt(len(t))
    ran = t[t["mfe_r"] >= 1.0]
    return {"n": len(t), "net_R": round(float(t["r"].sum()), 2), "mean_R": round(float(t["r"].mean()), 4),
            "t": round(float(t["r"].mean() / se), 2), "win_rate": round(float((t["r"] > 0).mean()), 3),
            "max_win_R": round(float(t["r"].max()), 2), "median_bars": int(t["bars"].median()),
            "stopped_at_initial": round(float((t["why"] == "stop").mean()), 3),
            "runners_1R": len(ran),
            "runner_median_kept_share": round(float((ran["r_gross"] / ran["mfe_r"]).median()), 2) if len(ran) else None,
            "by_half": {f"{a[:4]}-{b[:4]}": round(float(t[(t.date >= a) & (t.date <= b)]["r"].sum()), 2) for a, b in HALVES}}


def outcomes() -> None:
    diffs = rbn.rate_diffs()
    fidelity(diffs)
    df = frame()
    idx = pd.period_range(START, df["date"].max(), freq="Q")
    sets = {M: run(df, M, diffs) for M in ARMS}
    q = {M: rae.quarterly(t, idx) for M, t in sets.items()}
    years = np.array([p.year for p in idx])
    res = {"arms": {}}
    for M, t in sets.items():
        dq = q[M] - q[BASE]
        res["arms"][f"M{M}"] = {**summ(t), "delta_R": round(float(dq.sum()), 2),
                                "ci95": rae.boot_ci(dq) if M != BASE else None}
    dq = q[PRIMARY] - q[BASE]
    ci = rae.boot_ci(dq)
    halves = {f"{a[:4]}-{b[:4]}": round(float(dq[(years >= int(a[:4])) & (years <= int(b[:4]))].sum()), 2)
              for a, b in HALVES}
    drop3 = float(np.sort(dq)[:-3].sum())
    p = res["arms"][f"M{PRIMARY}"]
    crit = {"1_delta_gt0_ci_excludes0": bool(dq.sum() > 0 and ci[0] > 0),
            "2_primary_mean_gt0_t_ge_2": bool(p["mean_R"] > 0 and p["t"] >= 2.0),
            "3_delta_positive_both_halves": bool(all(v > 0 for v in halves.values())),
            "4_delta_positive_without_top3_quarters": bool(drop3 > 0),
            "5_neighbours_M8_M20_both_positive": bool(res["arms"]["M8"]["delta_R"] > 0 and res["arms"]["M20"]["delta_R"] > 0)}
    res["verdict"] = {"delta_M13_vs_M27": round(float(dq.sum()), 2), "ci95": ci, "halves": halves,
                      "drop_top3_quarters": round(drop3, 2), "criteria": crit,
                      "passed": int(sum(crit.values())), "result": "PASS" if all(crit.values()) else "FAIL"}
    # Exploratory context for the live-loser question: the long leg under the same engine,
    # and how often a stopped trade was "right" (price went 1R beyond entry within 120 bars).
    buy = run(df, BASE, diffs, "BUY")
    res["exploratory"] = {
        "BUY_M27": summ(buy),
        "stopped_then_1R_beyond_entry_within_120_bars": {
            "SELL": {"n_stopped": int((sets[BASE]["why"] == "stop").sum()),
                     "share": round(float(sets[BASE]["stop_then_1R"].mean()), 3)},
            "BUY": {"n_stopped": int((buy["why"] == "stop").sum()),
                    "share": round(float(buy["stop_then_1R"].mean()), 3)}}}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


STOP_ARMS = (2.0, 2.5, 3.0, 4.0)
STOP_BASE, STOP_PRIMARY = 2.0, 3.0
STOP_OUT = rbn.DATA / "gold_short_stop_outcomes.json"
CAP_GBP, MIN_SIZE = 45.0, 1.0      # config max_risk_gbp; IG minimum £/pt on Gold


def stop_count() -> None:
    diffs = rbn.rate_diffs()
    fidelity(diffs)
    for k in STOP_ARMS:
        t = run(frame(), BASE, diffs, k=k)
        recent = t[t["date"] >= "2025-01-01"]
        print(f"k={k}: n = {len(t)} | stop > £{CAP_GBP:.0f} at £{MIN_SIZE:.0f}/pt: all-time "
              f"{(t['sd'] * MIN_SIZE > CAP_GBP).mean():.1%}, since 2025 {(recent['sd'] * MIN_SIZE > CAP_GBP).mean():.1%} (n={len(recent)})")
    print(f"power gate (base n >= 100): {'PASS' if len(run(frame(), BASE, diffs, k=STOP_BASE)) >= 100 else 'UNDERPOWERED'}")


def stop_outcomes() -> None:
    diffs = rbn.rate_diffs()
    fidelity(diffs)
    df = frame()
    idx = pd.period_range(START, df["date"].max(), freq="Q")
    sets = {k: run(df, BASE, diffs, k=k) for k in STOP_ARMS}
    q = {k: rae.quarterly(t, idx) for k, t in sets.items()}
    years = np.array([p.year for p in idx])
    res = {"arms": {}}
    for k, t in sets.items():
        dq = q[k] - q[STOP_BASE]
        recent = t[t["date"] >= "2025-01-01"]
        res["arms"][f"k{k}"] = {**summ(t), "delta_R": round(float(dq.sum()), 2),
                                "ci95": rae.boot_ci(dq) if k != STOP_BASE else None,
                                "fixed_size_net_in_base_R": round(float(t["r_fixed"].sum()), 2),
                                "share_over_cap_since_2025": round(float((recent["sd"] * MIN_SIZE > CAP_GBP).mean()), 3),
                                "stopped_then_1R_share": round(float(t["stop_then_1R"].mean()), 3)}
    dq = q[STOP_PRIMARY] - q[STOP_BASE]
    ci = rae.boot_ci(dq)
    halves = {f"{a[:4]}-{b[:4]}": round(float(dq[(years >= int(a[:4])) & (years <= int(b[:4]))].sum()), 2)
              for a, b in HALVES}
    drop3 = float(np.sort(dq)[:-3].sum())
    p = res["arms"][f"k{STOP_PRIMARY}"]
    crit = {"1_delta_gt0_ci_excludes0": bool(dq.sum() > 0 and ci[0] > 0),
            "2_primary_mean_gt0_t_ge_2": bool(p["mean_R"] > 0 and p["t"] >= 2.0),
            "3_delta_positive_both_halves": bool(all(v > 0 for v in halves.values())),
            "4_delta_positive_without_top3_quarters": bool(drop3 > 0),
            "5_neighbours_k2.5_k4_both_positive": bool(res["arms"]["k2.5"]["delta_R"] > 0 and res["arms"]["k4.0"]["delta_R"] > 0)}
    res["verdict"] = {"delta_k3_vs_k2": round(float(dq.sum()), 2), "ci95": ci, "halves": halves,
                      "drop_top3_quarters": round(drop3, 2), "criteria": crit,
                      "passed": int(sum(crit.values())), "result": "PASS" if all(crit.values()) else "FAIL"}
    STOP_OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--outcomes", action="store_true")
    g.add_argument("--stop-count", action="store_true")
    g.add_argument("--stop-outcomes", action="store_true")
    a = ap.parse_args()
    {"count": count, "outcomes": outcomes, "stop_count": stop_count, "stop_outcomes": stop_outcomes}[
        next(k for k, v in vars(a).items() if v)]()
