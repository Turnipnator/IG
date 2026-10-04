#!/usr/bin/env python3
"""Would a NON-LIVE 1h breakout market earn a slot if it traded only on
scheduled-event days?

Pre-registered in research_notes.md ("Event-day mode switching", a503a40) with
amendments A1-A6 logged before any outcome. Read that before changing anything.

    venv/bin/python scripts/replay_event_day.py --count      # §5 gate, outcome-blind
    venv/bin/python scripts/replay_event_day.py --outcomes   # only if the gate passed

Trades: EUR/USD + Crude from the 09-22 replay cache (replay_trades.pkl, live gate,
measured spreads, C3 financing); Silver from replay_breakout_metals at 3.0 pt.
An entry is an EVENT DAY when the UTC date of its decision time holds a mapped
core event. Null: whole-week rotation of the event days (k = ±1..±26), dropping
any shifted day that is itself a real event day for that market.
Offline; touches no bot state.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_breakout_news as rbn  # noqa: E402
import replay_breakout_metals as rbm  # noqa: E402

EUR, CRUDE, SILVER = "CS.D.EURUSD.TODAY.IP", "CC.D.CL.USS.IP", rbm.SILVER
GOLD, GBP = "CS.D.USCGC.TODAY.IP", "CS.D.GBPUSD.TODAY.IP"
PRIMARY = (EUR, CRUDE, SILVER)
DESCRIPTIVE = (GOLD, GBP)
CCYS = {EUR: ("USD", "EUR"), CRUDE: ("USD",), SILVER: ("USD",),
        GOLD: ("USD",), GBP: ("USD", "GBP")}
NAMES = {EUR: "EUR/USD", CRUDE: "Crude Oil", SILVER: "Silver", GOLD: "Gold", GBP: "GBP/USD"}
HIGH_SPREAD = {EUR: 0.9, CRUDE: 4.5, SILVER: 4.0}    # A1: 1.5x base for EUR/Crude; silver as 10-02
START = pd.Timestamp("2005-01-01")
SIGMA_R_PRIOR = 2.1
MAX_POOLED_MDE = 0.30
MIN_EVENT_N = 60
SLIP_R = 0.25
SHIFTS = [k for k in range(-26, 27) if k != 0]
GATE = rbn.DATA / "event_day_count.json"
OUT = rbn.DATA / "event_day_outcomes.json"


# ----------------------------------------------------------------------------- tagging (times only)

def event_days(ev: pd.DataFrame, epic: str) -> np.ndarray:
    """Sorted unique UTC dates holding at least one mapped core event."""
    sel = ev[(ev["tier"] == "core") & ev["currency"].isin(CCYS[epic])]
    return np.unique(sel["utc"].dt.tz_convert("UTC").dt.tz_localize(None)
                     .dt.normalize().to_numpy("datetime64[D]"))


def decision_dates(decisions: pd.Series) -> np.ndarray:
    return decisions.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[D]")


def tag_event_day(decisions: pd.Series, days: np.ndarray) -> np.ndarray:
    return np.isin(decision_dates(decisions), days)


def shifted_days(real: np.ndarray, weeks: int) -> np.ndarray:
    """Event days moved by whole weeks; a shifted day that is itself a real
    event day is dropped, so a control day is never a real news day."""
    sh = real + np.timedelta64(7 * weeks, "D")
    return sh[~np.isin(sh, real)]


# ----------------------------------------------------------------------------- trades

def load_trades() -> pd.DataFrame:
    rbm.register()
    cache = pd.read_pickle(rbn.DATA / "replay_trades.pkl")
    assert cache.groupby("epic").size().to_dict() == {
        EUR: 1191, CRUDE: 745, GOLD: 1052, GBP: 1038}, "replay cache != Step-1 counts"
    cache = cache[cache["date"] >= START].copy()
    cache["cost_r"] = cache["r_gross"] - cache["r"]
    sil = rbm.run(SILVER, 3.0)
    sil["market"] = "Silver"
    sil["blocked_core-USD"] = rbn.tag_blocked(
        sil["decision_utc"], rbn.load_events().query("currency == 'USD'")["utc"]
        .dt.tz_localize(None).to_numpy("datetime64[ns]"))
    sil["blocked_core-GBP/EUR"] = False
    t = pd.concat([cache, sil], ignore_index=True)
    t["near_release"] = t["blocked_core-USD"].astype(bool) | t["blocked_core-GBP/EUR"].astype(bool)
    return t


def high_cost_r(epic: str) -> pd.DataFrame:
    """Re-run one market at its HIGH spread (A1)."""
    if epic == SILVER:
        return rbm.run(SILVER, HIGH_SPREAD[SILVER])
    rbn.SPREAD[epic] = HIGH_SPREAD[epic]
    df = rbn.load_market(epic)
    df["htf"] = rbn.htf_day_fast(df)
    t = rbn.run_gated(df, epic, rbn.rate_diffs())
    return t[t["date"] >= START].reset_index(drop=True)


# ----------------------------------------------------------------------------- stats

def t_one_sided_p(r: np.ndarray) -> tuple[float, float]:
    """(t, one-sided p for mean > 0); normal approximation, n >= 60 (A2)."""
    n = len(r)
    if n < 2:
        return float("nan"), float("nan")
    t = float(r.mean() / (r.std(ddof=1) / math.sqrt(n)))
    return t, 0.5 * math.erfc(t / math.sqrt(2))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = running
    return out


def delta(t: pd.DataFrame, flag: np.ndarray) -> float:
    return float(t.loc[flag, "r"].mean() - t.loc[~flag, "r"].mean())


# ----------------------------------------------------------------------------- stages

def tag_all(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ev = rbn.load_events()
    days = {e: event_days(ev, e) for e in CCYS}
    t["event_day"] = False
    for e in CCYS:
        sel = (t["epic"] == e).to_numpy()
        t.loc[sel, "event_day"] = tag_event_day(t.loc[sel, "decision_utc"], days[e])
    return t, days


def count() -> None:
    t, _ = tag_all(load_trades())
    res = {}
    for e in CCYS:
        x = t[t["epic"] == e]
        res[NAMES[e]] = {"n": int(len(x)), "n_event_day": int(x["event_day"].sum()),
                         "share": round(float(x["event_day"].mean()), 3),
                         "n_event_day_near_release": int((x["event_day"] & x["near_release"]).sum()),
                         "from": str(x["date"].min().date()), "to": str(x["date"].max().date())}
    pooled_n = sum(res[NAMES[e]]["n_event_day"] for e in PRIMARY)
    mde = 2.487 * SIGMA_R_PRIOR / math.sqrt(pooled_n)
    out = {"by_market": res, "pooled_primary_event_n": pooled_n, "pooled_mde_R": round(mde, 3),
           "gate_passes": bool(mde <= MAX_POOLED_MDE),
           "markets_eligible": [NAMES[e] for e in PRIMARY if res[NAMES[e]]["n_event_day"] >= MIN_EVENT_N]}
    GATE.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))                     # no R-derived number


def market_block(x: pd.DataFrame, hi: pd.DataFrame, days: np.ndarray) -> dict:
    ed = x[x["event_day"]]
    r = ed["r"].to_numpy(float)
    tstat, p = t_one_sided_p(r)
    halves = {"2005-2015": float(ed.loc[ed["date"] <= "2015-12-31", "r"].mean()),
              "2016-2026": float(ed.loc[ed["date"] >= "2016-01-01", "r"].mean())}
    ex_top3 = float(np.sort(r)[:-3].sum())
    ex_near = float(ed.loc[~ed["near_release"], "r"].mean())
    slip = float((ed["r"] - np.where(ed["why"] == "stop", SLIP_R, 0.0)).mean())
    hi_ed = hi[tag_event_day(hi["decision_utc"], days)]
    return {"n": int(len(ed)), "mean_r": round(float(r.mean()), 4), "t": round(tstat, 2), "p_one": p,
            "total_r": round(float(r.sum()), 1), "win_rate": round(float((r > 0).mean()), 3),
            "rest_mean_r": round(float(x.loc[~x["event_day"], "r"].mean()), 4),
            "delta": round(delta(x, x["event_day"].to_numpy()), 4),
            "halves": {k: round(v, 4) for k, v in halves.items()},
            "total_ex_top3": round(ex_top3, 1), "mean_ex_near_release": round(ex_near, 4),
            "n_ex_near_release": int((~ed["near_release"]).sum()),
            "mean_with_slip": round(slip, 4), "mean_cost_r": round(float(ed["cost_r"].mean()), 4),
            "mean_r_high_cost": round(float(hi_ed["r"].mean()), 4)}


def outcomes() -> None:
    gate = json.loads(GATE.read_text())
    if not gate["gate_passes"]:
        sys.exit("§5 gate failed — outcome run not permitted")
    t, days = tag_all(load_trades())
    prim = t[t["epic"].isin(PRIMARY)].reset_index(drop=True)

    # Pooled primary: event-day mean and Δ, rotation null on Δ (and on the mean, descriptive).
    flag = prim["event_day"].to_numpy()
    pr = prim.loc[flag, "r"].to_numpy(float)
    p_t, p_p = t_one_sided_p(pr)
    d_obs = delta(prim, flag)
    d_null, m_null = [], []
    for k in SHIFTS:
        f = np.zeros(len(prim), bool)
        for e in PRIMARY:
            sel = (prim["epic"] == e).to_numpy()
            f[sel] = tag_event_day(prim.loc[sel, "decision_utc"], shifted_days(days[e], k))
        real = flag                                           # a control trade must not be a real event-day trade
        f &= ~real
        rest = ~f & ~real
        d_null.append(float(prim.loc[f, "r"].mean() - prim.loc[rest, "r"].mean()))
        m_null.append(float(prim.loc[f, "r"].mean()))
    d_null = np.array(d_null)
    p_rot = float((np.abs(d_null) >= abs(d_obs)).mean())
    pooled = {"n": int(flag.sum()), "mean_r": round(float(pr.mean()), 4), "t": round(p_t, 2),
              "rest_mean_r": round(float(prim.loc[~flag, "r"].mean()), 4),
              "delta": round(d_obs, 4), "rotation_p_two_sided": round(p_rot, 3),
              "null_delta_median": round(float(np.median(d_null)), 4),
              "null_delta_95": [round(float(np.quantile(d_null, q)), 4) for q in (0.025, 0.975)],
              "null_eventday_mean_median": round(float(np.median(m_null)), 4)}
    pooled_ok = pr.mean() > 0 and p_t >= 2.0 and p_rot <= 0.05

    per = {}
    for e in PRIMARY:
        per[e] = market_block(t[t["epic"] == e], high_cost_r(e), days[e])
    hp = holm({e: per[e]["p_one"] for e in PRIMARY})
    verdicts = {}
    for e in PRIMARY:
        b = per[e]
        b["p_holm"] = round(hp[e], 4)
        b["p_one"] = round(b["p_one"], 4)
        crit = {"1_pooled_mean_t2_and_rotation": bool(pooled_ok),
                "2_own_mean_pos_holm_le_0.05": b["mean_r"] > 0 and hp[e] <= 0.05,
                "3_both_halves_pos": all(v > 0 for v in b["halves"].values()),
                "4_total_ex_top3_pos": b["total_ex_top3"] > 0,
                "5_pos_ex_near_release": b["mean_ex_near_release"] > 0,
                "6_pos_with_0.25R_stop_slip": b["mean_with_slip"] > 0,
                "7_cost_le_0.10_and_pos_high_cost": b["mean_cost_r"] <= 0.10 and b["mean_r_high_cost"] > 0,
                "n_ge_60": b["n"] >= MIN_EVENT_N}
        b["criteria"] = crit
        b["verdict"] = "SWITCH-WORTHY" if all(crit.values()) else "NO EDGE"
        verdicts[NAMES[e]] = b["verdict"]

    desc = {}
    for e in DESCRIPTIVE:
        x = t[t["epic"] == e]
        ed = x[x["event_day"]]
        desc[NAMES[e]] = {"n": int(len(ed)), "mean_r": round(float(ed["r"].mean()), 4),
                          "rest_mean_r": round(float(x.loc[~x["event_day"], "r"].mean()), 4),
                          "delta": round(delta(x, x["event_day"].to_numpy()), 4)}

    out = {"pooled_primary": pooled, "per_market": {NAMES[e]: per[e] for e in PRIMARY},
           "verdicts": verdicts, "descriptive_live_arms": desc}
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--outcomes", action="store_true")
    a = ap.parse_args()
    count() if a.count else outcomes()
