#!/usr/bin/env python3
"""Can DXY breakout pass the ≤0.10R cost gate on some frame, with an edge?

Pre-registered in research_notes.md ("Can DXY breakout pass the cost gate…",
0ce8e70) with amendments D1-D6 (a223fd8). This file runs the DAILY frame on
Yahoo DX-Y.NYB daily bars (D1); the 1h/4h frames wait for HistData m1.

    venv/bin/python scripts/replay_dxy_frames.py --count      # §6, outcome-blind
    venv/bin/python scripts/replay_dxy_frames.py --outcomes   # §7 + D6 null

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import daily_trend as dt  # noqa: E402

DATA = ROOT / "data" / "dxy"
CFG = dt.DailyTrendConfig()
START = pd.Timestamp("2005-01-01")
END = pd.Timestamp("2026-09-22")
SPREAD, SPREAD_LO, SPREAD_HI = 5.0, 3.0, 7.1
ADMIN = 0.025
CAP_DAILY_GBP = 250.0
MIN_N = 80
HOLM_P = 0.05 / 3          # D3: strictest step while 1h/4h are unrun
N_NULL = 500
COUNT_OUT = DATA / "daily_count.json"
OUT = DATA / "daily_outcomes.json"


# ----------------------------------------------------------------------------- data

def load_daily() -> pd.DataFrame:
    y = pd.read_csv(DATA / "yahoo_dxy_d1.csv", parse_dates=["date"])
    for c in ("open", "high", "low", "close"):
        y[c] = y[c] * 100.0                       # IG points
    y = y[y["date"] <= END]
    return dt.prepare_daily(y)


def us3m() -> pd.Series:
    x = pd.read_csv(ROOT / "data" / "fx_carry" / "IR3TIB01USM156N.csv")
    s = pd.Series(pd.to_numeric(x.iloc[:, 1], errors="coerce").values / 100.0,
                  index=pd.to_datetime(x.iloc[:, 0]).dt.to_period("M")).dropna()
    idx = pd.period_range("2003-01", "2026-12", freq="M")
    return s.reindex(idx).ffill()


# ----------------------------------------------------------------------------- engine (D4)

def sym_replay(df: pd.DataFrame, cfg: dt.DailyTrendConfig, shorts: bool = True, longs: bool = True,
               rng=None, p_enter=None, p_buy=None) -> list[dict]:
    """daily_trend.replay made symmetric. With shorts=False and no random entry it
    must equal dt.replay exactly (harness). With rng/p_enter it is the D6 null:
    on a flat bar, enter with probability p_enter[i], direction BUY w.p. p_buy;
    stop and channel exits are unchanged."""
    d = dt.add_levels(dt.prepare_daily(df), cfg)
    d["exit_hi"] = d["high"].rolling(cfg.n_exit).max().shift(1)
    d["entry_lo"] = d["low"].rolling(cfg.n_entry).min().shift(1)
    dates = d["date"].values
    o, h, l, c, atr = (d[x].values.astype(float) for x in ("open", "high", "low", "close", "atr"))
    hi, lo = d["entry_hi"].values.astype(float), d["exit_lo"].values.astype(float)
    ehi, elo = d["exit_hi"].values.astype(float), d["entry_lo"].values.astype(float)
    trades: list[dict] = []
    pos = 0; pend = 0; e_px = e_atr = stop = None; e_i = None
    for i in range(1, len(d)):
        if pend != pos:
            if pos:
                trades.append(_trade(dates, e_i, i, e_px, o[i], e_atr, "channel", pos))
                pos = 0
            if pend:
                pos = pend; e_px = o[i]; e_atr = atr[i - 1]; e_i = i
                stop = e_px - pos * cfg.stop_mult * e_atr
        if pos == 1 and l[i] <= stop:
            trades.append(_trade(dates, e_i, i, e_px, min(o[i], stop), e_atr, "stop", 1)); pos = 0; pend = 0
        elif pos == -1 and h[i] >= stop:
            trades.append(_trade(dates, e_i, i, e_px, max(o[i], stop), e_atr, "stop", -1)); pos = 0; pend = 0
        if not (np.isfinite(hi[i]) and np.isfinite(lo[i]) and np.isfinite(atr[i])
                and np.isfinite(ehi[i]) and np.isfinite(elo[i])):
            pend = pos; continue
        if pos == 0:
            if rng is not None:
                pend = 0
                if dates[i] >= START and rng.random() < p_enter[i]:
                    pend = 1 if rng.random() < p_buy else -1
            elif longs and c[i] > hi[i]:
                pend = 1
            elif shorts and c[i] < elo[i]:
                pend = -1
            else:
                pend = 0
        elif pos == 1:
            pend = 0 if c[i] < lo[i] else 1
        else:
            pend = 0 if c[i] > ehi[i] else -1
    return trades


def _trade(dates, e_i, x_i, e_px, x_px, e_atr, reason, side) -> dict:
    t = dt._trade(dates, e_i, x_i, e_px, x_px, e_atr, reason)
    if side == -1:
        t["pts"] = float(e_px - x_px)
        t["r_gross"] = float((e_px - x_px) / (2.0 * e_atr)) if e_atr else float("nan")
    t["d"] = "BUY" if side == 1 else "SELL"
    return t


def cost(t: pd.DataFrame, rates: pd.Series, spread: float, both_pay: bool) -> pd.DataFrame:
    t = t.copy()
    b = t["entry_date"].dt.to_period("M").map(rates).astype(float)
    rate = np.where((t["d"] == "BUY") | both_pay, b + ADMIN, ADMIN - b)
    nights = (t["exit_date"] - t["entry_date"]).dt.days
    sd = 2.0 * t["atr_at_entry"]
    fin = nights * rate / 365.0 * t["entry"]
    t["cost_r"] = (spread + fin) / sd
    t["r"] = t["r_gross"] - t["cost_r"]
    t["sd"] = sd
    return t


def trades_frame(df, shorts=True) -> pd.DataFrame:
    t = pd.DataFrame(sym_replay(df, CFG, shorts=shorts))
    return t[t["entry_date"] >= START].reset_index(drop=True)


# ----------------------------------------------------------------------------- stages

def harness(df) -> None:
    ref = dt.replay(df, CFG)
    mine = sym_replay(df, CFG, shorts=False)
    keys = ("entry_date", "exit_date", "entry", "exit", "reason")
    assert len(ref) == len(mine), f"harness: {len(ref)} vs {len(mine)} trades"
    for a, b in zip(ref, mine):
        assert all(a[k] == b[k] for k in keys), f"harness mismatch {a} vs {b}"
    print(f"Harness OK: long side == daily_trend.replay ({len(ref)} trades)")


def count() -> None:
    df = load_daily()
    harness(df)
    t = trades_frame(df)
    n = len(t)
    out = {"n": n, "by_direction": t["d"].value_counts().to_dict(),
           "per_year": round(n / ((END - START).days / 365.25), 2),
           "from": str(t["entry_date"].min().date()), "to": str(t["entry_date"].max().date()),
           "mde_R": round(2.487 * 2.1 / math.sqrt(n), 3) if n else None,
           "eligible": n >= MIN_N}
    COUNT_OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))              # no R-derived number


def stats(t: pd.DataFrame) -> dict:
    r = t["r"].to_numpy(float); n = len(r)
    tstat = float(r.mean() / (r.std(ddof=1) / math.sqrt(n)))
    return {"n": n, "mean_r": round(float(r.mean()), 4), "t": round(tstat, 2),
            "p_one": round(0.5 * math.erfc(tstat / math.sqrt(2)), 4),
            "total_r": round(float(r.sum()), 1), "win_rate": round(float((r > 0).mean()), 3)}


def outcomes() -> None:
    if not json.loads(COUNT_OUT.read_text())["eligible"]:
        sys.exit("§6: n < 80 — reported, cannot PASS; outcome run not permitted")
    df = load_daily(); rates = us3m()
    raw = trades_frame(df)
    base = cost(raw, rates, SPREAD, both_pay=False)
    hi = cost(raw, rates, SPREAD_HI, both_pay=True)
    lo = cost(raw, rates, SPREAD_LO, both_pay=False)
    s = stats(base)
    halves = {"2005-2015": round(float(base.loc[base.entry_date <= "2015-12-31", "r"].mean()), 4),
              "2016-2026": round(float(base.loc[base.entry_date >= "2016-01-01", "r"].mean()), 4)}
    ex10 = round(float(np.sort(base["r"].to_numpy())[:-10].sum()), 1)

    # D6 random-timing null
    d = dt.add_levels(dt.prepare_daily(df), CFG)
    real_pos = np.zeros(len(d), bool)
    idx = {pd.Timestamp(x): i for i, x in enumerate(d["date"])}
    for _, r in raw.iterrows():
        real_pos[idx[r.entry_date]:idx[r.exit_date] + 1] = True
    years = d["date"].dt.year.to_numpy()
    n_y = raw["entry_date"].dt.year.value_counts()
    flat_y = pd.Series(~real_pos, index=years).groupby(level=0).sum()
    p_enter = np.array([n_y.get(y, 0) / max(flat_y.get(y, 1), 1) for y in years])
    p_buy = float((raw["d"] == "BUY").mean())
    rng = np.random.default_rng(20261008)
    null = []
    for _ in range(N_NULL):
        nt = pd.DataFrame(sym_replay(df, CFG, rng=rng, p_enter=p_enter, p_buy=p_buy))
        nt = nt[nt["entry_date"] >= START]
        null.append(float(cost(nt, rates, SPREAD, False)["r"].mean()) if len(nt) else np.nan)
    null = np.array(null)
    q95 = float(np.nanquantile(null, 0.95))

    crit = {"1_mean_pos_p_le_0.0167": s["mean_r"] > 0 and s["p_one"] <= HOLM_P,
            "2_both_halves_pos": all(v > 0 for v in halves.values()),
            "3_total_ex_top10_pos": ex10 > 0,
            "4_cost_le_0.10R": float(base["cost_r"].mean()) <= 0.10,
            "5_pos_at_high_cost": float(hi["r"].mean()) > 0,
            "6_beats_random_timing_q95": s["mean_r"] > q95,
            "n_ge_80": s["n"] >= MIN_N}
    out = {"frame": "daily", **s, "halves": halves, "total_ex_top10": ex10,
           "mean_cost_r": round(float(base["cost_r"].mean()), 4),
           "median_cost_r": round(float(base["cost_r"].median()), 4),
           "mean_spread_r": round(float((SPREAD / base["sd"]).mean()), 4),
           "mean_r_high_cost": round(float(hi["r"].mean()), 4),
           "mean_r_low_cost": round(float(lo["r"].mean()), 4),
           "mean_gross_r": round(float(base["r_gross"].mean()), 4),
           "mean_nights": round(float((base.exit_date - base.entry_date).dt.days.mean()), 1),
           "by_direction": {k: stats(g) for k, g in base.groupby("d")},
           "null_mean_median": round(float(np.nanmedian(null)), 4), "null_q95": round(q95, 4),
           "null_p": round(float((null >= s["mean_r"]).mean()), 3),
           "share_over_250_cap_at_1gbp": round(float((base["sd"] * 1.0 > CAP_DAILY_GBP).mean()), 3),
           "median_stop_pts": round(float(base["sd"].median()), 1),
           "criteria": crit, "verdict": "PASS" if all(crit.values()) else "FAIL -> NO EDGE"}
    OUT.write_text(json.dumps(out, indent=1, default=str))
    base.to_csv(DATA / "daily_trades.csv", index=False)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--outcomes", action="store_true")
    a = ap.parse_args()
    count() if a.count else outcomes()
