#!/usr/bin/env python3
"""Does an ATR roll-over during a profitable breakout say "tighten the trail now"?

Pre-registered in research_notes.md ("Does ATR 'rolling over' tell us when to tighten
the breakout trail?", 2026-10-10). Read that before changing anything.

    venv/bin/python scripts/replay_breakout_atr_exit.py --count      # outcome-blind
    venv/bin/python scripts/replay_breakout_atr_exit.py --outcomes   # the verdict

Engine, data and costs are replay_breakout_news.run_gated's; only the exit differs per
arm. The baseline arm (both directions) must reproduce data/news_events/replay_trades.pkl
before anything else is computed. Offline; touches no bot state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_breakout_news as rbn  # noqa: E402
from config import MARKETS  # noqa: E402
from src.breakout import BREAKOUT_CONFIGS  # noqa: E402
from src.indicators import calculate_atr  # noqa: E402

START = pd.Timestamp("2005-01-01")
ROLL, FLAT_LAG, TIGHT_M, CHAND_K = 0.80, 5, 8, 3.0
HALVES = [("2005-01-01", "2015-12-31"), ("2016-01-01", "2026-12-31")]
OUT = rbn.DATA / "atr_exit_outcomes.json"
# arm -> (signal kind, "runs well" threshold in R, act on it?)
ARMS = {"base": (None, 0.0, False),
        "P1": ("P", 1.0, True), "P2": ("P", 2.0, True),
        "A1": ("A", 1.0, True), "A2": ("A", 2.0, True),
        "B1": ("B", 1.0, True), "B2": ("B", 2.0, True),
        "C1": ("C", 1.0, True), "C2": ("C", 2.0, True),
        "X1": ("X", 1.0, True)}
PRIMARY, CONTROL = "A1", "P1"


def run(df: pd.DataFrame, epic: str, diffs, kind=None, thr=0.0, act=False,
        directions=("BUY", "SELL")) -> pd.DataFrame:
    """rbn.run_gated with a switchable exit. Everything a rule reads comes from bars
    CLOSED before bar j; the level it sets is in force during bar j. With act=False the
    signal is only recorded (trigger bar + close) and the baseline exit is kept."""
    cfg = BREAKOUT_CONFIGS[epic]
    mk = next(m for m in MARKETS if m.epic == epic)
    N, M, k = cfg.n, cfg.m, cfg.stop_atr_mult
    h_lo, h_hi = (mk.trading_start + 1) % 24, (mk.trading_end + 1) % 24
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = calculate_atr(df["high"], df["low"], df["close"], 14).to_numpy(float)
    htf = df["htf"].to_numpy(object)
    dates = df["date"].to_numpy("datetime64[ns]")
    hours = df["date"].dt.hour.to_numpy()
    cost = rbn.SPREAD[epic]
    out = []
    i, n = N + 1, len(df)
    while i < n:
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1; continue
        upper, lower = hi[i - N:i].max(), lo[i - N:i].min()
        d = "BUY" if hi[i] >= upper else "SELL" if lo[i] <= lower else None
        if d is None or d not in directions:
            i += 1; continue
        hh = hours[i]
        if (hh < h_lo or hh >= h_hi) if h_lo < h_hi else (h_hi <= hh < h_lo):
            i += 1; continue
        if htf[i] != ("BULLISH" if d == "BUY" else "BEARISH"):
            i += 1; continue
        s = 1.0 if d == "BUY" else -1.0
        entry = cl[i]
        sd = max(a * k, mk.min_stop_distance)
        stop = entry - s * sd
        mfe, peak_atr, trig, trig_bar, chand = 0.0, a, False, -1, None
        j, xp, why = i + 1, None, None
        while j < n:
            p = j - 1                                   # last closed bar
            if p > i:
                mfe = max(mfe, s * ((hi[p] if d == "BUY" else lo[p]) - entry))
            if np.isfinite(atr[p]):
                peak_atr = max(peak_atr, atr[p])
            if kind and not trig and mfe >= thr * sd:
                if kind in ("P", "C"):
                    trig = True
                elif kind in ("A", "X"):
                    trig = atr[p] <= ROLL * peak_atr
                elif kind == "B":
                    trig = atr[p] < atr[p - FLAT_LAG]
                if trig:
                    trig_bar = p
                    if act and kind == "X":
                        xp, why = o[j], "atr-exit"; break
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * stop:
                xp, why = (min(o[j], stop) if d == "BUY" else max(o[j], stop)), "stop"; break
            m = TIGHT_M if (act and trig and kind in ("P", "A", "B")) else M
            lvl = lo[j - m:j].min() if d == "BUY" else hi[j - m:j].max()
            if act and trig and kind == "C":
                c = entry + s * mfe - s * CHAND_K * atr[p]
                chand = c if chand is None else (max(chand, c) if d == "BUY" else min(chand, c))
                lvl = max(lvl, chand) if d == "BUY" else min(lvl, chand)
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * lvl:
                xp, why = (min(o[j], lvl) if d == "BUY" else max(o[j], lvl)), "trail"; break
            j += 1
        if xp is None:
            xp, why, j = cl[-1], "open-at-end", n - 1
        g = s * (xp - entry)
        mfe_all = max(mfe, s * ((hi[i + 1:j + 1].max() if d == "BUY" else lo[i + 1:j + 1].min()) - entry)) if j > i else 0.0
        nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
        fin = nights * rbn.financing_rate(epic, d, pd.Timestamp(dates[i]), diffs) / 365.0 * entry
        out.append(dict(epic=epic, market=mk.name, date=pd.Timestamp(dates[i]), d=d, why=why, sd=sd,
                        r=(g - cost - fin) / sd, r_gross=g / sd, mfe_r=max(mfe_all, 0.0) / sd,
                        bars=j - i, trig=trig,
                        cont_r=(s * (xp - cl[trig_bar]) / sd) if trig else np.nan))
        i = j + 1
    return pd.DataFrame(out)


_FRAMES: dict = {}


def frames() -> dict:
    if not _FRAMES:
        for epic in rbn.UNIVERSE:
            df = rbn.load_market(epic)
            df["htf"] = rbn.htf_day_fast(df)
            _FRAMES[epic] = df
    return _FRAMES


def trades(arm: str, directions=("BUY",), act=None) -> pd.DataFrame:
    kind, thr, a = ARMS[arm]
    diffs = rbn.rate_diffs()
    t = pd.concat([run(df, e, diffs, kind, thr, a if act is None else act, directions)
                   for e, df in frames().items()], ignore_index=True)
    return t[t["date"] >= START].reset_index(drop=True)


def fidelity() -> None:
    ref = pd.read_pickle(rbn.DATA / "replay_trades.pkl")
    diffs = rbn.rate_diffs()
    mine = pd.concat([run(df, e, diffs) for e, df in frames().items()], ignore_index=True)
    mine = mine[mine["date"] <= ref["date"].max()]
    j = ref.merge(mine, on=["epic", "date", "d"], how="outer", suffixes=("_ref", ""), indicator=True)
    bad = int((j["_merge"] != "both").sum())
    worst = float((j["r_ref"] - j["r"]).abs().max())
    assert bad == 0 and worst < 1e-9, f"baseline does not reproduce replay_trades.pkl ({bad} unmatched, max |dr| {worst})"
    print(f"fidelity: {len(ref)} trades reproduced exactly")


def count() -> None:
    fidelity()
    base = trades("base")
    print(f"BUY-only baseline trades from {START.date()}: {len(base)}")
    print(base.groupby("market").size().to_string())
    for arm in ("P1", "P2", "A1", "A2", "B1", "B2"):
        t = trades(arm, act=False)              # signal observed on baseline trades
        print(f"{arm}: fires on {int(t['trig'].sum())} of {len(t)}  "
              + str(t[t["trig"]].groupby("market").size().to_dict()))
    n = int(trades(PRIMARY, act=False)["trig"].sum())
    print(f"gate (A1 >= 100 fires): {'PASS' if n >= 100 else 'UNDERPOWERED'}")


def quarterly(t: pd.DataFrame, idx) -> np.ndarray:
    return t.groupby(t["date"].dt.to_period("Q"))["r"].sum().reindex(idx, fill_value=0.0).to_numpy()


def boot_ci(x: np.ndarray, seed: int = 0, n: int = 10_000) -> list:
    rng = np.random.default_rng(seed)
    sums = x[rng.integers(0, len(x), size=(n, len(x)))].sum(axis=1)
    return [round(float(np.percentile(sums, 2.5)), 2), round(float(np.percentile(sums, 97.5)), 2)]


def summarise(t: pd.DataFrame) -> dict:
    return {"n": len(t), "net_R": round(float(t["r"].sum()), 2), "mean_R": round(float(t["r"].mean()), 4),
            "win_rate": round(float((t["r"] > 0).mean()), 3), "max_win_R": round(float(t["r"].max()), 2)}


def outcomes() -> None:
    fidelity()
    idx = pd.period_range(START, "2026-09-30", freq="Q")
    res = {}
    for label, dirs in (("BUY-only (primary)", ("BUY",)), ("both directions (robustness)", ("BUY", "SELL"))):
        sets = {arm: trades(arm, dirs) for arm in ARMS}
        q = {arm: quarterly(t, idx) for arm, t in sets.items()}
        block = {"arms": {}}
        for arm, t in sets.items():
            dq = q[arm] - q["base"]
            block["arms"][arm] = {**summarise(t), "delta_R": round(float(dq.sum()), 2),
                                  "ci95": boot_ci(dq) if arm != "base" else None,
                                  "by_market_delta": {m: round(float(t[t.market == m]["r"].sum()
                                                      - sets["base"][sets["base"].market == m]["r"].sum()), 2)
                                                      for m in sorted(t.market.unique())}}
        if dirs == ("BUY",):
            dq = q[PRIMARY] - q["base"]
            dc = q[PRIMARY] - q[CONTROL]
            years = np.array([p.year for p in idx])
            halves = {f"{a[:4]}-{b[:4]}": round(float(dq[(years >= int(a[:4])) & (years <= int(b[:4]))].sum()), 2)
                      for a, b in HALVES}
            gold = block["arms"][PRIMARY]["by_market_delta"]["Gold"]
            drop3 = float(np.sort(dq)[:-3].sum())
            ci, cic = boot_ci(dq), boot_ci(dc)
            crit = {"1_delta_gt0_ci_excludes0": bool(dq.sum() > 0 and ci[0] > 0),
                    "2_beats_profit_only_control": bool(dc.sum() > 0 and cic[0] > 0),
                    "3_positive_both_halves": bool(all(v > 0 for v in halves.values())),
                    "4_positive_on_gold": bool(gold > 0),
                    "5_positive_without_top3_quarters": bool(drop3 > 0)}
            block["verdict"] = {"delta_vs_base": round(float(dq.sum()), 2), "ci95": ci,
                                "delta_vs_control": round(float(dc.sum()), 2), "ci95_control": cic,
                                "halves": halves, "gold": gold, "drop_top3_quarters": round(drop3, 2),
                                "criteria": crit, "result": "PASS" if all(crit.values()) else "FAIL"}
            # Reported regardless: give-back on baseline, continuation value from the trigger bar.
            b = sets["base"]
            ran = b[b["mfe_r"] >= 1.0]
            block["giveback"] = {
                "trades": len(b), "reached_1R": len(ran), "reached_2R": int((b["mfe_r"] >= 2).sum()),
                "of_1R_runners": {"median_mfe_R": round(float(ran["mfe_r"].median()), 2),
                                  "median_exit_R": round(float(ran["r_gross"].median()), 2),
                                  "median_kept_share": round(float((ran["r_gross"] / ran["mfe_r"]).median()), 2),
                                  "closed_at_or_below_zero": round(float((ran["r_gross"] <= 0).mean()), 3)},
                "by_market": {m: {"reached_1R": len(g), "closed_at_or_below_zero": round(float((g["r_gross"] <= 0).mean()), 3),
                                  "median_kept_share": round(float((g["r_gross"] / g["mfe_r"]).median()), 2)}
                              for m, g in ran.groupby("market")}}
            cont = {}
            for arm in (CONTROL, PRIMARY, "B1"):
                c = trades(arm, dirs, act=False)["cont_r"].dropna()
                cont[arm] = {"n": len(c), "mean_R": round(float(c.mean()), 3),
                             "se": round(float(c.std(ddof=1) / np.sqrt(len(c))), 3),
                             "median_R": round(float(c.median()), 3), "share_positive": round(float((c > 0).mean()), 3)}
            block["continuation_from_trigger_bar"] = cont
        res[label] = block
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--outcomes", action="store_true")
    a = ap.parse_args()
    count() if a.count else outcomes()
