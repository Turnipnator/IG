"""Daily volatility-squeeze test for 1h breakouts — pre-registered in research_notes.md
("Does a daily volatility squeeze predict better 1h breakouts?", commit 20187d6).

  python scripts/replay_breakout_squeeze.py --step1      # tag + counts + power, OUTCOME-BLIND
  python scripts/replay_breakout_squeeze.py --outcomes   # the verdict

Trades are the news replay's live-gated entries (data/news_events/replay_trades.pkl);
nothing is re-simulated. The squeeze rank for a trade is taken from the last completed
UTC day STRICTLY before the decision's UTC date.
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

START = pd.Timestamp("2005-01-01")
BB_N, RANK_N, CUT = 20, 250, 1 / 3
MIN_HOURS = 12
SHIFTS = [k for k in range(-36, 37) if abs(k) >= 3]          # months, 66 shifts
HALVES = [("2005-01-01", "2015-12-31"), ("2016-01-01", "2026-12-31")]
SIGMA_R_PRIOR = 2.0
OUT = rbn.DATA / "squeeze_outcomes.json"


def daily_rank(epic: str) -> pd.Series:
    """Squeeze rank per UTC day (index = UTC date), from the hourly Dukascopy file."""
    h = rbn.load_market(epic)
    h["day"] = h["utc"].dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    g = h.groupby("day")
    d = pd.DataFrame({"close": g["close"].last(), "n": g.size()})
    d = d[d["n"] >= MIN_HOURS]
    sma = d["close"].rolling(BB_N).mean()
    sd = d["close"].rolling(BB_N).std(ddof=0)
    width = 4 * sd / sma
    rank = width.rolling(RANK_N).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    return rank.dropna()


def tag(trades: pd.DataFrame, ranks: dict, months: int = 0) -> pd.Series:
    """Rank of the last completed day < decision UTC date; optional whole-month rotation."""
    out = pd.Series(np.nan, index=trades.index)
    for epic, r in ranks.items():
        sel = trades["epic"] == epic
        if not sel.any():
            continue
        series = r
        if months:
            series = pd.Series(r.values, index=r.index + pd.DateOffset(months=months)).sort_index()
            series = series[~series.index.duplicated()]
        dec_day = trades.loc[sel, "decision_utc"].dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
        pos = series.index.searchsorted(dec_day.values, side="left") - 1   # strictly before
        ok = pos >= 0
        vals = np.full(len(pos), np.nan)
        vals[ok] = series.values[pos[ok]]
        if not months:  # look-ahead guard on the real tagging
            used = series.index[pos[ok]]
            assert (used < dec_day.values[ok]).all(), "squeeze rank uses the entry day"
            stale = (dec_day.values[ok] - used) > np.timedelta64(7, "D")
            vals[np.where(ok)[0][stale]] = np.nan   # data gap: no recent completed day
        out[sel] = vals
    return out


def load() -> tuple[pd.DataFrame, dict]:
    t = pd.read_pickle(rbn.DATA / "replay_trades.pkl")
    t = t[t["date"] >= START].reset_index(drop=True)
    ranks = {e: daily_rank(e) for e in rbn.UNIVERSE}
    t["rank"] = tag(t, ranks)
    return t, ranks


def step1() -> None:
    t, _ = load()
    dropped = int(t["rank"].isna().sum())
    t = t.dropna(subset=["rank"])
    sq = t["rank"] <= CUT
    n_s, n_r = int(sq.sum()), int((~sq).sum())
    mde = 2.8 * SIGMA_R_PRIOR * np.sqrt(1 / n_s + 1 / n_r)
    print(f"entries from {START.date()}: {len(t) + dropped}; dropped (no 250d history/gap): {dropped}")
    print(f"SQUEEZE n={n_s}  rest n={n_r}  share={n_s / len(t):.1%}")
    print(t.assign(sq=sq).groupby(["market", "sq"]).size().unstack())
    print(f"MDE(80%, a=.05, sigma_R=2.0) = {mde:.3f}R -> {'UNDERPOWERED' if mde > 0.25 else 'adequately powered'}")


def delta(r: np.ndarray, s: np.ndarray) -> float:
    return float(r[s].mean() - r[~s].mean()) if s.any() and (~s).any() else float("nan")


def outcomes() -> None:
    t, ranks = load()
    t = t.dropna(subset=["rank"]).reset_index(drop=True)
    r = t["r"].to_numpy(float)
    s = (t["rank"] <= CUT).to_numpy()
    obs = delta(r, s)

    null = []
    for k in SHIFTS:
        rk = tag(t, ranks, k)
        ok = rk.notna().to_numpy()
        null.append(delta(r[ok], (rk[ok] <= CUT).to_numpy()))
    null = np.array([x for x in null if np.isfinite(x)])
    p = float((np.abs(null) >= abs(obs)).mean())

    halves = {}
    for a, b in HALVES:
        m = ((t["date"] >= a) & (t["date"] <= b)).to_numpy()
        halves[f"{a[:4]}-{b[:4]}"] = {"delta": round(delta(r[m], s[m]), 4), "n_sq": int(s[m].sum()), "n_rest": int((~s[m]).sum())}
    per_mkt = {}
    for mk, g in t.groupby("market"):
        i = g.index.to_numpy()
        per_mkt[mk] = {"delta": round(delta(r[i], s[i]), 4), "mean_sq": round(float(r[i][s[i]].mean()), 4),
                       "mean_rest": round(float(r[i][~s[i]].mean()), 4), "n_sq": int(s[i].sum()), "n_rest": int((~s[i]).sum()),
                       "sum_sq": round(float(r[i][s[i]].sum()), 2), "sum_rest": round(float(r[i][~s[i]].sum()), 2)}
    keep = np.ones(len(r), bool)
    keep[np.argsort(r)[-10:]] = False
    drop10 = delta(r[keep], s[keep])

    sign = np.sign(obs)
    crit = {
        "1_abs_delta_ge_0.15": abs(obs) >= 0.15,
        "2_p_lt_0.05": p < 0.05,
        "3_same_sign_halves": all(np.sign(h["delta"]) == sign for h in halves.values()),
        "4_same_sign_gold_gbp": all(np.sign(per_mkt[m]["delta"]) == sign for m in ("Gold", "GBP/USD")),
        "5_drop_top10_abs_ge_0.10": sign * drop10 >= 0.10,
    }
    passed = all(crit.values())
    filter6 = float(r[~s].sum()) <= 0 if obs > 0 else float(r[s].sum()) <= 0
    res = {
        "n": len(r), "n_sq": int(s.sum()), "n_rest": int((~s).sum()),
        "mean_sq": round(float(r[s].mean()), 4), "mean_rest": round(float(r[~s].mean()), 4),
        "sum_sq": round(float(r[s].sum()), 2), "sum_rest": round(float(r[~s].sum()), 2),
        "delta": round(obs, 4), "p_rotation_two_sided": round(p, 4), "n_shifts": int(len(null)),
        "null_p2.5_p97.5": [round(float(np.percentile(null, 2.5)), 4), round(float(np.percentile(null, 97.5)), 4)],
        "halves": halves, "per_market": per_mkt, "delta_drop_top10": round(drop10, 4),
        "criteria": {k: bool(v) for k, v in crit.items()},
        "verdict": ("PASS (H1 squeeze)" if obs > 0 else "PASS (H2 expansion)") if passed else "FAIL -> NO EFFECT",
        "filter_criterion_6": bool(filter6) if passed else None,
    }
    # Descriptive only, after the verdict: mean r by rank tercile.
    terc = pd.cut(t["rank"], [0, 1 / 3, 2 / 3, 1.0001], labels=["low", "mid", "high"], include_lowest=True)
    res["descriptive_terciles"] = {str(k): {"n": int(len(g)), "mean_r": round(float(g.mean()), 4)}
                                   for k, g in t["r"].groupby(terc, observed=True)}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--step1", action="store_true")
    g.add_argument("--outcomes", action="store_true")
    a = ap.parse_args()
    step1() if a.step1 else outcomes()
