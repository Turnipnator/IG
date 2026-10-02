"""Silver variant search with a holdout — pre-registered in research_notes.md (ef2fe7e).

  python scripts/silver_variants.py

Family A: Donchian breakout over timeframe x entry N x stop k x HTF gate x direction
(108). Family B: daily pullback-in-trend (6). Every variant is scored on 2005-2015
(discovery) and 2016-2026 (holdout); the best discovery variant is judged on the
holdout only. The engine generalises replay_breakout_news.run_gated and is asserted
to reproduce its silver baseline (02ce154) exactly before anything else runs.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_breakout_metals as rbm  # noqa: E402
import replay_breakout_news as rbn  # noqa: E402
from src.indicators import calculate_atr  # noqa: E402

EPIC = rbm.SILVER
SPLIT = pd.Timestamp("2016-01-01")
START = pd.Timestamp("2005-01-01")
SPREADS = {"low": 2.0, "base": 3.0, "high": 4.0}
FIN = {"low": 0.03, "base": rbn.GOLD_ALL_IN, "high": 0.08}
MIN_STOP, CAP_INTRADAY, CAP_DAILY = 4.0, 45.0, 250.0
MIN_DISC_TRADES = 30
OUT = rbn.DATA / "silver_variants.json"


# ----------------------------------------------------------------------------- data
def frames() -> dict:
    """1h (as loaded), 4h and 1D bars on London time, each with the DAY-HTF label of
    the last hourly bar inside it (the label in force at that bar's close)."""
    h = rbn.load_market(EPIC)
    h["htf"] = rbn.htf_day_fast(h)
    out = {"1h": h}
    for tf, rule in (("4h", "4h"), ("1D", "1D")):
        g = h.set_index("date").resample(rule)
        f = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                          "close": g["close"].last(), "htf": g["htf"].last(),
                          "last_hour": g["close"].apply(lambda s: s.index.max() if len(s) else pd.NaT)}).dropna()
        f = f.reset_index().rename(columns={"date": "date"})
        out[tf] = f
    return out


# ----------------------------------------------------------------------------- engines
def breakout(df, n, k, htf_on, direction, hours_gate, spread, fin):
    """run_gated's loop, generalised: entry on the bar close that breaks the prior-N
    extreme, stop k*ATR14 (>= IG min stop), Donchian-M trail (M = N//2), one position."""
    m = max(2, n // 2)
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = calculate_atr(df["high"], df["low"], df["close"], 14).to_numpy(float)
    htf = df["htf"].to_numpy(object)
    dates = df["date"].to_numpy("datetime64[ns]")
    hours = df["date"].dt.hour.to_numpy()
    out, i, nb = [], n + 1, len(df)
    while i < nb:
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1; continue
        upper, lower = hi[i - n:i].max(), lo[i - n:i].min()
        d = "BUY" if hi[i] >= upper else "SELL" if lo[i] <= lower else None
        if d is None or (direction == "long" and d == "SELL") or (direction == "short" and d == "BUY"):
            i += 1; continue
        if hours_gate and hours[i] >= 22:              # live window 23->21 UTC = London 00-22
            i += 1; continue
        if htf_on and htf[i] != ("BULLISH" if d == "BUY" else "BEARISH"):
            i += 1; continue
        entry = cl[i]
        sd = max(a * k, MIN_STOP)
        stop = entry - sd if d == "BUY" else entry + sd
        j, xp = i + 1, None
        while j < nb:
            if d == "BUY" and lo[j] <= stop:  xp = min(o[j], stop); break
            if d == "SELL" and hi[j] >= stop: xp = max(o[j], stop); break
            if j - m >= 0:
                if d == "BUY":
                    lvl = lo[j - m:j].min()
                    if lo[j] <= lvl: xp = min(o[j], lvl); break
                else:
                    lvl = hi[j - m:j].max()
                    if hi[j] >= lvl: xp = max(o[j], lvl); break
            j += 1
        if xp is None:
            xp, j = cl[-1], nb - 1
        g = (xp - entry) if d == "BUY" else (entry - xp)
        nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
        cost = spread + nights * fin / 365.0 * entry
        out.append((pd.Timestamp(dates[i]), d, sd, (g - cost) / sd, g / sd, entry))
        i = j + 1
    return pd.DataFrame(out, columns=["date", "d", "sd", "r", "r_gross", "entry"])


def pullback(df, lookback, direction, spread, fin):
    """Daily pullback-in-trend (S&P rule). Long: close > SMA200 and close < prior
    L-day low -> exit close > prior L-day high, 10 sessions, or 3xATR20 stop.
    Short is the mirror. R unit = 2xATR20 (as the S&P study)."""
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    sma = pd.Series(cl).rolling(200).mean().to_numpy()
    atr = calculate_atr(df["high"], df["low"], df["close"], 20).to_numpy(float)
    dates = df["date"].to_numpy("datetime64[ns]")
    out, i, nb = [], max(200, lookback + 1), len(df)
    while i < nb:
        a = atr[i]
        if not (np.isfinite(a) and a > 0 and np.isfinite(sma[i])):
            i += 1; continue
        lowL, highL = lo[i - lookback:i].min(), hi[i - lookback:i].max()
        d = None
        if direction in ("long", "both") and cl[i] > sma[i] and cl[i] < lowL:
            d = "BUY"
        elif direction in ("short", "both") and cl[i] < sma[i] and cl[i] > highL:
            d = "SELL"
        if d is None:
            i += 1; continue
        entry, unit = cl[i], 2 * a
        sd = max(3 * a, MIN_STOP)
        stop = entry - sd if d == "BUY" else entry + sd
        j, xp = i + 1, None
        while j < nb:
            if d == "BUY" and lo[j] <= stop:  xp = min(o[j], stop); break
            if d == "SELL" and hi[j] >= stop: xp = max(o[j], stop); break
            if d == "BUY" and cl[j] > hi[j - lookback:j].max(): xp = cl[j]; break
            if d == "SELL" and cl[j] < lo[j - lookback:j].min(): xp = cl[j]; break
            if j - i >= 10: xp = cl[j]; break
            j += 1
        if xp is None:
            xp, j = cl[-1], nb - 1
        g = (xp - entry) if d == "BUY" else (entry - xp)
        nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
        cost = spread + nights * fin / 365.0 * entry
        out.append((pd.Timestamp(dates[i]), d, sd, (g - cost) / unit, g / unit, entry))
        i = j + 1
    return pd.DataFrame(out, columns=["date", "d", "sd", "r", "r_gross", "entry"])


# ----------------------------------------------------------------------------- scoring
def spearmanr(x, y, n_perm=10000, seed=42):
    """Rank correlation, with a two-sided permutation p-value (no scipy in the lock)."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


def stats(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return {"n": 0, "mean": np.nan, "t": np.nan, "total": 0.0}
    r = t["r"].to_numpy(float)
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return {"n": len(r), "mean": float(r.mean()), "t": float(r.mean() / (sd / np.sqrt(len(r)))) if sd and sd > 0 else np.nan,
            "total": float(r.sum()), "gross": float(t["r_gross"].mean()), "win": float((r > 0).mean())}


def split(t):
    t = t[t["date"] >= START]
    return t[t["date"] < SPLIT], t[t["date"] >= SPLIT]


def variants():
    for tf, n, k, htf_on, dirn in itertools.product(["1h", "4h", "1D"], [20, 55, 100], [2, 3],
                                                    [True, False], ["both", "long", "short"]):
        yield {"family": "A", "tf": tf, "n": n, "k": k, "htf": htf_on, "dir": dirn}
    for L, dirn in itertools.product([5, 10], ["long", "short", "both"]):
        yield {"family": "B", "tf": "1D", "L": L, "dir": dirn}


def simulate(v, fr, spread, fin):
    if v["family"] == "A":
        return breakout(fr[v["tf"]], v["n"], v["k"], v["htf"], v["dir"], v["tf"] == "1h", spread, fin)
    return pullback(fr["1D"], v["L"], v["dir"], spread, fin)


def label(v):
    if v["family"] == "A":
        return f"A {v['tf']:>2s} N{v['n']:<3d} k{v['k']} {'HTF' if v['htf'] else 'noHTF'} {v['dir']}"
    return f"B pullback L{v['L']} {v['dir']}"


def main():
    rbm.register()
    fr = frames()

    # 1. Engine check: reproduce the 02ce154 silver baseline exactly.
    base = rbm.run(EPIC, 3.0)
    mine = breakout(fr["1h"], 55, 2, True, "both", True, 3.0, rbn.GOLD_ALL_IN)
    mine = mine[mine["date"] >= START]
    print(f"engine check: n={len(mine)} mean {mine.r.mean():+.4f} vs baseline n={len(base)} mean {base.r.mean():+.4f}")
    assert len(mine) == len(base) and abs(mine.r.mean() - base.r.mean()) < 1e-9

    rows = []
    for v in variants():
        t = simulate(v, fr, SPREADS["base"], FIN["base"])
        disc, hold = split(t)
        sd_today = (t["sd"] / t["entry"]).median() * fr["1h"]["close"].iloc[-1] if len(t) else np.nan
        rows.append({**v, "label": label(v),
                     **{f"d_{k}": x for k, x in stats(disc).items()},
                     **{f"h_{k}": x for k, x in stats(hold).items()},
                     "gbp_risk_today": round(float(sd_today), 1)})
    g = pd.DataFrame(rows)

    # 2. Selection on discovery only.
    elig = g[g["d_n"] >= MIN_DISC_TRADES]
    best = elig.sort_values("d_mean", ascending=False).iloc[0]
    vbest = {k: best[k] for k in ("family", "tf", "n", "k", "htf", "dir", "L") if k in best and pd.notna(best[k])}
    vbest = {k: (int(x) if isinstance(x, (np.integer, float)) and k in ("n", "k", "L") else x) for k, x in vbest.items()}
    t_base = simulate(vbest, fr, SPREADS["base"], FIN["base"])
    t_hi = simulate(vbest, fr, SPREADS["high"], FIN["high"])
    _, hold = split(t_base)
    _, hold_hi = split(t_hi)
    hr = np.sort(hold["r"].to_numpy(float))
    hs = stats(hold)

    ok = g.dropna(subset=["d_mean", "h_mean"])
    rho, p_rho = spearmanr(ok["d_mean"], ok["h_mean"])
    crit = {"1_holdout_mean_pos_t_ge_2": bool(hs["mean"] > 0 and hs["t"] >= 2.0),
            "2_holdout_ex_top5_pos": bool(hr[:-5].sum() > 0) if len(hr) > 5 else False,
            "3_holdout_pos_high_cost": bool(stats(hold_hi)["mean"] > 0),
            "4_rank_persistence": bool(rho > 0 and p_rho < 0.05)}

    top10 = elig.sort_values("d_mean", ascending=False).head(10)
    res = {"selected": {"label": best["label"], "discovery": {k[2:]: best[k] for k in best.index if k.startswith("d_")},
                        "holdout": hs, "holdout_ex_top5": float(hr[:-5].sum()) if len(hr) > 5 else None,
                        "holdout_high_cost_mean": stats(hold_hi)["mean"], "gbp_risk_today": best["gbp_risk_today"]},
           "spearman_disc_vs_hold": {"rho": float(rho), "p": float(p_rho), "n_variants": int(len(ok))},
           "criteria": crit, "verdict": "PASS -> shadow arm" if all(crit.values()) else "FAIL -> no silver strategy",
           "top10_by_discovery": top10[["label", "d_n", "d_mean", "h_n", "h_mean", "h_t", "gbp_risk_today"]].round(3).to_dict("records"),
           "share_variants_holdout_positive": float((ok["h_mean"] > 0).mean()),
           "share_variants_holdout_t_ge_2": float((ok["h_t"] >= 2).mean()),
           "best_holdout_any": ok.sort_values("h_mean", ascending=False).head(5)[["label", "d_mean", "h_n", "h_mean", "h_t"]].round(3).to_dict("records")}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    g.round(4).to_csv(rbn.DATA / "silver_variants_grid.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(json.dumps(res, indent=2, default=str))
    print("\nper family / timeframe (holdout mean R, share positive):")
    g["grp"] = g["family"] + " " + g["tf"]
    print(g.groupby("grp").agg(n=("label", "size"), disc_mean=("d_mean", "mean"), hold_mean=("h_mean", "mean"),
                                hold_pos=("h_mean", lambda s: (s > 0).mean())).round(3))
    print(g.groupby("dir").agg(disc_mean=("d_mean", "mean"), hold_mean=("h_mean", "mean")).round(3))


if __name__ == "__main__":
    main()
