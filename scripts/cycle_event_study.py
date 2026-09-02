#!/usr/bin/env python3
"""Phase 1b: test the PFO cross DATES against PRICE, independent of the bot.

For each cross date D and offset o in {-1,0,+1}, at trading day t = D+o:
  rev5  : 5-bar move INTO t and 5-bar move OUT of t have opposite sign
          (close[t] is a 5-bar turning point)
  piv5  : close[t] is the max or min close within [t-5, t+5]  (strict pivot)
  mag5  : |close[t+5]-close[t]| / ATR14[t]                      (volatility)
Baselines: (a) every trading day 2022 -> 2026-08-31; (b) June-Aug 2026 trading
days NOT within +/-1 of a cross. Permutation test: per month, redraw the same
number of days at random, 10k reps.

PRE-COMMITTED (before running): rev5 on cross dates >= baseline(b) + 15 points
at the SAME offset, for BOTH Gold and S&P, June-Aug 2026. September is OOS.
"""
import argparse, json, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cycles import load_cycles  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lo", type=date.fromisoformat, default=date(2026, 6, 1))
ap.add_argument("--hi", type=date.fromisoformat, default=date(2026, 8, 31),
                help="window of cross dates to test (Sept OOS: --lo 2026-09-01 --hi 2026-09-30)")
ap.add_argument("--archive-dir", type=Path, default=Path("data/candle_archive"),
                help="IG candle archive for the IG-native cross-check (Gold lives on the VPS)")
ap.add_argument("--reps", type=int, default=10000)
ARGS = ap.parse_args()

H = 5                      # horizon (bars); sensitivity at 3 and 10 below
LO, HI = ARGS.lo, ARGS.hi
MKTS = {"Gold": ("GC=F", "CS.D.USCGC.TODAY.IP"),
        "S&P 500": ("^GSPC", "IX.D.SPTRD.DAILY.IP")}
rng = np.random.default_rng(20260902)

# ---------------------------------------------------------------- data
def yahoo_daily(tk):
    h = yf.Ticker(tk).history(start="2022-01-01", auto_adjust=False)
    df = pd.DataFrame({"o": h["Open"].values, "h": h["High"].values,
                       "l": h["Low"].values, "c": h["Close"].values},
                      index=pd.to_datetime(h.index.date))
    return df[~df.index.duplicated()]

def ig_daily(epic, min_bars=50):
    rows = [json.loads(l) for l in open(ARGS.archive_dir / f"{epic}.jsonl")]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["timestamp"])
    g = df.groupby(df["ts"].dt.date)
    d = pd.DataFrame({"o": g["open"].first(), "h": g["high"].max(),
                      "l": g["low"].min(), "c": g["close"].last(),
                      "n": g.size()})
    d = d[d["n"] >= min_bars].drop(columns="n")     # drop Sunday stubs
    d.index = pd.to_datetime(d.index)
    return d

def add_stats(df, h=H):
    c = df["c"]
    tr = pd.concat([df["h"] - df["l"], (df["h"] - c.shift()).abs(),
                    (df["l"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    pre = c - c.shift(h)             # move INTO t
    post = c.shift(-h) - c           # move OUT of t
    df[f"rev{h}"] = np.where(pre.isna() | post.isna(), np.nan,
                             ((np.sign(pre) != np.sign(post)) & (pre != 0) & (post != 0)).astype(float))
    rmax = c.rolling(2*h+1, center=True).max()
    rmin = c.rolling(2*h+1, center=True).min()
    df[f"piv{h}"] = np.where(rmax.isna(), np.nan, ((c == rmax) | (c == rmin)).astype(float))
    df[f"mag{h}"] = post.abs() / df["atr"]
    df["prior_up"] = np.sign(pre)
    return df

# ---------------------------------------------------------------- cycle dates
cyc = load_cycles(cross_tolerance_days=0)
def crosses(epic):
    return sorted(d for d, s in cyc.get(epic, {}).items() if s.cross and LO <= d <= HI)
def week_days(epic):
    return {d for d, s in cyc.get(epic, {}).items() if s.week and LO <= d <= HI}

# ---------------------------------------------------------------- core
def cells(df, dates, off, col):
    """Stat at D+off for each D; NaN if not a trading day or stat undefined."""
    out = []
    for D in dates:
        t = pd.Timestamp(D + timedelta(days=off))
        out.append(df.at[t, col] if t in df.index else np.nan)
    return np.array(out, dtype=float)

def perm_p(df, dates, off, col, obs, reps=None):
    reps = reps or ARGS.reps
    """Redraw, per month, the same number of computable days at random."""
    if np.isnan(obs):
        return np.nan
    pools, ks = [], []
    for m in sorted({D.month for D in dates}):
        ds = [D for D in dates if D.month == m]
        k = int(np.sum(~np.isnan(cells(df, ds, off, col))))
        if k == 0:
            continue
        pool = df.loc[(df.index.month == m) & (df.index.year == LO.year), col].dropna().values
        pools.append(pool); ks.append(k)
    if not pools:
        return np.nan
    sims = np.empty(reps)
    for r in range(reps):
        draw = np.concatenate([rng.choice(p, k, replace=False) for p, k in zip(pools, ks)])
        sims[r] = draw.mean()
    return float((sims >= obs).mean())

def report(name, df, epic, src):
    xs = crosses(epic)
    wk = week_days(epic)
    near = {D + timedelta(days=o) for D in xs for o in (-1, 0, 1)}
    ins = df.loc[(df.index >= pd.Timestamp(LO)) & (df.index <= pd.Timestamp(HI))]
    base_b = ins[~ins.index.map(lambda t: t.date() in near)]
    base_a = df.loc[df.index <= pd.Timestamp(HI)]
    print(f"\n{'='*78}\n{name}  [{src}]   crosses {LO}..{HI}: {len(xs)}   "
          f"trading days in window: {len(ins)}   non-cross(b): {len(base_b)}   full(a): {len(base_a)}")
    for col in (f"rev{H}", f"piv{H}", f"mag{H}"):
        ba, bb = base_a[col].mean(), base_b[col].mean()
        print(f"\n  {col:5s}  baseline(a) full={ba:.3f}   baseline(b) in-window non-cross={bb:.3f}")
        print(f"  {'offset':8s} {'n':>3s}  {'cross':>6s}  {'vs(b)':>7s}  {'vs(a)':>7s}  {'perm p':>7s}")
        for off in (-1, 0, 1):
            v = cells(df, xs, off, col)
            n = int(np.sum(~np.isnan(v)))
            obs = np.nanmean(v) if n else np.nan
            p = perm_p(df, xs, off, col, obs)
            flag = ""
            if col.startswith("rev") and not np.isnan(obs) and obs >= bb + 0.15:
                flag = "  <-- meets pre-commit"
            print(f"  D{off:+d}       {n:3d}  {obs:6.3f}  {obs-bb:+7.3f}  {obs-ba:+7.3f}  {p:7.3f}{flag}")
        # "anywhere within +/-1" for the pivot stat (the tolerance claim)
        if col.startswith("piv"):
            anyv = []
            for D in xs:
                vv = [cells(df, [D], o, col)[0] for o in (-1, 0, 1)]
                vv = [x for x in vv if not np.isnan(x)]
                anyv.append(max(vv) if vv else np.nan)
            anyv = np.array(anyv)
            # baseline: any pivot in a random 3-day window
            pv = df[col].values
            any_base = np.nanmean([np.nanmax(pv[i-1:i+2]) for i in range(1, len(pv)-1)
                                   if not np.all(np.isnan(pv[i-1:i+2]))])
            print(f"  any +/-1   {int(np.sum(~np.isnan(anyv))):3d}  {np.nanmean(anyv):6.3f}"
                  f"  (random 3-day window baseline {any_base:.3f})")
    # direction: does the sheet mark tops or bottoms? does post fade the prior?
    print(f"\n  prior-trend split at D+0 (rev{H}):")
    for lbl, sgn in (("prior UP  (candidate top)", 1), ("prior DOWN(candidate bottom)", -1)):
        pu = cells(df, xs, 0, "prior_up"); rv = cells(df, xs, 0, f"rev{H}")
        m = (pu == sgn) & ~np.isnan(rv)
        if m.sum():
            print(f"    {lbl:30s} n={int(m.sum()):2d}  reversal={rv[m].mean():.3f}")
    # red weeks: volatility claim
    inwk = ins[ins.index.map(lambda t: t.date() in wk)]
    outwk = ins[~ins.index.map(lambda t: t.date() in wk)]
    if len(inwk):
        print(f"\n  red-week days: n={len(inwk)}  mean |{H}d move|/ATR = {inwk[f'mag{H}'].mean():.3f}"
              f"   vs other Jun-Aug days n={len(outwk)}  {outwk[f'mag{H}'].mean():.3f}")
    # horizon sensitivity (offset 0, reversal only)
    print(f"\n  horizon sensitivity (D+0 reversal, cross vs Jun-Aug non-cross):")
    for h in (3, 5, 10):
        d2 = add_stats(df[["o", "h", "l", "c"]].copy(), h)
        i2 = d2.loc[(d2.index >= pd.Timestamp(LO)) & (d2.index <= pd.Timestamp(HI))]
        bb2 = i2[~i2.index.map(lambda t: t.date() in near)][f"rev{h}"].mean()
        v = cells(d2, xs, 0, f"rev{h}"); n = int(np.sum(~np.isnan(v)))
        print(f"    h={h:2d}  n={n:2d}  cross={np.nanmean(v):.3f}  base={bb2:.3f}  gap={np.nanmean(v)-bb2:+.3f}")

for name, (tk, epic) in MKTS.items():
    report(name, add_stats(yahoo_daily(tk)), epic, f"Yahoo {tk}")
    if (ARGS.archive_dir / f"{epic}.jsonl").exists():
        report(name, add_stats(ig_daily(epic)), epic,
               f"IG archive {epic} (from 2026-06-12; baseline (a) == the archive span)")
    else:
        print(f"\n  ({epic}: no archive at {ARGS.archive_dir}, IG cross-check skipped "
              f"-- scp it from the VPS and pass --archive-dir)")
