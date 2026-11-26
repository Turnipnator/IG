#!/usr/bin/env python3
"""Break-even stop on the breakout engine. Pre-registered in research_notes.md (2026-11-27).

    venv/bin/python scripts/replay_breakout_breakeven.py

Engine, data and costs are replay_breakout_atr_exit's; the only change is that once MFE on
closed bars reaches X x the initial stop distance, the stop is raised to the entry price.
Offline; touches no bot state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_breakout_news as rbn  # noqa: E402
import replay_breakout_atr_exit as rx  # noqa: E402
from config import MARKETS  # noqa: E402
from src.breakout import BREAKOUT_CONFIGS  # noqa: E402
from src.indicators import calculate_atr  # noqa: E402

START = pd.Timestamp("2005-01-01")


def run(df, epic, diffs, be=None, directions=("BUY",)) -> pd.DataFrame:
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
    out, i, n = [], N + 1, len(df)
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
        stop, mfe, armed = entry - s * sd, 0.0, False
        j, xp, why = i + 1, None, None
        while j < n:
            p = j - 1
            if p > i:
                mfe = max(mfe, s * ((hi[p] if d == "BUY" else lo[p]) - entry))
            if be is not None and not armed and mfe >= be * sd:
                armed, stop = True, entry
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * stop:
                xp = min(o[j], stop) if d == "BUY" else max(o[j], stop)
                why = "be" if armed else "stop"; break
            lvl = lo[j - M:j].min() if d == "BUY" else hi[j - M:j].max()
            if s * (lo[j] if d == "BUY" else hi[j]) <= s * lvl:
                xp, why = (min(o[j], lvl) if d == "BUY" else max(o[j], lvl)), "trail"; break
            j += 1
        if xp is None:
            xp, why, j = cl[-1], "open-at-end", n - 1
        g = s * (xp - entry)
        nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
        fin = nights * rbn.financing_rate(epic, d, pd.Timestamp(dates[i]), diffs) / 365.0 * entry
        out.append(dict(epic=epic, market=mk.name, date=pd.Timestamp(dates[i]), d=d, why=why,
                        r=(g - cost - fin) / sd))
        i = j + 1
    t = pd.DataFrame(out)
    return t[t["date"] >= START].reset_index(drop=True)


def maxdd(t):
    eq = t.sort_values("date")["r"].cumsum().to_numpy()
    return float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max())


def main():
    fr, diffs = rx.frames(), rbn.rate_diffs()
    idx = pd.period_range(START, "2026-09-30", freq="Q")
    years = np.array([p.year for p in idx])
    for e, df in fr.items():
        name = next(m for m in MARKETS if m.epic == e).name
        for d in ("BUY", "SELL"):
            base = run(df, e, diffs, None, (d,))
            qb = rx.quarterly(base, idx)
            print(f"\n=== {name} {d}  baseline n={len(base)} net={base.r.sum():+.1f}R WR={100*(base.r>0).mean():.0f}% "
                  f"maxDD={maxdd(base):.1f}R worstYear={base.groupby(base.date.dt.year).r.sum().min():+.1f}R max={base.r.max():.1f}R")
            assert abs(run(df, e, diffs, None, (d,)).r.sum() - rx.run(df, e, diffs, directions=(d,)).query("date >= @START").r.sum()) < 1e-9
            for be in (1.0, 2.0):
                t = run(df, e, diffs, be, (d,))
                dq = rx.quarterly(t, idx) - qb
                h1, h2 = dq[years <= 2015].sum(), dq[years >= 2016].sum()
                ci = rx.boot_ci(dq)
                drop3 = float(np.sort(dq)[:-3].sum())
                ok = dq.sum() > 0 and ci[0] > 0 and h1 > 0 and h2 > 0 and drop3 > 0
                print(f"BE{be:.0f}: n={len(t)} net={t.r.sum():+.1f}R  delta={dq.sum():+.1f}R CI{ci}  halves {h1:+.1f}/{h2:+.1f}  "
                      f"ex-top3q {drop3:+.1f}  WR={100*(t.r>0).mean():.0f}%  BE-exits={int((t.why=='be').sum())}  "
                      f"maxDD={maxdd(t):.1f}R worstYear={t.groupby(t.date.dt.year).r.sum().min():+.1f}R max={t.r.max():.1f}R  "
                      f"since2023 {t[t.date>='2023-01-01'].r.sum():+.1f} vs {base[base.date>='2023-01-01'].r.sum():+.1f}  "
                      f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
