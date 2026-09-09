#!/usr/bin/env python3
"""Offline parity check: IG DAY bars (the daily-trend store) vs the Yahoo GC=F bars
the 22-year study was run on, over their overlap. Run BEFORE trusting a live signal.

    python scripts/daily_trend_parity.py data/daily_bars/CS.D.USCGC.TODAY.IP.jsonl \
        tests/fixtures/gc_f_daily.csv

Reports close agreement, the two channels and ATR as IG/Yahoo ratios, and — the
thing that matters — whether the two series would have put the bot in the SAME
state (flat/long) on each overlapping day. IG's bar is 00:00-24:00 London with a
Sunday stub (dropped); Yahoo's is the CME settle day. Expect closes within ~0.3%
and identical state on all but a handful of boundary days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import daily_bars                                   # noqa: E402
from src.daily_trend import DailyTrendConfig, add_levels, prepare_daily   # noqa: E402


def state_series(d: pd.DataFrame, cfg: DailyTrendConfig) -> pd.Series:
    """Flat/long state at each bar close under the rule (signal on close)."""
    st = []; pos = 0
    for _, r in d.iterrows():
        if not np.isfinite(r["entry_hi"]) or not np.isfinite(r["exit_lo"]):
            st.append(np.nan); continue
        if pos == 0 and r["close"] > r["entry_hi"]: pos = 1
        elif pos == 1 and r["close"] < r["exit_lo"]: pos = 0
        st.append(pos)
    return pd.Series(st, index=d["date"].values)


def main(store_path: str, yahoo_csv: str) -> int:
    cfg = DailyTrendConfig()
    ig = prepare_daily(pd.read_json(store_path, lines=True)) if store_path.endswith(".jsonl") \
        else prepare_daily(pd.read_csv(store_path, parse_dates=["date"]))
    ya = prepare_daily(pd.read_csv(yahoo_csv, parse_dates=["date"]))
    lo, hi = ig["date"].min(), ig["date"].max()
    ya_w = ya[(ya["date"] >= lo - pd.Timedelta(days=400)) & (ya["date"] <= hi)]
    ig_l, ya_l = add_levels(ig, cfg), add_levels(ya_w, cfg)
    m = ig_l.merge(ya_l, on="date", suffixes=("_ig", "_ya"))
    m = m[np.isfinite(m["entry_hi_ig"]) & np.isfinite(m["entry_hi_ya"])]
    print(f"IG store: {len(ig)} bars {ig['date'].iloc[0].date()} -> {ig['date'].iloc[-1].date()}")
    print(f"overlap with levels on both sides: {len(m)} days")
    if m.empty:
        print("NOT ENOUGH OVERLAP — need >= 56 IG bars with Yahoo coverage"); return 2
    cl = (m["close_ig"] / m["close_ya"] - 1) * 100
    print(f"close: median |diff| {cl.abs().median():.3f}%  p95 {cl.abs().quantile(.95):.3f}%  mean bias {cl.mean():+.3f}%")
    for k in ("entry_hi", "exit_lo", "atr"):
        r = m[f"{k}_ig"] / m[f"{k}_ya"]
        print(f"{k:9s} IG/Yahoo ratio: median {r.median():.4f}  min {r.min():.4f}  max {r.max():.4f}")
    s_ig = state_series(ig_l, cfg).reindex(m["date"].values)
    s_ya = state_series(ya_l, cfg).reindex(m["date"].values)
    both = pd.DataFrame({"ig": s_ig.values, "ya": s_ya.values}).dropna()
    agree = (both.ig == both.ya).mean() * 100 if len(both) else float("nan")
    print(f"same flat/long state on {agree:.1f}% of {len(both)} overlapping days "
          f"(IG long {both.ig.mean()*100:.0f}%, Yahoo long {both.ya.mean()*100:.0f}%)")
    last = ig_l.iloc[-1]
    print(f"IG latest bar {last['date'].date()}: close {last['close']:.1f} | 55d hi {last['entry_hi']:.1f} | "
          f"20d lo {last['exit_lo']:.1f} | ATR20 {last['atr']:.1f} -> stop 2xATR {2*last['atr']:.1f} pts "
          f"(£{2*last['atr']:.0f} at size 1.0)")
    ok = cl.abs().median() < 0.5 and agree >= 90
    print("PARITY", "OK" if ok else "SUSPECT — inspect before trusting a live signal")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
