#!/usr/bin/env python3
"""Forex breakout — multi-year / walk-forward ROBUSTNESS (2026-06-19). The decider.

The cost stress test (backtest_forex_breakout_costs.py) left ONE survivor shape:
slow + HTF-filtered + large-N Donchian, led by GBP/USD N55. But that was a SINGLE
365d window. The killer question for any breakout edge is regime stability: does the
SAME fixed config stay positive across DIFFERENT years/regimes, or did it just catch
one good trending year? Breakout edges are notoriously cyclical, so this is the gate.

Method: fetch ~730d of Yahoo 1h (the max), apply a realistic 3-pip round-trip cost,
and run each survivor config UNCHANGED across non-overlapping sub-periods (full / 2
halves / 4 quarters). NOT re-optimised per window — we want the stability of a FIXED
rule out-of-sample. Look for SIGN CONSISTENCY (every window ≥ breakeven) over raw P&L;
a config that's +5% in one quarter and −3% in the others is a fluke, not an edge.

Caveats: ~2yr is still short for a breakout edge; sub-windows are thin (N55 ~ 14-27
trades/window); Yahoo cash != IG, flat cost flatters fast breaks. Necessary gate,
not final proof.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr, calculate_ema
from scripts.backtest_forex_breakout import breakout_sim, rstats, htf_series

logging.basicConfig(level=logging.ERROR)

TICKER_MAP.update({"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
                   "AUD/USD": "AUDUSD=X", "AUD/JPY": "AUDJPY=X", "NZD/USD": "NZDUSD=X"})

DAYS, STOP_K, COST = 730, 2.0, 3.0   # 3-pip realistic round-trip

# (name, pip, N, exit_mode, use_filter, label)
CONFIGS = [
    ("AUD/USD", 0.0001, 55, "donchian", True,  "N55 HTF Donchian (CANDIDATE)"),
    ("NZD/USD", 0.0001, 55, "donchian", True,  "N55 HTF Donchian (CANDIDATE-2)"),
    ("AUD/JPY", 0.01,   55, "donchian", True,  "N55 HTF Donchian (cand-3)"),
    ("GBP/USD", 0.0001, 55, "donchian", True,  "N55 HTF Donchian (live ref)"),
    ("EUR/USD", 0.0001, 55, "donchian", True,  "N55 HTF Donchian (live ref)"),
    ("USD/JPY", 0.01,   40, "donchian", False, "N40 naked Donchian (control)"),
]


def load(name):
    bt = Backtester(params={"ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
                            "rsi_period": 7, "atr_period": 14})
    raw = bt.fetch_data(name, DAYS, "1h")
    if raw is None or len(raw) < 500:
        return None
    raw = raw.copy()
    raw["atr"] = calculate_atr(raw["high"], raw["low"], raw["close"], 14)
    hs = htf_series(bt, name, days=DAYS)  # shared helper — was a copy-paste
                                     # duplicate carrying the same look-ahead
                                     # bug (fixed 2026-08-13).
    df = (pd.merge_asof(raw.sort_values("date"), hs.sort_values("date"),
                        on="date", direction="backward")
          if hs is not None else raw.assign(htf="NEUTRAL"))
    return df


def windows(df, k):
    """k equal-time non-overlapping [lo, hi) slices over the data's date span."""
    lo, hi = df["date"].iloc[0], df["date"].iloc[-1]
    span = (hi - lo) / k
    out = []
    for i in range(k):
        a = lo + span * i
        b = hi + pd.Timedelta(seconds=1) if i == k - 1 else lo + span * (i + 1)
        out.append((a, b))
    return out


def run(df, lo, hi, n, exit_mode, use_filter, pip):
    sl = df[(df["date"] >= lo) & (df["date"] < hi)].reset_index(drop=True)
    if len(sl) < n + 50:
        return None
    min_stop = float(sl["close"].median()) * 0.0003
    return rstats(breakout_sim(sl, n, STOP_K, exit_mode, use_filter, min_stop, COST, pip))


def show(tag, s):
    if s is None:
        print(f"    {tag:<22} (thin/no data)"); return
    sign = "✓" if s["pnl"] > 0 else ("·" if s["pnl"] > -0.5 else "✗")
    print(f"    {tag:<22} {sign}  n={s['n']:>3} WR={s['wr']:>3.0f}% P&L={s['pnl']:>+7.2f}% PF={s['pf']:>5.2f}")


def main():
    cache = {}
    for name, pip, n, exit_mode, use_filter, label in CONFIGS:
        if name not in cache:
            cache[name] = load(name)
        df = cache[name]
        if df is None:
            print(f"\n{name}: NO/THIN data"); continue
        span_d = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        print(f"\n{'='*80}\n{name} — {label}  |  {span_d}d of 1h ({len(df)} candles), 3-pip cost\n{'='*80}")
        lo, hi = df["date"].iloc[0], df["date"].iloc[-1] + pd.Timedelta(seconds=1)
        show("FULL period", run(df, lo, hi, n, exit_mode, use_filter, pip))
        print("  -- 2 halves (older → recent) --")
        for i, (a, b) in enumerate(windows(df, 2), 1):
            show(f"H{i} {a.date()}→{b.date()}" if False else f"half {i}", run(df, a, b, n, exit_mode, use_filter, pip))
        print("  -- 4 quarters (older → recent) --")
        for i, (a, b) in enumerate(windows(df, 4), 1):
            show(f"Q{i}", run(df, a, b, n, exit_mode, use_filter, pip))


if __name__ == "__main__":
    main()
