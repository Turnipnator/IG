#!/usr/bin/env python3
"""Forex BREAKOUT vs MOMENTUM head-to-head — firm up the thin live sample (2026-07-01).

Live so far, GBP/USD + EUR/USD only (USD/JPY disabled both ways):
  breakout  5 trades  +£13.23  PF 1.26  — but leans on ONE +£61 winner (#192)
  momentum 12 trades  -£39.52  PF 0.54  — the retired path, fuller sample, net loser

5 breakout trades is too thin to trust. This runs BOTH live configs UNCHANGED on the
same long Yahoo 1h window (and 4 walk-forward quarters for sign-consistency), same
3-pip round-trip cost charged to BOTH, so the comparison is like-for-like.

  MOMENTUM = live `forex` profile via the faithful simulate() (stop 1.0xATR, R:R 2.0,
             BE 0.7, ATR-trail 1.5, ranging-ADX exit, HTF-required, immediate entry).
  BREAKOUT = live Donchian config (N55, 2.0xATR stop, HTF-filtered, Donchian-M trail,
             NO take-profit) via breakout_sim().

Reuses the validated sims from backtest_forex_breakout.py + backtest_forex_pullback.py.
Yahoo cash 1h (EURUSD=X / GBPUSD=X), ZERO IG API cost. Caveats: Yahoo != IG DFB; flat
per-trade pip cost (real IG spread widens off-session); quarters lose ~N candles of
channel warmup at each slice edge. Read the DELTAS + sign-consistency, not the levels.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr
from scripts.backtest_forex_breakout import htf_series, breakout_sim, rstats
from scripts.backtest_forex_pullback import (
    MARKETS, simulate as tf_simulate, _build_htf,
)

logging.basicConfig(level=logging.ERROR)
TICKER_MAP.update({"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X"})

PAIRS = ["EUR/USD", "GBP/USD"]
DAYS = 725          # just under Yahoo's hard 730d 1h boundary (EURUSD=X 404s at 730)
INTERVAL = "1h"
N, STOP_K = 55, 2.0  # live breakout config
COST_PIPS = 3.0      # realistic IG round-trip (spread + slippage), charged to BOTH
PIP = 0.0001


def _spec(name):
    return next(m for m in MARKETS if m[0] == name)


def prep(bt, name):
    """Return (mom_df, htf_mom, brk_df, conf, adx_thr, min_stop, cooldown) — both
    sims fed off the SAME raw candles."""
    _, _, htf_i, _, params, conf, stop, be, trail, rdrop = _spec(name)
    raw = bt.fetch_data(name, DAYS, INTERVAL)
    if raw is None or len(raw) < 300:
        return None
    # momentum frame (full indicator set) + daily HTF frame
    mom_df = bt.add_indicators(raw.copy())
    htf_mom = _build_htf(bt, name, htf_i, DAYS)
    # breakout frame: raw + ATR + merged daily HTF column
    brk = raw.copy()
    brk["atr"] = calculate_atr(brk["high"], brk["low"], brk["close"], 14)
    hs = htf_series(bt, name)
    brk_df = pd.merge_asof(brk.sort_values("date"), hs.sort_values("date"),
                           on="date", direction="backward") if hs is not None else brk.assign(htf="NEUTRAL")
    min_stop = float(mom_df["close"].median()) * 0.0003
    cooldown = max(1, round(60 / 60))  # 1h candle -> 60-min cooldown = 1 candle
    return dict(mom_df=mom_df, htf_mom=htf_mom, brk_df=brk_df, conf=conf,
                stop=stop, be=be, trail=trail, rdrop=rdrop,
                adx_thr=params["adx_threshold"], min_stop=min_stop, cooldown=cooldown,
                median=float(mom_df["close"].median()))


def run_momentum(bt, p, mom_df):
    tr, _ = tf_simulate(mom_df, p["htf_mom"], bt, p["conf"], p["stop"], p["be"],
                        p["trail"], p["rdrop"], p["adx_thr"], None, 0, 0,
                        p["min_stop"], p["cooldown"])
    # charge the same round-trip cost momentum pays live (flat % off each trade)
    cost_pct = COST_PIPS * PIP / p["median"] * 100
    for t in tr:
        t["pnlp"] -= cost_pct
    return rstats(tr)


def run_breakout(brk_df):
    tr = breakout_sim(brk_df, N, STOP_K, "donchian", True, 0.0,
                      cost_pips=COST_PIPS, pip=PIP)
    return rstats(tr)


def fmt(s):
    if s is None:
        return "     no trades"
    return (f"n={s['n']:>3}  WR={s['wr']:>3.0f}%  P&L={s['pnl']:>+7.2f}%  PF={s['pf']:>5.2f}  "
            f"avgW={s['avg_w']:>+5.2f}  avgL={s['avg_l']:>+6.2f}  maxW={s['maxw']:>+6.2f}")


def main():
    print(f"\nForex BREAKOUT vs MOMENTUM — Yahoo {INTERVAL}/{DAYS}d, {COST_PIPS:.0f}-pip "
          f"round-trip charged to BOTH\n")
    agg = {"mom": [], "brk": []}
    for name in PAIRS:
        bt = Backtester(params=_spec(name)[4].copy())  # spec[4] = FOREX params dict
        p = prep(bt, name)
        if p is None:
            print(f"{name}: NO/THIN data\n"); continue
        mom_df, brk_df = p["mom_df"], p["brk_df"]
        mom_full = run_momentum(bt, p, mom_df)
        brk_full = run_breakout(brk_df)
        print(f"{'='*100}\n{name}  ({len(mom_df)} candles)\n{'='*100}")
        print(f"  MOMENTUM (live forex)   {fmt(mom_full)}")
        print(f"  BREAKOUT (live N55)     {fmt(brk_full)}")

        # walk-forward: 4 contiguous quarters, same configs UNCHANGED
        q = len(mom_df) // 4
        qb = len(brk_df) // 4
        print(f"  -- walk-forward quarters (sign-consistency) --")
        mom_signs, brk_signs, mom_qn, brk_qn = [], [], [], []
        for k in range(4):
            msl = mom_df.iloc[k * q:(k + 1) * q].reset_index(drop=True)
            bsl = brk_df.iloc[k * qb:(k + 1) * qb].reset_index(drop=True)
            ms = run_momentum(bt, p, msl)
            bs = run_breakout(bsl)
            mp = ms["pnl"] if ms else 0.0
            bp = bs["pnl"] if bs else 0.0
            mom_signs.append(mp)
            brk_signs.append(bp)
            mom_qn.append(ms["n"] if ms else 0)
            brk_qn.append(bs["n"] if bs else 0)
            print(f"     Q{k+1}:  mom {mp:>+7.2f}% (n={ms['n'] if ms else 0:>2})    "
                  f"brk {bp:>+7.2f}% (n={bs['n'] if bs else 0:>2})")
        # count only quarters that actually traded (Yahoo's early forex data is sparse)
        mtraded = [x for x, s in zip(mom_signs, mom_qn) if s > 0]
        btraded = [x for x, s in zip(brk_signs, brk_qn) if s > 0]
        mpos = sum(1 for x in mtraded if x > 0)
        bpos = sum(1 for x in btraded if x > 0)
        print(f"     quarters positive / with-trades:  "
              f"momentum {mpos}/{len(mtraded)}   breakout {bpos}/{len(btraded)}\n")
    print("Read: which strategy is net-positive AND sign-consistent across quarters.\n")


if __name__ == "__main__":
    main()
