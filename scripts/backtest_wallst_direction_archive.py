#!/usr/bin/env python3
"""Wall Street long/short asymmetry — IG-archive walk-forward via live analyze().

WHY THIS SCRIPT EXISTS
----------------------
The live journal shows Wall Street longs at -0.242R (n=34) and shorts at +0.252R
(n=16), t=2.22. Tempting. But the sides are NOT interleaved: 2026-07 was 10 longs,
-6.31R and ZERO shorts, so "longs are bad" and "July was bad" are inseparable in
that sample. The 2026-07-01 Yahoo sweep found Wall Street flat BOTH ways in BOTH
windows, and its memory note prescribes exactly this test for a definitive read:
the IG candle archive, the real instrument, the live interval, 24h sessions, driven
through the live analyze() — not a Yahoo cross-section.

The question a walk-forward answers that a single run cannot: does the BUY-vs-SELL
verdict computed in one window still hold in the next? The prior sweep's verdict
FLIPPED SIGN between windows on three of four markets. That is the signature of
noise, and it is what we are testing for here.

NO LOOK-AHEAD
-------------
htf_trend is computed from 1h bars that have CLOSED at or before the current 5m bar
(resample -> shift(1) -> asof-merge). Getting this wrong is not hypothetical: a
silent look-ahead in htf_series() invalidated every forex-breakout number quoted
before 2026-08-13. The HTF rule mirrors main.update_htf_trends exactly: EMA9 vs
EMA21 plus close vs EMA21 on the hourly frame.

COST
----
Charged as a fixed round-trip spread on every trade. Note the direction question is
cost-INVARIANT — spread hits longs and shorts identically — so the BUY-vs-SELL delta
survives any cost assumption. Cost only affects the absolute PF.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import statistics as st

import pandas as pd

from archive_loader import load_archive
from config import MARKETS, get_strategy_for_market
from src.indicators import calculate_ema
from src.strategy import Signal, TradingStrategy, should_close_position

LOOKBACK = 300          # bars handed to analyze(), as live passes a bounded window

# Per-market round-trip spread in the instrument's own points. A FIRST VERSION OF
# THIS SCRIPT USED ONE VALUE (4.8) FOR ALL THREE and the S&P control came out at
# PF 0.13 -- S&P's median stop is 8.8 points, so 4.8 was charging it 0.55R of pure
# cost per trade. The control failing is what exposed it. Measured out-of-hours, so
# treat as indicative.
SPREADS = {"IX.D.DOW.DAILY.IP": 4.8, "IX.D.SPTRD.DAILY.IP": 0.6,
           "IX.D.NASDAQ.CASH.IP": 2.0, "IX.D.NIKKEI.DAILY.IP": 7.0,
           "IX.D.HANGSENG.DAILY.IP": 30.0}
SP500_EPIC = "IX.D.SPTRD.DAILY.IP"


def htf_series_no_lookahead(df: pd.DataFrame) -> pd.Series:
    """BULLISH/BEARISH/NEUTRAL per 5m bar, from CLOSED hourly bars only.

    resample -> shift(1) is the load-bearing pair: without the shift, the hour a
    bar belongs to includes that bar's own future.
    """
    h = (df.set_index("date")
           .resample("1h")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
           .dropna())
    ema9, ema21 = calculate_ema(h["close"], 9), calculate_ema(h["close"], 21)
    trend = pd.Series("NEUTRAL", index=h.index, dtype=object)
    trend[(ema9 > ema21) & (h["close"] > ema21)] = "BULLISH"
    trend[(ema9 < ema21) & (h["close"] < ema21)] = "BEARISH"
    trend = trend.shift(1)                      # only hours that have CLOSED
    out = pd.merge_asof(df[["date"]].sort_values("date"),
                        trend.rename("htf").reset_index().sort_values("date"),
                        on="date", direction="backward")
    return out["htf"].fillna("NEUTRAL")


def regime_series(df: pd.DataFrame) -> pd.Series:
    """The GLOBAL direction gate, reconstructed from the S&P 500 archive.

    THIS IS THE GATE THAT MATTERS FOR A DIRECTION QUESTION and the first version of
    this script did not model it at all. main.py:1685 blocks EVERY SELL while the
    S&P regime is BULLISH, every BUY while it is BEARISH, and all trades while it is
    NEUTRAL. So a market's live long/short mix is not a free choice the strategy
    makes -- it is dictated by what the S&P was doing at the time. Replaying without
    it measures a strategy the bot has never run.
    """
    sp = load_archive(SP500_EPIC).reset_index(drop=True)
    reg = htf_series_no_lookahead(sp)          # same EMA9/21 rule main.py uses
    out = pd.merge_asof(df[["date"]].sort_values("date"),
                        pd.DataFrame({"date": sp["date"], "reg": reg}).sort_values("date"),
                        on="date", direction="backward")
    return out["reg"].fillna("NEUTRAL")


def replay(df: pd.DataFrame, market, spread: float, sides=("BUY", "SELL"),
           apply_regime_gate: bool = True):
    """Walk the frame bar by bar through the live analyze()/should_close_position."""
    strat = TradingStrategy()
    htf = htf_series_no_lookahead(df)
    regime = regime_series(df) if apply_regime_gate else None
    # The live gate is max(market.min_confidence, regime_params.min_confidence)
    # (main.py:1751-1757). The regime floor is not modelled here; omitting it can
    # only ADMIT trades live would reject, so this run is the permissive bound.
    min_conf = market.min_confidence
    trades, pos = [], None

    for i in range(LOOKBACK, len(df)):
        row = df.iloc[i]
        window = df.iloc[i - LOOKBACK:i + 1]
        t = htf.iloc[i]

        if pos is not None:
            # Intrabar stop/limit first — the broker would fill these before any
            # indicator-based exit could be evaluated at the close.
            if pos["dir"] == "BUY":
                if row["low"] <= pos["stop"]:
                    trades.append({**pos, "exit": pos["stop"], "why": "stop"}); pos = None
                elif pos["limit"] and row["high"] >= pos["limit"]:
                    trades.append({**pos, "exit": pos["limit"], "why": "limit"}); pos = None
            else:
                if row["high"] >= pos["stop"]:
                    trades.append({**pos, "exit": pos["stop"], "why": "stop"}); pos = None
                elif pos["limit"] and row["low"] <= pos["limit"]:
                    trades.append({**pos, "exit": pos["limit"], "why": "limit"}); pos = None
            if pos is not None:
                close_it, why = should_close_position(
                    window.copy(), pos["dir"], market=market, htf_trend=t)
                if close_it:
                    trades.append({**pos, "exit": float(row["close"]), "why": why[:26]})
                    pos = None
            continue

        sig = strat.analyze(window.copy(), market, float(row["close"]), htf_trend=t)
        if sig.signal in (Signal.BUY, Signal.SELL) and sig.signal.value in sides:
            if sig.confidence < min_conf or sig.stop_distance <= 0:
                continue
            if regime is not None:
                r = regime.iloc[i]
                if r == "NEUTRAL":
                    continue
                if r == "BULLISH" and sig.signal == Signal.SELL:
                    continue
                if r == "BEARISH" and sig.signal == Signal.BUY:
                    continue
            d = sig.signal.value
            entry = float(row["close"])
            pos = {"dir": d, "entry": entry, "stop_dist": sig.stop_distance,
                   "stop": entry - sig.stop_distance if d == "BUY" else entry + sig.stop_distance,
                   "limit": (entry + sig.limit_distance if d == "BUY"
                             else entry - sig.limit_distance) if sig.limit_distance else None,
                   "date": row["date"], "conf": sig.confidence}

    out = []
    for t in trades:
        gross = (t["exit"] - t["entry"]) if t["dir"] == "BUY" else (t["entry"] - t["exit"])
        out.append({**t, "R": (gross - spread) / t["stop_dist"]})
    return out


def summarise(trades, label):
    if not trades:
        return f"  {label:22s} n= 0"
    rs = [t["R"] for t in trades]
    w = [r for r in rs if r > 0]
    l = [r for r in rs if r <= 0]
    pf = (sum(w) / abs(sum(l))) if l and sum(l) else float("inf")
    return (f"  {label:22s} n={len(rs):3d} WR={100*len(w)/len(rs):5.1f}% "
            f"sumR={sum(rs):+7.2f} avgR={st.mean(rs):+.3f} PF={pf:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epic", default="IX.D.DOW.DAILY.IP")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--spread", type=float, default=None)
    ap.add_argument("--no-regime-gate", action="store_true",
                    help="replay WITHOUT the S&P regime gate (what the first, wrong version did)")
    a = ap.parse_args()

    market = next(m for m in MARKETS if m.epic == a.epic)
    if a.spread is None:
        a.spread = SPREADS.get(a.epic, 4.8)
    gate = not a.no_regime_gate
    df = load_archive(a.epic).reset_index(drop=True)
    print(f"=== {market.name} ({a.epic}) — IG archive, live analyze() ===")
    print(f"    {len(df)} bars  {df.date.iloc[0]} -> {df.date.iloc[-1]}  "
          f"spread={a.spread}  min_conf={market.min_confidence}  regime_gate={gate}")

    all_t = replay(df, market, a.spread, apply_regime_gate=gate)
    print("\n--- FULL PERIOD ---")
    print(summarise(all_t, "both sides"))
    print(summarise([t for t in all_t if t["dir"] == "BUY"], "  BUY only"))
    print(summarise([t for t in all_t if t["dir"] == "SELL"], "  SELL only"))

    print(f"\n--- WALK-FORWARD: {a.folds} consecutive windows ---")
    print("    (the test: does the BUY-vs-SELL verdict hold from one window to the next?)")
    n = len(df) // a.folds
    verdicts = []
    for k in range(a.folds):
        seg = df.iloc[k * n:(k + 1) * n].reset_index(drop=True)
        if len(seg) <= LOOKBACK + 10:
            continue
        ts = replay(seg, market, a.spread, apply_regime_gate=gate)
        b = [t["R"] for t in ts if t["dir"] == "BUY"]
        s = [t["R"] for t in ts if t["dir"] == "SELL"]
        v = ("BUY" if sum(b) > sum(s) else "SELL") if (b or s) else "-"
        verdicts.append(v)
        print(f"  fold {k+1} {str(seg.date.iloc[0])[:10]}..{str(seg.date.iloc[-1])[:10]}  "
              f"BUY n={len(b):2d} sumR={sum(b):+6.2f} | SELL n={len(s):2d} sumR={sum(s):+6.2f} "
              f"-> better: {v}")

    print(f"\n  fold verdicts: {verdicts}")
    if verdicts and len(set(v for v in verdicts if v != '-')) > 1:
        print("  ⚠️  VERDICT FLIPS BETWEEN WINDOWS — this is the signature of noise,")
        print("      not a structural asymmetry. Do NOT restrict a side on this evidence.")
    elif verdicts:
        print("  Verdict is sign-consistent across every window.")


if __name__ == "__main__":
    main()
