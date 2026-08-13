#!/usr/bin/env python3
"""Forex BREAKOUT concept test (2026-06-19) — a different strategy CLASS.

Our whole bot is EMA-momentum CONFIRMATION (enter after the trend is confirmed →
structurally late; sells the bottom / buys the top of a leg). A breakout system is
the opposite: enter ON the break of an N-bar range to catch the initial thrust we
keep missing (the move the friend's chart showed). This tests whether the CONCEPT
has an edge on forex, benchmarked head-to-head against our current trend-follower.

Strategy: classic Donchian / Turtle channel breakout.
  ENTRY  — buy-stop at the prior N-bar HIGH (long) / sell-stop at the prior N-bar
           LOW (short). Fills at the channel level (or the open on a gap through it).
  STOP   — k × ATR from entry (Turtle "N"; k swept).
  EXIT   — (a) DONCHIAN trail: exit long on break of the prior M-bar low (M=N/2) —
               no fixed target, lets winners run (the whole point of breakout);
           (b) fixed 2:1 R:R target — for contrast.
  FILTER — none  vs  HTF-aligned (only longs in a daily-BULLISH market, shorts in
           daily-BEARISH) — reuses our existing edge to kill counter-trend false breaks.

Breakout signature to look for: LOW win-rate but avg-win >> avg-loss (high realised
R:R, big max-win) — the inverse of our current book's "wins tiny / loses big".

Yahoo cash (EURUSD=X / GBPUSD=X / USDJPY=X), 1h, ZERO IG API cost. Caveats: Yahoo !=
IG DFB, no spread/slippage modelled (breakout's wider ATR stops are MORE spread-
robust than our tight forex stops, but still not live), no IG confirmation possible
yet (thin archive). CONCEPT lead only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr, calculate_ema
from scripts.backtest_forex_pullback import MARKETS, simulate as tf_simulate, _build_htf

logging.basicConfig(level=logging.ERROR)

TICKER_MAP.update({"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X"})

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]
DAYS = 365
INTERVAL = "1h"
N_SWEEP = [20, 40, 55]
STOP_K = 2.0
RR = 2.0


def htf_series(bt, name, days=None):
    """Daily BULLISH/BEARISH/NEUTRAL trend, as-of-mergeable onto the 1h frame.

    Two bugs fixed 2026-08-13 — both silently corrupted every caller, and the
    corrupted numbers were the basis of the 2026-07-24 GBP/USD demotion.

    (1) LOOK-AHEAD. Yahoo timestamps a daily bar at 00:00 of its OWN day, so a
        plain merge_asof(direction="backward") attaches day D's row — whose
        `close` is D's CLOSE — to the 1h bars of day D. Every intraday entry was
        therefore gated by a trend computed from that day's closing price. Live
        does NOT do this: update_htf_trends fetches 30 COMPLETED DAY bars. Fix =
        shift the label one bar, so a day is gated by the last completed day.
        Cost on GBP/USD (730d, 3-pip): quarters read 1.41→2.16→1.45→0.78 with the
        bug, 1.76→1.82→1.13→1.19 without. It moved individual quarters by up to
        0.7 PF in BOTH directions — it changes WHICH trades fire, not just P&L.

    (2) HARD-CODED WINDOW. This used the module-level DAYS (365) regardless of the
        caller's own window, so a 725/730-day 1h backtest merged against only 365
        days of daily HTF; every 1h bar older than that got NaN and could never
        produce an entry. `days` now defaults to the module value but callers
        running a longer window MUST pass their own.
    """
    d = bt.fetch_data(name, days or DAYS, "1d")
    if d is None or d.empty:
        return None
    d = d.copy()
    d["ema_9"] = calculate_ema(d["close"], 9)
    d["ema_21"] = calculate_ema(d["close"], 21)
    def trend(r):
        if pd.isna(r["ema_9"]) or pd.isna(r["ema_21"]):
            return "NEUTRAL"
        if r["ema_9"] > r["ema_21"] and r["close"] > r["ema_21"]:
            return "BULLISH"
        if r["ema_9"] < r["ema_21"] and r["close"] < r["ema_21"]:
            return "BEARISH"
        return "NEUTRAL"
    # .shift(1) = gate today with the last COMPLETED daily bar (see (1) above).
    d["htf"] = d.apply(trend, axis=1).shift(1).fillna("NEUTRAL")
    return d[["date", "htf"]]


def breakout_sim(df, n, stop_k, exit_mode, use_filter, min_stop, cost_pips=0.0, pip=0.0001):
    """exit_mode: 'donchian' (trail on M-bar opposite extreme) or 'rr' (fixed 2:1).
    cost_pips = round-trip spread + slippage charged per trade (stop-entry breakouts
    slip on fast breaks). Deducted from each trade's gross %P&L."""
    cost_frac = cost_pips * pip
    m = max(2, n // 2)
    high, low = df["high"], df["low"]
    upper = high.rolling(n).max().shift(1)
    lower = low.rolling(n).min().shift(1)
    exit_low = low.rolling(m).min().shift(1)
    exit_high = high.rolling(m).max().shift(1)
    trades, pos = [], None
    for i in range(n + 1, len(df)):
        row = df.iloc[i]
        o, h, l, atr = row["open"], row["high"], row["low"], row["atr"]
        if pos:
            d = pos["dir"]
            reason = price = None
            if d == "BUY":
                if l <= pos["stop"]:
                    reason, price = "Stop", pos["stop"]
                elif exit_mode == "rr" and h >= pos["limit"]:
                    reason, price = "TP", pos["limit"]
                elif exit_mode == "donchian" and not pd.isna(exit_low.iloc[i]) and l <= exit_low.iloc[i]:
                    reason, price = "DonExit", exit_low.iloc[i]
            else:
                if h >= pos["stop"]:
                    reason, price = "Stop", pos["stop"]
                elif exit_mode == "rr" and l <= pos["limit"]:
                    reason, price = "TP", pos["limit"]
                elif exit_mode == "donchian" and not pd.isna(exit_high.iloc[i]) and h >= exit_high.iloc[i]:
                    reason, price = "DonExit", exit_high.iloc[i]
            if reason:
                pnlp = ((price - pos["entry"]) if d == "BUY" else (pos["entry"] - price)) / pos["entry"] * 100
                pnlp -= cost_frac / pos["entry"] * 100   # spread + slippage haircut
                trades.append(dict(dir=d, pnlp=pnlp, reason=reason))
                pos = None
            continue
        if pd.isna(atr) or pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
            continue
        long_ok = (not use_filter) or row["htf"] == "BULLISH"
        short_ok = (not use_filter) or row["htf"] == "BEARISH"
        if h >= upper.iloc[i] and long_ok:
            entry = max(o, upper.iloc[i])           # gap-through fills at the open
            sd = max(atr * stop_k, min_stop)
            pos = dict(dir="BUY", entry=entry, stop=entry - sd,
                       limit=entry + sd * RR if exit_mode == "rr" else None)
        elif l <= lower.iloc[i] and short_ok:
            entry = min(o, lower.iloc[i])
            sd = max(atr * stop_k, min_stop)
            pos = dict(dir="SELL", entry=entry, stop=entry + sd,
                       limit=entry - sd * RR if exit_mode == "rr" else None)
    return trades


def rstats(trades):
    n = len(trades)
    if not n:
        return None
    wins = [t for t in trades if t["pnlp"] > 0]
    losses = [t for t in trades if t["pnlp"] < 0]
    gp = sum(t["pnlp"] for t in wins)
    gl = -sum(t["pnlp"] for t in losses)
    avg_w = gp / len(wins) if wins else 0.0
    avg_l = -gl / len(losses) if losses else 0.0
    return dict(n=n, wr=len(wins) / n * 100, pnl=sum(t["pnlp"] for t in trades),
                pf=gp / gl if gl > 0 else 999.0, avg_w=avg_w, avg_l=avg_l,
                rr=(avg_w / -avg_l) if avg_l < 0 else 0.0,
                maxw=max((t["pnlp"] for t in trades), default=0.0))


def line(tag, s):
    if s is None:
        print(f"  {tag:<30} no trades"); return
    print(f"  {tag:<30} n={s['n']:>3} WR={s['wr']:>3.0f}% P&L={s['pnl']:>+7.2f}% PF={s['pf']:>5.2f}  "
          f"avgW={s['avg_w']:>+5.2f} avgL={s['avg_l']:>+6.2f} R:R={s['rr']:>4.2f} maxW={s['maxw']:>+5.2f}")


def benchmark(name):
    """Our CURRENT trend-follower on identical data (immediate entry, live exits)."""
    spec = next((m for m in MARKETS if m[0] == name), None)
    if not spec:
        return None
    _, interval, htf_i, days, params, conf, stop, be, trail, rdrop = spec
    bt = Backtester(params=params.copy())
    raw = bt.fetch_data(name, days, interval)
    if raw is None or len(raw) < 100:
        return None
    df = bt.add_indicators(raw)
    htf = _build_htf(bt, name, htf_i, days)
    min_stop = float(df["close"].median()) * 0.0003
    cooldown = max(1, round(60 / ({"15m": 15, "1h": 60}[interval])))
    tr, _ = tf_simulate(df, htf, bt, conf, stop, be, trail, rdrop, params["adx_threshold"],
                        None, 0, 0, min_stop, cooldown)
    return rstats(tr)


def main():
    for name in PAIRS:
        bt = Backtester(params={"ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
                                "rsi_period": 7, "atr_period": 14})
        raw = bt.fetch_data(name, DAYS, INTERVAL)
        if raw is None or len(raw) < 200:
            print(f"\n{name}: NO/THIN data"); continue
        raw = raw.copy()
        raw["atr"] = calculate_atr(raw["high"], raw["low"], raw["close"], 14)
        hs = htf_series(bt, name)
        df = pd.merge_asof(raw.sort_values("date"), hs.sort_values("date"),
                           on="date", direction="backward") if hs is not None else raw.assign(htf="NEUTRAL")
        min_stop = float(df["close"].median()) * 0.0003
        print(f"\n{'='*92}\n{name} — Yahoo {INTERVAL}/{DAYS}d, {len(df)} candles. "
              f"Donchian breakout, stop {STOP_K}xATR\n{'='*92}")
        line(">> BENCHMARK: current trend-follower", benchmark(name))
        for exit_mode in ("donchian", "rr"):
            tag = "Donchian-trail exit" if exit_mode == "donchian" else "fixed 2:1 exit"
            print(f"  -- breakout, {tag} --")
            for use_filter in (False, True):
                ftag = "HTF-filtered" if use_filter else "naked      "
                for n in N_SWEEP:
                    s = rstats(breakout_sim(df, n, STOP_K, exit_mode, use_filter, min_stop))
                    line(f"  N={n:<3} {ftag}", s)


if __name__ == "__main__":
    main()
