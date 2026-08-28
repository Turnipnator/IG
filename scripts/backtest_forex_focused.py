#!/usr/bin/env python3
"""Focused forex diagnosis (2026-06-19) — three hypotheses from the live journal.

The current-era forex book is ~40% WR / ≈ -£100 over ~20 trades. Journal patterns:
  (a) SHORTS (5W/6L) beat LONGS (3W/9L) — is forex better short-only / one-sided?
  (b) GBP/USD is the worst pair (1W/5L, -£42) — disable/restrict candidate?
  (c) USD/JPY wins tiny (+£3.4 avg) but loses big (-£17 avg) — a ~1:5 inverted
      realized R:R. Are the BE/trail/ranging exits scratching winners before the
      2.0 R:R target, while losers run to the full stop?

Reuses the FAITHFUL forex simulator from backtest_forex_pullback.py (live exits:
stop + TP + breakeven + ATR-trail + ranging-ADX, no MACD). Immediate entry only
(window=0) = exactly how forex trades live. Yahoo cash, ZERO IG API cost.

Caveats (same as before): Yahoo != IG DFB, no spread/regime/hours modelled, n is
small — directional lead, not a live-P&L promise. Read the DELTAS, not the levels.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from src.backtest import Backtester
from scripts.backtest_forex_pullback import MARKETS, simulate, _build_htf

logging.basicConfig(level=logging.ERROR)


def rstats(trades):
    """Rich stats: separates TP wins from trailed/BE/ranging exits to expose
    whether winners are being cut before the 2.0 R:R target."""
    n = len(trades)
    if not n:
        return None
    wins = [t for t in trades if t["pnlp"] > 0]
    losses = [t for t in trades if t["pnlp"] < 0]
    gp = sum(t["pnlp"] for t in wins)
    gl = -sum(t["pnlp"] for t in losses)
    pf = gp / gl if gl > 0 else 999.0
    avg_w = gp / len(wins) if wins else 0.0
    avg_l = -gl / len(losses) if losses else 0.0
    tp = sum(1 for t in trades if t["reason"] == "TP")
    rng = sum(1 for t in trades if t["reason"] == "Ranging")
    # "Stop"-reason exits split by sign: real stop-out (loss) vs trail/BE lock (>=0)
    stop_loss = sum(1 for t in trades if t["reason"] == "Stop" and t["pnlp"] < 0)
    trail_lock = sum(1 for t in trades if t["reason"] == "Stop" and t["pnlp"] >= 0)
    rr = (avg_w / -avg_l) if avg_l < 0 else 0.0
    return dict(n=n, wr=len(wins) / n * 100, pnl=sum(t["pnlp"] for t in trades),
                pf=pf, avg_w=avg_w, avg_l=avg_l, rr=rr,
                tp=tp, rng=rng, stop_loss=stop_loss, trail_lock=trail_lock)


def line(tag, s):
    if s is None:
        print(f"  {tag:<22} no trades"); return
    print(f"  {tag:<22} n={s['n']:>3} WR={s['wr']:>3.0f}% P&L={s['pnl']:>+7.2f}% "
          f"PF={s['pf']:>5.2f}  avgW={s['avg_w']:>+5.2f} avgL={s['avg_l']:>+6.2f} "
          f"realR:R={s['rr']:>4.2f}  [TP={s['tp']} trailLock={s['trail_lock']} "
          f"rng={s['rng']} stopLoss={s['stop_loss']}]")


def load(name, interval, htf_i, days, params):
    bt = Backtester(params=params.copy())
    raw = bt.fetch_data(name, days, interval)
    if raw is None or len(raw) < 100:
        return None
    df = bt.add_indicators(raw)
    htf = _build_htf(bt, name, htf_i, days)
    min_stop = float(df["close"].median()) * 0.0003
    interval_min = {"15m": 15, "1h": 60}[interval]
    cooldown = max(1, round(60 / interval_min))
    return bt, df, htf, min_stop, cooldown, interval_min


def main():
    print("#" * 88)
    print("# HYPOTHESIS (a) + (b): direction split per pair (immediate entry = live)")
    print("#" * 88)
    agg = {None: [], "SELL": [], "BUY": []}
    for name, interval, htf_i, days, params, conf, stop, be, trail, rdrop in MARKETS:
        L = load(name, interval, htf_i, days, params)
        if L is None:
            print(f"\n{name}: NO/THIN data"); continue
        bt, df, htf, min_stop, cooldown, _ = L
        adx_t = params["adx_threshold"]
        print(f"\n=== {name}  ({interval}, ADX {adx_t}, stop {stop}x, BE {be}, trail {trail}x) ===")
        for allowed, tag in [(None, "both directions"), ("SELL", "SELL-only"), ("BUY", "BUY-only")]:
            tr, _ = simulate(df, htf, bt, conf, stop, be, trail, rdrop, adx_t,
                             allowed, 0, 0, min_stop, cooldown)
            line(tag, rstats(tr))
            agg[allowed].extend(tr)
    print(f"\n--- FOREX BOOK AGGREGATE (all 3 pairs) ---")
    for allowed, tag in [(None, "both directions"), ("SELL", "SELL-only"), ("BUY", "BUY-only")]:
        line(tag, rstats(agg[allowed]))

    print("\n" + "#" * 88)
    print("# HYPOTHESIS (c): are the exits cutting winners? (both directions)")
    print("#  current live exits  vs  pure stop/TP (no BE/trail/ranging)  vs  R:R 1.5 / 3.0")
    print("#" * 88)
    import scripts.backtest_forex_pullback as fp
    for name, interval, htf_i, days, params, conf, stop, be, trail, rdrop in MARKETS:
        L = load(name, interval, htf_i, days, params)
        if L is None:
            continue
        bt, df, htf, min_stop, cooldown, _ = L
        adx_t = params["adx_threshold"]
        print(f"\n=== {name} ===")
        # current live config
        tr, _ = simulate(df, htf, bt, conf, stop, be, trail, rdrop, adx_t, None, 0, 0, min_stop, cooldown)
        line("current (live)", rstats(tr))
        # pure stop/TP: be_trig huge (never arms → no trail), ranging off
        tr, _ = simulate(df, htf, bt, conf, stop, 999, 0.0, 0, adx_t, None, 0, 0, min_stop, cooldown)
        line("pure stop/TP", rstats(tr))
        # no ranging exit only (keep BE/trail)
        tr, _ = simulate(df, htf, bt, conf, stop, be, trail, 0, adx_t, None, 0, 0, min_stop, cooldown)
        line("no ranging exit", rstats(tr))
        # R:R sweep (pure stop/TP so the target is the only variable)
        for rr in (1.5, 3.0):
            old = fp.RR
            fp.RR = rr
            tr, _ = simulate(df, htf, bt, conf, stop, 999, 0.0, 0, adx_t, None, 0, 0, min_stop, cooldown)
            fp.RR = old
            line(f"pure stop/TP, R:R {rr}", rstats(tr))


if __name__ == "__main__":
    main()
