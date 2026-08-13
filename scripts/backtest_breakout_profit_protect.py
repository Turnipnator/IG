#!/usr/bin/env python3
"""Forex BREAKOUT open-profit protection A/B (review item 10, 2026-06-26).

Trigger: 2026-06-24 GBP/USD breakout SELL #199 ran +46pts in favour then round-
tripped the whole lot back to ~flat. Breakouts run NO take-profit by design — the
ONLY live exits are the Donchian-M trail (M=N//2=27 bars ≈ 27h on 1h, far too wide
to lock a one-day spike) and the k×ATR stop. (The "RSI-extreme" bank that saved EUR
#192 was an artifact of the pre-item-11 lost-tag bug routing it to the momentum
exit; the designed breakout path has no such exit.) So a medium 20–50pt move that
reverses gives everything back — the "dead zone".

This A/Bs three profit-protection layers against the current naked Donchian trail,
on the LIVE config (GBP/USD + EUR/USD, N55, HTF-filtered, stop 2.0×ATR, 1h), with a
realistic 3-pip IG DFB round-trip cost:
  (a) partial  — bank fraction F of the position at +k×ATR, let the rest ride the trail
  (b) tighten  — once +k×ATR open profit, switch the trail channel M from 27 → m_tight
  (c) lock     — once +k×ATR open profit, ratchet the stop to entry + lock×(k×ATR) to
                 lock profit (mirrors the momentum breakeven_lock_pct Gold uses)

The turtle design exists to catch the rare fat-tail runner, so the PASS bar is:
improve net P&L / PF AND cut the round-trip rate WITHOUT materially clipping maxW
(the runner). A candidate that lifts P&L only by capping the big winner is a FAIL.

Yahoo cash (GBPUSD=X / EURUSD=X), 1h. Zero IG cost. Caveats: Yahoo != IG DFB; flat
3-pip cost flatters fast fills; one 365d window (the walk-forward showed the GBP edge
fading — re-run on more data before trusting magnitudes). A/B DELTAS are the read.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr
from scripts.backtest_forex_breakout import htf_series

logging.basicConfig(level=logging.ERROR)
TICKER_MAP.update({"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X"})

PAIRS = ["GBP/USD", "EUR/USD"]
DAYS, INTERVAL = 365, "1h"
N, STOP_K = 55, 2.0
COST_PIPS, PIP = 3.0, 0.0001


def sim(df, protect=None, n=N, stop_k=STOP_K, min_stop=0.0, cost_pips=COST_PIPS, pip=PIP):
    """Donchian breakout (N55, HTF-filtered, k×ATR stop) with optional open-profit
    protection. protect=None is the live baseline (naked Donchian-M trail).

    protect dict modes (k = trigger in ATR-at-entry units):
      {'mode':'partial','k':1.5,'frac':0.5}
      {'mode':'tighten','k':1.5,'m_tight':8}
      {'mode':'lock','k':1.5,'lock':0.5}
    """
    cost_frac = cost_pips * pip
    m = max(2, n // 2)
    high, low = df["high"], df["low"]
    upper = high.rolling(n).max().shift(1)
    lower = low.rolling(n).min().shift(1)
    exit_low = low.rolling(m).min().shift(1)
    exit_high = high.rolling(m).max().shift(1)
    mt = (protect or {}).get("m_tight", m)
    exit_low_t = low.rolling(mt).min().shift(1)
    exit_high_t = high.rolling(mt).max().shift(1)

    trades, pos = [], None
    for i in range(n + 1, len(df)):
        row = df.iloc[i]
        o, h, l, atr = row["open"], row["high"], row["low"], row["atr"]
        if pos:
            d, e, ae = pos["dir"], pos["entry"], pos["atr_e"]
            # peak favourable excursion in ATR units (round-trip diagnosis)
            fav = ((h - e) if d == "BUY" else (e - l)) / ae
            pos["peak"] = max(pos["peak"], fav)
            trig = pos["k"] * ae if pos.get("k") else None

            # (c) lock / (b) tighten arm: once open profit >= k×ATR, ratchet/​switch.
            # (partial does its own arming inline in the BUY/SELL branch below, so it
            # must NOT consume the armed flag here.)
            if protect and protect["mode"] in ("lock", "tighten") and not pos["armed"] and trig is not None:
                hit = (h >= e + trig) if d == "BUY" else (l <= e - trig)
                if hit:
                    pos["armed"] = True
                    if protect["mode"] == "lock":
                        lk = protect["lock"] * trig
                        pos["stop"] = (e + lk) if d == "BUY" else (e - lk)

            reason = price = None
            ex_lo = exit_low_t.iloc[i] if (protect and protect["mode"] == "tighten" and pos["armed"]) else exit_low.iloc[i]
            ex_hi = exit_high_t.iloc[i] if (protect and protect["mode"] == "tighten" and pos["armed"]) else exit_high.iloc[i]
            if d == "BUY":
                # (a) partial: bank frac at +k×ATR before checking stop/trail
                if protect and protect["mode"] == "partial" and not pos["armed"] and h >= e + trig:
                    lvl = e + trig
                    pos["banked"] += protect["frac"] * ((lvl - e) / e * 100 - cost_frac / e * 100)
                    pos["rem"] -= protect["frac"]; pos["armed"] = True
                if l <= pos["stop"]:
                    reason, price = "Stop", pos["stop"]
                elif not pd.isna(ex_lo) and l <= ex_lo:
                    reason, price = "DonExit", ex_lo
            else:
                if protect and protect["mode"] == "partial" and not pos["armed"] and l <= e - trig:
                    lvl = e - trig
                    pos["banked"] += protect["frac"] * ((e - lvl) / e * 100 - cost_frac / e * 100)
                    pos["rem"] -= protect["frac"]; pos["armed"] = True
                if h >= pos["stop"]:
                    reason, price = "Stop", pos["stop"]
                elif not pd.isna(ex_hi) and h >= ex_hi:
                    reason, price = "DonExit", ex_hi
            if reason:
                move = ((price - e) if d == "BUY" else (e - price)) / e * 100
                legpnl = pos["rem"] * (move - cost_frac / e * 100)
                total = pos["banked"] + legpnl
                trades.append(dict(dir=d, pnlp=total, reason=reason, peak=pos["peak"]))
                pos = None
            continue
        if pd.isna(atr) or pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
            continue
        if h >= upper.iloc[i] and row["htf"] == "BULLISH":
            entry = max(o, upper.iloc[i]); sd = max(atr * stop_k, min_stop)
            pos = dict(dir="BUY", entry=entry, stop=entry - sd, atr_e=atr, peak=0.0,
                       armed=False, banked=0.0, rem=1.0, k=(protect or {}).get("k"))
        elif l <= lower.iloc[i] and row["htf"] == "BEARISH":
            entry = min(o, lower.iloc[i]); sd = max(atr * stop_k, min_stop)
            pos = dict(dir="SELL", entry=entry, stop=entry + sd, atr_e=atr, peak=0.0,
                       armed=False, banked=0.0, rem=1.0, k=(protect or {}).get("k"))
    return trades


def stats(trades):
    n = len(trades)
    if not n:
        return None
    wins = [t for t in trades if t["pnlp"] > 0]
    losses = [t for t in trades if t["pnlp"] < 0]
    gp = sum(t["pnlp"] for t in wins); gl = -sum(t["pnlp"] for t in losses)
    # round-trip: ran >=1.5 ATR favourable but closed <=0
    rt = sum(1 for t in trades if t["peak"] >= 1.5 and t["pnlp"] <= 0)
    return dict(n=n, wr=len(wins) / n * 100, pnl=sum(t["pnlp"] for t in trades),
                pf=gp / gl if gl > 0 else 999.0,
                maxw=max((t["pnlp"] for t in trades), default=0.0),
                roundtrip=rt)


def line(tag, s, base=None):
    if s is None:
        print(f"  {tag:<26} no trades"); return
    dpf = f" Δ{s['pnl']-base['pnl']:+.2f}%" if base else ""
    flag = ""
    if base and s is not base:
        if s["maxw"] < base["maxw"] * 0.8:
            flag = "  ⚠ clips runner"
    print(f"  {tag:<26} n={s['n']:>3} WR={s['wr']:>3.0f}% P&L={s['pnl']:>+7.2f}%{dpf:>9} "
          f"PF={s['pf']:>5.2f} maxW={s['maxw']:>+5.2f} roundtrips={s['roundtrip']:>2}{flag}")


def main():
    arms = [
        ("BASELINE naked trail", None),
        ("partial k1.5 f0.5", {"mode": "partial", "k": 1.5, "frac": 0.5}),
        ("partial k2.0 f0.5", {"mode": "partial", "k": 2.0, "frac": 0.5}),
        ("partial k2.0 f0.33", {"mode": "partial", "k": 2.0, "frac": 0.33}),
        ("partial k2.0 f0.25", {"mode": "partial", "k": 2.0, "frac": 0.25}),
        ("partial k2.5 f0.25", {"mode": "partial", "k": 2.5, "frac": 0.25}),
        ("tighten k1.5 m8", {"mode": "tighten", "k": 1.5, "m_tight": 8}),
        ("tighten k2.0 m8", {"mode": "tighten", "k": 2.0, "m_tight": 8}),
        ("lock k1.5 l0.5", {"mode": "lock", "k": 1.5, "lock": 0.5}),
        ("lock k2.0 l0.5", {"mode": "lock", "k": 2.0, "lock": 0.5}),
    ]
    for name in PAIRS:
        bt = Backtester(params={"ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
                                "rsi_period": 7, "atr_period": 14})
        raw = bt.fetch_data(name, DAYS, INTERVAL)
        if raw is None or len(raw) < 200:
            print(f"\n{name}: NO/THIN data"); continue
        raw = raw.copy()
        raw["atr"] = calculate_atr(raw["high"], raw["low"], raw["close"], 14)
        hs = htf_series(bt, name, days=DAYS)
        df = pd.merge_asof(raw.sort_values("date"), hs.sort_values("date"),
                           on="date", direction="backward") if hs is not None else raw.assign(htf="NEUTRAL")
        min_stop = float(df["close"].median()) * 0.0003
        print(f"\n{'='*100}\n{name} — Yahoo {INTERVAL}/{DAYS}d, {len(df)} candles. "
              f"N{N} HTF Donchian, stop {STOP_K}×ATR, {COST_PIPS}-pip cost\n{'='*100}")
        base = stats(sim(df, None, min_stop=min_stop))
        for tag, p in arms:
            s = stats(sim(df, p, min_stop=min_stop))
            line(tag, s, base if p is not None else None)
    print("\nPASS bar: lifts P&L/PF AND cuts roundtrips WITHOUT '⚠ clips runner' "
          "(maxW < 80% of baseline). Read DELTAS, not levels (one 365d window).")


if __name__ == "__main__":
    main()
