#!/usr/bin/env python3
"""Forex breakout CANDIDATE scout (2026-06-25).

We currently run the Donchian breakout on EUR/USD + GBP/USD only (USD/JPY
disabled — walk-forward rejected + −£93 live). Question: are there OTHER liquid
majors whose breakout edge clears the IG forex spread? Same discipline that
disabled Crude/USD-JPY: run each pair through the live breakout config
(N=55, 2.0×ATR, HTF-filtered, Donchian-trail exit) and a SPREAD-COST sweep —
keep only pairs that stay positive after a realistic IG round-trip spread.

Incumbents (EUR/USD, GBP/USD) included as in-sample references; USD/JPY as a
negative control (should fail). Reuses the validated breakout_sim/rstats from
backtest_forex_breakout.py. Yahoo 1h/365d, ZERO IG API cost.
Caveat: Yahoo cash ≠ IG DFB; spread modelled as a flat per-trade pip haircut
(real IG spread varies + widens off-session). A pass here → walk-forward next
(the stricter gate that killed USD/JPY), NOT a deploy.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr
from scripts.backtest_forex_breakout import breakout_sim, rstats, htf_series

logging.basicConfig(level=logging.ERROR)

DAYS, INTERVAL, N, STOP_K = 365, "1h", 55, 2.0

# name: (yahoo ticker, pip size, representative IG DFB round-trip spread in pips)
CANDIDATES = {
    "EUR/USD": ("EURUSD=X", 0.0001, 0.6),   # incumbent winner (live +£42) — reference
    "GBP/USD": ("GBPUSD=X", 0.0001, 0.9),   # incumbent — reference (live mixed)
    "USD/JPY": ("USDJPY=X", 0.01,   1.0),   # DISABLED — negative control, expect fail
    "AUD/USD": ("AUDUSD=X", 0.0001, 0.6),   # commodity ccy, trends
    "USD/CAD": ("USDCAD=X", 0.0001, 1.5),   # oil-linked, trends
    "NZD/USD": ("NZDUSD=X", 0.0001, 1.5),   # commodity ccy
    "USD/CHF": ("USDCHF=X", 0.0001, 1.5),   # haven
    "EUR/GBP": ("EURGBP=X", 0.0001, 1.0),   # range-bound — expect poor for breakout
    "AUD/JPY": ("AUDJPY=X", 0.01,   1.8),   # risk-on cross, trends
    "EUR/JPY": ("EURJPY=X", 0.01,   1.5),   # JPY cross
    "GBP/JPY": ("GBPJPY=X", 0.01,   2.5),   # "the dragon" — very trendy/volatile
}

TICKER_MAP.update({name: tk for name, (tk, _p, _s) in CANDIDATES.items()})


def load(name):
    bt = Backtester(params={"ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
                            "rsi_period": 7, "atr_period": 14})
    raw = bt.fetch_data(name, DAYS, INTERVAL)
    if raw is None or len(raw) < 200:
        return None
    raw = raw.copy()
    raw["atr"] = calculate_atr(raw["high"], raw["low"], raw["close"], 14)
    hs = htf_series(bt, name, days=DAYS)
    df = (pd.merge_asof(raw.sort_values("date"), hs.sort_values("date"), on="date",
                        direction="backward") if hs is not None else raw.assign(htf="NEUTRAL"))
    return df


def main():
    print(f"Forex breakout candidates | Yahoo {INTERVAL}/{DAYS}d | N={N}, {STOP_K}×ATR, HTF-filtered, Donchian-trail exit")
    print(f"{'pair':<9} {'spread':>6} | {'n':>3} {'WR':>4} {'R:R':>5} | "
          f"{'P&L@0':>8} {'PF@0':>5} | {'P&L@spr':>8} {'PF@spr':>6} | {'PF@2×spr':>8} | verdict")
    print("-" * 104)
    results = []
    for name, (tk, pip, spr) in CANDIDATES.items():
        df = load(name)
        if df is None:
            print(f"{name:<9} {'—':>6} | NO/THIN DATA"); continue
        min_stop = float(df["close"].median()) * 0.0003
        s0 = rstats(breakout_sim(df, N, STOP_K, "donchian", True, min_stop, cost_pips=0.0, pip=pip))
        s1 = rstats(breakout_sim(df, N, STOP_K, "donchian", True, min_stop, cost_pips=spr, pip=pip))
        s2 = rstats(breakout_sim(df, N, STOP_K, "donchian", True, min_stop, cost_pips=spr * 2, pip=pip))
        if not s0:
            print(f"{name:<9} {spr:>5.1f}p | no trades"); continue
        # verdict: positive P&L AND PF>1.1 after realistic spread, robust at 2×
        ok_spr = s1 and s1["pnl"] > 0 and s1["pf"] > 1.1
        ok_2x = s2 and s2["pf"] > 1.0
        if ok_spr and ok_2x:
            verdict = "✅ CANDIDATE"
        elif ok_spr:
            verdict = "🟡 fragile (fails 2×)"
        else:
            verdict = "❌ fails spread"
        incumbent = " [live]" if name in ("EUR/USD", "GBP/USD") else (" [disabled]" if name == "USD/JPY" else "")
        print(f"{name:<9} {spr:>5.1f}p | {s0['n']:>3} {s0['wr']:>3.0f}% {s0['rr']:>5.2f} | "
              f"{s0['pnl']:>+7.2f}% {s0['pf']:>5.2f} | {s1['pnl']:>+7.2f}% {s1['pf']:>6.2f} | "
              f"{(s2['pf'] if s2 else 0):>8.2f} | {verdict}{incumbent}")
        results.append((name, s1, verdict))
    print("-" * 104)
    print("Gate: ✅ = P&L>0 AND PF>1.1 after realistic IG spread, AND PF>1.0 at 2× spread (robust).")
    print("Incumbents EUR/USD (✅ expected) + GBP/USD anchor the read; USD/JPY (❌ expected) is the control.")
    print("Any ✅ candidate → run walk-forward next (the gate that killed USD/JPY) BEFORE proposing a deploy.")


if __name__ == "__main__":
    main()
