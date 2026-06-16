#!/usr/bin/env python3
"""AI-Index-only ADX-ceiling sweet-spot search (2026-06-16).

The shared-`indices` ceiling sweep (scripts/backtest_indices_adx_ceiling.py)
showed a profile-level cap HURTS Wall St/Japan/HK (their high-ADX band wins) while
HELPING AI Index — so the cap belongs on an AI-Index-ONLY profile, if anywhere.

This finds the AI Index sweet spot on its own data. AI Index has no Yahoo proxy →
the only price history is the free IG candle archive (~13d so far, ~6 sim trades),
so this is DIRECTIONAL on a thin sample — read alongside the 4 real journal trades
(losers clustered at the two highest ADX: 41.6, 47.1; winners 32.8, 41.2).

Faithful to the live `indices` profile; sweeps a FINE ceiling grid and dumps every
trade's entry ADX + outcome so the eyeball test beats the summary stats at n~6.
Run in-container on the VPS (the archive lives at /app/data/candle_archive).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester
from src.indicators import calculate_ema
from scripts.archive_loader import load_archive
from scripts.backtest_indices_adx_ceiling import PARAMS, simulate, stats, _build_htf

logging.basicConfig(level=logging.ERROR)

AIIDX_EPIC = "IX.D.AIIDX.DAILY.IP"
FINE_CEILINGS = [None, 55, 50, 48, 46, 45, 44, 42, 40, 38, 35]


def main():
    raw = load_archive(AIIDX_EPIC)
    if raw is None or raw.empty:
        print("No AI Index archive on this host — run in-container on the VPS.")
        return
    span_d = (raw["date"].iloc[-1] - raw["date"].iloc[0]).days or 1
    bt = Backtester(params=PARAMS.copy())
    df = bt.add_indicators(raw)
    htf = _build_htf(bt, "AI Index", df, span_d, "5m")
    min_stop = float(df["close"].median()) * 0.0005

    # Per-trade dump at NO cap — the ground truth distribution.
    base_trades, _ = simulate(df, htf, bt, None, min_stop)
    print(f"{'='*84}\nAI Index archive {span_d}d, {len(df)} candles — every sim trade (no cap)\n{'='*84}")
    print(f"  {'#':>2} {'dir':>4} {'entryADX':>9} {'pnl%':>8}  {'reason'}")
    for i, t in enumerate(base_trades, 1):
        flag = "  <-- high-ADX" if (not pd.isna(t["adx"]) and t["adx"] > 45) else ""
        print(f"  {i:>2} {t['dir']:>4} {t['adx']:>9.1f} {t['pnlp']:>+8.2f}  {t['reason']}{flag}")
    wins = [t for t in base_trades if t["pnlp"] > 0]
    losses = [t for t in base_trades if t["pnlp"] <= 0]
    if wins:
        print(f"  WINNERS entryADX mean={sum(t['adx'] for t in wins)/len(wins):.1f} (n={len(wins)})")
    if losses:
        print(f"  LOSERS  entryADX mean={sum(t['adx'] for t in losses)/len(losses):.1f} (n={len(losses)})")

    # Fine ceiling grid.
    print(f"\n{'='*84}\nFine ADX-ceiling grid (AI Index only)\n{'='*84}")
    print(f"{'ceiling':>8}{'Trades':>8}{'WR':>8}{'P&L%':>9}{'PF':>7}{'removed':>9}{'dP&L%':>9}  note")
    print("-" * 84)
    baseline = None
    for ceil in FINE_CEILINGS:
        trades, removed = simulate(df, htf, bt, ceil, min_stop)
        s = stats(trades)
        if ceil is None:
            baseline = s
            print(f"{'none':>8}{s['n']:>8}{s['wr']:>7.0f}%{s['pnl']:>+8.2f}%{s['pf']:>7.2f}"
                  f"{'':>9}{'':>9}  <= BASELINE")
        else:
            dp = s["pnl"] - baseline["pnl"]
            note = "removes a LOSER" if dp > 0.01 else ("removes a winner" if dp < -0.01 else "no change")
            print(f"{ceil:>8}{s['n']:>8}{s['wr']:>7.0f}%{s['pnl']:>+8.2f}%{s['pf']:>7.2f}"
                  f"{removed:>9}{dp:>+8.2f}%  {note}")


if __name__ == "__main__":
    main()
