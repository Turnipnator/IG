#!/usr/bin/env python3
"""
A/B test: indices (ADX 30) vs indices_selective (ADX 40) on Russell 2000 and Germany 40.

These two markets were moved to indices_selective in commit 1434207 based on
analogy to S&P's fix. This script validates the change directly rather than
waiting for live trades to accumulate.
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MARKETS, STRATEGY_PROFILES
from backtest import run_backtest


def format_row(label, r):
    if r is None or not r.trades:
        return f"{label:<28} (no trades)"
    n = len(r.trades)
    wins = [t for t in r.trades if t.pnl > 0.5]
    losses = [t for t in r.trades if t.pnl < -0.5]
    bes = n - len(wins) - len(losses)
    gw = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    wr = len(wins) / n * 100 if n else 0
    avg_w = gw / len(wins) if wins else 0
    avg_l = -gl / len(losses) if losses else 0
    return (
        f"{label:<28} {n:>3} {len(wins)}/{len(losses)}/{bes:<3} "
        f"{wr:>5.1f} £{r.total_pnl:>+8.2f} {pf:>5.2f} "
        f"£{avg_w:>+6.2f} £{avg_l:>+6.2f}"
    )


def main():
    days = 59
    targets = ["US Russell 2000", "Germany 40"]
    base = STRATEGY_PROFILES["indices"]           # ADX 30
    tight = STRATEGY_PROFILES["indices_selective"]  # ADX 40

    print(f"\nADX A/B test — indices (ADX 30) vs indices_selective (ADX 40)")
    print(f"Window: {days}d of 5m Yahoo data")
    print("=" * 110)
    header = f"{'Market / Profile':<28} {'N':>3} {'W/L/BE':>9} {'WR%':>6} {'P&L':>10} {'PF':>6} {'AvgW':>8} {'AvgL':>8}"
    print(header)
    print("-" * len(header))

    for name in targets:
        mc = next(m for m in MARKETS if m.name == name)
        r_adx30 = run_backtest(mc, strategy_override=base, days=days)
        r_adx40 = run_backtest(mc, strategy_override=tight, days=days)
        print(format_row(f"{name} (ADX 30)", r_adx30))
        print(format_row(f"{name} (ADX 40)", r_adx40))
        if r_adx30 and r_adx40:
            d_pnl = r_adx40.total_pnl - r_adx30.total_pnl
            print(f"  -> delta: £{d_pnl:+.2f} ({'ADX 40 better' if d_pnl > 0 else 'ADX 30 better'})")
        print()


if __name__ == "__main__":
    main()
