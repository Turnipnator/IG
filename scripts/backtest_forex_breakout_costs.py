#!/usr/bin/env python3
"""Forex breakout — spread+slippage STRESS test (2026-06-19). The make-or-break.

The gross breakout backtest (backtest_forex_breakout.py) looked materially better
than our trend-follower, but modelled NO trading cost. Breakout is uniquely exposed:
its entries are stop-orders at the channel that SLIP on fast breaks. This re-runs
the winning configs across realistic round-trip costs (spread + slippage, in pips)
to see whether the edge survives — the test that decides if the concept is real.

cost_pips = total round-trip (cross the spread once + entry-slip + exit-slip).
IG DFB majors: spread ~0.6-1.5 pip; add 1-3 pip breakout slippage → 2-5 pip is the
honest band. 0 = the gross ceiling from the previous run. Yahoo 1h/365d, ZERO IG cost.
Still one regime / one year — survival here is necessary, not sufficient.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr, calculate_ema
from scripts.backtest_forex_breakout import breakout_sim, rstats, htf_series, benchmark

logging.basicConfig(level=logging.ERROR)

TICKER_MAP.update({"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X"})

DAYS, INTERVAL, STOP_K = 365, "1h", 2.0
COSTS = [0.0, 2.0, 3.0, 5.0]   # round-trip pips

# pip size per pair in Yahoo price units, and the strong configs to stress
# (name, pip, [(N, exit_mode, use_filter, label), ...])
PAIRS = [
    ("EUR/USD", 0.0001, [(55, "donchian", True, "N55 HTF Donchian"),
                          (40, "donchian", True, "N40 HTF Donchian"),
                          (40, "donchian", False, "N40 naked Donchian")]),
    ("GBP/USD", 0.0001, [(55, "donchian", True, "N55 HTF Donchian"),
                          (40, "donchian", True, "N40 HTF Donchian"),
                          (55, "donchian", False, "N55 naked Donchian")]),
    ("USD/JPY", 0.01,   [(40, "donchian", False, "N40 naked Donchian"),
                          (40, "rr", False, "N40 naked 2:1"),
                          (40, "donchian", True, "N40 HTF Donchian")]),
]


def main():
    for name, pip, configs in PAIRS:
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
        b = benchmark(name)
        print(f"\n{'='*94}\n{name} — Yahoo 1h/365d. Cost = round-trip pips (spread+slippage). pip={pip}\n{'='*94}")
        if b:
            print(f"  current trend-follower (ref): n={b['n']} WR={b['wr']:.0f}% "
                  f"P&L={b['pnl']:+.2f}% PF={b['pf']:.2f}")
        print(f"  {'config':<22}{'cost':>6}  {'n':>4} {'WR':>5} {'P&L%':>8} {'PF':>6} {'avgW':>6} {'avgL':>7} {'R:R':>5}")
        print("  " + "-" * 78)
        for n, exit_mode, use_filter, label in configs:
            for c in COSTS:
                s = rstats(breakout_sim(df, n, STOP_K, exit_mode, use_filter, min_stop, c, pip))
                if not s:
                    continue
                flag = "  <= gross" if c == 0 else ""
                print(f"  {label:<22}{c:>5.0f}p  {s['n']:>4} {s['wr']:>4.0f}% {s['pnl']:>+7.2f}% "
                      f"{s['pf']:>6.2f} {s['avg_w']:>+5.2f} {s['avg_l']:>+6.2f} {s['rr']:>5.2f}{flag}")
            print()


if __name__ == "__main__":
    main()
