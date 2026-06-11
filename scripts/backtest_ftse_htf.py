#!/usr/bin/env python3
"""FTSE: does a DAILY trend gate stop the counter-trend pullback-shorts?
(2026-06-11). Live shorts fire on an HOURLY-bearish classification, but FTSE's
dominant trend is daily-UP — so hourly pullbacks in a daily uptrend trigger
shorts that get faded. Test: compare HTF=1h (live) vs HTF=1d for both/long/short.
If 1d removes the short losers without gutting the long edge, htf_resolution=DAY
is the per-EPIC fix. Yahoo ^FTSE 5m/59d, regime override neutralised, profile
stop forced. Yahoo cash != IG DFB but the HTF-resolution effect transfers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_cull_review as CR   # side-gate patch + harness
import scripts.backtest_adx_ceiling_all as H
from src.backtest import Backtester

NAME, TICKER, CI, FLOOR, STOP, RR, CONF = "FTSE 100", "^FTSE", 5, 30, 1.5, 2.0, 0.55


def run(htf_int, side):
    CR._SIDE["v"] = side
    H._FORCED_STOP["v"] = STOP
    H._CEIL["v"] = None
    params = {**H.DEFAULT_PARAMS, **H.INDICATOR_PARAMS}
    params["adx_threshold"] = FLOOR
    params["stop_atr_multiplier"] = STOP
    bt = Backtester(params=params)
    return bt.run(NAME, days=59, interval="5m", htf_interval=htf_int,
                  require_htf_alignment=True)


def main():
    H.prep(NAME, TICKER, CONF, RR)
    bt0 = Backtester(params={**H.DEFAULT_PARAMS, **H.INDICATOR_PARAMS})
    df0 = bt0.fetch_data(NAME, 59, "5m")
    H.MIN_STOP_DISTANCE_MAP[NAME] = float(df0["close"].median()) * 0.005

    print("=" * 70)
    print(f"FTSE HTF-resolution test — ^FTSE 5m/59d, floor {FLOOR}, stop {STOP}x")
    print("=" * 70)
    print(f"{'HTF':>6}{'side':>7}{'t':>5}{'WR':>6}{'P&L%':>9}{'PF':>7}")
    for htf_int, htf_lbl in (("1h", "1h"), ("1d", "1d")):
        for label, side in (("both", None), ("long", "BUY"), ("short", "SELL")):
            r = run(htf_int, side)
            print(f"{htf_lbl:>6}{label:>7}{r.total_trades:>5}{r.win_rate:>5.0%}"
                  f"{r.total_pnl:>+8.2f}%{r.profit_factor:>7.2f}")
        print("   " + "-" * 36)


if __name__ == "__main__":
    main()
