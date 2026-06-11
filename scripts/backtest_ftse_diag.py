#!/usr/bin/env python3
"""FTSE regressor diagnosis (2026-06-11). Live current-era (indices_tight, 2.0x)
is 4/4 SELL losses, all faded (3 MACD-reversal exits, price rose into each
short) — dies on DIRECTION, not stop width. Question: does ^FTSE have a
persistent long-bias the bot is fighting, or is short just unlucky lately?

Reuses the all-EPIC harness (forced profile stop, regime override neutralised).
Runs both/long/short at the live floor (30) across stop widths, on the max
Yahoo 5m window (59d). Yahoo ^FTSE != IG FTSE DFB but direction asymmetry and
stop-width ordering transfer as leads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backtest_cull_review as CR   # brings the side-gate patch + harness
import scripts.backtest_adx_ceiling_all as H
from src.backtest import Backtester

NAME, TICKER, CI, FLOOR, RR, CONF = "FTSE 100", "^FTSE", 5, 30, 2.0, 0.55
STOPS = [1.0, 1.5, 2.0, 2.5]


def main():
    yf_int, days, htf = H._intraday(CI)
    H.prep(NAME, TICKER, CONF, RR)
    bt0 = Backtester(params={**H.DEFAULT_PARAMS, **H.INDICATOR_PARAMS})
    df0 = bt0.fetch_data(NAME, days, yf_int)
    if df0 is None or len(df0) < 100:
        print("NO/THIN Yahoo data — abort"); return
    H.MIN_STOP_DISTANCE_MAP[NAME] = float(df0["close"].median()) * 0.005

    print("=" * 76)
    print(f"FTSE DIAGNOSIS — ^FTSE {yf_int}/{days}d, floor {FLOOR}, R:R {RR}")
    print("=" * 76)
    print(f"{'stop':>5}{'side':>7}{'t':>5}{'WR':>6}{'P&L%':>9}{'PF':>7}")
    spec = (NAME, TICKER, CI, FLOOR, None, RR, CONF)
    for stop in STOPS:
        for label, side in (("both", None), ("long", "BUY"), ("short", "SELL")):
            CR._SIDE["v"] = side
            r = H.run_one(NAME, CI, FLOOR, stop, None)
            print(f"{stop:>5.1f}{label:>7}{r.total_trades:>5}{r.win_rate:>5.0%}"
                  f"{r.total_pnl:>+8.2f}%{r.profit_factor:>7.2f}")
        print("   " + "-" * 40)


if __name__ == "__main__":
    main()
