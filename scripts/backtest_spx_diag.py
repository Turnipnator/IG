#!/usr/bin/env python3
"""S&P 500 profitability diagnosis (2026-06-11). Live `indices_selective` (EMA
5/12/26, ADX 40, hourly HTF, MACD exit) is PF 0.56 over 9 trades — barely active
and net-negative. Both directions lose on tiny samples; the worst trade was a
06-09 SELL at ADX 57.2 that got faded +15pts (classic high-ADX exhaustion short,
the pattern S&P was excluded from in the ADX-ceiling work).

Faithful ^GSPC 5m/59d via the full engine (confidence + hourly-HTF gate + the
real 5/12/26 EMAs), regime stop-override neutralised, profile stop forced.
Step 1: direction × ADX floor. Step 2: ADX ceiling on the best cell. Yahoo
^GSPC cash != IG S&P DFB → directional lead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import scripts.backtest_cull_review as CR          # side-gate patch
import scripts.backtest_adx_ceiling_all as H       # forced-stop + ceiling + caches + maps

logging.basicConfig(level=logging.ERROR)

# Use the real indices_selective indicator set, not the harness's generic 9/21/50.
H.INDICATOR_PARAMS = {
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45, "atr_period": 14,
}
NAME, TICKER, CI, STOP, RR, CONF = "S&P 500", "^GSPC", 5, 1.5, 2.0, 0.55
ADX = [25, 30, 35, 40]


def run(side, adx, ceiling):
    CR._SIDE["v"] = side
    return H.run_one(NAME, CI, adx, STOP, ceiling)


def line(tag, r):
    print(f"   {tag:<16}{r.total_trades:>4}t WR {r.win_rate:>4.0%} "
          f"P&L {r.total_pnl:>+7.2f}% PF {r.profit_factor:.2f}", flush=True)


def main():
    H.prep(NAME, TICKER, CONF, RR)
    bt0 = H.Backtester(params={**H.DEFAULT_PARAMS, **H.INDICATOR_PARAMS})
    df0 = bt0.fetch_data(NAME, 59, "5m")
    if df0 is None or len(df0) < 100:
        print("NO/THIN ^GSPC data — abort"); return
    H.MIN_STOP_DISTANCE_MAP[NAME] = float(df0["close"].median()) * 0.005

    print("=" * 64)
    print(f"S&P 500 DIAGNOSIS — ^GSPC 5m/59d, EMA 5/12/26, stop {STOP}x, R:R {RR}, HTF-gated")
    print("=" * 64)
    best = None  # (pnl, side, adx, result)
    for side, lbl in (("both", "both"), ("BUY", "long"), ("SELL", "short")):
        print(f"  --- {lbl} ---", flush=True)
        for adx in ADX:
            r = run(side, adx, None)
            line(f"adx{adx}", r)
            if r.total_trades >= 12 and (best is None or r.total_pnl > best[0]):
                best = (r.total_pnl, side, adx, r)

    if best:
        _, bside, badx, br = best
        print(f"\n  BEST (n>=12): {bside} adx{badx} — "
              f"{br.total_trades}t WR {br.win_rate:.0%} P&L {br.total_pnl:+.2f}% PF {br.profit_factor:.2f}", flush=True)
        print(f"  ADX-ceiling test on {bside} adx{badx} (does capping high-ADX exhaustion help?):", flush=True)
        for ceil in (None, 55, 50, 45):
            r = run(bside, badx, ceil)
            line(f"ceil {ceil}", r)
    else:
        print("\n  No cell cleared n>=12 — S&P is too thin/choppy for this strategy.", flush=True)


if __name__ == "__main__":
    main()
