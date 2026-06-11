#!/usr/bin/env python3
"""Japan 225 / Hong Kong HS50 profitability diagnosis (2026-06-11). Both run the
generic `indices` profile (EMA 5/12/26, ADX 30, hourly HTF, MACD exit) with only
1-2 live trades each — never validated per-EPIC. Faithful ^N225 / ^HSI 5m/59d via
the full engine (confidence + hourly-HTF + real 5/12/26 EMAs), regime override
neutralised, profile stop forced. Direction × ADX floor, then ADX ceiling on the
best cell. (AI Index = IX.D.AIIDX, IG-proprietary, no Yahoo equivalent — skipped.)
Yahoo cash != IG DFB; foreign-index intraday history can be thin → directional.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import scripts.backtest_cull_review as CR
import scripts.backtest_adx_ceiling_all as H

logging.basicConfig(level=logging.ERROR)

H.INDICATOR_PARAMS = {  # the real `indices` indicator set
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45, "atr_period": 14,
}
MARKETS = [("Japan 225", "^N225"), ("Hong Kong HS50", "^HSI")]
CI, STOP, RR, CONF = 5, 1.5, 2.0, 0.55
ADX = [25, 30, 35]


def run(name, side, adx, ceiling):
    CR._SIDE["v"] = side
    return H.run_one(name, CI, adx, STOP, ceiling)


def line(tag, r):
    print(f"   {tag:<12}{r.total_trades:>4}t WR {r.win_rate:>4.0%} "
          f"P&L {r.total_pnl:>+7.2f}% PF {r.profit_factor:.2f}", flush=True)


def diag(name, ticker):
    H.prep(name, ticker, CONF, RR)
    bt0 = H.Backtester(params={**H.DEFAULT_PARAMS, **H.INDICATOR_PARAMS})
    df0 = bt0.fetch_data(name, 59, "5m")
    print(f"\n{'='*60}\n{name} ({ticker}) — 5m/{0 if df0 is None else len(df0)}c, EMA 5/12/26, stop {STOP}x, HTF-gated\n{'='*60}", flush=True)
    if df0 is None or len(df0) < 100:
        print("  NO/THIN Yahoo data — cannot diagnose (cash-session intraday too sparse).", flush=True)
        return
    H.MIN_STOP_DISTANCE_MAP[name] = float(df0["close"].median()) * 0.005
    best = None
    for side, lbl in (("BUY", "long"), ("SELL", "short")):
        print(f"  --- {lbl} ---", flush=True)
        for adx in ADX:
            r = run(name, side, adx, None)
            line(f"adx{adx}", r)
            if r.total_trades >= 10 and (best is None or r.total_pnl > best[0]):
                best = (r.total_pnl, side, adx, r)
    if best:
        _, bs, ba, br = best
        print(f"  BEST (n>=10): {bs} adx{ba} — {br.total_trades}t WR {br.win_rate:.0%} "
              f"P&L {br.total_pnl:+.2f}% PF {br.profit_factor:.2f}; ADX-ceiling test:", flush=True)
        for c in (None, 55, 50, 45):
            line(f"ceil {c}", run(name, bs, ba, c))
    else:
        print("  No cell cleared n>=10 — too thin/choppy for a verdict.", flush=True)


def main():
    for name, ticker in MARKETS:
        try:
            diag(name, ticker)
        except Exception as e:
            print(f"\n{name}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
