#!/usr/bin/env python3
"""Wall Street (Dow) LONG entries — does the high-ADX / extended-leg band that
produced trade #207 actually lose money, or was #207 the unlucky tail of a
net-winning setup?

Motivation (2026-06-25): live Wall St #207 = a VALID long (bullish EMA, RSI 49.9,
ADX 44.5, HTF bullish, conf 0.79) that was the 3rd long of the day, bought ~4xATR
above the earlier winner into tripled volatility (ATR 27->75->100), and got
stopped on a pullback (−£22.80). The question for the 06-26 review: is "high-ADX,
late-leg long" a losing pattern we should filter (leg-filter / ADX-ceiling), or a
+EV band we'd be wrong to clip? Wall St runs the bare `indices` profile — NO
leg-filter, NO ADX-ceiling — so nothing currently gates it.

Method: run ^DJI LONG-ONLY through the live `indices` profile (EMA 5/12/26, RSI7
band 30-55, ADX>=30, stop 1.5xATR forced past the regime override, R:R 2.0, HTF
1h-aligned, MACD-3 exit), then bucket every long by (a) ADX-at-entry and (b)
leg-extension = ATRs price had already risen over the prior 12 candles. Report
n / WR / PF / avg-P&L per bucket. If the 40-45 / 45+ ADX buckets and the >3xATR
leg bucket are net-positive, #207 was the tail of a good setup, not a bad entry.

Yahoo ^DJI only (5m/59d live timeframe + 1h/700d for sample power) — zero IG API
cost. Caveat: Yahoo cash != IG DFB (frictionless mid), and the regime tradeable/
confidence gate still differs slightly from live; this is a RELATIVE band read.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import logging

import numpy as np

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    TICKER_MAP,
    Backtester,
)

logging.basicConfig(level=logging.ERROR)

# Force the regime stop override to the live profile stop (the engine swaps in
# regime_params.stop_atr_multiplier; live `indices` uses its own 1.5x). Same
# device the stop-width / min-hold sweeps use.
_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": 1.5}


def _patched_get_rp(regime):
    p = _orig_get_rp(regime)
    if _FORCED_STOP["v"] is not None:
        return dataclasses.replace(p, stop_atr_multiplier=_FORCED_STOP["v"])
    return p


bt_module.get_regime_params = _patched_get_rp

_CACHE: dict = {}
_orig_fetch = Backtester.fetch_data


def _cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _CACHE:
        _CACHE[key] = _orig_fetch(self, market, days, interval)
    df = _CACHE[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = _cached_fetch

MARKET = "Wall Street"
TICKER_MAP[MARKET] = "^DJI"
MIN_STOP_DISTANCE_MAP[MARKET] = 8.0   # modest floor so ATR*1.5 drives the stop
MIN_CONFIDENCE_MAP[MARKET] = 0.55
REWARD_RISK_MAP[MARKET] = 2.0

INDICES = {            # live `indices` profile
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 30, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
    "long_only": True,
}

LEG_LOOKBACK = 12      # candles to measure prior leg extension (NASDAQ leg-filter lb)
ADX_BANDS = [(30, 35), (35, 40), (40, 45), (45, 999)]
LEG_BANDS = [(-999, 1.5), (1.5, 3.0), (3.0, 999)]


def pf(trades):
    gp = sum(t.pnl_percent for t in trades if t.pnl_percent > 0)
    gl = abs(sum(t.pnl_percent for t in trades if t.pnl_percent < 0))
    return gp / gl if gl > 0 else float("inf") if gp > 0 else 0.0


def line(tag, trades):
    if not trades:
        print(f"  {tag:<14}    n=  0")
        return
    n = len(trades)
    wr = sum(1 for t in trades if t.pnl_percent > 0) / n
    tot = sum(t.pnl_percent for t in trades)
    avg = tot / n
    print(f"  {tag:<14} n={n:>3}  WR={wr:>5.1%}  PF={pf(trades):>5.2f}  "
          f"sumP&L={tot:>+6.2f}%  avg={avg:>+6.3f}%")


def run_tf(days, interval):
    params = DEFAULT_PARAMS.copy()
    params.update(INDICES)
    bt = Backtester(params=params)
    res = bt.run(MARKET, days=days, interval=interval, require_htf_alignment=True)

    # Recompute the indicator df (deterministic, cached fetch) to read ADX/ATR and
    # the prior-leg extension at each trade's entry candle.
    df = bt.add_indicators(bt.fetch_data(MARKET, days, interval))
    df = df.reset_index(drop=True)
    close = df["close"].values
    atr = df["atr"].values
    leg_up = np.full(len(df), np.nan)
    for i in range(LEG_LOOKBACK, len(df)):
        if atr[i] and not np.isnan(atr[i]) and atr[i] > 0:
            leg_up[i] = (close[i] - close[i - LEG_LOOKBACK]) / atr[i]
    by_date = {d: k for k, d in enumerate(df["date"])}

    longs = [t for t in res.trades if t.direction == "BUY"]
    for t in longs:
        k = by_date.get(t.entry_time)
        t._adx = float(df["adx"].iloc[k]) if k is not None else float("nan")
        t._leg = float(leg_up[k]) if k is not None else float("nan")

    print("=" * 78)
    print(f"Wall Street (^DJI) LONG-only via `indices` profile — {interval}/{days}d, "
          f"HTF-aligned, stop 1.5xATR")
    print("=" * 78)
    line("ALL LONGS", longs)

    print("  -- by ADX-at-entry --")
    for lo, hi in ADX_BANDS:
        band = [t for t in longs if not np.isnan(t._adx) and lo <= t._adx < hi]
        tag = f"ADX {lo}-{hi if hi < 999 else '+'}"
        line(tag, band)

    print(f"  -- by prior leg-extension (ATRs risen over last {LEG_LOOKBACK} candles) --")
    for lo, hi in LEG_BANDS:
        band = [t for t in longs if not np.isnan(t._leg) and lo <= t._leg < hi]
        tag = (f"leg <{hi}x" if lo < 0 else f"leg {lo}-{hi}x" if hi < 999 else f"leg >{lo}x")
        line(tag, band)

    # Where does #207 sit? ADX 44.5, leg ~4xATR -> ADX 40-45 band + >3x leg band.
    hi_adx = [t for t in longs if not np.isnan(t._adx) and t._adx >= 40]
    print("  -- #207 cohort: ADX>=40 longs (its band) --")
    line("ADX>=40", hi_adx)
    return longs


def main():
    print("Does the high-ADX / extended-leg LONG band (where #207 sits) lose money?\n")
    primary = run_tf(59, "5m")     # live timeframe
    print()
    deep = run_tf(700, "1h")       # more trades for statistical power
    print("\n" + "=" * 78)
    print("Read: if ADX 40-45 / 45+ and the >3x leg bucket are net-positive (PF>1, "
          "+sumP&L),\n#207 was the unlucky tail of a +EV setup → do NOT add a "
          "ceiling/leg-filter to Wall St.\nIf those buckets bleed → a leg-filter or "
          "ADX-ceiling is worth enforcing. 5m is the live read;\n1h adds sample but "
          "ADX dynamics differ by timeframe.")


if __name__ == "__main__":
    main()
