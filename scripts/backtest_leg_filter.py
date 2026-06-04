#!/usr/bin/env python3
"""
Leg-size (exhaustion) filter sweep — does blocking "chasing" entries help?

Motivation (2026-06-04): the IG book's biggest loss bucket is "Stop/limit hit"
(-£98.94 over the rolling 100). Today's three index losers were all bearish
CONTINUATION shorts entered at very high ADX (37-47) with mid-range RSI — i.e.
shorting into the 5th-6th candle of an already-extended down-leg, then getting
whipsawed by the bounce. The Oanda_Gold bot fixed the same failure mode with a
leg-size filter (block entries when price already ran > 2.0× ATR in the trade
direction over the last 6 H1 candles; validated +$1,917 / dodged 4 of 4 big
losers on Gold H1).

This sweep ports that filter (now gated in src/backtest.py:
leg_filter_lookback / leg_filter_threshold) and tests it on the IG index book.

KEY RECALIBRATION: Oanda tuned 6 candles on H1 (= 6 hours). On 5m candles, 6
candles is only 30 min, so we sweep the *5m-equivalent* lookbacks 12/24/36
(=1h/2h/3h) and test Germany 40 on its native 1h with Oanda-like 6/10.

Data: Yahoo Finance only — ZERO IG API cost. Each market+interval is downloaded
once and cached, so the whole sweep is ~6 Yahoo fetches total.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import numpy as np
import pandas as pd

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    TICKER_MAP,
    Backtester,
)
from src.indicators import calculate_ema

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Fetch cache: download each (market, days, interval) once, reuse everywhere.
# Keeps the whole sweep to ~6 Yahoo hits instead of ~hundreds. (Yahoo is free,
# but be polite + fast.)
# ---------------------------------------------------------------------------
_FETCH_CACHE: dict = {}
_orig_fetch = Backtester.fetch_data


def _cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _FETCH_CACHE:
        _FETCH_CACHE[key] = _orig_fetch(self, market, days, interval)
    df = _FETCH_CACHE[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = _cached_fetch

# ---------------------------------------------------------------------------
# Markets under test — the whipsaw victims. Live strategy mirrored via the
# per-market override maps (R:R, min-stop, min-confidence) + stop_atr in params.
# ticker / min_stop are in the YAHOO price scale (not IG points).
# ---------------------------------------------------------------------------
MARKETS = {
    # name            ticker      min_stop  conf   rr   stop  interval days
    "NASDAQ 100":  dict(ticker="^NDX",    min_stop=100.0, conf=0.50, rr=1.5, stop=1.0, interval="5m", days=59),
    "S&P 500":     dict(ticker="^GSPC",   min_stop=30.0,  conf=0.50, rr=2.0, stop=1.0, interval="5m", days=59),
    "FTSE 100":    dict(ticker="^FTSE",   min_stop=40.0,  conf=0.50, rr=2.0, stop=1.0, interval="5m", days=59),
    "Wall Street": dict(ticker="^DJI",    min_stop=200.0, conf=0.50, rr=2.0, stop=1.5, interval="5m", days=59),
    "Germany 40":  dict(ticker="^GDAXI",  min_stop=120.0, conf=0.50, rr=2.0, stop=1.5, interval="1h", days=365),
}

# Base indices profile (ADX 30 like the live "indices*" profiles).
INDICES = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 30, "atr_period": 14,
    "reward_risk_ratio": 2.0,
}


def patch_market(name):
    cfg = MARKETS[name]
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[name] = cfg["conf"]
    REWARD_RISK_MAP[name] = cfg["rr"]


def _lookbacks(interval):
    # 1h: Oanda-like short legs. 5m: 6/12/24 candles = 30m/1h/2h.
    return [6, 10] if interval == "1h" else [6, 12, 24]


def _base_params(name, leg_lb=0, leg_thr=0.0):
    cfg = MARKETS[name]
    params = DEFAULT_PARAMS.copy()
    params.update(INDICES)
    params["stop_atr_multiplier"] = cfg["stop"]
    params["leg_filter_lookback"] = leg_lb
    params["leg_filter_threshold"] = leg_thr
    return params


def collect_leg_atr(name, leg_lb):
    """First pass: leg_atr at every in-direction raw entry signal (the entries
    the filter could block). Used to calibrate thresholds to percentiles so the
    knob is 'block the most-extended N%' rather than an unscaled constant."""
    cfg = MARKETS[name]
    params = _base_params(name, leg_lb=leg_lb, leg_thr=1e9)
    bt = Backtester(params=params)
    df = bt.add_indicators(bt.fetch_data(name, cfg["days"], cfg["interval"]))
    htf = bt.fetch_data(name, cfg["days"], "1h")
    if htf is not None and not htf.empty:
        htf["ema_9"] = calculate_ema(htf["close"], 9)
        htf["ema_21"] = calculate_ema(htf["close"], 21)
    vals = []
    for i in range(params["ema_slow"], len(df)):
        row = df.iloc[i]
        trend = bt.calculate_htf_trend(name, row["date"], htf)
        d, _, _ = bt.check_entry_signal(row, trend, require_htf_alignment=True)
        la = row.get("leg_atr")
        if d and la is not None and not pd.isna(la):
            short = bool(row.get("leg_is_short"))
            if (short and d == "SELL") or (not short and d == "BUY"):
                vals.append(float(la))
    return vals


def run_one(name, leg_lb, leg_thr):
    cfg = MARKETS[name]
    bt = Backtester(params=_base_params(name, leg_lb, leg_thr))
    return bt.run(
        name, days=cfg["days"], interval=cfg["interval"],
        require_htf_alignment=True,
    )


# Percentile of the in-direction entry distribution at which to set the leg_atr
# threshold → blocks the most-extended (100-pct)% of would-be entries.
BLOCK_PCTS = [50, 65, 80]  # block top 50% / 35% / 20%


def run_market(name):
    patch_market(name)
    cfg = MARKETS[name]

    print("=" * 108)
    print(f"{name} ({cfg['ticker']}) — {cfg['interval']} candles, {cfg['days']}d, "
          f"stop {cfg['stop']}x, R:R {cfg['rr']}, conf>={cfg['conf']}")
    print("=" * 108)
    print(f"{'Config':<30} {'Trades':>7} {'WR':>7} {'P&L%':>8} {'PF':>6} "
          f"{'ΔP&L':>8} {'ΔTrd':>6}  Note")
    print("-" * 108)

    base = run_one(name, 0, 0.0)
    print(f"{'BASELINE (no leg filter)':<30} {base.total_trades:>7} {base.win_rate:>6.1%} "
          f"{base.total_pnl:>+7.2f}% {base.profit_factor:>6.2f} {'—':>8} {'—':>6}")

    rows = []
    for lb in _lookbacks(cfg["interval"]):
        vals = collect_leg_atr(name, lb)
        if len(vals) < 5:
            print(f"  (lb={lb}: only {len(vals)} in-dir entries — skipping, too few to calibrate)")
            continue
        for pct in BLOCK_PCTS:
            thr = float(np.percentile(vals, pct))
            r = run_one(name, lb, thr)
            d_pnl = r.total_pnl - base.total_pnl
            d_trd = r.total_trades - base.total_trades
            if r.total_pnl > base.total_pnl and r.profit_factor > base.profit_factor:
                note = "↑ better P&L+PF"
            elif r.profit_factor > base.profit_factor + 0.01:
                note = "↑ PF (safer)"
            elif r.total_pnl < base.total_pnl - 0.01:
                note = "↓ worse"
            else:
                note = "≈ flat"
            hrs = lb * (60 if cfg["interval"] == "1h" else 5) / 60
            label = f"lb{lb}/{hrs:.0f}h blkTop{100-pct}% thr{thr:.1f}"
            if r.total_trades == 0:
                print(f"{label:<30} {'0':>7} {'-':>7} {'-':>8} {'-':>6} {'-':>8} {'-':>6}")
            else:
                print(f"{label:<30} {r.total_trades:>7} {r.win_rate:>6.1%} "
                      f"{r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                      f"{d_pnl:>+7.2f}% {d_trd:>+6d}  {note}")
            rows.append((label, lb, thr, r))

    leg_rows = [x for x in rows if x[3].total_trades > 0]
    best = None
    if leg_rows:
        best_pnl = max(leg_rows, key=lambda x: x[3].total_pnl)
        best_pf = max(leg_rows, key=lambda x: x[3].profit_factor)
        best = best_pnl
        print()
        print(f"  BASELINE: {base.total_trades}t, P&L {base.total_pnl:+.2f}%, PF {base.profit_factor:.2f}, WR {base.win_rate:.1%}")
        print(f"  BEST P&L: {best_pnl[0]} → P&L {best_pnl[3].total_pnl:+.2f}% "
              f"(Δ{best_pnl[3].total_pnl - base.total_pnl:+.2f}%), PF {best_pnl[3].profit_factor:.2f}")
        print(f"  BEST PF : {best_pf[0]} → PF {best_pf[3].profit_factor:.2f} "
              f"(Δ{best_pf[3].profit_factor - base.profit_factor:+.2f}), P&L {best_pf[3].total_pnl:+.2f}%")
    print()
    return base, best


def main():
    summary = []
    for name in MARKETS:
        base, best = run_market(name)
        if base:
            summary.append((name, base, best))

    print("=" * 104)
    print("SUMMARY — baseline vs best leg-filter config (by P&L)")
    print("=" * 104)
    print(f"{'Market':<14} {'Base P&L':>9} {'Base PF':>8} | {'Best cfg':<22} {'Best P&L':>9} {'Best PF':>8} {'Verdict':>10}")
    print("-" * 104)
    for name, base, best in summary:
        if not best:
            print(f"{name:<14} {base.total_pnl:>+8.2f}% {base.profit_factor:>8.2f} | {'(no trades w/ filter)':<22}")
            continue
        b = best[3]
        verdict = "HELPS" if (b.total_pnl > base.total_pnl and b.profit_factor >= base.profit_factor) else \
                  "neutral" if b.profit_factor > base.profit_factor else "no edge"
        print(f"{name:<14} {base.total_pnl:>+8.2f}% {base.profit_factor:>8.2f} | {best[0]:<22} "
              f"{b.total_pnl:>+8.2f}% {b.profit_factor:>8.2f} {verdict:>10}")


if __name__ == "__main__":
    main()
