#!/usr/bin/env python3
"""Swing-proximity sweep — does skipping entries INTO the recent extreme help?

Motivation (2026-06-09): the live S&P SELL at 18:00 (ADX 57) shorted ~17% off
the swing low — i.e. straight into support, on the bounce, and lost. A mate's
read: "strange place to enter, risk of bouncing to the Fib levels" (it bounced
to the 61.8% retracement and only then resumed down). This is the OBJECTIVE,
non-Fib formalisation of "don't sell into the support at the bottom of the leg":

  SELL: reject if (close - recent_N_bar_low)  < X × ATR   (too close above support)
  BUY:  reject if (recent_N_bar_high - close) < X × ATR   (too close below resistance)

A fresh breakout THROUGH the extreme (dist < 0) is NOT blocked — that's momentum,
not chasing. Swing low/high use the prior N bars (shift(1)) so no lookahead.

Sweeps lookback N and proximity X (ATR units); X=None is the no-filter baseline.
Each (N,X) vs baseline isolates the blocked band's realised contribution.

Overlap note: this targets the SAME exhaustion problem as the leg-filter and the
ADX-55 ceiling already live on NASDAQ — the real question a positive result must
answer is "does it add anything those two don't already catch?"

Live profiles mirrored: S&P indices_selective (ADX 40, 1.5x), NASDAQ
indices_wide (ADX 30, 2.0x), R:R 2.0, conf 0.55, 5m. Yahoo ^GSPC/^NDX 59d —
ZERO IG API cost.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import logging

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

logging.basicConfig(level=logging.WARNING)

# --- regime stop-override neutralisation (live uses the profile stop) ---
_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": None}


def _patched_get_rp(regime):
    p = _orig_get_rp(regime)
    if _FORCED_STOP["v"] is not None:
        return dataclasses.replace(p, stop_atr_multiplier=_FORCED_STOP["v"])
    return p


bt_module.get_regime_params = _patched_get_rp

# --- add prior-N swing low/high columns (shift(1) = no lookahead) ---
_SWING = {"n": 12, "x": None}
_orig_add = Backtester.add_indicators


def _patched_add(self, df):
    df = _orig_add(self, df)
    n = _SWING["n"]
    df["swing_low_n"] = df["low"].shift(1).rolling(n, min_periods=n).min()
    df["swing_high_n"] = df["high"].shift(1).rolling(n, min_periods=n).max()
    return df


Backtester.add_indicators = _patched_add

# --- proximity gate wrapping the real signal check ---
_orig_check = Backtester.check_entry_signal


def _patched_check(self, row, htf_trend, require_htf_alignment=False):
    direction, conf, reason = _orig_check(self, row, htf_trend, require_htf_alignment)
    x = _SWING["x"]
    if direction is None or x is None:
        return direction, conf, reason
    atr = row.get("atr")
    close = row.get("close")
    if atr is None or pd.isna(atr) or atr <= 0:
        return direction, conf, reason
    if direction == "SELL":
        lo = row.get("swing_low_n")
        if lo is not None and not pd.isna(lo):
            dist = close - lo
            if 0 <= dist < x * atr:
                return None, 0, f"Too close above swing low ({dist:.1f} < {x}xATR)"
    elif direction == "BUY":
        hi = row.get("swing_high_n")
        if hi is not None and not pd.isna(hi):
            dist = hi - close
            if 0 <= dist < x * atr:
                return None, 0, f"Too close below swing high ({dist:.1f} < {x}xATR)"
    return direction, conf, reason


Backtester.check_entry_signal = _patched_check

# --- cached Yahoo fetch ---
_CACHE: dict = {}
_orig_fetch = Backtester.fetch_data


def _cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _CACHE:
        _CACHE[key] = _orig_fetch(self, market, days, interval)
    df = _CACHE[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = _cached_fetch

MARKETS = {
    "S&P 500":    dict(ticker="^GSPC", min_stop=30.0,  conf=0.55, adx_floor=40, stop=1.5, rr=2.0),
    "NASDAQ 100": dict(ticker="^NDX",  min_stop=100.0, conf=0.55, adx_floor=30, stop=2.0, rr=2.0),
}

BASE_INDICES = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "atr_period": 14,
}

DAYS = 59
INTERVAL = "5m"
LOOKBACKS = [12, 20, 30]
# None = baseline (no proximity filter). X in ATR units.
XS = [None, 0.5, 1.0, 1.5, 2.0]


def patch(name):
    cfg = MARKETS[name]
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[name] = cfg["conf"]
    REWARD_RISK_MAP[name] = cfg["rr"]


def run_one(name, n, x):
    cfg = MARKETS[name]
    _SWING["n"] = n
    _SWING["x"] = x
    _FORCED_STOP["v"] = cfg["stop"]
    params = DEFAULT_PARAMS.copy()
    params.update(BASE_INDICES)
    params["adx_threshold"] = cfg["adx_floor"]
    params["stop_atr_multiplier"] = cfg["stop"]
    params["reward_risk_ratio"] = cfg["rr"]
    bt = Backtester(params=params)
    return bt.run(name, days=DAYS, interval=INTERVAL, require_htf_alignment=True)


def run_market(name):
    patch(name)
    cfg = MARKETS[name]
    # one baseline (filter off) — N is irrelevant when x is None
    base = run_one(name, LOOKBACKS[0], None)
    print("=" * 96)
    print(f"{name} ({cfg['ticker']}) — {INTERVAL}, {DAYS}d. LIVE: ADX floor {cfg['adx_floor']}, "
          f"stop {cfg['stop']}x, R:R {cfg['rr']}")
    print(f"  BASELINE (no proximity filter): {base.total_trades}t, WR {base.win_rate:.1%}, "
          f"P&L {base.total_pnl:+.2f}%, PF {base.profit_factor:.2f}")
    print("=" * 96)
    print(f"{'N':>4}{'X(ATR)':>8}{'Trades':>8}{'WR':>8}{'P&L%':>9}{'PF':>7}"
          f"{'removed':>9}{'ΔP&L%':>8}  note")
    print("-" * 96)
    for n in LOOKBACKS:
        for x in XS:
            if x is None:
                continue
            r = run_one(name, n, x)
            rem = base.total_trades - r.total_trades
            dp = r.total_pnl - base.total_pnl
            note = "blocked band NET LOSER" if dp > 0.01 else (
                "blocked band net winner" if dp < -0.01 else "")
            if r.total_trades == 0:
                print(f"{n:>4}{x:>8}{0:>8}  (all entries filtered)")
                continue
            print(f"{n:>4}{x:>8}{r.total_trades:>8}{r.win_rate:>7.1%}{r.total_pnl:>+8.2f}%"
                  f"{r.profit_factor:>7.2f}{rem:>9}{dp:>+8.2f}  {note}")
        print("-" * 96)
    print()


def main():
    for name in MARKETS:
        run_market(name)
    print("Reading: ΔP&L% > 0 means removing the entries within X×ATR of the recent")
    print("N-bar low/high IMPROVED results (those 'into-the-extreme' entries were net")
    print("losers). Compare attrition: a filter that needs to cut many trades for a")
    print("small gain is overfitting. Cross-check vs the ADX-55 / leg-filter already")
    print("live on NASDAQ — if it only removes trades those two already flag, it adds nothing.")


if __name__ == "__main__":
    main()
