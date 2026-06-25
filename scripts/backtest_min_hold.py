#!/usr/bin/env python3
"""Minimum-hold sweep for the momentum (MACD-3) exit — does forcing a position
to live >=N closed candles before the momentum exit may fire help or hurt the
index strategies?

Motivation (2026-06-25): live FTSE trade #205 opened 14:25:01 and closed 17s
later on "MACD histogram negative for 3 candles" (−£0.14, pure spread). Cause is
a candle-close RACE: the entry MACD pre-check passed on the candle window known
at 14:25:01 (the just-closing 14:20 candle had not yet committed), then that
candle committed seconds later, completed a bearish-3 run, and the monitor loop
fired the momentum exit immediately. The entry guard is evaluated one candle too
early relative to the exit, so it cannot see a candle that closes right after
entry.

The fix is a minimum-hold: the momentum exit (and the ranging-3 exit) may not
fire until the position has been held >=1 closed candle — which is exactly the
hold the BACKTEST ENGINE ALREADY ASSUMES (a position opens at candle i and the
earliest exit check is i+1). So min_hold=1 reproduces the validated baseline;
this sweep confirms (a) 1 candle is neutral vs the status quo and (b) holding
LONGER (2-4 candles) does not improve the indices — i.e. 1 is the correct,
minimal value the fix should enforce.

Yahoo only (^FTSE/^NDX/^DJI/^GSPC) — zero IG API cost. One fetch per market.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import logging

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

# Force the regime stop override to the live profile stop (same reason as the
# stop-width sweep: the engine replaces stop_atr_multiplier with the regime's,
# but live uses the profile stop directly).
_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": None}


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

# Live index profiles (post 16fea93): FTSE indices_tight + NASDAQ indices_wide
# both run 2.0x ATR stop / R:R 2.0. Wall St + S&P shown for breadth at their
# live-ish shape. min_stop kept modest so the ATR stop drives.
MARKETS = {
    "FTSE 100":    dict(ticker="^FTSE",  min_stop=3.0,  conf=0.55, stop=2.0, rr=2.0, adx=30),
    "NASDAQ 100":  dict(ticker="^NDX",   min_stop=10.0, conf=0.55, stop=2.0, rr=2.0, adx=30),
    "Wall Street": dict(ticker="^DJI",   min_stop=8.0,  conf=0.55, stop=1.5, rr=2.0, adx=25),
    "S&P 500":     dict(ticker="^GSPC",  min_stop=4.0,  conf=0.55, stop=1.5, rr=2.0, adx=25),
}

INDICES = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "atr_period": 14,
}

HOLDS = [1, 2, 3, 4]
DAYS = 59


def run_one(name, min_hold):
    cfg = MARKETS[name]
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[name] = cfg["conf"]
    REWARD_RISK_MAP[name] = cfg["rr"]
    _FORCED_STOP["v"] = cfg["stop"]
    params = DEFAULT_PARAMS.copy()
    params.update(INDICES)
    params["adx_threshold"] = cfg["adx"]
    params["stop_atr_multiplier"] = cfg["stop"]
    bt = Backtester(params=params)
    return bt.run(name, days=DAYS, interval="5m", require_htf_alignment=True,
                  min_hold_candles=min_hold)


def macd_exits(result):
    return sum(1 for t in result.trades if t.exit_reason in ("MACD bearish", "MACD bullish"))


def run_market(name):
    cfg = MARKETS[name]
    print("=" * 84)
    print(f"{name} ({cfg['ticker']}) — 5m, {DAYS}d, stop {cfg['stop']}x / R:R {cfg['rr']} / ADX{cfg['adx']}")
    print(f"{'min_hold':>9}{'Trades':>8}{'WR':>7}{'P&L%':>8}{'PF':>6}{'MACDexits':>10}  note")
    print("-" * 84)
    base = None
    for h in HOLDS:
        r = run_one(name, h)
        if base is None:
            base = r
        note = "<= FIX (=baseline)" if h == 1 else ""
        if r.total_trades == 0:
            print(f"{h:>9}{0:>8}")
            continue
        print(f"{h:>9}{r.total_trades:>8}{r.win_rate:>6.1%}{r.total_pnl:>+7.2f}%"
              f"{r.profit_factor:>6.2f}{macd_exits(r):>10}  {note}")


def main():
    print("Minimum-hold sweep | momentum exit may not fire until >=N closed candles held")
    print("min_hold=1 == current backtest baseline == the live fix. Looking for: 1 is")
    print("neutral-to-best; longer holds do not improve the indices.\n")
    for name in MARKETS:
        run_market(name)
    print("=" * 84)
    print("Read: if min_hold=1 matches/leads on PF and P&L, the fix is safe and minimal —")
    print("it only stops live from exiting on the entry candle (a 0-hold the engine never models).")


if __name__ == "__main__":
    main()
