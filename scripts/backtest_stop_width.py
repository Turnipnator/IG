#!/usr/bin/env python3
"""FTSE 100 + NASDAQ 100 stop-width sweep — does widening the 1.0x ATR stop
stop the fast 5m whipsaw-outs without killing the winners?

Motivation (2026-06-04): the recent FTSE/NASDAQ losses were rule-sound entries
(right direction, HTF-aligned, healthy RSI) that got full-stopped fast on tight
1.0x ATR stops (FTSE indices_tight = 1.0x, NASDAQ indices_wide = 1.0x). Same
signature the forex book had before GBP/USD moved off tight 5m stops.

Sweeps stop_atr_multiplier {1.0,1.5,2.0,2.5} x R:R {1.5,2.0,2.5}, holding HTF
alignment. Prints the realised average stop (% of price) per config to PROVE
the multiplier is binding (min_stop floor set low so ATR drives the stop).

Yahoo only (^FTSE / ^NDX) — zero IG API cost. One cached fetch per market.
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

logging.basicConfig(level=logging.WARNING)

# The backtester OVERRIDES self.params["stop_atr_multiplier"] with
# regime_params.stop_atr_multiplier — so a raw stop sweep does nothing (every
# config gets the regime's stop). LIVE does NOT do this: strategy.analyze() uses
# the profile's stop_atr_mult directly. To test stop width the live way, wrap
# get_regime_params to force the stop to the swept value while leaving the rest
# of the regime behaviour (tradeable / confidence / sizing) untouched.
_orig_get_rp = bt_module.get_regime_params
_FORCED_STOP = {"v": None}


def _patched_get_rp(regime):
    p = _orig_get_rp(regime)
    if _FORCED_STOP["v"] is not None:
        return dataclasses.replace(p, stop_atr_multiplier=_FORCED_STOP["v"])
    return p


bt_module.get_regime_params = _patched_get_rp

# Cache: one Yahoo download per (market, days, interval), reused across configs.
_CACHE: dict = {}
_orig_fetch = Backtester.fetch_data


def _cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _CACHE:
        _CACHE[key] = _orig_fetch(self, market, days, interval)
    df = _CACHE[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = _cached_fetch

# min_stop kept LOW (price units) so atr*multiplier is the binding constraint —
# otherwise every stop-width config floors to the same value and the sweep is
# meaningless. Live floor is ~0.1% of price; here we go lower to isolate width.
MARKETS = {
    "FTSE 100":   dict(ticker="^FTSE", min_stop=3.0,  conf=0.55, live_stop=1.0, live_rr=2.0),
    "NASDAQ 100": dict(ticker="^NDX",  min_stop=10.0, conf=0.55, live_stop=1.0, live_rr=1.5),
}

INDICES = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 30, "atr_period": 14,
}

STOPS = [1.0, 1.5, 2.0, 2.5]
RRS = [1.5, 2.0, 2.5]
DAYS = 59


def patch(name):
    cfg = MARKETS[name]
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[name] = cfg["conf"]


def avg_stop_pct(result):
    """Mean stop distance as % of entry price across the trades (validates that
    the multiplier actually moved the stop)."""
    ds = [abs(t.entry_price - t.stop_price) / t.entry_price * 100
          for t in result.trades if t.entry_price]
    return sum(ds) / len(ds) if ds else 0.0


def run_one(name, stop_mult, rr):
    cfg = MARKETS[name]
    REWARD_RISK_MAP[name] = rr
    _FORCED_STOP["v"] = stop_mult  # force the regime override to use our stop
    params = DEFAULT_PARAMS.copy()
    params.update(INDICES)
    params["stop_atr_multiplier"] = stop_mult
    bt = Backtester(params=params)
    return bt.run(name, days=DAYS, interval="5m", require_htf_alignment=True)


def run_market(name):
    patch(name)
    cfg = MARKETS[name]
    print("=" * 92)
    print(f"{name} ({cfg['ticker']}) — 5m, {DAYS}d. "
          f"LIVE = stop {cfg['live_stop']}x / R:R {cfg['live_rr']}")
    print("=" * 92)
    print(f"{'stop':>5}{'R:R':>5}{'avgStop%':>9}{'Trades':>7}{'WR':>7}{'P&L%':>8}{'PF':>6}"
          f"{'AvgWin':>8}{'AvgLoss':>8}  note")
    print("-" * 92)
    best = None
    for stop_mult in STOPS:
        for rr in RRS:
            r = run_one(name, stop_mult, rr)
            is_live = (stop_mult == cfg["live_stop"] and rr == cfg["live_rr"])
            note = "<= CURRENT" if is_live else ""
            if r.total_trades == 0:
                print(f"{stop_mult:>5}{rr:>5}{'-':>9}{0:>7}")
                continue
            print(f"{stop_mult:>5}{rr:>5}{avg_stop_pct(r):>8.2f}%{r.total_trades:>7}"
                  f"{r.win_rate:>6.1%}{r.total_pnl:>+7.2f}%{r.profit_factor:>6.2f}"
                  f"{r.avg_win:>+7.2f}%{r.avg_loss:>+7.2f}%  {note}")
            if best is None or r.profit_factor > best[1].profit_factor:
                best = ((stop_mult, rr), r)
    if best:
        (sm, rr), r = best
        print(f"\n  BEST PF: stop {sm}x / R:R {rr} → PF {r.profit_factor:.2f}, "
              f"P&L {r.total_pnl:+.2f}%, WR {r.win_rate:.1%}, {r.total_trades}t")
    print()


def main():
    for name in MARKETS:
        run_market(name)


if __name__ == "__main__":
    main()
