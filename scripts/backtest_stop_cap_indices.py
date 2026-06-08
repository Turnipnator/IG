#!/usr/bin/env python3
"""Wall Street + Germany 40 stop-width sweep — does the min_stop*20 cap hurt?

Motivation (2026-06-08): the live stop ceiling `max_stop = min_stop * 20`
(strategy.py) truncates the ATR-based stop on every high-ATR index. Measured
live:
  Germany 40  ATR~122  intended 1.5x = ~183  ->  capped to  40  (0.33x effective)
  Wall Street ATR~ 99  intended 1.5x = ~148  ->  capped to  80  (0.80x effective)
The backtester applies NO such cap (backtest.py: stop = max(atr_stop, min_stop)),
so live has been diverging from every backtest. This sweep asks: at the FULL,
uncapped ATR stop the profile intends (1.5x+), do these markets do better than
the tight stop the cap is forcing on them (~0.5x)?

Yahoo only (^DJI 5m / ^GDAXI 1h) — zero IG API cost. One cached fetch per market.
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

# Same regime-override neutralisation as backtest_stop_width.py: the backtester
# replaces stop_atr_multiplier with the regime's; force it to the swept value so
# the sweep is not inert (live uses the profile stop directly, no override).
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

# min_stop kept LOW (price units) so atr*multiplier is the binding constraint.
MARKETS = {
    "Wall Street": dict(ticker="^DJI",   min_stop=20.0, conf=0.55, interval="5m", days=59,  live_stop=1.5, live_rr=2.0),
    "Germany 40":  dict(ticker="^GDAXI", min_stop=20.0, conf=0.55, interval="1h", days=365, live_stop=1.5, live_rr=2.0),
}

INDICES = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 30, "atr_period": 14,
}

# 0.5x approximates the tight stop the min_stop*20 cap is currently forcing;
# 1.5x is the profile intent; 2.0/2.5 test going wider still.
STOPS = [0.5, 1.0, 1.5, 2.0, 2.5]
RRS = [1.5, 2.0, 2.5]


def patch(name):
    cfg = MARKETS[name]
    TICKER_MAP[name] = cfg["ticker"]
    MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[name] = cfg["conf"]


def avg_stop_pct(result):
    ds = [abs(t.entry_price - t.stop_price) / t.entry_price * 100
          for t in result.trades if t.entry_price]
    return sum(ds) / len(ds) if ds else 0.0


def run_one(name, stop_mult, rr):
    cfg = MARKETS[name]
    REWARD_RISK_MAP[name] = rr
    _FORCED_STOP["v"] = stop_mult
    params = DEFAULT_PARAMS.copy()
    params.update(INDICES)
    params["stop_atr_multiplier"] = stop_mult
    bt = Backtester(params=params)
    return bt.run(name, days=cfg["days"], interval=cfg["interval"],
                  require_htf_alignment=True)


def run_market(name):
    patch(name)
    cfg = MARKETS[name]
    print("=" * 92)
    print(f"{name} ({cfg['ticker']}) — {cfg['interval']}, {cfg['days']}d. "
          f"LIVE profile = stop {cfg['live_stop']}x / R:R {cfg['live_rr']} "
          f"(but cap forces ~0.3-0.8x effective)")
    print("=" * 92)
    print(f"{'stop':>5}{'R:R':>5}{'avgStop%':>9}{'Trades':>7}{'WR':>7}{'P&L%':>8}{'PF':>6}"
          f"{'AvgWin':>8}{'AvgLoss':>8}  note")
    print("-" * 92)
    best = None
    for stop_mult in STOPS:
        for rr in RRS:
            r = run_one(name, stop_mult, rr)
            is_live = (stop_mult == cfg["live_stop"] and rr == cfg["live_rr"])
            note = "<= PROFILE" if is_live else ("~capped now" if stop_mult == 0.5 and rr == RRS[0] else "")
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
