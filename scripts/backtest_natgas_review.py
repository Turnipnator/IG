#!/usr/bin/env python3
"""
Natural Gas profitability review (2026-05-21).

NatGas has ZERO live trades in the journal (added in bad376d with 1h candles,
never fired live), so the keep/drop decision rests entirely on backtest. The 1h
ADX sweep flagged it: -5.29% over 365d (every ADX negative), only +5.06% over
720d. This drills in:
  1) Period attribution — bucket the 720d trades by month to see WHEN it broke.
  2) Rescue sweep — can ANY config (stop / R:R / long-only / ADX) make the
     recent 365d profitable, or is it structurally broken now?

Yahoo data (NG=F) — no IG cost. Backtester models fixed stop/limit + MACD exit
only (no live BE / ATR-trailing); trust relative comparisons.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collections
import logging

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    Backtester,
)
from config import STRATEGY_PROFILES

logging.basicConfig(level=logging.ERROR)

MARKET = "Natural Gas"
TICKER = "NG=F"
P = STRATEGY_PROFILES["natgas"]

bt_module.TICKER_MAP[MARKET] = TICKER
MIN_STOP_DISTANCE_MAP[MARKET] = 0.0
MIN_CONFIDENCE_MAP[MARKET] = 0.55
REWARD_RISK_MAP[MARKET] = P.reward_risk

# Cache Yahoo fetches
_orig = Backtester.fetch_data
_cache: dict = {}
def cached(self, market, days=30, interval="5m"):
    k = (market, days, interval)
    if k not in _cache:
        _cache[k] = _orig(self, market, days, interval)
    df = _cache[k]
    return df.copy() if df is not None else None
Backtester.fetch_data = cached


def base_params(**over):
    b = DEFAULT_PARAMS.copy()
    b.update({
        "ema_fast": P.ema_fast, "ema_medium": P.ema_medium, "ema_slow": P.ema_slow,
        "rsi_period": P.rsi_period, "rsi_overbought": P.rsi_overbought,
        "rsi_oversold": P.rsi_oversold, "rsi_buy_max": P.rsi_buy_max,
        "rsi_sell_min": P.rsi_sell_min, "adx_threshold": P.adx_threshold,
        "atr_period": 14, "stop_atr_multiplier": P.stop_atr_mult,
        "reward_risk_ratio": P.reward_risk,
    })
    b.update(over)
    return b


def run(days, **over):
    REWARD_RISK_MAP[MARKET] = over.get("reward_risk_ratio", P.reward_risk)
    bt = Backtester(params=base_params(**over))
    return bt.run(MARKET, days=days, interval="1h", htf_interval="1d",
                  require_htf_alignment=True)


print("=" * 84)
print(f"Natural Gas ({TICKER}) review — current profile: ADX{P.adx_threshold} "
      f"stop{P.stop_atr_mult}x R:R{P.reward_risk} EMA{P.ema_fast}/{P.ema_medium}/{P.ema_slow}")
print("=" * 84)

# 1) Period attribution from a single 720d run (current config)
print("\n--- Period attribution (current config, 720d) — P&L by entry month ---")
r = run(720)
by_month = collections.defaultdict(lambda: [0, 0.0])  # month -> [trades, pnl%]
for t in r.trades:
    key = t.entry_time.strftime("%Y-%m")
    by_month[key][0] += 1
    by_month[key][1] += t.pnl_percent
cum = 0.0
print(f"  {'Month':<9} {'Trades':>7} {'P&L%':>9} {'Cumulative%':>12}")
for m in sorted(by_month):
    n, pnl = by_month[m]
    cum += pnl
    print(f"  {m:<9} {n:>7} {pnl:>+8.2f}% {cum:>+11.2f}%")
print(f"  {'TOTAL':<9} {r.total_trades:>7} {r.total_pnl:>+8.2f}%  PF {r.profit_factor:.2f}  WR {r.win_rate:.1%}")

# 2) Rescue sweep at 365d (recent): can any config make it profitable?
print("\n--- Rescue sweep @ 365d (recent) — can any config save it? ---")
print(f"  {'Config':<34} {'Trades':>7} {'WinRate':>8} {'P&L%':>9} {'PF':>6}")
print("  " + "-" * 66)
rescue = [
    ("CURRENT ADX25 2.5x R:R3.0", {}),
    ("ADX25 1.5x R:R2.0",         {"stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0}),
    ("ADX25 2.0x R:R2.0",         {"stop_atr_multiplier": 2.0, "reward_risk_ratio": 2.0}),
    ("ADX25 3.0x R:R3.0",         {"stop_atr_multiplier": 3.0, "reward_risk_ratio": 3.0}),
    ("ADX20 2.5x R:R3.0",         {"adx_threshold": 20}),
    ("ADX30 2.5x R:R3.0",         {"adx_threshold": 30}),
    ("CURRENT + long-only",       {"long_only": True}),
    ("CURRENT + short-only",      {"long_only": False, "_short": True}),
]
for label, over in rescue:
    short_only = over.pop("_short", False)
    try:
        rr = run(365, **over)
        if short_only:
            trades = [t for t in rr.trades if t.direction == "SELL"]
            n = len(trades)
            pnl = sum(t.pnl_percent for t in trades)
            wins = [t for t in trades if t.pnl_percent > 0]
            wr = len(wins) / n if n else 0
            gp = sum(t.pnl_percent for t in wins)
            gl = abs(sum(t.pnl_percent for t in trades if t.pnl_percent < 0)) or 1
            pf = gp / gl
            print(f"  {label:<34} {n:>7} {wr:>7.1%} {pnl:>+8.2f}% {pf:>6.2f}")
        elif rr.total_trades == 0:
            print(f"  {label:<34} {'0':>7} {'—':>8} {'—':>9} {'—':>6}")
        else:
            print(f"  {label:<34} {rr.total_trades:>7} {rr.win_rate:>7.1%} "
                  f"{rr.total_pnl:>+8.2f}% {rr.profit_factor:>6.2f}")
    except Exception as e:
        print(f"  {label:<34} ERROR: {e}")
