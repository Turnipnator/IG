#!/usr/bin/env python3
"""
Gold ranging-exit A/B (2026-05-31).

Live journal shows 6 of 36 Gold trades exited on "Market turned ranging
(ADX dropped)" for a combined -£69.60 — biggest single loss bucket on
the strategy. Live behaviour is hardcoded: drop=10 (35→25), consecutive=3.

Test variants on Gold-specific config (EMA 3/8/21, RSI 85/15, ADX 35,
stop 1.5x ATR, R:R 3.0, 0.3% pullback):
  A. drop=10, consec=3              — matches live
  B. drop=0                         — disabled entirely
  C. drop=10, consec=3, declining   — strictest (must also trend down)
  D. drop=5, consec=3               — gentler (exit at 30, not 25)

Yahoo data only — zero IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    Backtester,
)
from config import MARKETS, STRATEGY_PROFILES

logging.basicConfig(level=logging.ERROR)

GOLD_TICKER = "GC=F"
MARKET = "Gold"

# Cache Yahoo fetches across configs
_orig_fetch = Backtester.fetch_data
_cache: dict = {}


def cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _cache:
        _cache[key] = _orig_fetch(self, market, days, interval)
    df = _cache[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = cached_fetch


def params_from_profile(p) -> dict:
    base = DEFAULT_PARAMS.copy()
    base.update({
        "ema_fast": p.ema_fast,
        "ema_medium": p.ema_medium,
        "ema_slow": p.ema_slow,
        "rsi_period": p.rsi_period,
        "rsi_overbought": p.rsi_overbought,
        "rsi_oversold": p.rsi_oversold,
        "rsi_buy_max": p.rsi_buy_max,
        "rsi_sell_min": p.rsi_sell_min,
        "adx_threshold": p.adx_threshold,
        "atr_period": 14,
        "stop_atr_multiplier": p.stop_atr_mult,
        "reward_risk_ratio": p.reward_risk,
    })
    return base


VARIANTS = [
    ("A. live (drop=10, consec=3)",          {"ranging_exit_drop": 10, "ranging_exit_consecutive": 3, "ranging_exit_require_declining": False}),
    ("B. disabled (drop=0)",                 {"ranging_exit_drop": 0,  "ranging_exit_consecutive": 0, "ranging_exit_require_declining": False}),
    ("C. strict (consec=3 + declining)",     {"ranging_exit_drop": 10, "ranging_exit_consecutive": 3, "ranging_exit_require_declining": True}),
    ("D. gentler (drop=5, consec=3)",        {"ranging_exit_drop": 5,  "ranging_exit_consecutive": 3, "ranging_exit_require_declining": False}),
]


def fmt(r):
    if not r or r.total_trades == 0:
        return "0 trades"
    return (f"{r.total_trades:>3}t {r.win_rate:>5.0%} "
            f"{r.total_pnl:>+7.2f}% PF{r.profit_factor:>5.2f} "
            f"DD{r.max_drawdown:>4.0%}")


def count_ranging_exits(r):
    if not r or not r.trades:
        return 0
    return sum(1 for t in r.trades if t.exit_reason == "Market turned ranging")


def ranging_exit_pnl(r):
    if not r or not r.trades:
        return 0.0
    return sum(t.pnl_percent for t in r.trades if t.exit_reason == "Market turned ranging")


def main():
    gold_mc = next(m for m in MARKETS if m.name == MARKET)
    gold_profile = STRATEGY_PROFILES[gold_mc.strategy]
    base_params = params_from_profile(gold_profile)
    bt_module.TICKER_MAP[MARKET] = GOLD_TICKER
    MIN_STOP_DISTANCE_MAP[MARKET] = 0.0
    MIN_CONFIDENCE_MAP[MARKET] = gold_mc.min_confidence
    REWARD_RISK_MAP[MARKET] = gold_profile.reward_risk

    print("=" * 110)
    print(f"Gold ranging-exit A/B — current live profile")
    print(f"EMA {gold_profile.ema_fast}/{gold_profile.ema_medium}/{gold_profile.ema_slow}, "
          f"RSI {gold_profile.rsi_oversold}/{gold_profile.rsi_overbought}, ADX{gold_profile.adx_threshold}, "
          f"stop {gold_profile.stop_atr_mult}x ATR, R:R {gold_profile.reward_risk}, "
          f"min_conf {gold_mc.min_confidence}")
    print("=" * 110)

    test_cells = [
        ("5m / 1h HTF",  "5m",  "1h", 55),
        ("1h / 1d HTF",  "1h",  "1d", 180),
        ("1h / 1d HTF",  "1h",  "1d", 365),
    ]

    for label, primary, htf, days in test_cells:
        print(f"\n--- {label}, {days}d ---")
        print(f"  {'variant':<38}  {'result':>34}    {'ranging exits':>22}")
        print("  " + "-" * 105)
        for variant_label, overrides in VARIANTS:
            p = base_params.copy()
            p.update(overrides)
            bt = Backtester(params=p)
            r = bt.run(MARKET, days=days, interval=primary,
                       htf_interval=htf, require_htf_alignment=True)
            re_count = count_ranging_exits(r)
            re_pnl = ranging_exit_pnl(r)
            re_str = f"{re_count}× ({re_pnl:+.2f}%)" if re_count else "none"
            print(f"  {variant_label:<38}  {fmt(r):>34}    {re_str:>22}")

    print("\n" + "=" * 110)
    print("READING THE TABLE")
    print("=" * 110)
    print("""
  - 'A. live' is the current production behaviour — baseline to beat.
  - 'B. disabled' removes the ranging exit entirely. Trades only exit via
    SL/TP/HTF-reversal.
  - 'C. strict' adds a require-declining check on top of 3-consecutive.
  - 'D. gentler' raises the exit threshold (exits at ADX 30 instead of 25).
  - 'ranging exits' shows how many trades hit this exit and total %P&L
    they contributed in each variant.

  If B beats A by a meaningful margin on the 5m row, the live exit is
  doing net damage and worth removing on Gold. If the rows are equivalent,
  the exit is neutral and not the cause of Gold's underperformance.
""")


if __name__ == "__main__":
    main()
