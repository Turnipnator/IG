#!/usr/bin/env python3
"""
Natural Gas — comprehensive strategy search before enabling live.

Same approach used to evaluate Russell 2000:
  - Multiple timeframes (5m/15m/30m/1h)
  - Multiple strategy profiles (crude / gold / indices_adx35 / default)
  - Tighter and wider stops, long/short variants
  - Long entry-history window for 1h to get a meaningful sample

Goal: pick the profile that wins by P&L AND has enough trades to be
trustworthy (≥10 trades preferred). If nothing clears the bar, don't
enable — same logic that retired Russell.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)


MARKET = "Natural Gas"
TICKER = "NG=F"
MIN_STOP_PRICE = 0.05  # ~1.5% of $3.50 — gas moves big


# Strategy profiles (mirroring config.py)
INDICES_ADX35 = {
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}

CRUDE = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 80, "rsi_oversold": 20,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 30, "atr_period": 14,
    "stop_atr_multiplier": 1.0, "reward_risk_ratio": 2.0,
}

GOLD = {
    "ema_fast": 3, "ema_medium": 8, "ema_slow": 21,
    "rsi_period": 7, "rsi_overbought": 85, "rsi_oversold": 15,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 3.0,
}

DEFAULT = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 25, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}


VARIANTS = [
    ("indices_adx35",            INDICES_ADX35, {}),
    ("indices_adx35 ADX 30",     INDICES_ADX35, {"adx_threshold": 30}),
    ("indices_adx35 ADX 40",     INDICES_ADX35, {"adx_threshold": 40}),
    ("crude profile",            CRUDE, {}),
    ("crude long-only",          CRUDE, {"long_only": True}),
    ("crude wider 2.0× R:R 3",   CRUDE, {"stop_atr_multiplier": 2.0, "reward_risk_ratio": 3.0}),
    ("gold profile",             GOLD, {}),
    ("gold ADX 40",              GOLD, {"adx_threshold": 40}),
    ("default profile",          DEFAULT, {}),
    ("default ADX 35",           DEFAULT, {"adx_threshold": 35}),
    ("default wide 2.5× R:R 3",  DEFAULT, {"stop_atr_multiplier": 2.5, "reward_risk_ratio": 3.0}),
]


# (label, entry_interval, htf_interval, days)
TIMEFRAMES = [
    ("5m / 1h HTF",  "5m",  "1h", 60),
    ("15m / 1h HTF", "15m", "1h", 60),
    ("30m / 1h HTF", "30m", "1h", 60),
    ("1h / 1d HTF",  "1h",  "1d", 730),
]


def patch():
    bt_module.TICKER_MAP[MARKET] = TICKER
    bt_module.MIN_STOP_DISTANCE_MAP[MARKET] = MIN_STOP_PRICE
    MIN_CONFIDENCE_MAP[MARKET] = 0.55
    REWARD_RISK_MAP[MARKET] = 2.0


def run_one(interval, htf_interval, days, base_profile, overrides):
    params = DEFAULT_PARAMS.copy()
    params.update(base_profile)
    params.update(overrides)
    rr = overrides.get("reward_risk_ratio", base_profile.get("reward_risk_ratio", 2.0))
    REWARD_RISK_MAP[MARKET] = rr
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, interval=interval,
                  htf_interval=htf_interval, require_htf_alignment=True)


def main():
    patch()

    all_results = {}  # (tf, variant) -> result

    for tf_label, interval, htf_interval, days in TIMEFRAMES:
        print()
        print("=" * 110)
        print(f"Natural Gas — {tf_label} ({days}d, HTF aligned)")
        print("=" * 110)
        print(f"{'Variant':<32} {'Trades':>7} {'WinRate':>8} {'P&L%':>8} "
              f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>7}")
        print("-" * 110)

        for v_label, base, overrides in VARIANTS:
            try:
                r = run_one(interval, htf_interval, days, base, overrides)
                all_results[(tf_label, v_label)] = r
                if r.total_trades == 0:
                    print(f"{v_label:<32} {'0':>7} {'—':>8} {'—':>8} {'—':>6} "
                          f"{'—':>8} {'—':>9} {'—':>7}")
                else:
                    print(f"{v_label:<32} {r.total_trades:>7} {r.win_rate:>7.1%} "
                          f"{r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                          f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% {r.max_drawdown:>6.1%}")
            except Exception as e:
                print(f"{v_label:<32} ERROR: {e}")

    # Summary: top by P&L with at least 10 trades, then top with at least 5
    print()
    print("=" * 110)
    print("TOP 5 BY P&L (≥10 trades)")
    print("=" * 110)
    candidates_10 = [(tf, v, r) for (tf, v), r in all_results.items() if r.total_trades >= 10]
    candidates_10.sort(key=lambda x: x[2].total_pnl, reverse=True)
    if candidates_10:
        for tf, v, r in candidates_10[:5]:
            print(f"  {tf:<14} | {v:<28} | P&L {r.total_pnl:+6.2f}%  PF {r.profit_factor:5.2f}  "
                  f"WR {r.win_rate:5.1%}  trades {r.total_trades}")
    else:
        print("  (no variants with ≥10 trades)")

    print()
    print("TOP 5 BY P&L (≥5 trades)")
    candidates_5 = [(tf, v, r) for (tf, v), r in all_results.items() if r.total_trades >= 5]
    candidates_5.sort(key=lambda x: x[2].total_pnl, reverse=True)
    if candidates_5:
        for tf, v, r in candidates_5[:5]:
            print(f"  {tf:<14} | {v:<28} | P&L {r.total_pnl:+6.2f}%  PF {r.profit_factor:5.2f}  "
                  f"WR {r.win_rate:5.1%}  trades {r.total_trades}")
    else:
        print("  (no variants with ≥5 trades)")


if __name__ == "__main__":
    main()
