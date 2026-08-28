#!/usr/bin/env python3
"""
USD/JPY long-only diagnostic.

Live shows 4 trades 2W/2L, -£11.63. 3 of 4 trades were SELLs and the
recent 2 SELLs both got reversed (-£16.45, -£9.95). USD/JPY has been in
a structural uptrend on rates differentials, so we're testing whether
the SELL side is a net drag.

Same approach used to retire Russell: run the live profile across
multiple timeframes and windows, compare long+short vs long-only vs
short-only. If long-only beats long+short by a meaningful margin, flip
the live config; otherwise leave it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    DISABLE_MACD_EXIT,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    TICKER_MAP,
    Backtester,
)

logging.basicConfig(level=logging.WARNING)


MARKET = "USD/JPY"
TICKER = "JPY=X"
MIN_STOP = 0.10  # ~6 pips at 156-ish

USDJPY_PROFILE = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 80, "rsi_oversold": 20,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}


# (label, short_only, long_only)
VARIANTS = [
    ("long + short (current)", False, False),
    ("long-only",              False, True),
    ("short-only",             True,  False),
]

# (label, entry_interval, htf_interval, days)
TIMEFRAMES = [
    ("15m / 1h HTF",  "15m", "1h", 60),
    ("30m / 1h HTF",  "30m", "1h", 60),
    ("1h / 1d HTF",   "1h",  "1d", 730),
]


def patch():
    TICKER_MAP[MARKET] = TICKER
    MIN_STOP_DISTANCE_MAP[MARKET] = MIN_STOP
    MIN_CONFIDENCE_MAP[MARKET] = 0.55
    REWARD_RISK_MAP[MARKET] = 2.0
    # Live USD/JPY uses non-MACD exit (use_macd_exit=False) — match it.
    DISABLE_MACD_EXIT.add(MARKET)


def run_one(interval: str, htf_interval: str, days: int,
            short_only: bool, long_only: bool):
    params = DEFAULT_PARAMS.copy()
    params.update(USDJPY_PROFILE)
    params["long_only"] = long_only
    params["short_only"] = short_only
    # Match the live ranging-exit fix we just deployed (3-candle confirmation)
    params["ranging_exit_consecutive"] = 3
    params["ranging_exit_drop"] = 10
    bt = Backtester(params=params)
    return bt.run(MARKET, days=days, interval=interval,
                  htf_interval=htf_interval, require_htf_alignment=True)


def main():
    patch()

    # We need to add short_only support to the backtester or filter results.
    # For simplicity, we'll achieve "short-only" by negating long_only behavior
    # via post-trade filtering on direction. But the cleanest path is to add
    # short_only to backtest. Let's check first if it's already supported.
    # If not, we'll filter trades after running.

    print()
    print("USD/JPY long-only diagnostic")
    print()

    for tf_label, interval, htf_interval, days in TIMEFRAMES:
        print("=" * 100)
        print(f"USD/JPY — {tf_label} ({days}d, HTF aligned, ranging exit 3-candle)")
        print("=" * 100)
        print(f"{'Variant':<24} {'Trades':>7} {'Long':>5} {'Short':>6} "
              f"{'WinRate':>8} {'P&L%':>8} {'PF':>6} "
              f"{'AvgWin%':>8} {'AvgLoss%':>9}")
        print("-" * 100)

        for v_label, short_only, long_only in VARIANTS:
            try:
                # If short_only requested and backtester doesn't support it,
                # run with long_only=False then post-filter trades.
                if short_only and not _backtester_supports_short_only():
                    r = run_one(interval, htf_interval, days, False, False)
                    r = _filter_to_shorts(r)
                else:
                    r = run_one(interval, htf_interval, days, short_only, long_only)

                long_n  = sum(1 for t in r.trades if t.direction == "BUY")
                short_n = sum(1 for t in r.trades if t.direction == "SELL")

                if r.total_trades == 0:
                    print(f"{v_label:<24} {'0':>7} {'—':>5} {'—':>6} "
                          f"{'—':>8} {'—':>8} {'—':>6} {'—':>8} {'—':>9}")
                else:
                    print(f"{v_label:<24} {r.total_trades:>7} {long_n:>5} {short_n:>6} "
                          f"{r.win_rate:>7.1%} {r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                          f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}%")
            except Exception as e:
                print(f"{v_label:<24} ERROR: {e}")
        print()


def _backtester_supports_short_only() -> bool:
    return "short_only" in DEFAULT_PARAMS


def _filter_to_shorts(result):
    """Recompute aggregates over short trades only."""
    shorts = [t for t in result.trades if t.direction == "SELL"]
    result.trades = shorts
    result.total_trades = len(shorts)
    if not shorts:
        result.win_rate = 0.0
        result.total_pnl = 0.0
        result.profit_factor = 0.0
        result.avg_win = 0.0
        result.avg_loss = 0.0
        return result
    wins = [t.pnl_percent for t in shorts if (t.pnl_percent or 0) > 0]
    losses = [t.pnl_percent for t in shorts if (t.pnl_percent or 0) <= 0]
    result.win_rate = len(wins) / len(shorts)
    result.total_pnl = sum(t.pnl_percent or 0 for t in shorts)
    sum_wins = sum(wins) if wins else 0.0
    sum_losses = abs(sum(losses)) if losses else 0.0
    result.profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else 0.0
    result.avg_win = (sum(wins) / len(wins)) if wins else 0.0
    result.avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    return result


if __name__ == "__main__":
    main()
