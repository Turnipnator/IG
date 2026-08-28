#!/usr/bin/env python3
"""
30-day backtest comparing baseline vs RSI-extreme cooldown filter.

The filter blocks entries that are likely counter-trend bounces:
  - SELL blocked if RSI dipped below `rsi_extreme_low` in the prior N candles
  - BUY  blocked if RSI poked above `rsi_extreme_high` in the prior N candles

This catches the failure mode observed live on 2026-05-01 EUR/USD:
  RSI was 14.5 at 07:35, rebounded to 46.2 by 07:45 (crossed sell-gate
  from below), bot fired SHORT, RSI hit 84 by 07:50 → instant stop-out.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from src.backtest import DEFAULT_PARAMS, TICKER_MAP, Backtester

logging.basicConfig(level=logging.WARNING)


_BASE_RSI = {
    "rsi_extreme_lookback": 3,
    "rsi_extreme_low": 25,
    "rsi_extreme_high": 75,
}

CONFIGS = [
    ("baseline (no filter)", {"rsi_extreme_lookback": 0}),
    ("ADX +5 only", {**_BASE_RSI, "rsi_extreme_adx_boost": 5}),
    ("ADX +10 only", {**_BASE_RSI, "rsi_extreme_adx_boost": 10}),
    ("Conf +0.10 only", {**_BASE_RSI, "rsi_extreme_confidence_boost": 0.10}),
    ("ADX +5 & Conf +0.10", {
        **_BASE_RSI,
        "rsi_extreme_adx_boost": 5,
        "rsi_extreme_confidence_boost": 0.10,
    }),
]


def run_one(market: str, days: int, overrides: dict):
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(market, days=days, require_htf_alignment=False)


def main():
    days = 30
    markets = list(TICKER_MAP.keys())

    # market -> { config_label -> result }
    grid: dict[str, dict] = {m: {} for m in markets}

    for market in markets:
        for label, overrides in CONFIGS:
            try:
                result = run_one(market, days, overrides)
                grid[market][label] = result
            except Exception as e:
                print(f"  {market} / {label}: ERROR — {e}")
                grid[market][label] = None

    # Per-market table
    print("\n" + "=" * 110)
    print(f"30-day backtest — RSI extreme cooldown filter")
    print("=" * 110)
    print(f"{'Market':<14} {'Config':<32} {'Trades':>7} {'WinRate':>8} {'P&L%':>8} "
          f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>7}")
    print("-" * 110)

    for market in markets:
        baseline = grid[market].get("baseline (no filter)")
        for label, _ in CONFIGS:
            r = grid[market].get(label)
            if r is None:
                print(f"{market:<14} {label:<32} {'—':>7} {'—':>8} {'—':>8} {'—':>6} "
                      f"{'—':>8} {'—':>9} {'—':>7}")
                continue
            print(f"{market:<14} {label:<32} {r.total_trades:>7} {r.win_rate:>7.1%} "
                  f"{r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                  f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% {r.max_drawdown:>6.1%}")
        print()

    # Aggregate by config
    print("=" * 110)
    print("AGGREGATE (all 6 markets summed)")
    print("=" * 110)
    print(f"{'Config':<32} {'Trades':>7} {'Wins':>6} {'WinRate':>8} {'Total P&L%':>11} {'Avg PF':>8}")
    print("-" * 110)
    for label, _ in CONFIGS:
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0
        pfs = []
        for m in markets:
            r = grid[m].get(label)
            if r is None or r.total_trades == 0:
                continue
            total_trades += r.total_trades
            total_wins += r.winning_trades
            total_pnl += r.total_pnl
            if r.profit_factor > 0:
                pfs.append(r.profit_factor)
        wr = total_wins / total_trades if total_trades else 0
        avg_pf = sum(pfs) / len(pfs) if pfs else 0
        print(f"{label:<32} {total_trades:>7} {total_wins:>6} {wr:>7.1%} "
              f"{total_pnl:>+10.2f}% {avg_pf:>8.2f}")

    # Diff vs baseline
    print()
    print("DIFF vs baseline (all 6 markets summed)")
    print("-" * 110)
    base = sum(
        grid[m]["baseline (no filter)"].total_pnl
        for m in markets
        if grid[m].get("baseline (no filter)") is not None
    )
    base_trades = sum(
        grid[m]["baseline (no filter)"].total_trades
        for m in markets
        if grid[m].get("baseline (no filter)") is not None
    )
    for label, _ in CONFIGS:
        if label == "baseline (no filter)":
            continue
        total_pnl = sum(
            grid[m][label].total_pnl
            for m in markets
            if grid[m].get(label) is not None
        )
        total_trades = sum(
            grid[m][label].total_trades
            for m in markets
            if grid[m].get(label) is not None
        )
        print(f"{label:<32}  ΔP&L={total_pnl - base:+.2f}%   "
              f"Δtrades={total_trades - base_trades:+d}   "
              f"({base_trades} → {total_trades})")


if __name__ == "__main__":
    main()
