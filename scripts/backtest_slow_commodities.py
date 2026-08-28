#!/usr/bin/env python3
"""
Positional commodity backtest: daily candles, EMA 20/50 cross + ADX 20.

Tests whether a slow-timeframe profile can capture multi-week trends on
Gold, Copper, Crude Oil, and Cotton — markets the live 15m strategy misses.

Usage:
    python scripts/backtest_slow_commodities.py
    python scripts/backtest_slow_commodities.py --years 5
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import calculate_ema, calculate_atr, calculate_adx


TICKERS = {
    "Gold":      "GC=F",
    "Copper":    "HG=F",
    "Crude Oil": "CL=F",
    "Cotton":    "CT=F",
}


@dataclass
class Trade:
    market: str
    direction: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    r_multiple: float = 0.0
    stop_distance: float = 0.0
    bars_held: int = 0


def fetch_daily(ticker: str, years: int) -> Optional[pd.DataFrame]:
    import yfinance as yf
    start = datetime.now() - timedelta(days=years * 365 + 30)
    data = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=False)
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = pd.DataFrame({
        "date":  data.index,
        "open":  data["Open"].values,
        "high":  data["High"].values,
        "low":   data["Low"].values,
        "close": data["Close"].values,
    }).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.dropna()


def add_indicators(df: pd.DataFrame, ema_fast: int, ema_slow: int, adx_period: int) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], ema_fast)
    df["ema_slow"] = calculate_ema(df["close"], ema_slow)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], period=14)
    df["adx"] = calculate_adx(df["high"], df["low"], df["close"], period=adx_period)
    return df


def simulate(
    df: pd.DataFrame,
    market: str,
    adx_min: float = 20.0,
    atr_stop_mult: float = 2.0,
    rr: float = 3.0,
) -> list[Trade]:
    """
    Daily EMA 20/50 trend-follow.

    Entry: EMA20 crosses above/below EMA50 AND ADX >= adx_min AND close on right side.
    Stop:  entry ± atr_stop_mult * ATR
    Target: rr * stop distance (e.g. 3R)
    Exit also on opposing EMA cross.
    """
    trades: list[Trade] = []
    active: Optional[Trade] = None

    df = df.dropna(subset=["ema_fast", "ema_slow", "adx", "atr"]).reset_index(drop=True)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        close = row["close"]
        high = row["high"]
        low = row["low"]

        if active is not None:
            active.bars_held += 1
            stop_price = (
                active.entry_price - active.stop_distance
                if active.direction == "BUY"
                else active.entry_price + active.stop_distance
            )
            limit_price = (
                active.entry_price + active.stop_distance * rr
                if active.direction == "BUY"
                else active.entry_price - active.stop_distance * rr
            )

            closed = False
            if active.direction == "BUY":
                if low <= stop_price:
                    active.exit_price = stop_price
                    active.exit_reason = "Stop"
                    active.r_multiple = -1.0
                    closed = True
                elif high >= limit_price:
                    active.exit_price = limit_price
                    active.exit_reason = "Target"
                    active.r_multiple = rr
                    closed = True
                elif row["ema_fast"] < row["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]:
                    active.exit_price = close
                    active.exit_reason = "Cross"
                    active.r_multiple = (close - active.entry_price) / active.stop_distance
                    closed = True
            else:
                if high >= stop_price:
                    active.exit_price = stop_price
                    active.exit_reason = "Stop"
                    active.r_multiple = -1.0
                    closed = True
                elif low <= limit_price:
                    active.exit_price = limit_price
                    active.exit_reason = "Target"
                    active.r_multiple = rr
                    closed = True
                elif row["ema_fast"] > row["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]:
                    active.exit_price = close
                    active.exit_reason = "Cross"
                    active.r_multiple = (active.entry_price - close) / active.stop_distance
                    closed = True

            if closed:
                active.exit_date = row["date"]
                trades.append(active)
                active = None

        if active is None:
            bull_cross = (
                prev["ema_fast"] <= prev["ema_slow"]
                and row["ema_fast"] > row["ema_slow"]
            )
            bear_cross = (
                prev["ema_fast"] >= prev["ema_slow"]
                and row["ema_fast"] < row["ema_slow"]
            )
            if row["adx"] >= adx_min:
                if bull_cross and close > row["ema_slow"]:
                    active = Trade(
                        market=market,
                        direction="BUY",
                        entry_date=row["date"],
                        entry_price=close,
                        stop_distance=row["atr"] * atr_stop_mult,
                    )
                elif bear_cross and close < row["ema_slow"]:
                    active = Trade(
                        market=market,
                        direction="SELL",
                        entry_date=row["date"],
                        entry_price=close,
                        stop_distance=row["atr"] * atr_stop_mult,
                    )

    return trades


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"count": 0}
    rs = np.array([t.r_multiple for t in trades])
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    total_r = rs.sum()

    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min() if len(dd) else 0.0

    returns = rs
    sharpe = (returns.mean() / returns.std() * np.sqrt(52)) if returns.std() > 0 else 0.0

    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    return {
        "count": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "total_r": total_r,
        "avg_r": rs.mean(),
        "avg_win_r": wins.mean() if len(wins) else 0.0,
        "avg_loss_r": losses.mean() if len(losses) else 0.0,
        "profit_factor": pf,
        "max_dd_r": max_dd,
        "sharpe": sharpe,
        "avg_bars": np.mean([t.bars_held for t in trades]),
    }


def print_results(market: str, stats: dict, start: datetime, end: datetime):
    years = (end - start).days / 365.25
    if stats["count"] == 0:
        print(f"  {market:<12} | NO TRADES")
        return
    print(
        f"  {market:<12} | "
        f"{stats['count']:>3} trades ({stats['count']/years:.1f}/yr)  "
        f"WR {stats['win_rate']:>5.1f}%  "
        f"Total {stats['total_r']:>+6.2f}R  "
        f"Avg {stats['avg_r']:>+5.2f}R  "
        f"PF {stats['profit_factor']:>4.2f}  "
        f"MaxDD {stats['max_dd_r']:>+6.2f}R  "
        f"Sharpe {stats['sharpe']:>5.2f}  "
        f"Hold {stats['avg_bars']:>4.1f}d"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-slow", type=int, default=50)
    parser.add_argument("--adx-min", type=float, default=20.0)
    parser.add_argument("--atr-mult", type=float, default=2.0)
    parser.add_argument("--rr", type=float, default=3.0)
    args = parser.parse_args()

    print(
        f"\nPositional Commodity Backtest — {args.years}y daily candles\n"
        f"Entry: EMA{args.ema_fast}/{args.ema_slow} cross + ADX>={args.adx_min}  "
        f"Stop: {args.atr_mult}xATR  Target: {args.rr}R  Exit also on opposing cross\n"
        + "=" * 110
    )

    all_trades: list[Trade] = []
    for market, ticker in TICKERS.items():
        df = fetch_daily(ticker, args.years)
        if df is None or len(df) < args.ema_slow + 20:
            print(f"  {market:<12} | NO DATA")
            continue
        df = add_indicators(df, args.ema_fast, args.ema_slow, adx_period=14)
        trades = simulate(
            df, market,
            adx_min=args.adx_min,
            atr_stop_mult=args.atr_mult,
            rr=args.rr,
        )
        all_trades.extend(trades)
        stats = summarize(trades)
        print_results(market, stats, df["date"].iloc[0], df["date"].iloc[-1])

    print("-" * 110)
    agg = summarize(all_trades)
    if agg["count"]:
        years = args.years
        print(
            f"  {'PORTFOLIO':<12} | "
            f"{agg['count']:>3} trades ({agg['count']/years:.1f}/yr)  "
            f"WR {agg['win_rate']:>5.1f}%  "
            f"Total {agg['total_r']:>+6.2f}R  "
            f"Avg {agg['avg_r']:>+5.2f}R  "
            f"PF {agg['profit_factor']:>4.2f}  "
            f"MaxDD {agg['max_dd_r']:>+6.2f}R  "
            f"Sharpe {agg['sharpe']:>5.2f}"
        )

    print("\nExit breakdown:")
    reasons: dict[str, int] = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r:<10} {c:>3}")
    print()


if __name__ == "__main__":
    main()
