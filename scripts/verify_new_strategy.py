#!/usr/bin/env python3
"""
Verify the new dual-strategy configuration performs as expected.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed")
    sys.exit(1)

from src.indicators import calculate_ema, calculate_rsi, calculate_adx, calculate_macd, calculate_atr

logging.basicConfig(level=logging.WARNING)

TICKER_MAP = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "Dollar Index": "DX-Y.NYB",
}

# The two strategy profiles
STRATEGIES = {
    "indices": {  # S&P 500, NASDAQ 100
        "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
        "rsi_period": 7, "rsi_buy_max": 65, "rsi_sell_min": 35,
        "adx_threshold": 20, "stop_mult": 1.5, "rr": 2.0,
        "min_conf": 0.4, "macd_exit": True, "require_htf": True,
    },
    "default": {  # Gold, EUR/USD, Dollar Index, Crude Oil
        "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
        "rsi_period": 7, "rsi_buy_max": 60, "rsi_sell_min": 40,
        "adx_threshold": 25, "stop_mult": 1.5, "rr": 4.0,
        "min_conf": 0.4, "macd_exit": False, "require_htf": False,
    },
}

MARKET_STRATEGY = {
    "S&P 500": "indices",
    "NASDAQ 100": "indices",
    "Gold": "default",
    "Crude Oil": "default",
    "EUR/USD": "default",
    "Dollar Index": "default",
}

POINT_VALUE = {
    "S&P 500": 1.0, "NASDAQ 100": 1.0, "Gold": 1.0,
    "Crude Oil": 1.0, "EUR/USD": 10000.0, "Dollar Index": 1.0,
}

MIN_STOP = {
    "S&P 500": 30.0, "NASDAQ 100": 100.0, "Gold": 25.0,
    "Crude Oil": 0.35, "EUR/USD": 0.005, "Dollar Index": 0.50,
}


def fetch_data(market, days=30, interval="5m"):
    ticker = TICKER_MAP.get(market)
    if not ticker:
        return None
    try:
        end = datetime.now()
        start = end - timedelta(days=min(days, 60))
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        df = df.reset_index()
        for col in df.columns:
            if col.lower() in ("datetime", "date", "index"):
                df = df.rename(columns={col: "date"})
                break
        return df
    except:
        return None


def run_backtest(market, df, htf_df, account_size=10000, risk_pct=0.01):
    strategy = STRATEGIES[MARKET_STRATEGY[market]]

    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], strategy["ema_fast"])
    df["ema_medium"] = calculate_ema(df["close"], strategy["ema_medium"])
    df["ema_slow"] = calculate_ema(df["close"], strategy["ema_slow"])
    df["rsi"] = calculate_rsi(df["close"], strategy["rsi_period"])
    df["adx"] = calculate_adx(df["high"], df["low"], df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)

    if htf_df is not None:
        htf_df = htf_df.copy()
        htf_df["ema_9"] = calculate_ema(htf_df["close"], 9)
        htf_df["ema_21"] = calculate_ema(htf_df["close"], 21)

    trades = []
    position = None
    equity = account_size

    for i in range(strategy["ema_slow"] + 5, len(df)):
        row = df.iloc[i]
        close = row["close"]
        atr = row["atr"]
        current_time = row["date"]

        # Exit check
        if position:
            exit_price = None

            if position["dir"] == "BUY":
                if close <= position["stop"]:
                    exit_price = position["stop"]
                elif close >= position["limit"]:
                    exit_price = position["limit"]
            else:
                if close >= position["stop"]:
                    exit_price = position["stop"]
                elif close <= position["limit"]:
                    exit_price = position["limit"]

            # MACD exit only if strategy uses it
            if strategy["macd_exit"] and not exit_price and i >= 3:
                last3 = [df.iloc[i-j]["macd_hist"] for j in range(3)]
                if position["dir"] == "BUY" and all(h < 0 for h in last3 if not pd.isna(h)):
                    exit_price = close
                elif position["dir"] == "SELL" and all(h > 0 for h in last3 if not pd.isna(h)):
                    exit_price = close

            if exit_price:
                pv = POINT_VALUE.get(market, 1.0)
                if position["dir"] == "BUY":
                    pts = exit_price - position["entry"]
                else:
                    pts = position["entry"] - exit_price
                pnl = pts * position["size"] * pv
                equity += pnl
                trades.append({"pnl": pnl, "dir": position["dir"]})
                position = None
            continue

        # Entry check
        ema_f, ema_m, ema_s = row["ema_fast"], row["ema_medium"], row["ema_slow"]
        rsi, adx = row["rsi"], row["adx"]

        if pd.isna(ema_s) or pd.isna(adx) or pd.isna(rsi) or pd.isna(atr):
            continue
        if adx < strategy["adx_threshold"]:
            continue

        # HTF trend
        htf = "NEUTRAL"
        if htf_df is not None:
            mask = htf_df["date"] <= current_time
            if mask.any():
                h = htf_df[mask].iloc[-1]
                if not pd.isna(h.get("ema_9")) and not pd.isna(h.get("ema_21")):
                    if h["ema_9"] > h["ema_21"] and h["close"] > h["ema_21"]:
                        htf = "BULLISH"
                    elif h["ema_9"] < h["ema_21"] and h["close"] < h["ema_21"]:
                        htf = "BEARISH"

        # HTF filter
        if strategy["require_htf"] and htf == "NEUTRAL":
            continue

        direction = None
        conf = 0

        # Buy signal
        if ema_f > ema_m > ema_s and close > ema_s and 30 < rsi < strategy["rsi_buy_max"]:
            if htf != "BEARISH":
                conf = min((adx - strategy["adx_threshold"]) / 50, 0.3) + (0.4 if htf == "BULLISH" else 0.2)
                direction = "BUY"

        # Sell signal
        elif ema_f < ema_m < ema_s and close < ema_s and strategy["rsi_sell_min"] < rsi < 70:
            if htf != "BULLISH":
                conf = min((adx - strategy["adx_threshold"]) / 50, 0.3) + (0.4 if htf == "BEARISH" else 0.2)
                direction = "SELL"

        if not direction or conf < strategy["min_conf"]:
            continue

        # Position sizing
        stop_dist = max(atr * strategy["stop_mult"], MIN_STOP.get(market, 0))
        limit_dist = stop_dist * strategy["rr"]
        pv = POINT_VALUE.get(market, 1.0)
        size = min((equity * risk_pct) / (stop_dist * pv), 5)

        if direction == "BUY":
            stop, limit = close - stop_dist, close + limit_dist
        else:
            stop, limit = close + stop_dist, close - limit_dist

        position = {"dir": direction, "entry": close, "stop": stop, "limit": limit, "size": size}

    # Close open position
    if position:
        pv = POINT_VALUE.get(market, 1.0)
        exit_price = df.iloc[-1]["close"]
        if position["dir"] == "BUY":
            pts = exit_price - position["entry"]
        else:
            pts = position["entry"] - exit_price
        pnl = pts * position["size"] * pv
        trades.append({"pnl": pnl, "dir": position["dir"]})

    return trades


def main():
    print("=" * 70)
    print("VERIFICATION: New Dual-Strategy Configuration")
    print("=" * 70)
    print("\nStrategy Assignments:")
    print("  INDICES (Momentum):   S&P 500, NASDAQ 100")
    print("    - Fast EMAs (5/12/26), ADX 20, R:R 2.0, MACD exit ON")
    print("  DEFAULT (Big Winners): Gold, EUR/USD, Dollar Index, Crude Oil")
    print("    - Standard EMAs (9/21/50), ADX 25, R:R 4.0, MACD exit OFF")

    markets = ["S&P 500", "NASDAQ 100", "Gold", "Crude Oil", "EUR/USD", "Dollar Index"]
    account_size = 10000
    days = 30

    print(f"\nFetching {days} days of data...")

    all_trades = []
    results = {}

    for market in markets:
        df = fetch_data(market, days, "5m")
        htf = fetch_data(market, days, "1h")

        if df is None:
            print(f"  {market}: No data")
            continue

        trades = run_backtest(market, df, htf, account_size)
        pnl = sum(t["pnl"] for t in trades)
        wins = len([t for t in trades if t["pnl"] > 0])
        wr = wins / len(trades) * 100 if trades else 0

        results[market] = {"pnl": pnl, "trades": len(trades), "wins": wins, "wr": wr}
        all_trades.extend(trades)

        strategy_name = MARKET_STRATEGY[market].upper()
        print(f"  {market:<15} [{strategy_name:<8}]: {len(trades):>3} trades, {wr:>4.0f}% win, £{pnl:>+8,.0f}")

    # Summary
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_trades = len(all_trades)
    total_wins = len([t for t in all_trades if t["pnl"] > 0])
    total_wr = total_wins / total_trades * 100 if total_trades else 0

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # By strategy type
    indices_pnl = sum(results.get(m, {}).get("pnl", 0) for m in ["S&P 500", "NASDAQ 100"])
    indices_trades = sum(results.get(m, {}).get("trades", 0) for m in ["S&P 500", "NASDAQ 100"])

    default_pnl = sum(results.get(m, {}).get("pnl", 0) for m in ["Gold", "Crude Oil", "EUR/USD", "Dollar Index"])
    default_trades = sum(results.get(m, {}).get("trades", 0) for m in ["Gold", "Crude Oil", "EUR/USD", "Dollar Index"])

    print(f"\nBy Strategy:")
    print(f"  INDICES (Momentum):    {indices_trades:>3} trades, £{indices_pnl:>+8,.0f}")
    print(f"  DEFAULT (Big Winners): {default_trades:>3} trades, £{default_pnl:>+8,.0f}")

    print(f"\nTotal Performance:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {total_wr:.0f}%")
    print(f"  Total P&L: £{total_pnl:>+,.0f}")
    print(f"  Monthly Return: {total_pnl/account_size*100:+.1f}%")

    # Expected vs actual
    expected_pnl = 2462 + 505  # Big Winners + Momentum from earlier tests
    print(f"\nExpected P&L (from optimization): ~£{expected_pnl:,}")
    print(f"Actual P&L: £{total_pnl:+,.0f}")

    if total_pnl > 0:
        print("\n✓ Strategy is profitable!")
    else:
        print("\n✗ Strategy needs review")


if __name__ == "__main__":
    main()
