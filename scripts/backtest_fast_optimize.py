#!/usr/bin/env python3
"""
Fast strategy optimization - focused on high-impact parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
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

MIN_STOP_MAP = {
    "S&P 500": 30.0, "NASDAQ 100": 100.0, "Gold": 25.0,
    "Crude Oil": 0.35, "EUR/USD": 0.005, "Dollar Index": 0.50,
}

POINT_VALUE_MAP = {
    "S&P 500": 1.0, "NASDAQ 100": 1.0, "Gold": 1.0,
    "Crude Oil": 1.0, "EUR/USD": 10000.0, "Dollar Index": 1.0,
}


@dataclass
class Trade:
    market: str
    direction: str
    entry_price: float
    exit_price: float
    stop_price: float
    limit_price: float
    size: float
    pnl: float = 0.0
    exit_reason: str = ""


def fetch_data(market: str, days: int = 60, interval: str = "5m"):
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


def run_backtest(market, df, htf_df, params, account_size=10000, risk_pct=0.01):
    """Run single backtest."""
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], params["ema_fast"])
    df["ema_medium"] = calculate_ema(df["close"], params["ema_medium"])
    df["ema_slow"] = calculate_ema(df["close"], params["ema_slow"])
    df["rsi"] = calculate_rsi(df["close"], params["rsi_period"])
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
    peak = account_size
    max_dd = 0

    for i in range(params["ema_slow"] + 5, len(df)):
        row = df.iloc[i]
        close = row["close"]
        atr = row["atr"]
        current_time = row["date"]

        # Exit check
        if position:
            exit_price = None
            exit_reason = None

            if position["dir"] == "BUY":
                if close <= position["stop"]:
                    exit_price, exit_reason = position["stop"], "Stop"
                elif close >= position["limit"]:
                    exit_price, exit_reason = position["limit"], "TP"
            else:
                if close >= position["stop"]:
                    exit_price, exit_reason = position["stop"], "Stop"
                elif close <= position["limit"]:
                    exit_price, exit_reason = position["limit"], "TP"

            # MACD exit
            if params.get("macd_exit", True) and not exit_price and i >= 3:
                hist = row["macd_hist"]
                last3 = [df.iloc[i-j]["macd_hist"] for j in range(3)]
                if position["dir"] == "BUY" and all(h < 0 for h in last3 if not pd.isna(h)):
                    exit_price, exit_reason = close, "MACD"
                elif position["dir"] == "SELL" and all(h > 0 for h in last3 if not pd.isna(h)):
                    exit_price, exit_reason = close, "MACD"

            if exit_price:
                pv = POINT_VALUE_MAP.get(market, 1.0)
                if position["dir"] == "BUY":
                    pts = exit_price - position["entry"]
                else:
                    pts = position["entry"] - exit_price
                pnl = pts * position["size"] * pv
                equity += pnl
                trades.append(Trade(market, position["dir"], position["entry"], exit_price,
                                   position["stop"], position["limit"], position["size"], pnl, exit_reason))
                if equity > peak:
                    peak = equity
                max_dd = max(max_dd, (peak - equity) / peak)
                position = None
            continue

        # Entry check
        ema_f, ema_m, ema_s = row["ema_fast"], row["ema_medium"], row["ema_slow"]
        rsi, adx = row["rsi"], row["adx"]

        if pd.isna(ema_s) or pd.isna(adx) or pd.isna(rsi) or pd.isna(atr):
            continue
        if adx < params["adx_threshold"]:
            continue

        # HTF trend
        htf = "NEUTRAL"
        if htf_df is not None:
            mask = htf_df["date"] <= current_time
            if mask.any():
                h = htf_df[mask].iloc[-1]
                if h["ema_9"] > h["ema_21"] and h["close"] > h["ema_21"]:
                    htf = "BULLISH"
                elif h["ema_9"] < h["ema_21"] and h["close"] < h["ema_21"]:
                    htf = "BEARISH"

        direction = None
        conf = 0

        # Buy signal
        if ema_f > ema_m > ema_s and close > ema_s and params["rsi_os"] < rsi < params["rsi_buy_max"]:
            if htf != "BEARISH":
                conf = min((adx - 25) / 50, 0.3) + (0.4 if htf == "BULLISH" else 0.2)
                direction = "BUY"

        # Sell signal
        elif ema_f < ema_m < ema_s and close < ema_s and params["rsi_sell_min"] < rsi < params["rsi_ob"]:
            if htf != "BULLISH":
                conf = min((adx - 25) / 50, 0.3) + (0.4 if htf == "BEARISH" else 0.2)
                direction = "SELL"

        if not direction or conf < params["min_conf"]:
            continue

        # Position sizing
        stop_dist = max(atr * params["stop_mult"], MIN_STOP_MAP.get(market, 0))
        limit_dist = stop_dist * params["rr"]
        pv = POINT_VALUE_MAP.get(market, 1.0)
        size = min((equity * risk_pct) / (stop_dist * pv), 5)

        if direction == "BUY":
            stop, limit = close - stop_dist, close + limit_dist
        else:
            stop, limit = close + stop_dist, close - limit_dist

        position = {"dir": direction, "entry": close, "stop": stop, "limit": limit, "size": size}

    # Close open position
    if position:
        pv = POINT_VALUE_MAP.get(market, 1.0)
        exit_price = df.iloc[-1]["close"]
        if position["dir"] == "BUY":
            pts = exit_price - position["entry"]
        else:
            pts = position["entry"] - exit_price
        pnl = pts * position["size"] * pv
        trades.append(Trade(market, position["dir"], position["entry"], exit_price,
                           position["stop"], position["limit"], position["size"], pnl, "EOT"))

    return trades, max_dd


def main():
    print("=" * 80)
    print("STRATEGY OPTIMIZATION - Finding Maximum Profit Configuration")
    print("=" * 80)

    markets = ["EUR/USD", "Dollar Index", "S&P 500", "NASDAQ 100", "Gold", "Crude Oil"]
    account_size = 10000
    days = 30

    print(f"\nAccount: £{account_size:,} | Period: {days} days")
    print("\nFetching data...")

    market_data = {}
    htf_data = {}
    for m in markets:
        df = fetch_data(m, days, "5m")
        htf = fetch_data(m, days, "1h")
        if df is not None:
            market_data[m] = df
            htf_data[m] = htf
            print(f"  {m}: {len(df)} candles")

    # Key parameters to optimize (reduced grid for speed)
    test_configs = [
        # Current strategy
        {"name": "Current", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.5, "macd_exit": True},

        # Lower ADX threshold (more trades)
        {"name": "ADX 20", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 20, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.5, "macd_exit": True},

        # Lower confidence (more trades)
        {"name": "Conf 0.4", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.4, "macd_exit": True},

        # Higher R:R (bigger winners)
        {"name": "RR 3.0", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 3.0, "min_conf": 0.5, "macd_exit": True},

        # Tighter stops (less risk per trade)
        {"name": "Stop 1.0x", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.0, "rr": 2.0, "min_conf": 0.5, "macd_exit": True},

        # Wider stops (more room to breathe)
        {"name": "Stop 2.0x", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.0, "rr": 2.0, "min_conf": 0.5, "macd_exit": True},

        # No MACD exit (let winners run)
        {"name": "No MACD Exit", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.5, "macd_exit": False},

        # Faster EMAs (more signals)
        {"name": "Fast EMA 5/15/40", "ema_fast": 5, "ema_medium": 15, "ema_slow": 40, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.5, "macd_exit": True},

        # Slower EMAs (higher quality signals)
        {"name": "Slow EMA 12/26/60", "ema_fast": 12, "ema_medium": 26, "ema_slow": 60, "rsi_period": 14,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.5, "macd_exit": True},

        # Aggressive (low ADX, low conf, high RR)
        {"name": "Aggressive", "ema_fast": 5, "ema_medium": 15, "ema_slow": 40, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 65, "rsi_sell_min": 35,
         "adx_threshold": 20, "stop_mult": 1.0, "rr": 2.5, "min_conf": 0.4, "macd_exit": False},

        # Conservative (high ADX, high conf, tight risk)
        {"name": "Conservative", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 14,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 55, "rsi_sell_min": 45,
         "adx_threshold": 30, "stop_mult": 2.0, "rr": 1.5, "min_conf": 0.6, "macd_exit": True},

        # Momentum (wider RSI bands, lower ADX)
        {"name": "Momentum", "ema_fast": 5, "ema_medium": 12, "ema_slow": 30, "rsi_period": 7,
         "rsi_os": 25, "rsi_ob": 75, "rsi_buy_max": 65, "rsi_sell_min": 35,
         "adx_threshold": 20, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.4, "macd_exit": True},

        # Swing (slower, higher RR)
        {"name": "Swing", "ema_fast": 12, "ema_medium": 26, "ema_slow": 50, "rsi_period": 14,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.0, "rr": 3.0, "min_conf": 0.5, "macd_exit": False},

        # Scalper (tight stops, quick exits)
        {"name": "Scalper", "ema_fast": 5, "ema_medium": 12, "ema_slow": 26, "rsi_period": 5,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 20, "stop_mult": 0.75, "rr": 1.5, "min_conf": 0.4, "macd_exit": True},

        # High RR + Low Conf (few big winners)
        {"name": "Big Winners", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 4.0, "min_conf": 0.4, "macd_exit": False},
    ]

    print(f"\nTesting {len(test_configs)} strategy configurations...\n")

    results = []

    for config in test_configs:
        all_trades = []
        total_dd = 0

        for m in market_data:
            trades, dd = run_backtest(m, market_data[m], htf_data.get(m), config, account_size)
            all_trades.extend(trades)
            total_dd = max(total_dd, dd)

        total_pnl = sum(t.pnl for t in all_trades)
        wins = len([t for t in all_trades if t.pnl > 0])
        win_rate = wins / len(all_trades) if all_trades else 0

        results.append({
            "name": config["name"],
            "config": config,
            "trades": len(all_trades),
            "wins": wins,
            "win_rate": win_rate,
            "pnl": total_pnl,
            "max_dd": total_dd,
            "all_trades": all_trades,
        })

        print(f"  {config['name']:<20} | {len(all_trades):>3} trades | {win_rate:>5.0%} win | £{total_pnl:>+8,.0f} | DD {total_dd:>5.1%}")

    # Sort by P&L
    results.sort(key=lambda x: x["pnl"], reverse=True)

    print("\n" + "=" * 80)
    print("RESULTS RANKED BY PROFIT")
    print("=" * 80)

    print(f"\n{'Rank':<5} {'Strategy':<20} {'Trades':>8} {'Win Rate':>10} {'P&L':>12} {'Max DD':>10}")
    print("-" * 75)

    for i, r in enumerate(results):
        print(f"#{i+1:<4} {r['name']:<20} {r['trades']:>8} {r['win_rate']:>9.0%} £{r['pnl']:>+11,.0f} {r['max_dd']:>9.1%}")

    # Best strategy details
    best = results[0]
    print("\n" + "=" * 80)
    print(f"BEST STRATEGY: {best['name']}")
    print("=" * 80)

    print(f"\nPerformance:")
    print(f"  Total P&L: £{best['pnl']:+,.2f}")
    print(f"  Monthly Return: {best['pnl']/account_size*100:+.1f}%")
    print(f"  Total Trades: {best['trades']}")
    print(f"  Win Rate: {best['win_rate']:.0%}")
    print(f"  Max Drawdown: {best['max_dd']:.1%}")

    if best['all_trades']:
        wins = [t.pnl for t in best['all_trades'] if t.pnl > 0]
        losses = [t.pnl for t in best['all_trades'] if t.pnl <= 0]
        print(f"  Avg Win: £{np.mean(wins):+,.2f}" if wins else "")
        print(f"  Avg Loss: £{np.mean(losses):,.2f}" if losses else "")

    print(f"\nParameters:")
    for k, v in best['config'].items():
        if k != "name":
            print(f"  {k}: {v}")

    # P&L by market
    print(f"\nP&L by Market:")
    market_pnl = {}
    for t in best['all_trades']:
        if t.market not in market_pnl:
            market_pnl[t.market] = {"pnl": 0, "trades": 0, "wins": 0}
        market_pnl[t.market]["pnl"] += t.pnl
        market_pnl[t.market]["trades"] += 1
        if t.pnl > 0:
            market_pnl[t.market]["wins"] += 1

    for m, stats in sorted(market_pnl.items(), key=lambda x: -x[1]["pnl"]):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"  {m:<15}: £{stats['pnl']:>+8,.0f} ({stats['trades']:>2} trades, {wr:>3.0f}% win)")

    # Compare to current
    current = next((r for r in results if r["name"] == "Current"), None)
    if current:
        improvement = best["pnl"] - current["pnl"]
        print(f"\n" + "=" * 80)
        print("IMPROVEMENT vs CURRENT STRATEGY")
        print("=" * 80)
        print(f"\n  Current P&L:  £{current['pnl']:>+,.0f}")
        print(f"  Optimal P&L:  £{best['pnl']:>+,.0f}")
        print(f"  Improvement:  £{improvement:>+,.0f} ({improvement/account_size*100:+.1f}% of account)")

        if improvement > 0:
            print(f"\n  Key changes from current:")
            for k in best['config']:
                if k != "name" and best['config'][k] != current['config'].get(k):
                    print(f"    {k}: {current['config'].get(k)} -> {best['config'][k]}")


if __name__ == "__main__":
    main()
