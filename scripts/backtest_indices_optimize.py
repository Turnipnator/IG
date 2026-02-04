#!/usr/bin/env python3
"""
Optimize strategy specifically for S&P 500 and NASDAQ 100.
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

TICKER_MAP = {"S&P 500": "^GSPC", "NASDAQ 100": "^NDX"}
MIN_STOP_MAP = {"S&P 500": 30.0, "NASDAQ 100": 100.0}
POINT_VALUE_MAP = {"S&P 500": 1.0, "NASDAQ 100": 1.0}


def fetch_data(market, days=60, interval="5m"):
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
                last3 = [df.iloc[i-j]["macd_hist"] for j in range(3)]
                if position["dir"] == "BUY" and all(h < 0 for h in last3 if not pd.isna(h)):
                    exit_price, exit_reason = close, "MACD"
                elif position["dir"] == "SELL" and all(h > 0 for h in last3 if not pd.isna(h)):
                    exit_price, exit_reason = close, "MACD"

            # Trailing stop (if enabled)
            if params.get("trailing_stop", False) and not exit_price:
                trail_dist = atr * params.get("trail_mult", 2.0)
                if position["dir"] == "BUY":
                    new_stop = close - trail_dist
                    if new_stop > position["stop"]:
                        position["stop"] = new_stop
                else:
                    new_stop = close + trail_dist
                    if new_stop < position["stop"]:
                        position["stop"] = new_stop

            if exit_price:
                pv = POINT_VALUE_MAP.get(market, 1.0)
                if position["dir"] == "BUY":
                    pts = exit_price - position["entry"]
                else:
                    pts = position["entry"] - exit_price
                pnl = pts * position["size"] * pv
                equity += pnl
                trades.append({"pnl": pnl, "reason": exit_reason, "dir": position["dir"]})
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
                if not pd.isna(h.get("ema_9")) and not pd.isna(h.get("ema_21")):
                    if h["ema_9"] > h["ema_21"] and h["close"] > h["ema_21"]:
                        htf = "BULLISH"
                    elif h["ema_9"] < h["ema_21"] and h["close"] < h["ema_21"]:
                        htf = "BEARISH"

        # Only trade with HTF (indices are trend-following)
        if params.get("require_htf", True) and htf == "NEUTRAL":
            continue

        direction = None
        conf = 0

        # Buy signal - only in bullish HTF
        if ema_f > ema_m > ema_s and close > ema_s and params["rsi_os"] < rsi < params["rsi_buy_max"]:
            if htf == "BULLISH":
                conf = min((adx - 25) / 50, 0.3) + 0.4
                direction = "BUY"

        # Sell signal - only in bearish HTF
        elif ema_f < ema_m < ema_s and close < ema_s and params["rsi_sell_min"] < rsi < params["rsi_ob"]:
            if htf == "BEARISH":
                conf = min((adx - 25) / 50, 0.3) + 0.4
                direction = "SELL"

        if not direction or conf < params["min_conf"]:
            continue

        # Position sizing
        stop_dist = max(atr * params["stop_mult"], MIN_STOP_MAP.get(market, 0))
        limit_dist = stop_dist * params["rr"]
        pv = POINT_VALUE_MAP.get(market, 1.0)
        size = min((equity * risk_pct) / (stop_dist * pv), params.get("max_size", 5))

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
        trades.append({"pnl": pnl, "reason": "EOT", "dir": position["dir"]})

    return trades, max_dd


def main():
    print("=" * 80)
    print("INDICES-SPECIFIC STRATEGY OPTIMIZATION")
    print("S&P 500 and NASDAQ 100")
    print("=" * 80)

    markets = ["S&P 500", "NASDAQ 100"]
    account_size = 10000
    days = 30

    print(f"\nFetching data...")
    market_data = {}
    htf_data = {}
    for m in markets:
        df = fetch_data(m, days, "5m")
        htf = fetch_data(m, days, "1h")
        if df is not None:
            market_data[m] = df
            htf_data[m] = htf
            print(f"  {m}: {len(df)} candles")

    # Indices-specific configurations to test
    test_configs = [
        # Baseline (current with indices defaults)
        {"name": "Current", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.5,
         "macd_exit": True, "require_htf": False, "max_size": 5},

        # HTF Required (only trade with trend)
        {"name": "HTF Required", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.5,
         "macd_exit": True, "require_htf": True, "max_size": 5},

        # Wider stops for indices (more volatile)
        {"name": "Wide Stops 2.5x", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.5, "rr": 2.0, "min_conf": 0.5,
         "macd_exit": True, "require_htf": True, "max_size": 5},

        # Lower RR for indices (take profits quicker)
        {"name": "RR 1.5", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.0, "rr": 1.5, "min_conf": 0.5,
         "macd_exit": True, "require_htf": True, "max_size": 5},

        # Conservative with HTF
        {"name": "Conservative HTF", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 14,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 55, "rsi_sell_min": 45,
         "adx_threshold": 30, "stop_mult": 2.0, "rr": 1.5, "min_conf": 0.6,
         "macd_exit": True, "require_htf": True, "max_size": 5},

        # Momentum indices
        {"name": "Momentum", "ema_fast": 5, "ema_medium": 12, "ema_slow": 26, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 65, "rsi_sell_min": 35,
         "adx_threshold": 20, "stop_mult": 1.5, "rr": 2.0, "min_conf": 0.4,
         "macd_exit": True, "require_htf": True, "max_size": 5},

        # Trend following (slower EMAs, bigger targets)
        {"name": "Trend Follow", "ema_fast": 12, "ema_medium": 26, "ema_slow": 50, "rsi_period": 14,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.0, "rr": 3.0, "min_conf": 0.5,
         "macd_exit": False, "require_htf": True, "max_size": 5},

        # No MACD exit + HTF
        {"name": "No MACD + HTF", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.0, "rr": 2.5, "min_conf": 0.5,
         "macd_exit": False, "require_htf": True, "max_size": 5},

        # Breakout style (high ADX, tight entry)
        {"name": "Breakout", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 35, "rsi_ob": 65, "rsi_buy_max": 55, "rsi_sell_min": 45,
         "adx_threshold": 30, "stop_mult": 1.5, "rr": 2.5, "min_conf": 0.5,
         "macd_exit": False, "require_htf": True, "max_size": 5},

        # Swing trade (slower, bigger RR)
        {"name": "Swing", "ema_fast": 12, "ema_medium": 26, "ema_slow": 60, "rsi_period": 14,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.5, "rr": 3.0, "min_conf": 0.5,
         "macd_exit": False, "require_htf": True, "max_size": 5},

        # Higher ADX filter
        {"name": "ADX 35", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 35, "stop_mult": 2.0, "rr": 2.0, "min_conf": 0.5,
         "macd_exit": False, "require_htf": True, "max_size": 5},

        # Quick scalp (low RR, high win rate target)
        {"name": "Scalp", "ema_fast": 5, "ema_medium": 12, "ema_slow": 26, "rsi_period": 5,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 20, "stop_mult": 1.0, "rr": 1.2, "min_conf": 0.4,
         "macd_exit": True, "require_htf": True, "max_size": 5},

        # Big moves only
        {"name": "Big Moves", "ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
         "rsi_os": 30, "rsi_ob": 70, "rsi_buy_max": 60, "rsi_sell_min": 40,
         "adx_threshold": 25, "stop_mult": 2.0, "rr": 4.0, "min_conf": 0.4,
         "macd_exit": False, "require_htf": True, "max_size": 5},

        # Very conservative
        {"name": "Very Conservative", "ema_fast": 12, "ema_medium": 26, "ema_slow": 50, "rsi_period": 14,
         "rsi_os": 35, "rsi_ob": 65, "rsi_buy_max": 55, "rsi_sell_min": 45,
         "adx_threshold": 35, "stop_mult": 2.5, "rr": 1.5, "min_conf": 0.6,
         "macd_exit": True, "require_htf": True, "max_size": 3},
    ]

    print(f"\nTesting {len(test_configs)} configurations...\n")

    results = []

    for config in test_configs:
        all_trades = []
        total_dd = 0
        market_results = {}

        for m in market_data:
            trades, dd = run_backtest(m, market_data[m], htf_data.get(m), config, account_size)
            all_trades.extend(trades)
            total_dd = max(total_dd, dd)

            m_pnl = sum(t["pnl"] for t in trades)
            m_wins = len([t for t in trades if t["pnl"] > 0])
            market_results[m] = {"pnl": m_pnl, "trades": len(trades), "wins": m_wins}

        total_pnl = sum(t["pnl"] for t in all_trades)
        wins = len([t for t in all_trades if t["pnl"] > 0])
        win_rate = wins / len(all_trades) if all_trades else 0

        results.append({
            "name": config["name"],
            "config": config,
            "trades": len(all_trades),
            "wins": wins,
            "win_rate": win_rate,
            "pnl": total_pnl,
            "max_dd": total_dd,
            "by_market": market_results,
        })

        print(f"  {config['name']:<20} | {len(all_trades):>3} trades | {win_rate:>5.0%} win | £{total_pnl:>+8,.0f} | DD {total_dd:>5.1%}")

    # Sort by P&L
    results.sort(key=lambda x: x["pnl"], reverse=True)

    print("\n" + "=" * 80)
    print("RESULTS RANKED BY PROFIT")
    print("=" * 80)

    print(f"\n{'Rank':<5} {'Strategy':<22} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'DD':>7}")
    print("-" * 65)

    for i, r in enumerate(results):
        print(f"#{i+1:<4} {r['name']:<22} {r['trades']:>7} {r['win_rate']:>6.0%} £{r['pnl']:>+9,.0f} {r['max_dd']:>6.1%}")

    # Best profitable strategy
    profitable = [r for r in results if r["pnl"] > 0]

    if profitable:
        best = profitable[0]
        print("\n" + "=" * 80)
        print(f"BEST STRATEGY FOR INDICES: {best['name']}")
        print("=" * 80)

        print(f"\nPerformance:")
        print(f"  Total P&L: £{best['pnl']:+,.2f}")
        print(f"  Monthly Return: {best['pnl']/account_size*100:+.1f}%")
        print(f"  Total Trades: {best['trades']}")
        print(f"  Win Rate: {best['win_rate']:.0%}")
        print(f"  Max Drawdown: {best['max_dd']:.1%}")

        print(f"\nBy Market:")
        for m, stats in best["by_market"].items():
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            print(f"  {m}: £{stats['pnl']:+,.0f} ({stats['trades']} trades, {wr:.0f}% win)")

        print(f"\nParameters:")
        for k, v in best['config'].items():
            if k != "name":
                print(f"  {k}: {v}")

        # Compare to current
        current = next((r for r in results if r["name"] == "Current"), None)
        if current:
            improvement = best["pnl"] - current["pnl"]
            print(f"\nImprovement vs Current: £{improvement:+,.0f}")
    else:
        print("\n" + "=" * 80)
        print("NO PROFITABLE STRATEGIES FOUND FOR INDICES")
        print("=" * 80)
        print("\nAll tested configurations produced losses on S&P 500 and NASDAQ 100.")
        print("Consider:")
        print("  1. Trading indices only during strong trends (VIX < 20)")
        print("  2. Using a longer timeframe (15m or 1H candles)")
        print("  3. Reducing position sizes significantly")
        print("  4. Only trading indices during US market hours")

        # Show least bad option
        least_bad = results[0]
        print(f"\nLeast losing strategy: {least_bad['name']} (£{least_bad['pnl']:+,.0f})")


if __name__ == "__main__":
    main()
