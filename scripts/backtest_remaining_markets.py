#!/usr/bin/env python3
"""
Unified backtest for all markets not retuned today.

Already addressed in this session (skipped here):
  Germany 40, Copper, Gold — config changes shipped
  USD/JPY — left as-is after sweep
  Natural Gas — already on 1h, well-validated
  Japan 225, Hong Kong HS50 — already swept in prior session
  Russell 2000 — already parked

Tested here:
  Indices: S&P 500, NASDAQ 100, Wall Street, FTSE 100
  Commodities: Crude Oil, NY Cocoa, NY Cotton
  Forex: EUR/USD, GBP/USD
  Rates: US 2-Year T-Note, US 10-Year T-Note

Skipped: AI Index (no clean yfinance ticker for IG's custom basket).

Per market: current vs higher confidence (0.70) vs higher ADX (+5)
vs 1h conversion (where currently sub-1h).

yfinance only — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)


# Strategy profile snapshots — pulled from config.py STRATEGY_PROFILES.
# Each value is the param overrides relative to DEFAULT_PARAMS.
STRATEGY_PARAMS = {
    "indices": {
        "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 30, "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
    },
    "indices_adx35": {
        "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 35, "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
    },
    "indices_selective": {
        "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 40, "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
    },
    "indices_wide": {
        "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 30, "stop_atr_multiplier": 1.0, "reward_risk_ratio": 1.5,
    },
    "indices_tight": {
        "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 30, "stop_atr_multiplier": 1.0, "reward_risk_ratio": 2.0,
    },
    "forex": {
        "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 30, "stop_atr_multiplier": 1.0, "reward_risk_ratio": 2.0,
    },
    "crude": {
        "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
        "rsi_overbought": 80, "rsi_oversold": 20,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 30, "stop_atr_multiplier": 1.0, "reward_risk_ratio": 2.0,
    },
    "default": {
        "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "rsi_buy_max": 55, "rsi_sell_min": 45,
        "adx_threshold": 30, "stop_atr_multiplier": 1.8, "reward_risk_ratio": 2.0,
    },
}


# (market_name, ticker, strategy_profile, current_interval, min_stop_price)
MARKETS_TO_TEST = [
    ("S&P 500",          "^GSPC",     "indices_selective", "5m",  30.0),
    ("NASDAQ 100",       "^NDX",      "indices_wide",      "5m",  100.0),
    ("Wall Street",      "^DJI",      "indices",           "5m",  200.0),
    ("FTSE 100",         "^FTSE",     "indices_tight",     "5m",  45.0),
    ("Crude Oil",        "CL=F",      "crude",             "15m", 0.35),
    ("NY Cocoa",         "CC=F",      "default",           "15m", 30.0),
    ("NY Cotton",        "CT=F",      "default",           "15m", 0.5),
    ("EUR/USD",          "EURUSD=X",  "forex",             "5m",  0.005),
    ("GBP/USD",          "GBPUSD=X",  "forex",             "5m",  0.005),
    ("US 2-Year T-Note", "ZT=F",      "default",           "15m", 0.5),
    ("US 10-Year T-Note","ZN=F",      "default",           "15m", 0.5),
]


def patch_market(name, ticker, min_stop, min_conf=0.55, rr=2.0):
    bt_module.TICKER_MAP[name] = ticker
    bt_module.MIN_STOP_DISTANCE_MAP[name] = min_stop
    MIN_CONFIDENCE_MAP[name] = min_conf
    REWARD_RISK_MAP[name] = rr


def run_one(market, ticker, min_stop, overrides, interval, min_conf, days, htf=True):
    patch_market(market, ticker, min_stop, min_conf=min_conf,
                 rr=overrides.get("reward_risk_ratio", 2.0))
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(market, days=days, interval=interval, require_htf_alignment=htf)


def format_result(r):
    if r is None or r.total_trades == 0:
        return f"{'0':>6} {'-':>6} {'-':>8} {'-':>6} {'-':>7}"
    return (f"{r.total_trades:>6d} {r.win_rate:>5.0%} {r.total_pnl:>+7.2f}% "
            f"{r.profit_factor:>5.2f} {r.max_drawdown:>6.1%}")


def main():
    days_intraday = 59
    days_1h = 365

    print("=" * 130)
    print("Unified backtest — 11 markets, 4 variants each (current / 1h / conf>=0.70 / ADX+5)")
    print("=" * 130)
    print(f"{'Market':<20} {'Variant':<20} {'Bars':>4} {'Days':>5} "
          f"{'Trades':>6} {'WR':>6} {'P&L':>8} {'PF':>5} {'MaxDD':>7}")
    print("-" * 130)

    all_results = {}

    for name, ticker, strategy, current_interval, min_stop in MARKETS_TO_TEST:
        base_params = STRATEGY_PARAMS[strategy].copy()
        market_results = {}

        # Variant 1: CURRENT
        try:
            r = run_one(name, ticker, min_stop, base_params, current_interval,
                        0.55, days_intraday)
            market_results["CURRENT"] = r
            print(f"{name:<20} {'CURRENT ' + strategy:<20} {current_interval:>4} "
                  f"{days_intraday:>5} {format_result(r)}")
        except Exception as e:
            print(f"{name:<20} CURRENT ERROR: {e}")

        # Variant 2: Higher confidence (0.70)
        try:
            r = run_one(name, ticker, min_stop, base_params, current_interval,
                        0.70, days_intraday)
            market_results["conf 0.70"] = r
            print(f"{name:<20} {'conf >= 0.70':<20} {current_interval:>4} "
                  f"{days_intraday:>5} {format_result(r)}")
        except Exception as e:
            print(f"{name:<20} conf 0.70 ERROR: {e}")

        # Variant 3: +5 ADX (more selective)
        try:
            new_adx = base_params["adx_threshold"] + 5
            adx_up = {**base_params, "adx_threshold": new_adx}
            r = run_one(name, ticker, min_stop, adx_up, current_interval,
                        0.55, days_intraday)
            market_results["ADX+5"] = r
            label = f"ADX {new_adx} (+5)"
            print(f"{name:<20} {label:<20} {current_interval:>4} "
                  f"{days_intraday:>5} {format_result(r)}")
        except Exception as e:
            print(f"{name:<20} ADX+5 ERROR: {e}")

        # Variant 4: 1h candles (if not already 1h)
        if current_interval != "1h":
            try:
                r = run_one(name, ticker, min_stop, base_params, "1h",
                            0.55, days_1h)
                market_results["1h"] = r
                print(f"{name:<20} {'1h candles':<20} {'1h':>4} "
                      f"{days_1h:>5} {format_result(r)}")
            except Exception as e:
                print(f"{name:<20} 1h ERROR: {e}")

        print()
        all_results[name] = market_results

    # Summary by market: which variant won?
    print("=" * 130)
    print("WINNERS by Profit Factor (only variants with >= 10 trades):")
    print("=" * 130)
    for name, variants in all_results.items():
        viable = [(v, r) for v, r in variants.items()
                  if r is not None and r.total_trades >= 10]
        if not viable:
            print(f"  {name:<20} -- no variant produced 10+ trades")
            continue
        best = max(viable, key=lambda x: x[1].profit_factor)
        current = variants.get("CURRENT")
        cur_pf = current.profit_factor if current else 0
        cur_label = f"PF {cur_pf:.2f}" if current else "n/a"
        delta = best[1].profit_factor - cur_pf if current else 0
        flag = "→ CHANGE" if delta > 0.20 else "  hold"
        print(f"  {name:<20} current {cur_label}, best={best[0]} "
              f"PF {best[1].profit_factor:.2f}  Δ{delta:+.2f}  {flag}")


if __name__ == "__main__":
    main()
