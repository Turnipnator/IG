#!/usr/bin/env python3
"""
GBP/USD + EUR/USD forex backtest — diagnose the live losing book.

Live journal (rolling 100):
  GBP/USD  1W/4L  -£31.72   EUR/USD  1W/3L  -£15.15

Config-era split (why this matters — see feedback_check_config_era):
  * GBP/USD recent losers (2026-05-26, 05-28, 06-01) are ALL on the CURRENT
    config: 5m candles, stop 1.0x, min_stop 4.0pts, conf 0.56. Every one a
    3-28 min stop-out. This is a live, current-era bleed → the real target.
  * EUR/USD losers (04-14, 04-27, 05-01) are 1-2 min stop-outs from the OLD
    5m config. It has SINCE moved to 1h candles (config note: 1h PF 1.25,
    +0.46%/365d vs 5m PF 0.73, -0.28%). So those losses may already be fixed
    — this run is to CONFIRM 1h beats 5m and the current profile is sound.

Hypothesis: tight 1.0x stops on 5m forex get whipsawed. Test wider stops and
higher timeframes.

yfinance (GBPUSD=X / EURUSD=X) — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)

# Live "forex" profile (config.py STRATEGY_PROFILES["forex"])
FOREX = {
    "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 30, "atr_period": 14,
    "stop_atr_multiplier": 1.0, "reward_risk_ratio": 2.0,
}

# Yahoo tickers + realistic min-stop floors in PRICE units.
# Live floors: GBP/USD 4.0 IG pts (=0.0004), EUR/USD 3.0 IG pts (=0.0003).
# Set the backtest floor to match live so stop 1.0x reproduces the live whipsaw,
# then the multiplier sweep shows whether wider stops help.
MARKETS = {
    "GBP/USD": {"ticker": "GBPUSD=X", "min_stop": 0.0004},
    "EUR/USD": {"ticker": "EURUSD=X", "min_stop": 0.0003},
}


def patch_market(market, min_conf=0.55):
    cfg = MARKETS[market]
    bt_module.TICKER_MAP[market] = cfg["ticker"]
    bt_module.MIN_STOP_DISTANCE_MAP[market] = cfg["min_stop"]
    MIN_CONFIDENCE_MAP[market] = min_conf
    REWARD_RISK_MAP[market] = 2.0


def run_one(market, days, overrides, interval, min_conf, htf=True):
    patch_market(market, min_conf=min_conf)
    params = DEFAULT_PARAMS.copy()
    params.update(FOREX)
    params.update(overrides)
    bt = Backtester(params=params)
    return bt.run(market, days=days, interval=interval, require_htf_alignment=htf)


# (label, overrides, interval, min_conf)
GBP_CONFIGS = [
    ("CURRENT 5m stop 1.0x",          {"stop_atr_multiplier": 1.0}, "5m", 0.55),
    ("5m stop 1.5x",                  {"stop_atr_multiplier": 1.5}, "5m", 0.55),
    ("5m stop 2.0x",                  {"stop_atr_multiplier": 2.0}, "5m", 0.55),
    ("5m ADX 35 (stricter)",          {"stop_atr_multiplier": 1.0, "adx_threshold": 35}, "5m", 0.55),
    ("5m conf >= 0.65",               {"stop_atr_multiplier": 1.0}, "5m", 0.65),
    ("15m stop 1.0x",                 {"stop_atr_multiplier": 1.0}, "15m", 0.55),
    ("15m stop 1.5x",                 {"stop_atr_multiplier": 1.5}, "15m", 0.55),
    ("1h stop 1.0x",                  {"stop_atr_multiplier": 1.0}, "1h", 0.55),
    ("1h stop 1.5x",                  {"stop_atr_multiplier": 1.5}, "1h", 0.55),
    ("1h ADX 35",                     {"stop_atr_multiplier": 1.0, "adx_threshold": 35}, "1h", 0.55),
]

EUR_CONFIGS = [
    ("CURRENT 1h stop 1.0x",          {"stop_atr_multiplier": 1.0}, "1h", 0.55),
    ("1h stop 1.5x",                  {"stop_atr_multiplier": 1.5}, "1h", 0.55),
    ("1h ADX 35",                     {"stop_atr_multiplier": 1.0, "adx_threshold": 35}, "1h", 0.55),
    ("1h conf >= 0.65",               {"stop_atr_multiplier": 1.0}, "1h", 0.65),
    ("OLD 5m stop 1.0x",              {"stop_atr_multiplier": 1.0}, "5m", 0.55),
    ("5m stop 1.5x",                  {"stop_atr_multiplier": 1.5}, "5m", 0.55),
    ("15m stop 1.0x",                 {"stop_atr_multiplier": 1.0}, "15m", 0.55),
]


def run_market(market, configs):
    days_intraday = 59
    days_1h = 365

    print("=" * 100)
    print(f"{market} ({MARKETS[market]['ticker']}) — intraday=59d, 1h=365d")
    print("=" * 100)
    print(f"{'Config':<32} {'Bars':>4} {'Days':>5} {'Trades':>7} "
          f"{'WR':>7} {'P&L%':>8} {'PF':>6} {'MaxDD%':>7}")
    print("-" * 100)

    results = {}
    for label, overrides, interval, min_conf in configs:
        days = days_1h if interval == "1h" else days_intraday
        try:
            r = run_one(market, days, overrides, interval, min_conf)
            results[label] = r
            if r.total_trades == 0:
                print(f"{label:<32} {interval:>4} {days:>5} {'0':>7} "
                      f"{'-':>7} {'-':>8} {'-':>6} {'-':>7}")
            else:
                print(f"{label:<32} {interval:>4} {days:>5} {r.total_trades:>7} "
                      f"{r.win_rate:>6.1%} {r.total_pnl:>+7.2f}% "
                      f"{r.profit_factor:>6.2f} {r.max_drawdown:>6.1%}")
        except Exception as e:
            print(f"{label:<32} ERROR: {e}")

    ranked = sorted(
        [(l, r) for l, r in results.items() if r.total_trades > 0],
        key=lambda x: x[1].profit_factor,
        reverse=True,
    )
    if ranked:
        print()
        best_l, best_r = ranked[0]
        print(f"  BEST by PF: {best_l}  (PF {best_r.profit_factor:.2f}, "
              f"P&L {best_r.total_pnl:+.2f}%, WR {best_r.win_rate:.1%}, {best_r.total_trades}t)")
    print()
    return results


def main():
    run_market("GBP/USD", GBP_CONFIGS)
    run_market("EUR/USD", EUR_CONFIGS)


if __name__ == "__main__":
    main()
