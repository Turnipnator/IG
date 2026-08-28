#!/usr/bin/env python3
"""
60-day batch backtest — Germany 40, US Russell 2000, Gold.

Three live underperformers:
  Germany 40       7 trades, 43% WR, -£46.90  (avg win £2.17 vs loss -£13.35)
  US Russell 2000 13 trades, 38% WR, -£11.50  (mostly MACD exits, fast)
  Gold            30 trades, 43% WR, -£23.86  (largest sample; tightened 2026-04-07)

Goal: find the right ADX threshold and stop multiplier per market.
Yahoo data — no IG API cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import src.backtest as bt_module
from src.backtest import DEFAULT_PARAMS, MIN_CONFIDENCE_MAP, REWARD_RISK_MAP, Backtester

logging.basicConfig(level=logging.WARNING)


# Market -> (ticker, min_stop_price, default_min_conf, default_rr, profile_params)
INDICES_PROFILE = {
    "ema_fast": 5, "ema_medium": 12, "ema_slow": 26,
    "rsi_period": 7, "rsi_overbought": 70, "rsi_oversold": 30,
    "rsi_buy_max": 55, "rsi_sell_min": 45,
    "adx_threshold": 30, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 2.0,
}

GOLD_PROFILE = {
    "ema_fast": 3, "ema_medium": 8, "ema_slow": 21,
    "rsi_period": 7, "rsi_overbought": 85, "rsi_oversold": 15,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 3.0,
}


MARKETS = {
    "Germany 40": {
        "ticker": "^GDAXI",
        "min_stop": 25.0,    # ~0.1% of 24000
        "min_conf": 0.55,
        "rr": 2.0,
        "profile": INDICES_PROFILE,
        "configs": [
            ("Current (ADX 30)",   {}),
            ("ADX 35",             {"adx_threshold": 35}),
            ("ADX 40",             {"adx_threshold": 40}),
            ("ADX 35, stop 1.0×",  {"adx_threshold": 35, "stop_atr_multiplier": 1.0}),
            ("ADX 35, stop 2.0×",  {"adx_threshold": 35, "stop_atr_multiplier": 2.0}),
        ],
    },
    "US Russell 2000": {
        "ticker": "^RUT",
        "min_stop": 8.0,     # ~0.3% of 2200
        "min_conf": 0.55,
        "rr": 2.0,
        "profile": INDICES_PROFILE,
        "configs": [
            ("Current (ADX 30)",   {}),
            ("ADX 35",             {"adx_threshold": 35}),
            ("ADX 40",             {"adx_threshold": 40}),
            ("ADX 35, stop 1.0×",  {"adx_threshold": 35, "stop_atr_multiplier": 1.0}),
            ("ADX 35, stop 2.0×",  {"adx_threshold": 35, "stop_atr_multiplier": 2.0}),
        ],
    },
    "Gold": {
        "ticker": "GC=F",
        "min_stop": 25.0,
        "min_conf": 0.55,
        "rr": 3.0,
        "profile": GOLD_PROFILE,
        "configs": [
            ("Current (ADX 35, 1.5×, R:R 3)",  {}),
            ("ADX 30",             {"adx_threshold": 30}),
            ("ADX 40",             {"adx_threshold": 40}),
            ("Stop 1.0×",          {"stop_atr_multiplier": 1.0}),
            ("Stop 2.0×",          {"stop_atr_multiplier": 2.0}),
            ("R:R 2.0",            {"reward_risk_ratio": 2.0}),
            ("R:R 4.0",            {"reward_risk_ratio": 4.0}),
        ],
    },
}


def patch_markets():
    for name, cfg in MARKETS.items():
        bt_module.TICKER_MAP[name] = cfg["ticker"]
        bt_module.MIN_STOP_DISTANCE_MAP[name] = cfg["min_stop"]
        MIN_CONFIDENCE_MAP[name] = cfg["min_conf"]
        REWARD_RISK_MAP[name] = cfg["rr"]


def run_one(market: str, days: int, profile: dict, overrides: dict):
    params = DEFAULT_PARAMS.copy()
    params.update(profile)
    params.update(overrides)
    # Sync reward_risk_ratio override into REWARD_RISK_MAP since run() reads it from there
    if "reward_risk_ratio" in overrides:
        REWARD_RISK_MAP[market] = overrides["reward_risk_ratio"]
    else:
        REWARD_RISK_MAP[market] = profile["reward_risk_ratio"]
    bt = Backtester(params=params)
    return bt.run(market, days=days, require_htf_alignment=True)


def main():
    patch_markets()
    days = 60

    for market, cfg in MARKETS.items():
        print()
        print("=" * 100)
        print(f"{market} ({cfg['ticker']}) — {days}d, 5m, HTF aligned")
        print("=" * 100)
        print(f"{'Config':<36} {'Trades':>7} {'WinRate':>8} {'P&L%':>8} "
              f"{'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>7}")
        print("-" * 100)

        results = []
        for label, overrides in cfg["configs"]:
            try:
                r = run_one(market, days, cfg["profile"], overrides)
                results.append((label, r))
                if r.total_trades == 0:
                    print(f"{label:<36} {'0':>7} {'—':>8} {'—':>8} {'—':>6} "
                          f"{'—':>8} {'—':>9} {'—':>7}")
                else:
                    print(f"{label:<36} {r.total_trades:>7} {r.win_rate:>7.1%} "
                          f"{r.total_pnl:>+7.2f}% {r.profit_factor:>6.2f} "
                          f"{r.avg_win:>+7.2f}% {r.avg_loss:>+8.2f}% {r.max_drawdown:>6.1%}")
            except Exception as e:
                print(f"{label:<36} ERROR: {e}")

        ranked = sorted(
            [(l, r) for l, r in results if r.total_trades > 0],
            key=lambda x: x[1].total_pnl,
            reverse=True,
        )
        print()
        print(f"  Best: {ranked[0][0]} → {ranked[0][1].total_pnl:+.2f}% "
              f"(PF {ranked[0][1].profit_factor:.2f}, WR {ranked[0][1].win_rate:.1%}, "
              f"{ranked[0][1].total_trades} trades)" if ranked else "  No valid runs")


if __name__ == "__main__":
    main()
