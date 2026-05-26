#!/usr/bin/env python3
"""
Test the ADX-declining filter (2026-05-26).

Live `src/strategy.py` rejects entry when current ADX < prev_adx - 0.5
(line 169-181). Health-check 2026-05-26 showed it produced 13 HOLD rejections
in 5 days while only 3 trades fired live. Question: are those rejections
saving us from losses, or are they killing profitable trades?

The backtester does NOT replicate this filter by default. This script patches
`_check_signal` to honour `require_rising_adx=True`, runs each live market
both ways, and compares trade count / P&L% / PF / win rate.

Yahoo data only — zero IG API cost. Caveat: backtester models fixed
stop/limit + MACD exit only (no BE/ATR-trailing), so trust the relative
delta between the two configs, not absolute %.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import pandas as pd

import src.backtest as bt_module
from src.backtest import (
    DEFAULT_PARAMS,
    MIN_CONFIDENCE_MAP,
    MIN_STOP_DISTANCE_MAP,
    REWARD_RISK_MAP,
    Backtester,
)
from config import MARKETS, STRATEGY_PROFILES

logging.basicConfig(level=logging.ERROR)

# Yahoo ticker for each live market. AI Index is IG-proprietary, no Yahoo
# proxy, so it's skipped.
TICKERS = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Germany 40": "^GDAXI",
    "Wall Street": "^DJI",
    "FTSE 100": "^FTSE",
    "Japan 225": "^N225",
    "Hong Kong HS50": "^HSI",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "Copper": "HG=F",
    "NY Cocoa": "CC=F",
    "NY Cotton": "CT=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "US 2-Year T-Note": "ZT=F",
    "US 10-Year T-Note": "ZN=F",
}

# yfinance history caps: 5m/15m ~60d, 1h ~730d, 1d unlimited.
# Pick the deepest window each interval supports so trade counts are meaningful.
WINDOWS_BY_INTERVAL = {
    "5m":  [55],
    "15m": [55],
    "1h":  [180, 365],
    "1d":  [365, 720],
}

# --- Patch add_indicators to expose prev_adx -------------------------------

_orig_add = Backtester.add_indicators


def add_with_prev_adx(self, df):
    df = _orig_add(self, df)
    df["prev_adx"] = df["adx"].shift(1)
    return df


Backtester.add_indicators = add_with_prev_adx


# --- Patch check_entry_signal to honour require_rising_adx -----------------

_orig_check = Backtester.check_entry_signal


def check_with_filter(self, row, htf_trend, require_htf_alignment=False):
    direction, conf, reason = _orig_check(self, row, htf_trend, require_htf_alignment)
    if direction is None:
        return direction, conf, reason
    if not self.params.get("require_rising_adx", False):
        return direction, conf, reason
    prev = row.get("prev_adx")
    cur = row.get("adx")
    if prev is None or pd.isna(prev):
        return direction, conf, reason
    if cur < prev - 0.5:
        return None, 0, f"ADX declining ({prev:.1f}->{cur:.1f})"
    return direction, conf, reason


Backtester.check_entry_signal = check_with_filter


# --- Cache Yahoo fetches ---------------------------------------------------

_orig_fetch = Backtester.fetch_data
_cache: dict = {}


def cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _cache:
        _cache[key] = _orig_fetch(self, market, days, interval)
    df = _cache[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = cached_fetch


# --- Helpers --------------------------------------------------------------


def params_from_profile(p) -> dict:
    base = DEFAULT_PARAMS.copy()
    base.update({
        "ema_fast": p.ema_fast,
        "ema_medium": p.ema_medium,
        "ema_slow": p.ema_slow,
        "rsi_period": p.rsi_period,
        "rsi_overbought": p.rsi_overbought,
        "rsi_oversold": p.rsi_oversold,
        "rsi_buy_max": p.rsi_buy_max,
        "rsi_sell_min": p.rsi_sell_min,
        "adx_threshold": p.adx_threshold,
        "atr_period": 14,
        "stop_atr_multiplier": p.stop_atr_mult,
        "reward_risk_ratio": p.reward_risk,
    })
    return base


def run_one(market, params, interval, htf_interval, days):
    bt = Backtester(params=params)
    return bt.run(market, days=days, interval=interval,
                  htf_interval=htf_interval, require_htf_alignment=True)


def main():
    # Aggregate totals per filter setting across all markets, keyed by interval
    agg: dict = {}

    print("\n" + "=" * 110)
    print("ADX-declining filter test — per-market comparison")
    print("Format:  WITH filter (live)  vs  WITHOUT filter")
    print("=" * 110)

    for mc in MARKETS:
        if mc.name not in TICKERS:
            print(f"\n[skip] {mc.name} — no Yahoo proxy")
            continue
        prof = STRATEGY_PROFILES[mc.strategy]
        base = params_from_profile(prof)
        bt_module.TICKER_MAP[mc.name] = TICKERS[mc.name]
        MIN_STOP_DISTANCE_MAP[mc.name] = 0.0
        MIN_CONFIDENCE_MAP[mc.name] = mc.min_confidence
        REWARD_RISK_MAP[mc.name] = prof.reward_risk

        # candle_interval is INT minutes; htf_resolution is IG string
        ci = getattr(mc, "candle_interval", 5)
        htf_str = getattr(mc, "htf_resolution", "HOUR")
        yf_int = {5: "5m", 15: "15m", 30: "30m", 60: "1h"}.get(ci, "5m")
        yf_htf = {"MINUTE_5": "5m", "MINUTE_15": "15m", "MINUTE_30": "30m",
                  "HOUR": "1h", "HOUR_4": "1h", "DAY": "1d"}.get(htf_str, "1h")

        windows = WINDOWS_BY_INTERVAL.get(yf_int, [60])
        print(f"\n{mc.name}  ({TICKERS[mc.name]}, {yf_int}/{yf_htf}, "
              f"profile={mc.strategy} ADX{prof.adx_threshold})")
        print(f"  {'window':>7}  {'WITH filter':>30}    {'WITHOUT filter':>30}    delta")
        print("  " + "-" * 100)

        for d in windows:
            results = {}
            for require_rising in (True, False):
                key = (yf_int, d, require_rising)
                if key not in agg:
                    agg[key] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
                                "gross_profit": 0.0, "gross_loss": 0.0}
                p = base.copy()
                p["require_rising_adx"] = require_rising
                try:
                    r = run_one(mc.name, p, yf_int, yf_htf, d)
                except Exception as e:
                    results[require_rising] = f"ERR {e}"
                    continue
                results[require_rising] = r
                if r and r.total_trades > 0:
                    a = agg[key]
                    a["trades"] += r.total_trades
                    wins = [t for t in r.trades if t.pnl_percent > 0]
                    losses = [t for t in r.trades if t.pnl_percent < 0]
                    a["wins"] += len(wins)
                    a["losses"] += len(losses)
                    a["pnl"] += r.total_pnl
                    a["gross_profit"] += sum(t.pnl_percent for t in wins)
                    a["gross_loss"] += abs(sum(t.pnl_percent for t in losses))

            def fmt(r):
                if isinstance(r, str):
                    return r
                if not r or r.total_trades == 0:
                    return "0 trades"
                return (f"{r.total_trades:>2}t {r.win_rate:>5.0%} "
                        f"{r.total_pnl:>+6.2f}% PF{r.profit_factor:>5.2f}")
            rw = fmt(results[True])
            rwo = fmt(results[False])
            # Delta in trade count + P&L
            delta = ""
            if (not isinstance(results[True], str) and
                    not isinstance(results[False], str) and
                    results[True] and results[False]):
                dt = results[False].total_trades - results[True].total_trades
                dp = results[False].total_pnl - results[True].total_pnl
                delta = f"{dt:+d}t {dp:+.2f}%"
            print(f"  {d:>6}d  {rw:>30}    {rwo:>30}    {delta}")

    # Aggregate summary by interval bucket + window
    print("\n" + "=" * 110)
    print("AGGREGATE (sum across all tested markets, grouped by interval + window)")
    print("=" * 110)
    seen = sorted({(yi, d) for (yi, d, _) in agg.keys()})
    for yi, d in seen:
        for require_rising in (True, False):
            a = agg.get((yi, d, require_rising))
            if a is None:
                continue
            wr = a["wins"] / (a["wins"] + a["losses"]) if (a["wins"] + a["losses"]) else 0
            pf = a["gross_profit"] / a["gross_loss"] if a["gross_loss"] > 0 else float("inf")
            tag = "WITH " if require_rising else "WO   "
            print(f"  {tag} {yi:>3}/{d}d: {a['trades']:>3}t  {wr:>5.0%} WR  "
                  f"{a['pnl']:>+7.2f}%  PF {pf:>5.2f}")
        print()


if __name__ == "__main__":
    main()
