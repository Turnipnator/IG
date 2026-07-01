#!/usr/bin/env python3
"""All-EPIC direction-asymmetry sweep for the both-way indices (2026-07-01).

Follow-up to the forex breakout-vs-momentum run: "did you only look at forex?".
The other live markets run a single strategy, so the question isn't which strategy —
it's whether each has a DIRECTION asymmetry we're not exploiting (the AI-Index /
S&P / FTSE long-only pattern: equity baskets drift up, so shorts fight the tide).

Four live indices have NEVER been direction-checked and run BOTH ways:
  Wall Street (Dow), NASDAQ 100, Japan 225, Hong Kong HS50.
S&P 500 (long-only) + FTSE 100 (long-bias) are included as POSITIVE CONTROLS — the
method must re-discover their known BUY>SELL edge or it isn't trustworthy.

Method: run the live faithful Backtester.run() (real per-market strategy params, HTF
requirement, MACD/ranging exits, stops) then split the REALISED trades by side —
BUY-only vs SELL-only vs both. Yahoo 5m/60d (the live indices interval), ZERO IG API
cost. Caveats: Yahoo cash = regular-session only (IG DFB is ~24h), 60d is a medium
sample, splitting realised trades ignores how removing one side shifts re-entry timing.
Directional lead for the review, not a live-P&L promise. Read BUY-vs-SELL deltas.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import logging

import config
from src.backtest import Backtester, TICKER_MAP, DEFAULT_PARAMS

logging.basicConfig(level=logging.ERROR)

# StrategyConfig field -> Backtester param key (names diverge between the two schemas)
FIELD_MAP = {
    "ema_fast": "ema_fast", "ema_medium": "ema_medium", "ema_slow": "ema_slow",
    "rsi_period": "rsi_period", "rsi_overbought": "rsi_overbought",
    "rsi_oversold": "rsi_oversold", "rsi_buy_max": "rsi_buy_max",
    "rsi_sell_min": "rsi_sell_min", "adx_threshold": "adx_threshold",
    "stop_atr_mult": "stop_atr_multiplier", "reward_risk": "reward_risk_ratio",
}

# Yahoo tickers for the markets not already in TICKER_MAP
TICKER_MAP.update({
    "Wall Street": "^DJI",
    "FTSE 100": "^FTSE",
    "Japan 225": "^N225",
    "Hong Kong HS50": "^HSI",
})

# controls first (known long-biased), then the four real unknowns
TARGETS = ["S&P 500", "FTSE 100", "Wall Street", "NASDAQ 100", "Japan 225", "Hong Kong HS50"]
CONTROLS = {"S&P 500", "FTSE 100"}
DAYS = int(os.environ.get("SWEEP_DAYS", "60"))
INTERVAL = os.environ.get("SWEEP_INTERVAL", "5m")
HTF = "1d" if INTERVAL == "1h" else "1h"

def params_for(market):
    m = next(x for x in config.MARKETS if x.name == market)
    prof = config.STRATEGY_PROFILES[m.strategy]
    pd_ = dataclasses.asdict(prof) if dataclasses.is_dataclass(prof) else dict(vars(prof))
    p = dict(DEFAULT_PARAMS)
    for src_field, dst_key in FIELD_MAP.items():
        if pd_.get(src_field) is not None:
            p[dst_key] = pd_[src_field]
    return p, m, prof


def split_stats(trades, side=None):
    ts = [t for t in trades if side is None or t.direction == side]
    n = len(ts)
    if n == 0:
        return None
    wins = [t for t in ts if t.pnl_percent > 0]
    gp = sum(t.pnl_percent for t in wins)
    gl = -sum(t.pnl_percent for t in ts if t.pnl_percent < 0)
    return dict(n=n, wr=len(wins) / n * 100, pnl=sum(t.pnl_percent for t in ts),
                pf=gp / gl if gl > 0 else 999.0)


def fmt(s):
    if s is None:
        return "   no trades"
    return f"n={s['n']:>3}  WR={s['wr']:>3.0f}%  P&L={s['pnl']:>+7.2f}%  PF={s['pf']:>5.2f}"


def main():
    print(f"\nIndices DIRECTION-asymmetry sweep — Yahoo {INTERVAL}/{DAYS}d, live strategy params\n")
    for market in TARGETS:
        tag = "  [control]" if market in CONTROLS else ""
        try:
            p, m, prof = params_for(market)
            bt = Backtester(params=p)
            res = bt.run(market, days=DAYS, interval=INTERVAL, htf_interval=HTF,
                         require_htf_alignment=getattr(prof, "require_htf", True),
                         min_confidence=m.min_confidence)
        except Exception as e:
            print(f"{'='*88}\n{market}{tag}: ERROR {type(e).__name__}: {e}\n")
            continue
        tr = res.trades
        both = split_stats(tr)
        buy = split_stats(tr, "BUY")
        sell = split_stats(tr, "SELL")
        print(f"{'='*88}\n{market}{tag}   strat={m.strategy}  (live dir="
              f"{getattr(m,'allowed_direction','') or 'both'})")
        print(f"  BOTH       {fmt(both)}")
        print(f"  BUY-only   {fmt(buy)}")
        print(f"  SELL-only  {fmt(sell)}")
        # verdict
        if buy and sell:
            if sell['pnl'] < 0 and buy['pnl'] > 0:
                print(f"  >> ASYMMETRY: SELL side net-negative, BUY net-positive -> long-only candidate")
            elif buy['pnl'] < 0 and sell['pnl'] > 0:
                print(f"  >> ASYMMETRY: BUY side net-negative -> short-only candidate")
            else:
                print(f"  >> no clear asymmetry")
        print()


if __name__ == "__main__":
    main()
