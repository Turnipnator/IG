#!/usr/bin/env python3
"""
BE-then-trail backtest (2026-05-31).

Production has BE 0.7 + ATR×1.5 trail. The healthcheck noted the strategy
makes money on its big-winner exits (RSI overbought) and loses small on
stop-outs — 82 stop-out trades net -£253. Question: is BE locking too late
(letting winners scratch back to BE-or-loss), or just right?

The existing backtester models only fixed SL/TP. This script subclasses
Backtester and adds the live BE+trail logic to the exit loop, then sweeps
the BE trigger across {OFF, 0.3, 0.5, 0.7, 0.9} on Gold + NASDAQ 100 +
Germany 40 (worst loser among indices) at 5m/55d and 1h/365d.

Yahoo data only — zero IG API cost.
"""

import os
import sys
from datetime import timedelta
from typing import Optional

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
    Trade,
)
from config import MARKETS, STRATEGY_PROFILES

logging.basicConfig(level=logging.ERROR)

# Yahoo tickers
TICKERS = {
    "Gold": "GC=F",
    "NASDAQ 100": "^NDX",
    "Germany 40": "^GDAXI",
}

# Cache Yahoo fetches across configs
_orig_fetch = Backtester.fetch_data
_cache: dict = {}


def cached_fetch(self, market, days=30, interval="5m"):
    key = (market, days, interval)
    if key not in _cache:
        _cache[key] = _orig_fetch(self, market, days, interval)
    df = _cache[key]
    return df.copy() if df is not None else None


Backtester.fetch_data = cached_fetch


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
        "atr_trail_mult": p.atr_trail_mult,
    })
    return base


class BETrailBacktester(Backtester):
    """
    Backtester with BE-then-trail simulation matching live main.py logic:
      - On each candle while in position, compute unrealized profit.
      - If profit >= breakeven_trigger_pct * initial_risk and BE not yet
        applied, move stop to entry price.
      - After BE applied, on each candle compute trail = close ± ATR*trail_mult
        and ratchet stop only in the favorable direction.

    Set self.be_trigger_pct = None to disable (matches production's behaviour
    before the BE feature was added).
    """

    def __init__(self, params=None, be_trigger_pct: Optional[float] = 0.7,
                 trail_atr_mult: float = 1.5):
        super().__init__(params=params)
        self.be_trigger_pct = be_trigger_pct
        self.trail_atr_mult = trail_atr_mult

    def run(self, market, days=30, interval="5m", htf_interval="1h",
            require_htf_alignment=False, min_confidence=None,
            account_size=10000, risk_per_trade=0.01):
        # --- setup identical to parent ---
        self.trades = []
        if min_confidence is None:
            min_confidence = MIN_CONFIDENCE_MAP.get(market, 0.5)

        df = self.fetch_data(market, days, interval)
        if df is None or df.empty:
            return self._empty_result(market, days)
        df = self.add_indicators(df)

        htf_df = self.fetch_data(market, days, htf_interval)
        if htf_df is not None and not htf_df.empty:
            from src.indicators import calculate_ema
            htf_df["ema_9"] = calculate_ema(htf_df["close"], 9)
            htf_df["ema_21"] = calculate_ema(htf_df["close"], 21)

        position: Optional[Trade] = None
        # BE/trail state (per-position)
        initial_risk = 0.0
        be_applied = False
        equity = account_size
        peak_equity = account_size
        max_drawdown = 0.0
        cooldown_until = None

        for i in range(self.params["ema_slow"], len(df)):
            row = df.iloc[i]
            current_time = row["date"]
            close = row["close"]
            atr = row["atr"]

            if cooldown_until and current_time < cooldown_until:
                continue

            if position:
                exit_reason = None
                exit_price = None

                # --- BE + trail bookkeeping (BEFORE stop check) ---
                if self.be_trigger_pct is not None and not pd.isna(atr):
                    if position.direction == "BUY":
                        profit = close - position.entry_price
                    else:
                        profit = position.entry_price - close

                    # Move stop to BE once trigger hit
                    if (not be_applied
                            and profit >= initial_risk * self.be_trigger_pct):
                        be_applied = True
                        position.stop_price = position.entry_price

                    # After BE: trail at close ± ATR*trail_mult, ratchet only favorably
                    if be_applied:
                        trail_dist = atr * self.trail_atr_mult
                        if position.direction == "BUY":
                            new_trail = close - trail_dist
                            if new_trail > position.stop_price:
                                position.stop_price = new_trail
                        else:
                            new_trail = close + trail_dist
                            if new_trail < position.stop_price:
                                position.stop_price = new_trail

                # --- standard stop / TP checks ---
                if position.direction == "BUY" and close <= position.stop_price:
                    exit_reason = "BE-trail stop" if be_applied else "Stop loss"
                    exit_price = position.stop_price
                elif position.direction == "SELL" and close >= position.stop_price:
                    exit_reason = "BE-trail stop" if be_applied else "Stop loss"
                    exit_price = position.stop_price

                if position.direction == "BUY" and close >= position.limit_price:
                    exit_reason = "Take profit"
                    exit_price = position.limit_price
                elif position.direction == "SELL" and close <= position.limit_price:
                    exit_reason = "Take profit"
                    exit_price = position.limit_price

                if exit_reason:
                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    if position.direction == "BUY":
                        position.pnl_percent = (exit_price - position.entry_price) / position.entry_price * 100
                    else:
                        position.pnl_percent = (position.entry_price - exit_price) / position.entry_price * 100
                    position.pnl = position.pnl_percent * position.size
                    equity += position.pnl * account_size / 100
                    self.trades.append(position)
                    if position.pnl_percent < 0:
                        cooldown_until = current_time + timedelta(hours=1)
                    position = None
                    be_applied = False
                    initial_risk = 0.0
                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity
                    max_drawdown = max(max_drawdown, drawdown)
                continue

            # --- entry signal ---
            htf_trend = self.calculate_htf_trend(market, current_time, htf_df)
            direction, confidence, reason = self.check_entry_signal(
                row, htf_trend, require_htf_alignment)
            if direction and confidence >= min_confidence:
                if pd.isna(atr):
                    continue
                stop_atr_mult = self.params["stop_atr_multiplier"]
                atr_stop = atr * stop_atr_mult
                min_stop = MIN_STOP_DISTANCE_MAP.get(market, 0.0)
                stop_distance = max(atr_stop, min_stop)
                rr_ratio = REWARD_RISK_MAP.get(market, self.params["reward_risk_ratio"])
                limit_distance = stop_distance * rr_ratio
                if direction == "BUY":
                    stop_price = close - stop_distance
                    limit_price = close + limit_distance
                else:
                    stop_price = close + stop_distance
                    limit_price = close - limit_distance
                risk_amount = account_size * risk_per_trade
                size = risk_amount / stop_distance if stop_distance > 0 else 0
                position = Trade(
                    entry_time=current_time, exit_time=None, market=market,
                    direction=direction, entry_price=close, exit_price=None,
                    stop_price=stop_price, limit_price=limit_price, size=size,
                    htf_trend=htf_trend, confidence=confidence,
                )
                initial_risk = stop_distance
                be_applied = False

        if position:
            position.exit_time = df.iloc[-1]["date"]
            position.exit_price = df.iloc[-1]["close"]
            position.exit_reason = "End of test"
            if position.direction == "BUY":
                position.pnl_percent = (position.exit_price - position.entry_price) / position.entry_price * 100
            else:
                position.pnl_percent = (position.entry_price - position.exit_price) / position.entry_price * 100
            position.pnl = position.pnl_percent * position.size
            self.trades.append(position)

        return self._calculate_results(market, days, require_htf_alignment, max_drawdown)


def fmt(r):
    if not r or r.total_trades == 0:
        return "0 trades"
    return (f"{r.total_trades:>3}t {r.win_rate:>5.0%} "
            f"{r.total_pnl:>+7.2f}% PF{r.profit_factor:>5.2f} "
            f"DD{r.max_drawdown:>4.0%}")


def count_be_trail(r):
    if not r or not r.trades:
        return 0
    return sum(1 for t in r.trades if t.exit_reason == "BE-trail stop")


def be_trail_pnl(r):
    if not r or not r.trades:
        return 0.0
    return sum(t.pnl_percent for t in r.trades if t.exit_reason == "BE-trail stop")


BE_VARIANTS = [
    ("BE OFF (fixed SL/TP only)", None),
    ("BE 0.3 (Oanda-style)",      0.3),
    ("BE 0.5",                    0.5),
    ("BE 0.7 (LIVE)",             0.7),
    ("BE 0.9 (let it ride)",      0.9),
]


def run_market(market_name, primary, htf, days):
    mc = next(m for m in MARKETS if m.name == market_name)
    prof = STRATEGY_PROFILES[mc.strategy]
    base = params_from_profile(prof)
    bt_module.TICKER_MAP[market_name] = TICKERS[market_name]
    MIN_STOP_DISTANCE_MAP[market_name] = 0.0
    MIN_CONFIDENCE_MAP[market_name] = mc.min_confidence
    REWARD_RISK_MAP[market_name] = prof.reward_risk
    trail = prof.atr_trail_mult

    print(f"\n--- {market_name} | {primary}/{htf} HTF | {days}d | "
          f"profile={mc.strategy} ADX{prof.adx_threshold} R:R{prof.reward_risk} "
          f"trail={trail}x ---")
    print(f"  {'variant':<35}  {'result':>34}    {'BE-trail exits':>22}")
    print("  " + "-" * 100)
    for label, be in BE_VARIANTS:
        bt = BETrailBacktester(params=base, be_trigger_pct=be, trail_atr_mult=trail)
        r = bt.run(market_name, days=days, interval=primary,
                   htf_interval=htf, require_htf_alignment=True)
        bt_n = count_be_trail(r)
        bt_pnl = be_trail_pnl(r)
        bt_str = f"{bt_n}× ({bt_pnl:+.2f}%)" if bt_n else "n/a"
        print(f"  {label:<35}  {fmt(r):>34}    {bt_str:>22}")


def main():
    print("=" * 110)
    print("BE-then-trail sweep — adds live BE+trail logic to the backtester")
    print("=" * 110)
    test_cells = [
        ("Gold",        "5m",  "1h", 55),
        ("Gold",        "1h",  "1d", 365),
        ("NASDAQ 100",  "5m",  "1h", 55),
        ("NASDAQ 100",  "1h",  "1d", 365),
        ("Germany 40",  "5m",  "1h", 55),
        ("Germany 40",  "1h",  "1d", 365),
    ]
    for m, p, h, d in test_cells:
        run_market(m, p, h, d)

    print("\n" + "=" * 110)
    print("READING THE TABLE")
    print("=" * 110)
    print("""
  - 'BE OFF' is the no-BE baseline (fixed SL/TP only — what the bare backtester models).
  - 'BE 0.7 (LIVE)' is current production.
  - 'BE 0.3' is the Oanda-style aggressive lock-in.
  - 'BE 0.9' is let-it-ride — only locks if very close to TP.
  - 'BE-trail exits' = trades that exited via BE-or-trail (not the original SL).

  If BE 0.3 beats BE 0.7 on the loser markets (Gold, Germany 40) without
  hurting NASDAQ 100, that's evidence for narrowing the trigger. If results
  are flat across variants, BE timing isn't where the edge lives.
""")


if __name__ == "__main__":
    main()
