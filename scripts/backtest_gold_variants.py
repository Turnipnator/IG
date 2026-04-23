#!/usr/bin/env python3
"""
Gold exit-logic A/B test.

Runs the live Gold strategy through the existing Yahoo pipeline under
several variant configurations to isolate which exit rule(s) are hurting.

Variants tested:
  BASELINE                 — live config unchanged
  NO_RANGING_EXIT          — disable ADX ranging exit
  LOOSE_BE_TRAIL           — BE trigger 100%, ATR trail 2.5x
  NO_RANGING + LOOSE_TRAIL — both of the above
  LONG_COOLDOWN            — 8 candles (2h) after ANY exit
  EVERYTHING               — all fixes combined

All variants use identical entry logic + data, so differences are attributable
to the exit changes.
"""

import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MARKETS, get_strategy_for_market, STRATEGY_PROFILES
from src.indicators import add_all_indicators

from backtest import (  # reuse data pipeline
    fetch_data,
    fetch_htf_data,
    calculate_htf_trend,
    lookup_htf_trend,
    IG_POINTS_MULT,
)


@dataclass
class VTrade:
    direction: str
    entry_price: float
    entry_time: datetime
    stop_distance: float  # IG points
    limit_distance: float  # IG points
    size: float
    stop_price: float
    limit_price: float
    breakeven_applied: bool = False
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0


def simulate_gold(
    df: pd.DataFrame,
    htf_trends: pd.DataFrame,
    strategy,
    market_config,
    pts_mult: float,
    disable_ranging_exit: bool = False,
    breakeven_trigger_pct_override: Optional[float] = None,
    atr_trail_mult_override: Optional[float] = None,
    cooldown_candles_after_any_close: Optional[int] = None,
    stop_atr_mult_override: Optional[float] = None,
    reward_risk_override: Optional[float] = None,
) -> list[VTrade]:
    """
    Simulation that mirrors backtest.run_backtest's loop for a single market,
    with the exit-logic knobs we want to test exposed as parameters.
    """
    be_trigger = breakeven_trigger_pct_override or strategy.breakeven_trigger_pct
    trail_mult = atr_trail_mult_override or strategy.atr_trail_mult
    stop_mult = stop_atr_mult_override or strategy.stop_atr_mult
    rr = reward_risk_override or strategy.reward_risk

    indicator_params = {
        "ema_fast": strategy.ema_fast,
        "ema_medium": strategy.ema_medium,
        "ema_slow": strategy.ema_slow,
        "rsi_period": strategy.rsi_period,
        "rsi_overbought": strategy.rsi_overbought,
        "rsi_oversold": strategy.rsi_oversold,
        "rsi_buy_max": strategy.rsi_buy_max,
        "rsi_sell_min": strategy.rsi_sell_min,
        "adx_threshold": strategy.adx_threshold,
    }
    di = add_all_indicators(df.copy(), indicator_params)

    trades: list[VTrade] = []
    active: Optional[VTrade] = None
    cooldown_until = None
    min_start = max(strategy.ema_slow + 10, 50)

    for i in range(min_start, len(di)):
        row = di.iloc[i]
        ts = row["date"]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row["atr"] if "atr" in row and not pd.isna(row.get("atr", float("nan"))) else 0
        htf_trend = lookup_htf_trend(htf_trends, ts)

        if active is not None:
            t = active
            closed = False

            if t.direction == "BUY":
                if low <= t.stop_price:
                    t.exit_price = t.stop_price
                    t.exit_reason = "Stop"
                    closed = True
                elif high >= t.limit_price:
                    t.exit_price = t.limit_price
                    t.exit_reason = "Limit"
                    closed = True
            else:
                if high >= t.stop_price:
                    t.exit_price = t.stop_price
                    t.exit_reason = "Stop"
                    closed = True
                elif low <= t.limit_price:
                    t.exit_price = t.limit_price
                    t.exit_reason = "Limit"
                    closed = True

            if not closed:
                if not t.breakeven_applied:
                    profit_pts = (close - t.entry_price) if t.direction == "BUY" else (t.entry_price - close)
                    stop_dist = t.stop_distance / pts_mult
                    if profit_pts >= stop_dist * be_trigger:
                        t.stop_price = t.entry_price
                        t.breakeven_applied = True

                if t.breakeven_applied and atr > 0:
                    trail_dist = atr * trail_mult
                    if t.direction == "BUY":
                        new_trail = close - trail_dist
                        if new_trail > t.stop_price:
                            t.stop_price = new_trail
                    else:
                        new_trail = close + trail_dist
                        if new_trail < t.stop_price:
                            t.stop_price = new_trail

                rsi_val = row.get("rsi", 50)
                adx_now = row.get("adx", 50)
                if not pd.isna(rsi_val):
                    if t.direction == "BUY" and rsi_val > strategy.rsi_overbought:
                        t.exit_price = close
                        t.exit_reason = "RSI_OB"
                        closed = True
                    elif t.direction == "SELL" and rsi_val < strategy.rsi_oversold:
                        t.exit_price = close
                        t.exit_reason = "RSI_OS"
                        closed = True

                if (
                    not closed
                    and not strategy.use_macd_exit
                    and not disable_ranging_exit
                    and not pd.isna(adx_now)
                ):
                    adx_exit = strategy.adx_threshold - 5
                    if adx_now < adx_exit:
                        t.exit_price = close
                        t.exit_reason = "Ranging"
                        closed = True
                    elif t.direction == "BUY" and htf_trend == "BEARISH":
                        t.exit_price = close
                        t.exit_reason = "HTF_flip"
                        closed = True
                    elif t.direction == "SELL" and htf_trend == "BULLISH":
                        t.exit_price = close
                        t.exit_reason = "HTF_flip"
                        closed = True

            if closed:
                t.exit_time = ts
                if t.direction == "BUY":
                    pnl_pts = (t.exit_price - t.entry_price) * pts_mult
                else:
                    pnl_pts = (t.entry_price - t.exit_price) * pts_mult
                t.pnl = round(pnl_pts * t.size, 2)
                trades.append(t)
                active = None
                if cooldown_candles_after_any_close is not None:
                    cd = cooldown_candles_after_any_close
                else:
                    cd = 12 if t.pnl < 0 else 3
                cooldown_until = i + cd
                continue

        if active is not None:
            continue
        if cooldown_until and i < cooldown_until:
            continue

        ema_fast = row.get("ema_fast", 0)
        ema_medium = row.get("ema_medium", 0)
        ema_slow_val = row.get("ema_slow", 0)
        rsi = row.get("rsi", 50)
        adx_val = row.get("adx", 0)

        if pd.isna(ema_fast) or pd.isna(adx_val) or pd.isna(atr):
            continue
        if adx_val < strategy.adx_threshold:
            continue
        if i > 0:
            prev_adx = di.iloc[i - 1].get("adx", adx_val)
            if not pd.isna(prev_adx) and adx_val < prev_adx - 0.5:
                continue

        bullish = ema_fast > ema_medium > ema_slow_val and close > ema_slow_val
        bearish = ema_fast < ema_medium < ema_slow_val and close < ema_slow_val

        signal_dir = None
        if bullish and strategy.rsi_oversold < rsi < strategy.rsi_buy_max:
            if htf_trend != "BEARISH":
                signal_dir = "BUY"
        elif bearish and strategy.rsi_sell_min < rsi < strategy.rsi_overbought:
            if htf_trend != "BULLISH":
                signal_dir = "SELL"
        if signal_dir is None:
            continue

        if market_config.min_confidence > 0.5:
            ema_sep = abs(ema_fast - ema_slow_val) / abs(ema_slow_val) if ema_slow_val != 0 else 0
            conf = min(0.25, ema_sep * 10) + min(0.15, (adx_val - 25) / 100) + 0.4
            if htf_trend == signal_dir.replace("BUY", "BULLISH").replace("SELL", "BEARISH"):
                conf += 0.15
            if conf < market_config.min_confidence:
                continue

        stop_dist_price = max(atr * stop_mult, market_config.min_stop_distance / pts_mult)
        max_stop_price = (market_config.min_stop_distance * 20) / pts_mult
        stop_dist_price = min(stop_dist_price, max_stop_price)
        limit_dist_price = stop_dist_price * rr
        if stop_dist_price <= 0:
            continue

        if signal_dir == "BUY":
            stop_price = close - stop_dist_price
            limit_price = close + limit_dist_price
        else:
            stop_price = close + stop_dist_price
            limit_price = close - limit_dist_price

        active = VTrade(
            direction=signal_dir,
            entry_price=close,
            entry_time=ts,
            stop_distance=stop_dist_price * pts_mult,
            limit_distance=limit_dist_price * pts_mult,
            size=market_config.default_size,
            stop_price=stop_price,
            limit_price=limit_price,
        )

    return trades


def summarize(trades: list[VTrade]) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "exits": {}}
    wins = [t.pnl for t in trades if t.pnl > 0.50]
    losses = [t.pnl for t in trades if t.pnl < -0.50]
    bes = [t.pnl for t in trades if -0.50 <= t.pnl <= 0.50]
    total = sum(t.pnl for t in trades)
    gw = sum(wins)
    gl = abs(sum(losses))
    pf = gw / gl if gl > 0 else float("inf")
    exits: dict[str, int] = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
    return {
        "n": len(trades),
        "w": len(wins),
        "l": len(losses),
        "be": len(bes),
        "pnl": total,
        "wr": len(wins) / len(trades) * 100,
        "pf": pf,
        "avg_win": np.mean(wins) if wins else 0.0,
        "avg_loss": np.mean(losses) if losses else 0.0,
        "exits": exits,
    }


def fetch_window(market_name: str, interval: str, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
    """Direct yfinance fetch for an arbitrary historical window (bypasses disk cache)."""
    import yfinance as yf
    from backtest import TICKER_MAP
    ticker = TICKER_MAP.get(market_name)
    if not ticker:
        return None
    data = yf.download(ticker, start=start_dt, end=end_dt, interval=interval, progress=False)
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = pd.DataFrame({
        "date": data.index,
        "open": data["Open"].values,
        "high": data["High"].values,
        "low": data["Low"].values,
        "close": data["Close"].values,
        "volume": data["Volume"].values if "Volume" in data.columns else 0,
    }).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def run_window(label: str, df: pd.DataFrame, htf_df: pd.DataFrame, strategy, market_config, pts_mult):
    htf_trends = calculate_htf_trend(htf_df) if htf_df is not None and not htf_df.empty else pd.DataFrame()

    variants = [
        ("BASELINE (1.5x stop)",      dict()),
        ("STOP_2.0x",                 dict(stop_atr_mult_override=2.0)),
        ("STOP_2.5x",                 dict(stop_atr_mult_override=2.5)),
        ("STOP_3.0x",                 dict(stop_atr_mult_override=3.0)),
        ("STOP_2.0x + R:R_2.5",       dict(stop_atr_mult_override=2.0, reward_risk_override=2.5)),
        ("STOP_2.5x + R:R_2.0",       dict(stop_atr_mult_override=2.5, reward_risk_override=2.0)),
        ("NO_RANGING_EXIT",           dict(disable_ranging_exit=True)),
        ("LOOSE_BE_TRAIL",            dict(breakeven_trigger_pct_override=1.0, atr_trail_mult_override=2.5)),
        ("NO_RANGING + LOOSE_TRAIL",  dict(disable_ranging_exit=True, breakeven_trigger_pct_override=1.0, atr_trail_mult_override=2.5)),
        ("LONG_COOLDOWN_2H",          dict(cooldown_candles_after_any_close=8)),
        ("EVERYTHING",                dict(disable_ranging_exit=True, breakeven_trigger_pct_override=1.0, atr_trail_mult_override=2.5, cooldown_candles_after_any_close=8)),
    ]

    results = []
    for name, kwargs in variants:
        trades = simulate_gold(df, htf_trends, strategy, market_config, pts_mult, **kwargs)
        s = summarize(trades)
        results.append((name, s))

    print(f"\n=== {label} ===")
    header = f"{'Variant':<26} {'N':>3} {'W/L/BE':>9} {'WR%':>6} {'P&L':>9} {'PF':>6} {'AvgW':>7} {'AvgL':>7}"
    print(header)
    print("-" * len(header))
    for name, s in results:
        if s["n"] == 0:
            print(f"{name:<26} {'0':>3}")
            continue
        print(
            f"{name:<26} {s['n']:>3} "
            f"{s['w']}/{s['l']}/{s['be']:<3}   "
            f"{s['wr']:>5.1f} "
            f"£{s['pnl']:>+7.2f} "
            f"{s['pf']:>5.2f} "
            f"£{s['avg_win']:>+5.2f} "
            f"£{s['avg_loss']:>+5.2f}"
        )
    return results


def main():
    from datetime import timedelta
    market_name = "Gold"
    market_config = next(m for m in MARKETS if m.name == market_name)
    strategy = get_strategy_for_market(market_config)
    pts_mult = IG_POINTS_MULT.get(market_name, 1)
    candle_str = f"{market_config.candle_interval}m"

    print(f"\nGold exit-logic A/B test — two 60d {candle_str} windows")
    print(f"Live config: stop {strategy.stop_atr_mult}x, ADX {strategy.adx_threshold}, BE {strategy.breakeven_trigger_pct}, "
          f"ATR trail {strategy.atr_trail_mult}x, R:R {strategy.reward_risk}")
    print("=" * 115)

    now = datetime.now()
    recent_start = now - timedelta(days=59)
    recent_end = now
    older_start = now - timedelta(days=119)
    older_end = now - timedelta(days=60)

    recent_df = fetch_window(market_name, candle_str, recent_start, recent_end)
    recent_htf = fetch_window(market_name, "1h", recent_start - timedelta(days=30), recent_end)
    older_df = fetch_window(market_name, candle_str, older_start, older_end)
    older_htf = fetch_window(market_name, "1h", older_start - timedelta(days=30), older_end)

    if recent_df is None or recent_df.empty:
        print("No recent data")
        return
    if older_df is None or older_df.empty:
        print("No older-window data (Yahoo may not retain this far back for 5m)")
        return
    print(f"Recent window: {recent_df['date'].min()} -> {recent_df['date'].max()} ({len(recent_df)} candles)")
    print(f"Older  window: {older_df['date'].min()} -> {older_df['date'].max()} ({len(older_df)} candles)")

    run_window("RECENT (0-60d ago)", recent_df, recent_htf, strategy, market_config, pts_mult)
    run_window("OLDER  (60-120d ago)", older_df, older_htf, strategy, market_config, pts_mult)
    return



if __name__ == "__main__":
    main()
