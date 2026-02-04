#!/usr/bin/env python3
"""
Accurate regime backtest - matches live bot behavior.

Key difference: regime is classified on HOURLY data, signals on 5-minute data.
This matches how main.py works (update_htf_trends uses 1H for regime).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

from src.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_adx,
    calculate_macd,
    calculate_atr,
    add_all_indicators,
)
from src.regime import (
    MarketRegime,
    TrendState,
    VolatilityState,
    REGIME_PARAMS,
    RegimeParams,
    classify_regime,
    get_regime_params,
)

logging.basicConfig(level=logging.WARNING)


TICKER_MAP = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "Dollar Index": "DX-Y.NYB",
}

DEFAULT_PARAMS = {
    "ema_fast": 9,
    "ema_medium": 21,
    "ema_slow": 50,
    "rsi_period": 7,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_buy_max": 60,
    "rsi_sell_min": 40,
    "adx_threshold": 25,
    "atr_period": 14,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
}

# Relaxed regime params - allow trend following in ranging regimes
RELAXED_REGIME_PARAMS = {
    "TRENDING_NORMAL": REGIME_PARAMS["TRENDING_NORMAL"],
    "TRENDING_LOW": REGIME_PARAMS["TRENDING_LOW"],
    "TRENDING_HIGH": REGIME_PARAMS["TRENDING_HIGH"],
    "RANGING_NORMAL": RegimeParams(
        size_multiplier=0.5,
        stop_atr_multiplier=1.5,
        min_confidence=0.55,
        allow_trend_follow=True,  # CHANGED from False
        allow_mean_reversion=True,
        cooldown_multiplier=1.5,
    ),
    "RANGING_LOW": RegimeParams(
        size_multiplier=0.75,
        stop_atr_multiplier=1.0,
        min_confidence=0.5,
        allow_trend_follow=True,  # CHANGED from False
        allow_mean_reversion=True,
        cooldown_multiplier=1.0,
    ),
    "RANGING_HIGH": REGIME_PARAMS["RANGING_HIGH"],
}


@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    market: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    stop_price: float
    limit_price: float
    pnl_percent: float = 0.0
    exit_reason: str = ""
    regime: str = ""
    confidence: float = 0.0


def fetch_data(market: str, days: int = 30, interval: str = "5m") -> Optional[pd.DataFrame]:
    """Fetch historical data from Yahoo Finance."""
    ticker = TICKER_MAP.get(market)
    if not ticker:
        return None

    try:
        if interval in ["1m", "2m", "5m", "15m", "30m"]:
            days = min(days, 60)

        end = datetime.now()
        start = end - timedelta(days=days)

        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        df = df.reset_index()
        for col in df.columns:
            if col.lower() in ("datetime", "date", "index"):
                df = df.rename(columns={col: "date"})
                break

        if "date" not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "date"})

        return df
    except Exception as e:
        print(f"Error fetching {market}: {e}")
        return None


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add technical indicators."""
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], params["ema_fast"])
    df["ema_medium"] = calculate_ema(df["close"], params["ema_medium"])
    df["ema_slow"] = calculate_ema(df["close"], params["ema_slow"])
    df["rsi"] = calculate_rsi(df["close"], params["rsi_period"])
    df["adx"] = calculate_adx(df["high"], df["low"], df["close"], period=14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], params["atr_period"])
    return df


def get_hourly_regime_at_time(
    hourly_df: pd.DataFrame,
    current_time: datetime,
) -> Optional[MarketRegime]:
    """Get market regime from hourly data at a specific point in time."""
    if hourly_df is None or hourly_df.empty:
        return None

    # Get hourly data up to current_time
    mask = hourly_df["date"] <= current_time
    if not mask.any():
        return None

    df_slice = hourly_df[mask].tail(25).copy()  # Last 25 hours
    if len(df_slice) < 20:
        return None

    try:
        return classify_regime(df_slice)
    except:
        return None


def get_htf_trend(htf_df: pd.DataFrame, current_time: datetime) -> str:
    """Get higher timeframe trend."""
    if htf_df is None or htf_df.empty:
        return "NEUTRAL"

    mask = htf_df["date"] <= current_time
    if not mask.any():
        return "NEUTRAL"

    latest = htf_df[mask].iloc[-1]
    ema_9 = latest.get("ema_9")
    ema_21 = latest.get("ema_21")
    close = latest.get("close")

    if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(close):
        return "NEUTRAL"

    if ema_9 > ema_21 and close > ema_21:
        return "BULLISH"
    elif ema_9 < ema_21 and close < ema_21:
        return "BEARISH"
    return "NEUTRAL"


def run_backtest(market: str, use_relaxed: bool = False, days: int = 30) -> tuple[list[Trade], dict]:
    """
    Run backtest matching live bot behavior.

    Returns:
        tuple of (trades, stats) where stats includes blocked signal counts
    """
    params = DEFAULT_PARAMS.copy()
    regime_params_map = RELAXED_REGIME_PARAMS if use_relaxed else REGIME_PARAMS

    # Fetch 5-minute data for signals
    df = fetch_data(market, days, "5m")
    if df is None or df.empty:
        return [], {}

    df = add_indicators(df, params)

    # Fetch hourly data for regime classification (matches live bot)
    hourly_df = fetch_data(market, days, "1h")
    if hourly_df is not None and not hourly_df.empty:
        hourly_df = add_indicators(hourly_df, params)
        hourly_df["ema_9"] = calculate_ema(hourly_df["close"], 9)
        hourly_df["ema_21"] = calculate_ema(hourly_df["close"], 21)

    trades = []
    position = None
    stats = {
        "signals_generated": 0,
        "blocked_by_regime": 0,
        "blocked_by_confidence": 0,
        "blocked_regimes": {},
        "trades_by_regime": {},
    }

    for i in range(max(params["ema_slow"], 20), len(df)):
        row = df.iloc[i]
        current_time = row["date"]
        close = row["close"]
        atr = row["atr"]

        # Exit logic
        if position:
            exit_reason = None
            exit_price = None

            if position.direction == "BUY" and close <= position.stop_price:
                exit_reason, exit_price = "Stop loss", position.stop_price
            elif position.direction == "SELL" and close >= position.stop_price:
                exit_reason, exit_price = "Stop loss", position.stop_price
            elif position.direction == "BUY" and close >= position.limit_price:
                exit_reason, exit_price = "Take profit", position.limit_price
            elif position.direction == "SELL" and close <= position.limit_price:
                exit_reason, exit_price = "Take profit", position.limit_price

            # MACD exit
            macd_hist = row["macd_hist"]
            if not exit_reason and i >= 3:
                last_3 = [df.iloc[i-j]["macd_hist"] for j in range(3)]
                if position.direction == "BUY" and all(h < 0 for h in last_3 if not pd.isna(h)):
                    exit_reason, exit_price = "MACD bearish", close
                elif position.direction == "SELL" and all(h > 0 for h in last_3 if not pd.isna(h)):
                    exit_reason, exit_price = "MACD bullish", close

            if exit_reason:
                position.exit_time = current_time
                position.exit_price = exit_price
                position.exit_reason = exit_reason
                if position.direction == "BUY":
                    position.pnl_percent = (exit_price - position.entry_price) / position.entry_price * 100
                else:
                    position.pnl_percent = (position.entry_price - exit_price) / position.entry_price * 100
                trades.append(position)
                position = None
            continue

        # Check for signal (5-minute data)
        ema_fast = row["ema_fast"]
        ema_medium = row["ema_medium"]
        ema_slow = row["ema_slow"]
        rsi = row["rsi"]
        adx = row["adx"]
        macd_hist = row["macd_hist"]

        if pd.isna(ema_slow) or pd.isna(adx) or pd.isna(rsi):
            continue

        # ADX filter on 5-minute data
        if adx < params["adx_threshold"]:
            continue

        # Get HTF trend
        htf_trend = get_htf_trend(hourly_df, current_time)

        direction = None
        confidence = 0.0

        # Bullish signal
        if (ema_fast > ema_medium > ema_slow and
            close > ema_slow and
            params["rsi_oversold"] < rsi < params["rsi_buy_max"]):

            if not pd.isna(macd_hist) and macd_hist < 0:
                continue
            if htf_trend == "BEARISH":
                continue

            confidence = min((adx - 25) / 50, 0.3)
            confidence += max(0, min((60 - rsi) / 100, 0.3))
            if htf_trend == "BULLISH":
                confidence += 0.4
            elif htf_trend == "NEUTRAL":
                confidence += 0.2
            direction = "BUY"

        # Bearish signal
        elif (ema_fast < ema_medium < ema_slow and
              close < ema_slow and
              params["rsi_sell_min"] < rsi < params["rsi_overbought"]):

            if not pd.isna(macd_hist) and macd_hist > 0:
                continue
            if htf_trend == "BULLISH":
                continue

            confidence = min((adx - 25) / 50, 0.3)
            confidence += max(0, min((rsi - 40) / 100, 0.3))
            if htf_trend == "BEARISH":
                confidence += 0.4
            elif htf_trend == "NEUTRAL":
                confidence += 0.2
            direction = "SELL"

        if not direction:
            continue

        stats["signals_generated"] += 1

        # Get regime from HOURLY data (matches live bot)
        regime = get_hourly_regime_at_time(hourly_df, current_time)
        if not regime:
            continue

        regime_p = regime_params_map.get(regime.code, regime_params_map["RANGING_HIGH"])

        # Regime filters
        if not regime.is_tradeable:
            stats["blocked_by_regime"] += 1
            stats["blocked_regimes"][regime.code] = stats["blocked_regimes"].get(regime.code, 0) + 1
            continue

        if not regime_p.allow_trend_follow:
            stats["blocked_by_regime"] += 1
            stats["blocked_regimes"][regime.code] = stats["blocked_regimes"].get(regime.code, 0) + 1
            continue

        if confidence < regime_p.min_confidence:
            stats["blocked_by_confidence"] += 1
            continue

        # Trade allowed - calculate stops
        if pd.isna(atr):
            continue

        stop_distance = atr * regime_p.stop_atr_multiplier
        limit_distance = stop_distance * params["reward_risk_ratio"]

        if direction == "BUY":
            stop_price = close - stop_distance
            limit_price = close + limit_distance
        else:
            stop_price = close + stop_distance
            limit_price = close - limit_distance

        stats["trades_by_regime"][regime.code] = stats["trades_by_regime"].get(regime.code, 0) + 1

        position = Trade(
            entry_time=current_time,
            exit_time=None,
            market=market,
            direction=direction,
            entry_price=close,
            exit_price=None,
            stop_price=stop_price,
            limit_price=limit_price,
            regime=regime.code,
            confidence=confidence,
        )

    # Close open position
    if position:
        position.exit_time = df.iloc[-1]["date"]
        position.exit_price = df.iloc[-1]["close"]
        position.exit_reason = "End of test"
        if position.direction == "BUY":
            position.pnl_percent = (position.exit_price - position.entry_price) / position.entry_price * 100
        else:
            position.pnl_percent = (position.entry_price - position.exit_price) / position.entry_price * 100
        trades.append(position)

    return trades, stats


def main():
    print("=" * 80)
    print("ACCURATE REGIME BACKTEST - Matching Live Bot Behavior")
    print("=" * 80)
    print("\nKey: Regime is classified on HOURLY data, signals on 5-minute data.")
    print("This matches how main.py uses update_htf_trends() for regime classification.\n")

    markets = ["EUR/USD", "Dollar Index", "S&P 500", "NASDAQ 100"]

    all_conservative_trades = []
    all_relaxed_trades = []
    all_conservative_stats = {}
    all_relaxed_stats = {}

    for market in markets:
        print(f"\nProcessing {market}...")

        conservative_trades, conservative_stats = run_backtest(market, use_relaxed=False, days=30)
        relaxed_trades, relaxed_stats = run_backtest(market, use_relaxed=True, days=30)

        all_conservative_trades.extend(conservative_trades)
        all_relaxed_trades.extend(relaxed_trades)

        print(f"\n{'='*60}")
        print(f"{market}")
        print(f"{'='*60}")

        # Conservative
        con_wins = sum(1 for t in conservative_trades if t.pnl_percent > 0)
        con_pnl = sum(t.pnl_percent for t in conservative_trades)
        print(f"\nCONSERVATIVE:")
        print(f"  Signals generated: {conservative_stats.get('signals_generated', 0)}")
        print(f"  Blocked by regime: {conservative_stats.get('blocked_by_regime', 0)}")
        if conservative_stats.get('blocked_regimes'):
            for regime, count in conservative_stats['blocked_regimes'].items():
                print(f"    - {regime}: {count}")
        print(f"  Blocked by confidence: {conservative_stats.get('blocked_by_confidence', 0)}")
        print(f"  Trades executed: {len(conservative_trades)}, {con_wins} wins, P&L: {con_pnl:+.2f}%")

        # Relaxed
        rel_wins = sum(1 for t in relaxed_trades if t.pnl_percent > 0)
        rel_pnl = sum(t.pnl_percent for t in relaxed_trades)
        print(f"\nRELAXED:")
        print(f"  Signals generated: {relaxed_stats.get('signals_generated', 0)}")
        print(f"  Blocked by regime: {relaxed_stats.get('blocked_by_regime', 0)}")
        if relaxed_stats.get('blocked_regimes'):
            for regime, count in relaxed_stats['blocked_regimes'].items():
                print(f"    - {regime}: {count}")
        print(f"  Blocked by confidence: {relaxed_stats.get('blocked_by_confidence', 0)}")
        print(f"  Trades executed: {len(relaxed_trades)}, {rel_wins} wins, P&L: {rel_pnl:+.2f}%")

        # Additional trades in relaxed mode
        additional = len(relaxed_trades) - len(conservative_trades)
        if additional > 0:
            # Find trades that exist in relaxed but not conservative
            extra_trades = [t for t in relaxed_trades if t.regime in ("RANGING_NORMAL", "RANGING_LOW")]
            if extra_trades:
                extra_wins = sum(1 for t in extra_trades if t.pnl_percent > 0)
                extra_pnl = sum(t.pnl_percent for t in extra_trades)
                print(f"\nADDITIONAL RANGING TRADES (relaxed only):")
                print(f"  {len(extra_trades)} trades, {extra_wins} wins, P&L: {extra_pnl:+.2f}%")

                # Show trade details
                print(f"\n  Trade details:")
                for t in extra_trades[:10]:
                    time_str = t.entry_time.strftime("%m/%d %H:%M") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
                    win_loss = "WIN" if t.pnl_percent > 0 else "LOSS"
                    print(f"    {time_str} {t.direction:4} {t.regime:<16} conf={t.confidence:.2f} {t.exit_reason:<12} {t.pnl_percent:+.2f}% [{win_loss}]")

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    con_total = len(all_conservative_trades)
    con_wins = sum(1 for t in all_conservative_trades if t.pnl_percent > 0)
    con_pnl = sum(t.pnl_percent for t in all_conservative_trades)
    con_wr = con_wins / con_total * 100 if con_total > 0 else 0

    rel_total = len(all_relaxed_trades)
    rel_wins = sum(1 for t in all_relaxed_trades if t.pnl_percent > 0)
    rel_pnl = sum(t.pnl_percent for t in all_relaxed_trades)
    rel_wr = rel_wins / rel_total * 100 if rel_total > 0 else 0

    print(f"\n{'Metric':<25} {'Conservative':>15} {'Relaxed':>15} {'Difference':>15}")
    print("-" * 70)
    print(f"{'Total Trades':<25} {con_total:>15} {rel_total:>15} {rel_total - con_total:>+15}")
    print(f"{'Winning Trades':<25} {con_wins:>15} {rel_wins:>15} {rel_wins - con_wins:>+15}")
    print(f"{'Win Rate':<25} {con_wr:>14.1f}% {rel_wr:>14.1f}% {rel_wr - con_wr:>+14.1f}%")
    print(f"{'Total P&L':<25} {con_pnl:>+14.2f}% {rel_pnl:>+14.2f}% {rel_pnl - con_pnl:>+14.2f}%")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if rel_total > con_total:
        additional_trades = rel_total - con_total
        additional_pnl = rel_pnl - con_pnl

        if rel_pnl > con_pnl:
            print(f"\n  RELAXED filter is BETTER:")
            print(f"  - {additional_trades} more trades taken")
            print(f"  - {additional_pnl:+.2f}% additional P&L")
            print(f"  - Recommend: Allow trend-following in RANGING_NORMAL/LOW regimes")
        else:
            print(f"\n  CONSERVATIVE filter is BETTER:")
            print(f"  - {additional_trades} trades blocked were mostly losers")
            print(f"  - {-additional_pnl:+.2f}% P&L preserved by blocking")
            print(f"  - Keep current conservative settings")
    else:
        print(f"\n  No difference in trade count - regime filter may not be triggering")
        print(f"  during this period. Check if hourly ADX > 25 more often than expected.")

    print()


if __name__ == "__main__":
    main()
