#!/usr/bin/env python3
"""
Backtest comparison: Conservative vs Relaxed regime filters.

Compares the current regime settings with less restrictive ones
to see what trades would have been executed and their outcomes.
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
)
from src.regime import (
    MarketRegime,
    TrendState,
    VolatilityState,
    REGIME_PARAMS,
    RegimeParams,
)

logging.basicConfig(level=logging.WARNING)


# Ticker mappings
TICKER_MAP = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "Dollar Index": "DX-Y.NYB",
}

# Less conservative regime parameters - allow trend following in ranging markets
RELAXED_REGIME_PARAMS = {
    "TRENDING_NORMAL": RegimeParams(
        size_multiplier=1.0,
        stop_atr_multiplier=1.5,
        min_confidence=0.5,
        allow_trend_follow=True,
        allow_mean_reversion=False,
        cooldown_multiplier=1.0,
    ),
    "TRENDING_LOW": RegimeParams(
        size_multiplier=1.0,
        stop_atr_multiplier=1.2,
        min_confidence=0.5,
        allow_trend_follow=True,
        allow_mean_reversion=False,
        cooldown_multiplier=1.0,
    ),
    "TRENDING_HIGH": RegimeParams(
        size_multiplier=0.5,
        stop_atr_multiplier=2.5,
        min_confidence=0.6,
        allow_trend_follow=True,
        allow_mean_reversion=False,
        cooldown_multiplier=1.5,
    ),
    # KEY CHANGE: Allow trend-following in RANGING_NORMAL with reduced size
    "RANGING_NORMAL": RegimeParams(
        size_multiplier=0.5,       # Still reduce size
        stop_atr_multiplier=1.5,
        min_confidence=0.55,       # Lower from 0.6
        allow_trend_follow=True,   # CHANGED: Was False
        allow_mean_reversion=True,
        cooldown_multiplier=1.5,
    ),
    # Also allow in RANGING_LOW
    "RANGING_LOW": RegimeParams(
        size_multiplier=0.75,
        stop_atr_multiplier=1.0,
        min_confidence=0.5,        # Lower from 0.55
        allow_trend_follow=True,   # CHANGED: Was False
        allow_mean_reversion=True,
        cooldown_multiplier=1.0,
    ),
    # Still block RANGING_HIGH - that's dangerous
    "RANGING_HIGH": RegimeParams(
        size_multiplier=0.0,
        stop_atr_multiplier=2.0,
        min_confidence=1.0,
        allow_trend_follow=False,
        allow_mean_reversion=False,
        cooldown_multiplier=2.0,
    ),
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
    blocked_by_conservative: bool = False


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

        # Handle MultiIndex columns from newer yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # Reset index to get datetime as a column
        df = df.reset_index()

        # Normalize the datetime column name (could be "Datetime", "datetime", "Date", or "index")
        for col in df.columns:
            if col.lower() in ("datetime", "date", "index"):
                df = df.rename(columns={col: "date"})
                break

        # Ensure date column exists
        if "date" not in df.columns:
            # If still no date column, the first column after reset is likely the datetime
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


def classify_regime(df: pd.DataFrame, adx_threshold: float = 25.0) -> MarketRegime:
    """Classify market regime from data."""
    latest = df.iloc[-1]
    adx = float(latest["adx"])
    atr = float(latest["atr"])

    trend = TrendState.TRENDING if adx >= adx_threshold else TrendState.RANGING

    atr_median = df["atr"].rolling(20).median().iloc[-1]
    atr_ratio = atr / atr_median if atr_median > 0 else 1.0

    if atr_ratio < 0.8:
        volatility = VolatilityState.LOW
    elif atr_ratio > 1.3:
        volatility = VolatilityState.HIGH
    else:
        volatility = VolatilityState.NORMAL

    return MarketRegime(trend=trend, volatility=volatility, adx=adx, atr_ratio=round(atr_ratio, 2))


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


def check_signal(row: pd.Series, htf_trend: str, params: dict) -> tuple[Optional[str], float]:
    """Check for entry signal."""
    ema_fast = row["ema_fast"]
    ema_medium = row["ema_medium"]
    ema_slow = row["ema_slow"]
    rsi = row["rsi"]
    adx = row["adx"]
    close = row["close"]
    macd_hist = row["macd_hist"]

    if pd.isna(ema_slow) or pd.isna(adx) or pd.isna(rsi):
        return None, 0

    if adx < params["adx_threshold"]:
        return None, 0

    # Bullish
    if (ema_fast > ema_medium > ema_slow and
        close > ema_slow and
        params["rsi_oversold"] < rsi < params["rsi_buy_max"]):

        if not pd.isna(macd_hist) and macd_hist < 0:
            return None, 0
        if htf_trend == "BEARISH":
            return None, 0

        confidence = min((adx - 25) / 50, 0.3)
        confidence += max(0, min((60 - rsi) / 100, 0.3))
        if htf_trend == "BULLISH":
            confidence += 0.4
        elif htf_trend == "NEUTRAL":
            confidence += 0.2
        return "BUY", min(confidence, 1.0)

    # Bearish
    if (ema_fast < ema_medium < ema_slow and
        close < ema_slow and
        params["rsi_sell_min"] < rsi < params["rsi_overbought"]):

        if not pd.isna(macd_hist) and macd_hist > 0:
            return None, 0
        if htf_trend == "BULLISH":
            return None, 0

        confidence = min((adx - 25) / 50, 0.3)
        confidence += max(0, min((rsi - 40) / 100, 0.3))
        if htf_trend == "BEARISH":
            confidence += 0.4
        elif htf_trend == "NEUTRAL":
            confidence += 0.2
        return "SELL", min(confidence, 1.0)

    return None, 0


def run_backtest(market: str, use_relaxed: bool = False, days: int = 30) -> list[Trade]:
    """Run backtest with specified regime params."""
    params = DEFAULT_PARAMS.copy()
    regime_params_map = RELAXED_REGIME_PARAMS if use_relaxed else REGIME_PARAMS

    df = fetch_data(market, days, "5m")
    if df is None or df.empty:
        return []

    df = add_indicators(df, params)

    htf_df = fetch_data(market, days, "1h")
    if htf_df is not None and not htf_df.empty:
        htf_df["ema_9"] = calculate_ema(htf_df["close"], 9)
        htf_df["ema_21"] = calculate_ema(htf_df["close"], 21)

    trades = []
    position = None

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

        # Entry logic
        htf_trend = get_htf_trend(htf_df, current_time)
        direction, confidence = check_signal(row, htf_trend, params)

        if not direction:
            continue

        # Regime classification
        regime_df = df.iloc[:i+1].copy()
        try:
            regime = classify_regime(regime_df)
            regime_p = regime_params_map.get(regime.code, regime_params_map["RANGING_HIGH"])
        except:
            continue

        # Regime filters
        if not regime.is_tradeable:
            continue

        if not regime_p.allow_trend_follow:
            continue

        if confidence < regime_p.min_confidence:
            continue

        # Calculate stops
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

    return trades


def print_trade_details(trades: list[Trade], label: str):
    """Print trade details."""
    if not trades:
        print(f"\n{label}: No trades")
        return

    print(f"\n{label} - Trade Details:")
    print("-" * 100)
    print(f"{'Time':<20} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'P&L':>8} {'Regime':<18} {'Exit Reason':<15}")
    print("-" * 100)

    for t in trades:
        time_str = t.entry_time.strftime("%m/%d %H:%M") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
        print(f"{time_str:<20} {t.direction:<5} {t.entry_price:>10.2f} {t.exit_price:>10.2f} {t.pnl_percent:>+7.2f}% {t.regime:<18} {t.exit_reason:<15}")


def main():
    print("=" * 80)
    print("REGIME FILTER COMPARISON: Conservative vs Relaxed")
    print("=" * 80)
    print("\nKey differences in RELAXED mode:")
    print("  - RANGING_NORMAL: allow_trend_follow = True (was False)")
    print("  - RANGING_NORMAL: min_confidence = 0.55 (was 0.6)")
    print("  - RANGING_LOW: allow_trend_follow = True (was False)")
    print("  - RANGING_LOW: min_confidence = 0.5 (was 0.55)")
    print()

    # Focus on the markets that were generating blocked signals
    markets = ["EUR/USD", "Dollar Index", "S&P 500", "NASDAQ 100"]

    all_conservative = []
    all_relaxed = []

    for market in markets:
        print(f"\nFetching data for {market}...")

        conservative_trades = run_backtest(market, use_relaxed=False, days=30)
        relaxed_trades = run_backtest(market, use_relaxed=True, days=30)

        all_conservative.extend(conservative_trades)
        all_relaxed.extend(relaxed_trades)

        # Find additional trades in relaxed mode
        additional = [t for t in relaxed_trades if t.regime in ("RANGING_NORMAL", "RANGING_LOW")]

        print(f"\n{'='*60}")
        print(f"{market}")
        print(f"{'='*60}")

        # Conservative stats
        con_wins = sum(1 for t in conservative_trades if t.pnl_percent > 0)
        con_pnl = sum(t.pnl_percent for t in conservative_trades)
        print(f"\nCONSERVATIVE: {len(conservative_trades)} trades, {con_wins} wins, {con_pnl:+.2f}% P&L")

        # Relaxed stats
        rel_wins = sum(1 for t in relaxed_trades if t.pnl_percent > 0)
        rel_pnl = sum(t.pnl_percent for t in relaxed_trades)
        print(f"RELAXED:      {len(relaxed_trades)} trades, {rel_wins} wins, {rel_pnl:+.2f}% P&L")

        # Additional trades
        if additional:
            add_wins = sum(1 for t in additional if t.pnl_percent > 0)
            add_pnl = sum(t.pnl_percent for t in additional)
            print(f"\nADDITIONAL RANGING TRADES: {len(additional)} trades, {add_wins} wins, {add_pnl:+.2f}% P&L")
            print_trade_details(additional, f"{market} - Additional Trades in Ranging Markets")

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    con_total = len(all_conservative)
    con_wins = sum(1 for t in all_conservative if t.pnl_percent > 0)
    con_pnl = sum(t.pnl_percent for t in all_conservative)
    con_wr = con_wins / con_total * 100 if con_total > 0 else 0

    rel_total = len(all_relaxed)
    rel_wins = sum(1 for t in all_relaxed if t.pnl_percent > 0)
    rel_pnl = sum(t.pnl_percent for t in all_relaxed)
    rel_wr = rel_wins / rel_total * 100 if rel_total > 0 else 0

    print(f"\n{'Metric':<25} {'Conservative':>15} {'Relaxed':>15} {'Difference':>15}")
    print("-" * 70)
    print(f"{'Total Trades':<25} {con_total:>15} {rel_total:>15} {rel_total - con_total:>+15}")
    print(f"{'Winning Trades':<25} {con_wins:>15} {rel_wins:>15} {rel_wins - con_wins:>+15}")
    print(f"{'Win Rate':<25} {con_wr:>14.1f}% {rel_wr:>14.1f}% {rel_wr - con_wr:>+14.1f}%")
    print(f"{'Total P&L':<25} {con_pnl:>+14.2f}% {rel_pnl:>+14.2f}% {rel_pnl - con_pnl:>+14.2f}%")

    # Additional trades analysis
    additional_trades = [t for t in all_relaxed if t.regime in ("RANGING_NORMAL", "RANGING_LOW")]
    if additional_trades:
        add_wins = sum(1 for t in additional_trades if t.pnl_percent > 0)
        add_pnl = sum(t.pnl_percent for t in additional_trades)
        add_wr = add_wins / len(additional_trades) * 100

        print(f"\n{'='*70}")
        print("TRADES THAT WOULD HAVE EXECUTED (blocked by conservative filter):")
        print(f"{'='*70}")
        print(f"  Trades: {len(additional_trades)}")
        print(f"  Winners: {add_wins}")
        print(f"  Win Rate: {add_wr:.1f}%")
        print(f"  P&L Impact: {add_pnl:+.2f}%")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if rel_pnl > con_pnl:
        improvement = rel_pnl - con_pnl
        print(f"\n  The RELAXED regime filter shows {improvement:+.2f}% better P&L.")
        print(f"  Consider allowing trend-following in RANGING_NORMAL/LOW regimes")
        print(f"  with reduced position sizes (0.5x-0.75x normal).")
    else:
        print(f"\n  The CONSERVATIVE regime filter performs better.")
        print(f"  Keep blocking trend-following trades in ranging markets.")

    print()


if __name__ == "__main__":
    main()
