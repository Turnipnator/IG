#!/usr/bin/env python3
"""
Debug script to analyze what signals are being generated and what regimes
are blocking them. This helps understand why conservative vs relaxed
filters show identical results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
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


def analyze_blocked_signals(market: str, days: int = 30):
    """Analyze signals that would be blocked by regime filter."""
    params = DEFAULT_PARAMS.copy()

    print(f"\nFetching data for {market}...")
    df = fetch_data(market, days, "5m")
    if df is None or df.empty:
        print(f"  No data for {market}")
        return

    df = add_indicators(df, params)

    htf_df = fetch_data(market, days, "1h")
    if htf_df is not None and not htf_df.empty:
        htf_df["ema_9"] = calculate_ema(htf_df["close"], 9)
        htf_df["ema_21"] = calculate_ema(htf_df["close"], 21)

    blocked_signals = []
    regime_counts = {}
    signals_by_regime = {}

    for i in range(max(params["ema_slow"], 20), len(df)):
        row = df.iloc[i]
        current_time = row["date"]
        close = row["close"]

        ema_fast = row["ema_fast"]
        ema_medium = row["ema_medium"]
        ema_slow = row["ema_slow"]
        rsi = row["rsi"]
        adx = row["adx"]
        macd_hist = row["macd_hist"]

        if pd.isna(ema_slow) or pd.isna(adx) or pd.isna(rsi):
            continue

        # Check regime
        regime_df = df.iloc[:i+1].copy()
        try:
            regime = classify_regime(regime_df)
            regime_params = REGIME_PARAMS.get(regime.code, REGIME_PARAMS["RANGING_HIGH"])
        except:
            continue

        regime_counts[regime.code] = regime_counts.get(regime.code, 0) + 1

        # Skip ADX filter to see what signals exist
        # (normally would filter here, but we want to see blocked signals)

        direction = None
        confidence = 0.0
        reason = ""

        # Get HTF trend
        htf_trend = get_htf_trend(htf_df, current_time)

        # Check bullish
        if (ema_fast > ema_medium > ema_slow and
            close > ema_slow and
            params["rsi_oversold"] < rsi < params["rsi_buy_max"]):

            if not pd.isna(macd_hist) and macd_hist < 0:
                continue
            if htf_trend == "BEARISH":
                continue

            confidence = min((adx - 25) / 50, 0.3) if adx >= 25 else 0
            confidence += max(0, min((60 - rsi) / 100, 0.3))
            if htf_trend == "BULLISH":
                confidence += 0.4
            elif htf_trend == "NEUTRAL":
                confidence += 0.2
            direction = "BUY"

        # Check bearish
        elif (ema_fast < ema_medium < ema_slow and
              close < ema_slow and
              params["rsi_sell_min"] < rsi < params["rsi_overbought"]):

            if not pd.isna(macd_hist) and macd_hist > 0:
                continue
            if htf_trend == "BULLISH":
                continue

            confidence = min((adx - 25) / 50, 0.3) if adx >= 25 else 0
            confidence += max(0, min((rsi - 40) / 100, 0.3))
            if htf_trend == "BEARISH":
                confidence += 0.4
            elif htf_trend == "NEUTRAL":
                confidence += 0.2
            direction = "SELL"

        if direction:
            key = f"{regime.code}_{direction}"
            if key not in signals_by_regime:
                signals_by_regime[key] = []

            # Check if would be blocked
            blocked = False
            block_reason = ""

            # ADX filter
            if adx < params["adx_threshold"]:
                blocked = True
                block_reason = f"ADX {adx:.1f} < 25"

            # Regime tradeable check
            elif not regime.is_tradeable:
                blocked = True
                block_reason = f"Regime {regime.code} not tradeable"

            # Regime trend-follow filter
            elif not regime_params.allow_trend_follow:
                blocked = True
                block_reason = f"Trend-follow blocked in {regime.code}"

            # Confidence filter
            elif confidence < regime_params.min_confidence:
                blocked = True
                block_reason = f"Confidence {confidence:.2f} < {regime_params.min_confidence}"

            signals_by_regime[key].append({
                "time": current_time,
                "direction": direction,
                "regime": regime.code,
                "adx": adx,
                "confidence": confidence,
                "blocked": blocked,
                "block_reason": block_reason,
            })

    # Print analysis
    print(f"\n{'='*70}")
    print(f"SIGNAL ANALYSIS: {market}")
    print(f"{'='*70}")

    print(f"\nRegime Distribution (candles):")
    total_candles = sum(regime_counts.values())
    for regime, count in sorted(regime_counts.items()):
        pct = count / total_candles * 100
        print(f"  {regime:<20}: {count:>6} ({pct:>5.1f}%)")

    print(f"\nSignals by Regime and Direction:")
    print("-" * 70)

    for key in sorted(signals_by_regime.keys()):
        signals = signals_by_regime[key]
        total = len(signals)
        blocked = sum(1 for s in signals if s["blocked"])
        allowed = total - blocked

        print(f"\n{key}: {total} signals ({allowed} allowed, {blocked} blocked)")

        # Show block reasons
        block_reasons = {}
        for s in signals:
            if s["blocked"]:
                reason = s["block_reason"]
                block_reasons[reason] = block_reasons.get(reason, 0) + 1

        if block_reasons:
            print("  Block reasons:")
            for reason, count in sorted(block_reasons.items(), key=lambda x: -x[1]):
                print(f"    - {reason}: {count}")

        # Show sample of blocked signals
        blocked_signals = [s for s in signals if s["blocked"] and "Trend-follow" in s.get("block_reason", "")]
        if blocked_signals[:3]:
            print("  Sample blocked trend-follow signals:")
            for s in blocked_signals[:3]:
                time_str = s["time"].strftime("%m/%d %H:%M") if hasattr(s["time"], 'strftime') else str(s["time"])[:16]
                print(f"    {time_str} - {s['direction']} ADX={s['adx']:.1f} conf={s['confidence']:.2f}")


def main():
    print("=" * 70)
    print("REGIME BLOCKING ANALYSIS")
    print("=" * 70)
    print("\nThis shows what signals exist and why they're blocked.")
    print("Focus on 'Trend-follow blocked' in RANGING_NORMAL/LOW regimes.\n")

    # Analyze the markets that showed blocked signals in the live logs
    for market in ["EUR/USD", "Dollar Index", "S&P 500", "NASDAQ 100"]:
        analyze_blocked_signals(market, days=30)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nIf no trend-follow signals are blocked in RANGING regimes,")
    print("it means the ADX filter is catching them first (ADX < 25).")
    print("The regime filter only matters when ADX >= 25 but market is ranging.")


if __name__ == "__main__":
    main()
