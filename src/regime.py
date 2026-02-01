"""
Market regime detection for adaptive strategy switching.

Classifies markets into regimes based on trend strength (ADX) and
volatility (ATR vs historical median). Each regime maps to different
trading rules, position sizes, and risk parameters.

Usage:
    from src.regime import classify_regime, get_regime_params

    df = add_all_indicators(price_data, STRATEGY_PARAMS)
    regime = classify_regime(df)
    params = get_regime_params(regime)
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class TrendState(Enum):
    """Market trend classification based on ADX."""
    TRENDING = "TRENDING"  # ADX >= 25: Strong directional movement
    RANGING = "RANGING"    # ADX < 25: Sideways/choppy price action


class VolatilityState(Enum):
    """Market volatility classification based on ATR vs rolling median."""
    LOW = "LOW"       # ATR < 0.8x median: Compressed, potential breakout
    NORMAL = "NORMAL" # ATR 0.8-1.3x median: Typical conditions
    HIGH = "HIGH"     # ATR > 1.3x median: Elevated risk, wider stops needed


@dataclass
class MarketRegime:
    """
    Combined market regime classification.

    Attributes:
        trend: Trending or ranging based on ADX
        volatility: Low/normal/high based on ATR percentile
        adx: Raw ADX value for reference
        atr_ratio: ATR / median ATR ratio
    """
    trend: TrendState
    volatility: VolatilityState
    adx: float
    atr_ratio: float

    @property
    def code(self) -> str:
        """Combined regime code, e.g. 'TRENDING_NORMAL' or 'RANGING_HIGH'."""
        return f"{self.trend.value}_{self.volatility.value}"

    @property
    def is_tradeable(self) -> bool:
        """Whether this regime allows any trading."""
        # RANGING + HIGH volatility is the most dangerous - skip entirely
        return not (self.trend == TrendState.RANGING and
                   self.volatility == VolatilityState.HIGH)

    @property
    def favours_trend_follow(self) -> bool:
        """Whether regime favours trend-following strategies."""
        return self.trend == TrendState.TRENDING

    @property
    def favours_mean_reversion(self) -> bool:
        """Whether regime favours mean-reversion strategies."""
        return (self.trend == TrendState.RANGING and
                self.volatility != VolatilityState.HIGH)


@dataclass
class RegimeParams:
    """
    Trading parameters adjusted for current regime.

    These multiply or override the base strategy parameters.
    """
    size_multiplier: float      # Position size scaling (1.0 = normal)
    stop_atr_multiplier: float  # Stop distance as ATR multiple
    min_confidence: float       # Minimum signal confidence to enter
    allow_trend_follow: bool    # Whether trend-following entries allowed
    allow_mean_reversion: bool  # Whether mean-reversion entries allowed
    cooldown_multiplier: float  # Cooldown period scaling after losses


# Regime-specific parameter mappings
REGIME_PARAMS = {
    # Trending + Normal volatility: Ideal conditions for trend-following
    "TRENDING_NORMAL": RegimeParams(
        size_multiplier=1.0,
        stop_atr_multiplier=1.5,
        min_confidence=0.5,
        allow_trend_follow=True,
        allow_mean_reversion=False,
        cooldown_multiplier=1.0,
    ),

    # Trending + Low volatility: Good for breakouts, can size up slightly
    "TRENDING_LOW": RegimeParams(
        size_multiplier=1.0,
        stop_atr_multiplier=1.2,  # Tighter stops in low vol
        min_confidence=0.5,
        allow_trend_follow=True,
        allow_mean_reversion=False,
        cooldown_multiplier=1.0,
    ),

    # Trending + High volatility: Trade trend but reduce size, wider stops
    "TRENDING_HIGH": RegimeParams(
        size_multiplier=0.5,
        stop_atr_multiplier=2.5,  # Much wider stops needed
        min_confidence=0.6,       # Require stronger signals
        allow_trend_follow=True,
        allow_mean_reversion=False,
        cooldown_multiplier=1.5,  # Longer cooldown after losses
    ),

    # Ranging + Normal volatility: Selective mean-reversion only
    "RANGING_NORMAL": RegimeParams(
        size_multiplier=0.5,
        stop_atr_multiplier=1.5,
        min_confidence=0.6,
        allow_trend_follow=False,  # Don't chase in ranges
        allow_mean_reversion=True,
        cooldown_multiplier=1.5,
    ),

    # Ranging + Low volatility: Compression, potential breakout coming
    "RANGING_LOW": RegimeParams(
        size_multiplier=0.75,
        stop_atr_multiplier=1.0,  # Tight stops, quick exits
        min_confidence=0.55,
        allow_trend_follow=False,
        allow_mean_reversion=True,
        cooldown_multiplier=1.0,
    ),

    # Ranging + High volatility: DANGER - chop will destroy you
    "RANGING_HIGH": RegimeParams(
        size_multiplier=0.0,  # No new positions
        stop_atr_multiplier=2.0,
        min_confidence=1.0,   # Effectively blocks all entries
        allow_trend_follow=False,
        allow_mean_reversion=False,
        cooldown_multiplier=2.0,
    ),
}


def classify_regime(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,
    atr_low_threshold: float = 0.8,
    atr_high_threshold: float = 1.3,
    atr_lookback: int = 20,
) -> MarketRegime:
    """
    Classify market regime from OHLC data with indicators.

    Args:
        df: DataFrame with 'adx' and 'atr' columns (from add_all_indicators)
        adx_threshold: ADX level separating trending/ranging (default 25)
        atr_low_threshold: ATR ratio below this = LOW volatility
        atr_high_threshold: ATR ratio above this = HIGH volatility
        atr_lookback: Periods for ATR median calculation

    Returns:
        MarketRegime with trend, volatility, and raw values

    Raises:
        ValueError: If required columns missing or insufficient data
    """
    if "adx" not in df.columns or "atr" not in df.columns:
        raise ValueError("DataFrame must have 'adx' and 'atr' columns. "
                        "Run add_all_indicators() first.")

    if len(df) < atr_lookback:
        raise ValueError(f"Need at least {atr_lookback} rows for regime "
                        f"classification, got {len(df)}")

    latest = df.iloc[-1]
    adx = float(latest["adx"])
    atr = float(latest["atr"])

    # Trend classification
    if adx >= adx_threshold:
        trend = TrendState.TRENDING
    else:
        trend = TrendState.RANGING

    # Volatility classification: compare current ATR to rolling median
    atr_median = df["atr"].rolling(atr_lookback).median().iloc[-1]

    if atr_median > 0:
        atr_ratio = atr / atr_median
    else:
        atr_ratio = 1.0  # Default to normal if no valid median

    if atr_ratio < atr_low_threshold:
        volatility = VolatilityState.LOW
    elif atr_ratio > atr_high_threshold:
        volatility = VolatilityState.HIGH
    else:
        volatility = VolatilityState.NORMAL

    return MarketRegime(
        trend=trend,
        volatility=volatility,
        adx=adx,
        atr_ratio=round(atr_ratio, 2),
    )


def get_regime_params(regime: MarketRegime) -> RegimeParams:
    """
    Get trading parameters for a given regime.

    Args:
        regime: MarketRegime from classify_regime()

    Returns:
        RegimeParams with adjusted trading parameters
    """
    return REGIME_PARAMS.get(regime.code, REGIME_PARAMS["RANGING_HIGH"])


def format_regime_status(regime: MarketRegime) -> str:
    """
    Format regime for logging/Telegram messages.

    Args:
        regime: MarketRegime to format

    Returns:
        Human-readable string like "TRENDING (ADX 32.5) | NORMAL vol (1.1x)"
    """
    trend_str = f"{regime.trend.value} (ADX {regime.adx:.1f})"
    vol_str = f"{regime.volatility.value} vol ({regime.atr_ratio:.1f}x)"
    tradeable = "OK" if regime.is_tradeable else "SKIP"

    return f"{trend_str} | {vol_str} [{tradeable}]"
