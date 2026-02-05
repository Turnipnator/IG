"""
Trading strategy and decision engine.
Combines technical indicators to generate trading signals.

Supports multiple strategy profiles:
- "default" (Big Winners): High R:R, no MACD exit - for Gold, Forex, Crude Oil
- "indices" (Momentum): Fast EMAs, MACD exit ON - for S&P 500, NASDAQ 100
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from src.indicators import add_all_indicators
from config import STRATEGY_PARAMS, MarketConfig, StrategyConfig, get_strategy_for_market

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Complete trade signal with metadata."""
    signal: Signal
    epic: str
    market_name: str
    confidence: float  # 0-1 score
    entry_price: float
    stop_distance: float
    limit_distance: float
    reason: str


class TradingStrategy:
    """
    EMA Crossover + RSI Strategy (v1).

    Buy conditions:
    - Fast EMA > Medium EMA > Slow EMA (bullish alignment)
    - Price above slow EMA
    - RSI below overbought level (< 70)

    Sell (short) conditions:
    - Fast EMA < Medium EMA < Slow EMA (bearish alignment)
    - Price below slow EMA
    - RSI above oversold level (> 30)
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or STRATEGY_PARAMS

    def analyze(
        self,
        df: pd.DataFrame,
        market: MarketConfig,
        current_price: float,
        htf_trend: str = "NEUTRAL",
    ) -> TradeSignal:
        """
        Analyze market data and generate a trading signal.

        Uses market-specific strategy configuration for different approaches:
        - Indices (S&P 500, NASDAQ): Fast momentum strategy with MACD exits
        - Others (Gold, Forex): Big Winners strategy with high R:R

        Args:
            df: DataFrame with OHLCV data
            market: Market configuration
            current_price: Current market price
            htf_trend: Higher timeframe trend ("BULLISH", "BEARISH", "NEUTRAL")

        Returns:
            TradeSignal with recommendation
        """
        # Get market-specific strategy configuration
        strategy = get_strategy_for_market(market)

        # Build params dict for indicators (merge strategy config)
        params = {
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

        # Add indicators with strategy-specific parameters
        df = add_all_indicators(df, params)

        if len(df) < strategy.ema_slow:
            return TradeSignal(
                signal=Signal.HOLD,
                epic=market.epic,
                market_name=market.name,
                confidence=0.0,
                entry_price=current_price,
                stop_distance=market.min_stop_distance,
                limit_distance=market.min_stop_distance,
                reason="Insufficient data for analysis",
            )

        latest = df.iloc[-1]

        ema_fast = latest["ema_fast"]
        ema_medium = latest["ema_medium"]
        ema_slow = latest["ema_slow"]
        rsi = latest["rsi"]
        atr = latest["atr"]
        adx = latest["adx"]
        close = latest["close"]

        rsi_overbought = strategy.rsi_overbought
        rsi_oversold = strategy.rsi_oversold
        adx_threshold = strategy.adx_threshold

        # ADX filter: skip if market is ranging (no clear trend)
        if adx < adx_threshold:
            return TradeSignal(
                signal=Signal.HOLD,
                epic=market.epic,
                market_name=market.name,
                confidence=0.0,
                entry_price=current_price,
                stop_distance=market.min_stop_distance,
                limit_distance=market.min_stop_distance,
                reason=f"ADX too low ({adx:.1f} < {adx_threshold}), market ranging",
            )

        # ADX direction filter: skip if trend is weakening (ADX declining)
        prev_adx = df.iloc[-2]["adx"] if len(df) >= 2 else adx
        adx_declining = adx < prev_adx - 0.5  # Allow small fluctuations (0.5 tolerance)
        if adx_declining:
            return TradeSignal(
                signal=Signal.HOLD,
                epic=market.epic,
                market_name=market.name,
                confidence=0.0,
                entry_price=current_price,
                stop_distance=market.min_stop_distance,
                limit_distance=market.min_stop_distance,
                reason=f"ADX declining ({prev_adx:.1f} → {adx:.1f}), trend weakening",
            )

        # Calculate dynamic stop/limit based on ATR and strategy R:R
        stop_distance = max(atr * strategy.stop_atr_mult, market.min_stop_distance)
        limit_distance = stop_distance * strategy.reward_risk  # Strategy-specific R:R

        # RSI entry ranges from strategy config
        rsi_buy_max = strategy.rsi_buy_max
        rsi_sell_min = strategy.rsi_sell_min

        # Check for bullish setup
        bullish_ema = ema_fast > ema_medium > ema_slow
        price_above_ema = close > ema_slow
        rsi_buy_valid = rsi_oversold < rsi < rsi_buy_max

        # Check for bearish setup
        bearish_ema = ema_fast < ema_medium < ema_slow
        price_below_ema = close < ema_slow
        rsi_sell_valid = rsi_sell_min < rsi < rsi_overbought

        # Pullback filter: price must be near fast EMA (not extended)
        # For BUY: price should have dipped toward fast EMA (not too far above it)
        # For SELL: price should have bounced toward fast EMA (not too far below it)
        pullback_threshold = strategy.pullback_pct / 100
        price_distance_pct = (close - ema_fast) / ema_fast
        buy_pullback_valid = price_distance_pct <= pullback_threshold  # Price near/below fast EMA
        sell_pullback_valid = price_distance_pct >= -pullback_threshold  # Price near/above fast EMA

        # MACD pre-check: don't enter if exit condition is already true
        # This prevents opening and immediately closing (losing the spread)
        last_3_macd = [df.iloc[-i]["macd_hist"] for i in range(1, 4)] if len(df) >= 4 else [0]
        macd_already_bearish = all(h < 0 for h in last_3_macd)
        macd_already_bullish = all(h > 0 for h in last_3_macd)

        # Generate signal with multi-timeframe confirmation
        if bullish_ema and price_above_ema and rsi_buy_valid:
            # Pullback filter: price must be near fast EMA (not overextended)
            if not buy_pullback_valid:
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"Price too far from EMA ({price_distance_pct*100:.2f}% > {strategy.pullback_pct}%), wait for pullback",
                )

            # MACD pre-check: only if strategy uses MACD exit
            # Don't buy if exit would trigger immediately
            if strategy.use_macd_exit and macd_already_bearish:
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"MACD already bearish (would exit immediately)",
                )

            # Multi-timeframe filter: check based on strategy requirement
            if strategy.require_htf and htf_trend != "BULLISH":
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"HTF not aligned for BUY (HTF={htf_trend}, need BULLISH)",
                )

            # Even if HTF not required, never trade against it
            if htf_trend == "BEARISH":
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"HTF opposing (BEARISH) - don't BUY against trend",
                )

            confidence = self._calculate_confidence(
                df, "bullish", rsi, rsi_overbought, adx, htf_trend
            )
            return TradeSignal(
                signal=Signal.BUY,
                epic=market.epic,
                market_name=market.name,
                confidence=confidence,
                entry_price=current_price,
                stop_distance=round(stop_distance, 2),
                limit_distance=round(limit_distance, 2),
                reason=f"Bullish EMA alignment, RSI={rsi:.1f}, ADX={adx:.1f}, HTF={htf_trend}",
            )

        elif bearish_ema and price_below_ema and rsi_sell_valid:
            # Pullback filter: price must be near fast EMA (not overextended)
            if not sell_pullback_valid:
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"Price too far from EMA ({price_distance_pct*100:.2f}% < -{strategy.pullback_pct}%), wait for bounce",
                )

            # MACD pre-check: only if strategy uses MACD exit
            if strategy.use_macd_exit and macd_already_bullish:
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"MACD already bullish (would exit immediately)",
                )

            # Multi-timeframe filter: check based on strategy requirement
            if strategy.require_htf and htf_trend != "BEARISH":
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"HTF not aligned for SELL (HTF={htf_trend}, need BEARISH)",
                )

            # Even if HTF not required, never trade against it
            if htf_trend == "BULLISH":
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=current_price,
                    stop_distance=round(stop_distance, 2),
                    limit_distance=round(limit_distance, 2),
                    reason=f"HTF opposing (BULLISH) - don't SELL against trend",
                )

            confidence = self._calculate_confidence(
                df, "bearish", rsi, rsi_oversold, adx, htf_trend
            )
            return TradeSignal(
                signal=Signal.SELL,
                epic=market.epic,
                market_name=market.name,
                confidence=confidence,
                entry_price=current_price,
                stop_distance=round(stop_distance, 2),
                limit_distance=round(limit_distance, 2),
                reason=f"Bearish EMA alignment, RSI={rsi:.1f}, ADX={adx:.1f}, HTF={htf_trend}",
            )

        else:
            return TradeSignal(
                signal=Signal.HOLD,
                epic=market.epic,
                market_name=market.name,
                confidence=0.0,
                entry_price=current_price,
                stop_distance=round(stop_distance, 2),
                limit_distance=round(limit_distance, 2),
                reason=self._get_hold_reason(
                    bullish_ema, bearish_ema, rsi,
                    rsi_overbought, rsi_oversold, adx, adx_threshold
                ),
            )

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        direction: str,
        rsi: float,
        threshold: float,
        adx: float = 25.0,
        htf_trend: str = "NEUTRAL",
    ) -> float:
        """
        Calculate confidence score for a signal (0-1).

        Factors:
        - EMA separation (stronger trend = higher confidence)
        - RSI distance from threshold
        - MACD confirmation
        - ADX trend strength
        - Higher timeframe alignment
        """
        latest = df.iloc[-1]

        # EMA separation factor (0-0.25)
        ema_fast = latest["ema_fast"]
        ema_slow = latest["ema_slow"]
        ema_separation = abs(ema_fast - ema_slow) / ema_slow
        ema_factor = min(ema_separation * 10, 0.25)

        # RSI factor (0-0.2)
        if direction == "bullish":
            rsi_distance = (threshold - rsi) / threshold
        else:
            rsi_distance = (rsi - threshold) / (100 - threshold)
        rsi_factor = max(0, min(rsi_distance, 0.2))

        # MACD confirmation factor (0-0.2)
        macd_hist = latest["macd_hist"]
        if direction == "bullish" and macd_hist > 0:
            macd_factor = 0.2
        elif direction == "bearish" and macd_hist < 0:
            macd_factor = 0.2
        else:
            macd_factor = 0.0

        # ADX strength factor (0-0.15)
        # Stronger trend = higher confidence
        adx_factor = min((adx - 25) / 50, 0.15) if adx > 25 else 0.0

        # Higher timeframe alignment factor (0-0.2)
        if (direction == "bullish" and htf_trend == "BULLISH") or \
           (direction == "bearish" and htf_trend == "BEARISH"):
            htf_factor = 0.2
        elif htf_trend == "NEUTRAL":
            htf_factor = 0.1
        else:
            htf_factor = 0.0

        confidence = ema_factor + rsi_factor + macd_factor + adx_factor + htf_factor
        return round(min(confidence, 1.0), 2)

    def _get_hold_reason(
        self,
        bullish_ema: bool,
        bearish_ema: bool,
        rsi: float,
        rsi_overbought: float,
        rsi_oversold: float,
        adx: float = 0.0,
        adx_threshold: float = 25.0,
    ) -> str:
        """Generate explanation for HOLD signal."""
        reasons = []

        if not bullish_ema and not bearish_ema:
            reasons.append("EMAs not aligned")

        if adx < adx_threshold:
            reasons.append(f"ADX weak ({adx:.1f})")

        rsi_buy_max = self.params.get("rsi_buy_max", 60)
        rsi_sell_min = self.params.get("rsi_sell_min", 40)

        if rsi >= rsi_overbought:
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi <= rsi_oversold:
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif bullish_ema and rsi >= rsi_buy_max:
            reasons.append(f"RSI too extended for buy ({rsi:.1f})")
        elif bearish_ema and rsi <= rsi_sell_min:
            reasons.append(f"RSI too exhausted for sell ({rsi:.1f})")

        if not reasons:
            reasons.append("No clear signal")

        return ", ".join(reasons)


def should_close_position(
    df: pd.DataFrame,
    direction: str,
    params: Optional[dict] = None,
    market: Optional[MarketConfig] = None,
    htf_trend: str = "NEUTRAL",
) -> tuple[bool, str]:
    """
    Check if an existing position should be closed.

    Exit conditions depend on market's strategy:
    - Indices (Momentum): Use MACD exit after 3 consecutive opposite bars
    - Others (Big Winners): Use ADX/HTF dynamic exit - close if conditions deteriorate

    Args:
        df: DataFrame with indicators
        direction: Current position direction ("BUY" or "SELL")
        params: Strategy parameters (legacy, prefer market config)
        market: Market configuration (used to get strategy-specific settings)
        htf_trend: Current higher timeframe trend ("BULLISH", "BEARISH", "NEUTRAL")

    Returns:
        Tuple of (should_close, reason)
    """
    # Get strategy config if market provided
    if market:
        strategy = get_strategy_for_market(market)
        use_macd_exit = strategy.use_macd_exit
        rsi_overbought = strategy.rsi_overbought
        rsi_oversold = strategy.rsi_oversold
        adx_threshold = strategy.adx_threshold
        indicator_params = {
            "ema_fast": strategy.ema_fast,
            "ema_medium": strategy.ema_medium,
            "ema_slow": strategy.ema_slow,
            "rsi_period": strategy.rsi_period,
        }
    else:
        # Legacy fallback
        params = params or STRATEGY_PARAMS
        use_macd_exit = True  # Default to True for backward compatibility
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        adx_threshold = params.get("adx_threshold", 25)
        indicator_params = params

    df = add_all_indicators(df, indicator_params)

    if len(df) < 3:
        return False, ""

    latest = df.iloc[-1]
    rsi = latest["rsi"]
    adx = latest["adx"]

    # RSI extreme exit (always active - protects against overextended moves)
    if direction == "BUY" and rsi > rsi_overbought:
        return True, f"RSI overbought ({rsi:.1f})"
    if direction == "SELL" and rsi < rsi_oversold:
        return True, f"RSI oversold ({rsi:.1f})"

    # MACD exit only if strategy uses it (indices)
    if use_macd_exit:
        last_3_macd = [df.iloc[-i]["macd_hist"] for i in range(1, 4)]

        if direction == "BUY":
            if all(h < 0 for h in last_3_macd):
                return True, "MACD histogram negative for 3 candles"
        elif direction == "SELL":
            if all(h > 0 for h in last_3_macd):
                return True, "MACD histogram positive for 3 candles"

    # Dynamic exit for non-MACD strategies (Gold, Forex, etc.)
    # These strategies need protection when market conditions change
    if not use_macd_exit:
        # ADX ranging exit: close if market has gone ranging
        # Use a slightly lower threshold to avoid premature exits
        adx_exit_threshold = adx_threshold - 3  # e.g., 25 -> 22
        if adx < adx_exit_threshold:
            return True, f"Market turned ranging (ADX {adx:.1f} < {adx_exit_threshold})"

        # HTF reversal exit: close if higher timeframe trend reversed against us
        if direction == "BUY" and htf_trend == "BEARISH":
            return True, f"HTF trend reversed to BEARISH"
        if direction == "SELL" and htf_trend == "BULLISH":
            return True, f"HTF trend reversed to BULLISH"

    return False, ""
