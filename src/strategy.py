"""
Trading strategy and decision engine.
Combines technical indicators to generate trading signals.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from src.indicators import add_all_indicators
from config import STRATEGY_PARAMS, MarketConfig

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
    ) -> TradeSignal:
        """
        Analyze market data and generate a trading signal.

        Args:
            df: DataFrame with OHLCV data
            market: Market configuration
            current_price: Current market price

        Returns:
            TradeSignal with recommendation
        """
        # Add indicators
        df = add_all_indicators(df, self.params)

        if len(df) < self.params["ema_slow"]:
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
        close = latest["close"]

        rsi_overbought = self.params.get("rsi_overbought", 70)
        rsi_oversold = self.params.get("rsi_oversold", 30)

        # Calculate dynamic stop/limit based on ATR
        atr_multiplier = 2.0
        stop_distance = max(atr * atr_multiplier, market.min_stop_distance)
        limit_distance = stop_distance * 1.5  # 1.5:1 reward/risk ratio

        # Check for bullish setup
        bullish_ema = ema_fast > ema_medium > ema_slow
        price_above_ema = close > ema_slow
        rsi_not_overbought = rsi < rsi_overbought

        # Check for bearish setup
        bearish_ema = ema_fast < ema_medium < ema_slow
        price_below_ema = close < ema_slow
        rsi_not_oversold = rsi > rsi_oversold

        # Generate signal
        if bullish_ema and price_above_ema and rsi_not_overbought:
            confidence = self._calculate_confidence(
                df, "bullish", rsi, rsi_overbought
            )
            return TradeSignal(
                signal=Signal.BUY,
                epic=market.epic,
                market_name=market.name,
                confidence=confidence,
                entry_price=current_price,
                stop_distance=round(stop_distance, 2),
                limit_distance=round(limit_distance, 2),
                reason=f"Bullish EMA alignment, RSI={rsi:.1f}",
            )

        elif bearish_ema and price_below_ema and rsi_not_oversold:
            confidence = self._calculate_confidence(
                df, "bearish", rsi, rsi_oversold
            )
            return TradeSignal(
                signal=Signal.SELL,
                epic=market.epic,
                market_name=market.name,
                confidence=confidence,
                entry_price=current_price,
                stop_distance=round(stop_distance, 2),
                limit_distance=round(limit_distance, 2),
                reason=f"Bearish EMA alignment, RSI={rsi:.1f}",
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
                    rsi_overbought, rsi_oversold
                ),
            )

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        direction: str,
        rsi: float,
        threshold: float,
    ) -> float:
        """
        Calculate confidence score for a signal (0-1).

        Factors:
        - EMA separation (stronger trend = higher confidence)
        - RSI distance from threshold
        - MACD confirmation
        """
        latest = df.iloc[-1]

        # EMA separation factor (0-0.4)
        ema_fast = latest["ema_fast"]
        ema_slow = latest["ema_slow"]
        ema_separation = abs(ema_fast - ema_slow) / ema_slow
        ema_factor = min(ema_separation * 10, 0.4)

        # RSI factor (0-0.3)
        if direction == "bullish":
            rsi_distance = (threshold - rsi) / threshold
        else:
            rsi_distance = (rsi - threshold) / (100 - threshold)
        rsi_factor = max(0, min(rsi_distance, 0.3))

        # MACD confirmation factor (0-0.3)
        macd_hist = latest["macd_hist"]
        if direction == "bullish" and macd_hist > 0:
            macd_factor = 0.3
        elif direction == "bearish" and macd_hist < 0:
            macd_factor = 0.3
        else:
            macd_factor = 0.0

        confidence = ema_factor + rsi_factor + macd_factor
        return round(min(confidence, 1.0), 2)

    def _get_hold_reason(
        self,
        bullish_ema: bool,
        bearish_ema: bool,
        rsi: float,
        rsi_overbought: float,
        rsi_oversold: float,
    ) -> str:
        """Generate explanation for HOLD signal."""
        reasons = []

        if not bullish_ema and not bearish_ema:
            reasons.append("EMAs not aligned")

        if rsi >= rsi_overbought:
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi <= rsi_oversold:
            reasons.append(f"RSI oversold ({rsi:.1f})")

        if not reasons:
            reasons.append("No clear signal")

        return ", ".join(reasons)


def should_close_position(
    df: pd.DataFrame,
    direction: str,
    params: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Check if an existing position should be closed.

    Requires 3 consecutive candles with MACD histogram against the
    position direction to confirm momentum has truly shifted,
    avoiding premature exits on noise.

    Args:
        df: DataFrame with indicators
        direction: Current position direction ("BUY" or "SELL")
        params: Strategy parameters

    Returns:
        Tuple of (should_close, reason)
    """
    params = params or STRATEGY_PARAMS
    df = add_all_indicators(df, params)

    if len(df) < 3:
        return False, ""

    latest = df.iloc[-1]
    rsi = latest["rsi"]

    rsi_overbought = params.get("rsi_overbought", 70)
    rsi_oversold = params.get("rsi_oversold", 30)

    # Check last 3 candles for sustained MACD histogram against position
    last_3_macd = [df.iloc[-i]["macd_hist"] for i in range(1, 4)]

    if direction == "BUY":
        # Close long if RSI overbought
        if rsi > rsi_overbought:
            return True, f"RSI overbought ({rsi:.1f})"
        # Close long if MACD histogram negative for 3 consecutive candles
        if all(h < 0 for h in last_3_macd):
            return True, "MACD histogram negative for 3 candles"

    elif direction == "SELL":
        # Close short if RSI oversold
        if rsi < rsi_oversold:
            return True, f"RSI oversold ({rsi:.1f})"
        # Close short if MACD histogram positive for 3 consecutive candles
        if all(h > 0 for h in last_3_macd):
            return True, "MACD histogram positive for 3 candles"

    return False, ""
