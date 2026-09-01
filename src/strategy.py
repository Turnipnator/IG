"""
Trading strategy and decision engine.
Combines technical indicators to generate trading signals.

Supports multiple strategy profiles:
- "default" (Big Winners): High R:R, no MACD exit - for Gold, Forex, Crude Oil
- "indices" (Momentum): Fast EMAs, MACD exit ON - for S&P 500, NASDAQ 100
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from src.indicators import add_all_indicators

# Bars each indicator needs before it stops being NaN. add_all_indicators calls
# calculate_adx/calculate_atr with their DEFAULT period (14) — not a strategy
# param — so these are fixed for every profile. ADX double-smooths (dx is NaN for
# the first period, then ewm(min_periods=period) over that), hence 2x. MACD
# histogram needs the slow EMA plus the signal EMA.
ADX_WARMUP_BARS = 2 * 14
MACD_HIST_WARMUP_BARS = 26 + 9

# Consecutive opposing MACD-histogram bars required to close a momentum position.
# 2026-09-01: 3 -> 5. The 3-bar window was a tail-clipper of the same class as the
# ADX-ceiling / leg-filter / swing-proximity / breakout-profit-protection ideas this
# repo has refuted before: it fires on noise and truncates the winners the edge lives
# on. Measured on 80d of IG-native 5m archive at each market's REAL trading-hours
# spread, entry held fixed and only this value varied:
#   full sample PF   S&P 0.18->0.54   NASDAQ 0.75->0.91   Japan 1.11->1.27   HK 0.72->0.85
#   walk-forward     N=5 beat N=3 in 8/8 disjoint halves and 10/12 disjoint time-folds
# Independently corroborated by the 2026-06-25 Crude exit study (macd5 > macd3 at every
# cost level). Max risk/trade is UNCHANGED — the broker stop is untouched; what changes
# is the distribution (more full-stop losses, more take-profits), the same trade-off
# accepted for the breakout trail in 0057aa9.
# CAVEAT recorded so it is not lost: this does NOT rescue any market. Japan 225 is the
# only one profitable in both halves and it was profitable at N=3 too; S&P stays deeply
# unprofitable and NASDAQ flips by regime at BOTH settings. All folds sit inside one
# 80-day window, so a different regime is untested. See research_notes.md 2026-08-31(e).
MACD_EXIT_BARS = 5
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
    # Leg-size (exhaustion) filter annotations. leg_atr is the in-direction
    # leg size / ATR over the configured lookback; leg_would_block is True when
    # that exceeds the market's threshold. Populated only for markets that opt
    # in (MarketConfig.leg_filter_lookback > 0); observational unless enforced.
    leg_atr: float = 0.0
    leg_would_block: bool = False
    # ADX-ceiling (exhaustion) annotation. True when this entry's ADX exceeds
    # the market's adx_ceiling — observational unless adx_ceiling_enforce.
    adx_ceiling_would_block: bool = False
    # Entry indicator snapshot (computed inside analyze() on the indicator-laden
    # copy of df). Carried here because main.py's df has no indicator columns,
    # so the journal can record real entry values instead of 0.0.
    adx: float = 0.0
    rsi: float = 0.0
    atr: float = 0.0
    ema_fast: float = 0.0
    ema_medium: float = 0.0
    ema_slow: float = 0.0
    # Donchian breakout only: the CHANNEL BOUNDARY the break crossed (analyze_breakout's
    # `level`). Previously this survived only inside `reason` as free text, so nothing
    # downstream could measure the gap between the level and the price we actually paid
    # — the +0.143R/trade entry slip. Journalling it makes that slip queryable.
    # 0.0 for every non-breakout signal.
    break_level: float = 0.0


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

        # NaN gate. This MUST come before every other guard below, because each of
        # them is a comparison and every comparison against NaN is False — so a NaN
        # does not trip a guard, it walks through all of them: `atr > max_sane` and
        # `atr <= 0` are both False, `adx < adx_threshold` is False (reads as "trend
        # is strong"), `stop_distance > max_stop` is False (no cap).
        #
        # The source is warm-up, not corruption. calculate_adx double-smooths and is
        # NaN until ~2*period bars (first value at n=27 for period 14), but the
        # length gate above only tests strategy.ema_slow — which is 21 for gold and
        # 26 for all five indices_* profiles, i.e. the live momentum book. 26 bars of
        # ordinary trending S&P data cleared that gate, skipped the ADX filter
        # entirely, and returned BUY at 61% confidence whose own reason string read
        # "ADX=nan", with a finite stop that would have filled normally.
        #
        # Streaming callers are covered by their own len(df) < 50 check; the polling
        # path had none. Checked here rather than filled in indicators.py because NaN
        # is the honest value for an undefined indicator, and test_golden pins the
        # indicator output.
        indicator_values = {
            "ema_fast": ema_fast, "ema_medium": ema_medium, "ema_slow": ema_slow,
            "rsi": rsi, "atr": atr, "adx": adx, "close": close,
        }
        not_finite = [k for k, v in indicator_values.items() if not math.isfinite(v)]
        if not_finite:
            logger.warning(
                f"[{market.name}] Indicators not finite ({', '.join(not_finite)}) on "
                f"{len(df)} bars — holding. ADX needs {ADX_WARMUP_BARS} bars; the "
                f"length gate above only tests ema_slow={strategy.ema_slow}."
            )
            return TradeSignal(
                signal=Signal.HOLD,
                epic=market.epic,
                market_name=market.name,
                confidence=0.0,
                entry_price=current_price,
                stop_distance=market.min_stop_distance,
                limit_distance=market.min_stop_distance,
                reason=f"Indicators not finite: {', '.join(not_finite)}",
            )

        # Sanity check: detect corrupted/stale streaming data.
        # Real corruption shows up as an ATR many multiples of price (the
        # billion-point stops). Legitimate intraday ATR tops out near ~1.4% of
        # price, so a 5%-of-price ceiling never trips on real volatility while
        # still catching corruption. The min_stop*50 term keeps a floor for
        # low-priced instruments; the price term is what saves high-priced
        # indices with tiny min_stops — e.g. Germany 40 (min_stop 2 -> 100)
        # whose true 1h ATR is ~90-130 was being false-flagged as corrupt.
        # ADX > 80 alone is legitimate in strong trends, so only flag it
        # alongside bad ATR.
        max_sane_atr = max(market.min_stop_distance * 50, close * 0.05)
        if atr > max_sane_atr or atr <= 0:
            logger.warning(
                f"[{market.name}] Corrupted indicator data detected — "
                f"ADX={adx:.1f}, ATR={atr:.2f} (max sane={max_sane_atr:.1f}). Skipping."
            )
            return TradeSignal(
                signal=Signal.HOLD,
                epic=market.epic,
                market_name=market.name,
                confidence=0.0,
                entry_price=current_price,
                stop_distance=market.min_stop_distance,
                limit_distance=market.min_stop_distance,
                reason=f"Corrupted data: ADX={adx:.1f}, ATR={atr:.2f}",
            )

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

        # Calculate dynamic stop/limit based on ATR and strategy R:R
        stop_distance = max(atr * strategy.stop_atr_mult, market.min_stop_distance)
        # Ceiling guard against runaway stops from corrupt data. Price-relative,
        # mirroring the ATR corruption guard above (close*0.05): the old flat
        # min_stop*20 truncated LEGITIMATE ATR stops on high-priced indices with
        # tiny min_stops — e.g. Wall Street (min_stop 4 -> cap 80) whose true 1.5x
        # ATR stop is ~148, Germany 40 (cap 40) wanting ~183. That capped them to
        # 0.3-0.8x effective (the tight-stop whipsaw) AND made live diverge from
        # the backtest, which applies no such cap. The close*0.05 term gives the
        # full ATR stop room on real volatility (intraday ATR tops ~1.4% of price)
        # while still catching genuine corruption (>5% of price). min_stop*20 stays
        # as the floor-of-ceiling for low-priced instruments (forex) where close is
        # not in stop-point units.
        max_stop = max(market.min_stop_distance * 20, close * 0.05)
        if stop_distance > max_stop:
            logger.warning(
                f"[{market.name}] Stop distance {stop_distance:.2f} exceeds max "
                f"{max_stop:.1f} (price-relative). Capping."
            )
            stop_distance = max_stop
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

        # MACD pre-check: don't enter if the MACD-3 exit is armed or one candle
        # from arming. The original check blocked only 3/3 (exit already true),
        # but entering at 2/3 — the two most recent closed candles opposing —
        # self-destructs one candle later: the next opposing close completes the
        # streak and the exit fires at the first post-min-hold evaluation
        # (~5-11 min, spread + adverse drift = guaranteed small loss). 2026-07-24
        # review: replaying this 2/3 gate over 6wk of real post-cap journal
        # trades blocks 11 trades netting −£92.85 (9 losers incl. five Wall St
        # conf-0.56 bleeders vs 2 small wins) while touching NONE of the kept
        # winners (+£249). Archive sim: PF up on 5/7 index markets.
        last_2_macd = [df.iloc[-i]["macd_hist"] for i in range(1, 3)] if len(df) >= 3 else [0]
        macd_already_bearish = all(h < 0 for h in last_2_macd)
        macd_already_bullish = all(h > 0 for h in last_2_macd)

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
                    reason=(
                        f"MACD already bearish for {len(last_2_macd)} bars, "
                        f"exit fires at {MACD_EXIT_BARS}"
                    ),
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
            signal = TradeSignal(
                signal=Signal.BUY,
                epic=market.epic,
                market_name=market.name,
                confidence=confidence,
                entry_price=current_price,
                stop_distance=round(stop_distance, 2),
                limit_distance=round(limit_distance, 2),
                reason=f"Bullish EMA alignment, RSI={rsi:.1f}, ADX={adx:.1f}, HTF={htf_trend}",
                adx=float(adx), rsi=float(rsi), atr=float(atr),
                ema_fast=float(ema_fast), ema_medium=float(ema_medium), ema_slow=float(ema_slow),
            )
            return self._apply_adx_ceiling(
                self._apply_leg_filter(signal, df, market, atr), market
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
                    reason=(
                        f"MACD already bullish for {len(last_2_macd)} bars, "
                        f"exit fires at {MACD_EXIT_BARS}"
                    ),
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
            signal = TradeSignal(
                signal=Signal.SELL,
                epic=market.epic,
                market_name=market.name,
                confidence=confidence,
                entry_price=current_price,
                stop_distance=round(stop_distance, 2),
                limit_distance=round(limit_distance, 2),
                reason=f"Bearish EMA alignment, RSI={rsi:.1f}, ADX={adx:.1f}, HTF={htf_trend}",
                adx=float(adx), rsi=float(rsi), atr=float(atr),
                ema_fast=float(ema_fast), ema_medium=float(ema_medium), ema_slow=float(ema_slow),
            )
            return self._apply_adx_ceiling(
                self._apply_leg_filter(signal, df, market, atr), market
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

    def _apply_leg_filter(
        self,
        signal: TradeSignal,
        df: pd.DataFrame,
        market: MarketConfig,
        atr: float,
    ) -> TradeSignal:
        """Annotate (and optionally block) an entry that chases an exhausted move.

        Mirrors src/backtest.py and the Oanda_Gold _calculateLegInfo: over the
        last `leg_filter_lookback` candles, leg = max(high) - min(low); the leg's
        direction is sign(last_close - first_open); legATR = leg / ATR. If the
        leg ran in the SAME direction as the entry and legATR exceeds the
        threshold, the move is treated as exhausted.

        Sets signal.leg_atr / signal.leg_would_block for observation. Only when
        market.leg_filter_enforce is True is the signal actually converted to
        HOLD — by default this is log-only (main.py records would-blocks).
        """
        lookback = getattr(market, "leg_filter_lookback", 0)
        if not lookback or lookback <= 0:
            return signal
        try:
            if df is None or len(df) < lookback or atr is None or atr <= 0:
                return signal
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                return signal
            window = df.iloc[-lookback:]
            leg_size = float(window["high"].max() - window["low"].min())
            first_open = float(window.iloc[0]["open"])
            last_close = float(window.iloc[-1]["close"])
            leg_is_short = last_close < first_open
            leg_atr = leg_size / atr if atr else 0.0

            direction = signal.signal.value  # "BUY" or "SELL"
            leg_in_direction = (
                (leg_is_short and direction == "SELL")
                or (not leg_is_short and direction == "BUY")
            )
            threshold = getattr(market, "leg_filter_threshold", 5.0)
            would_block = leg_in_direction and leg_atr > threshold

            signal.leg_atr = round(leg_atr, 2)
            signal.leg_would_block = bool(would_block)

            if would_block and getattr(market, "leg_filter_enforce", False):
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=signal.entry_price,
                    stop_distance=signal.stop_distance,
                    limit_distance=signal.limit_distance,
                    reason=(
                        f"Leg filter: chasing exhausted move "
                        f"(legATR {leg_atr:.1f} > {threshold}, lookback {lookback})"
                    ),
                    leg_atr=round(leg_atr, 2),
                    leg_would_block=True,
                )
        except Exception as e:  # never let instrumentation break signal flow
            logger.debug(f"[{market.name}] leg filter check failed: {e}")
        return signal

    def _apply_adx_ceiling(
        self,
        signal: TradeSignal,
        market: MarketConfig,
    ) -> TradeSignal:
        """Annotate (and optionally block) an entry whose ADX marks exhaustion.

        Thesis: a momentum signal is strongest at a climax, so an extreme ADX
        often precedes mean-reversion rather than continuation. If signal.adx
        exceeds market.adx_ceiling, set adx_ceiling_would_block for observation.
        Only when market.adx_ceiling_enforce is True is the signal converted to
        HOLD — by default this is log-only (main.py records would-blocks).
        Mirrors _apply_leg_filter.
        """
        if signal.signal == Signal.HOLD:
            return signal
        ceiling = getattr(market, "adx_ceiling", 0.0)
        if not ceiling or ceiling <= 0:
            return signal
        # Only cap the side the market actually exhausts on (if configured).
        # Exhaustion is one-sided per market; capping the continuation side
        # removes winners (see adx_ceiling_direction on MarketConfig).
        side = getattr(market, "adx_ceiling_direction", "") or ""
        if side and signal.signal.value != side:
            return signal
        try:
            if signal.adx <= ceiling:
                return signal
            signal.adx_ceiling_would_block = True
            if getattr(market, "adx_ceiling_enforce", False):
                return TradeSignal(
                    signal=Signal.HOLD,
                    epic=market.epic,
                    market_name=market.name,
                    confidence=0.0,
                    entry_price=signal.entry_price,
                    stop_distance=signal.stop_distance,
                    limit_distance=signal.limit_distance,
                    reason=(
                        f"ADX ceiling: exhaustion climax "
                        f"(ADX {signal.adx:.1f} > {ceiling})"
                    ),
                    adx=signal.adx, rsi=signal.rsi, atr=signal.atr,
                    adx_ceiling_would_block=True,
                )
        except Exception as e:  # never let instrumentation break signal flow
            logger.debug(f"[{market.name}] ADX ceiling check failed: {e}")
        return signal

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
        ema_separation = abs(ema_fast - ema_slow) / abs(ema_slow) if ema_slow != 0 else 0.0
        ema_factor = min(max(ema_separation * 10, 0.0), 0.25)

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
        clamped = max(0.0, min(confidence, 1.0))
        if clamped != confidence:
            logger.warning(
                f"Confidence out of range: {confidence:.4f} (ema={ema_factor:.4f}, "
                f"rsi={rsi_factor:.4f}, macd={macd_factor:.1f}, adx={adx_factor:.4f}, htf={htf_factor:.1f})"
            )
        return round(clamped, 2)

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
    suppress_momentum_exit: bool = False,
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

    # Each exit below is gated on ITS OWN inputs being finite, rather than on one
    # global bar count. The indicators warm up at different rates (RSI ~8 bars, ADX
    # 28, MACD histogram 35), and a shared floor set to the slowest would disable
    # the RSI exit in the 8-34 bar band where it works perfectly well — reachable
    # after a restart, since check_positions_from_stream guards only df.empty and a
    # position can be live while candles re-accumulate.
    #
    # Where an input IS NaN the outcome is unchanged: every comparison against NaN
    # was already False, i.e. "don't exit", the safe direction with the broker stop
    # still on the book. Forcing exits on bad data would clip winners, the family of
    # change this book has refuted repeatedly. This only makes the no-op visible.
    def _finite(*values):
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)

    # RSI extreme exit (always active - protects against overextended moves)
    if _finite(rsi):
        if direction == "BUY" and rsi > rsi_overbought:
            return True, f"RSI overbought ({rsi:.1f})"
        if direction == "SELL" and rsi < rsi_oversold:
            return True, f"RSI oversold ({rsi:.1f})"
    else:
        logger.debug(f"RSI exit skipped on {len(df)} bars — RSI not finite")

    # MACD exit only if strategy uses it (indices). Suppressed during the
    # minimum-hold window (suppress_momentum_exit) so a candle committing on the
    # entry boundary can't fire an open-then-instant-close (FTSE #205). Stop/limit
    # (broker) and RSI-extreme (above) stay active throughout the hold.
    if use_macd_exit and not suppress_momentum_exit and len(df) >= MACD_EXIT_BARS:
        # The length guard is load-bearing, not defensive tidiness: df.iloc[-i] raises
        # IndexError once i exceeds the frame, so widening MACD_EXIT_BARS without it
        # crashes the exit path on any frame shorter than the window — i.e. exactly on
        # a cold start or restart. Caught by
        # tests/test_indicator_nan_guards.py::TestExitPathToleratesShortFrames.
        # A frame too short to evaluate must HOLD, matching the _finite gate below;
        # the broker stop is unaffected either way.
        last_macd = [df.iloc[-i]["macd_hist"] for i in range(1, MACD_EXIT_BARS + 1)]

        if not _finite(*last_macd):
            logger.debug(
                f"MACD exit skipped on {len(df)} bars — histogram needs "
                f"{MACD_HIST_WARMUP_BARS}. Broker stop unaffected."
            )
        elif direction == "BUY":
            if all(h < 0 for h in last_macd):
                # The bar count is interpolated, not hard-coded, so the journal
                # records WHICH window each trade closed under and the era split
                # stays legible after any future change to MACD_EXIT_BARS.
                return True, f"MACD histogram negative for {MACD_EXIT_BARS} candles"
        elif direction == "SELL":
            if all(h > 0 for h in last_macd):
                return True, f"MACD histogram positive for {MACD_EXIT_BARS} candles"

    # Dynamic exit for non-MACD strategies (Gold, Forex, etc.)
    # These strategies need protection when market conditions change
    if not use_macd_exit:
        # ADX ranging exit: require 3 consecutive candles below threshold.
        # Single-candle wobble was bleeding ~£14/trade on Gold (8 trades, -£99)
        # by exiting whipsaws within 15-90 min. The 3 rests on its own evidence:
        # it improved WR on Gold (44->47%) and USD/JPY (23->39%) in 60d backtest
        # with neutral aggregate P&L. It ALSO used to match the MACD exit; that
        # stopped being true at 9e2dd0c (MACD_EXIT_BARS -> 5). The two windows are
        # independent by design -- do not re-couple them without re-running that
        # backtest, and note this exit only runs when use_macd_exit is False.
        adx_exit_threshold = adx_threshold - 10  # e.g., 35 -> 25
        # Ranging-3 exit also gated by the minimum-hold window (HTF reversal below
        # is NOT — a higher-timeframe flip can't race a single entry candle).
        if not suppress_momentum_exit and len(df) >= 3:
            recent_adx = [df.iloc[-i]["adx"] for i in range(1, 4)]
            if not _finite(*recent_adx):
                logger.debug(
                    f"ADX ranging exit skipped on {len(df)} bars — ADX needs "
                    f"{ADX_WARMUP_BARS}. Broker stop unaffected."
                )
            elif all(a < adx_exit_threshold for a in recent_adx):
                return True, f"Market turned ranging (ADX {adx:.1f} < {adx_exit_threshold} for 3 candles)"

        # HTF reversal exit: close if higher timeframe trend reversed against us
        if direction == "BUY" and htf_trend == "BEARISH":
            return True, f"HTF trend reversed to BEARISH"
        if direction == "SELL" and htf_trend == "BULLISH":
            return True, f"HTF trend reversed to BULLISH"

    return False, ""
