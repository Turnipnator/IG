"""Forex breakout strategy — Donchian / Turtle channel breakout (built 2026-06-19).

The validated replacement for the RETIRED forex momentum profiles (net-losing, 40%
WR, ~-£94 current era). Backtests (scripts/backtest_forex_breakout*.py): GBP/USD
N55 HTF-filtered Donchian survived BOTH a 3-pip cost (+2.65% PF1.34) AND walk-
forward (positive every quarter over 2yr). EUR/USD marginal; USD/JPY weaker (a
walk-forward mirage) — included for shadow observation only. The edge is FADING, so
this runs ONLY behind the /forex toggle (telegram.forex_mode) and is rolled out
shadow-first (matches the bot's observational→enforce culture).

Entry: a buy-stop at the prior N-bar HIGH (long) / sell-stop at the prior N-bar LOW
(short), with an optional HTF-trend filter (daily for GBP/EUR, hourly for JPY) and a
k×ATR stop. Live acts on candle-CLOSE, so a break that already happened intra-candle
enters at market (the slippage the cost test modelled). The live exit (Stage 2b) is a
Donchian-trail on the M=N/2 opposite channel — this module emits the ENTRY signal
only; `exit_channel()` exposes the trail level for the position manager.
"""
from dataclasses import dataclass
import logging

import pandas as pd

from src.indicators import calculate_atr
from src.strategy import Signal, TradeSignal
from config import MarketConfig

logger = logging.getLogger(__name__)


@dataclass
class BreakoutConfig:
    """Per-market breakout parameters (N is in candles at the market's interval)."""
    n: int                      # entry channel lookback
    stop_atr_mult: float = 2.0  # k×ATR initial stop (Turtle "N")
    htf_filter: bool = True     # only break WITH the higher-timeframe trend
    safety_rr: float = 3.0      # far safety TP; real exit is the Donchian-M trail

    @property
    def m(self) -> int:
        """Exit channel lookback (Turtle exit = half the entry channel)."""
        return max(2, self.n // 2)


# Per-EPIC config. N matches each pair's live candle interval: GBP/EUR are 1h
# (N55 ≈ 55h), USD/JPY is 15m (N40 ≈ 10h). GBP is the validated lead.
BREAKOUT_CONFIGS: dict[str, BreakoutConfig] = {
    "CS.D.GBPUSD.TODAY.IP": BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # lead, cost+walk-forward validated
    "CS.D.EURUSD.TODAY.IP": BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # marginal
    "CS.D.USDJPY.TODAY.IP": BreakoutConfig(n=40, stop_atr_mult=2.0, htf_filter=True),  # weak — shadow-watch only
}


def has_breakout_config(epic: str) -> bool:
    return epic in BREAKOUT_CONFIGS


def _hold(market: MarketConfig, current_price: float, reason: str) -> TradeSignal:
    return TradeSignal(
        signal=Signal.HOLD, epic=market.epic, market_name=market.name,
        confidence=0.0, entry_price=current_price,
        stop_distance=market.min_stop_distance, limit_distance=market.min_stop_distance,
        reason=reason,
    )


def analyze_breakout(df, market: MarketConfig, current_price: float, htf_trend: str) -> TradeSignal:
    """Emit a Donchian breakout entry signal (BUY/SELL/HOLD) for a forex market.

    Faithful to scripts/backtest_forex_breakout.py: break of the prior N-bar channel,
    HTF-filtered, k×ATR stop. Returns HOLD with a descriptive reason when no break /
    indicators not ready / filtered out.
    """
    cfg = BREAKOUT_CONFIGS.get(market.epic)
    if cfg is None:
        return _hold(market, current_price, "No breakout config for this market")
    if df is None or len(df) < cfg.n + 2:
        n_have = 0 if df is None else len(df)
        return _hold(market, current_price, f"Insufficient candles for {cfg.n}-bar channel ({n_have})")

    atr_series = calculate_atr(df["high"], df["low"], df["close"], 14)
    atr = float(atr_series.iloc[-1])
    if pd.isna(atr) or atr <= 0:
        return _hold(market, current_price, "ATR not ready")

    # Prior N-bar channel = the N candles BEFORE the just-closed last candle.
    prior = df.iloc[-(cfg.n + 1):-1]
    upper = float(prior["high"].max())
    lower = float(prior["low"].min())
    last = df.iloc[-1]
    high, low = float(last["high"]), float(last["low"])

    stop_distance = max(atr * cfg.stop_atr_mult, market.min_stop_distance)
    limit_distance = stop_distance * cfg.safety_rr

    if high >= upper:
        if cfg.htf_filter and htf_trend != "BULLISH":
            return _hold(market, current_price,
                         f"BUY break of {cfg.n}-bar high blocked — HTF {htf_trend} (need BULLISH)")
        direction, level = Signal.BUY, upper
    elif low <= lower:
        if cfg.htf_filter and htf_trend != "BEARISH":
            return _hold(market, current_price,
                         f"SELL break of {cfg.n}-bar low blocked — HTF {htf_trend} (need BEARISH)")
        direction, level = Signal.SELL, lower
    else:
        return _hold(market, current_price,
                     f"No break ({cfg.n}-bar channel {lower:.1f}–{upper:.1f}, last {low:.1f}–{high:.1f})")

    return TradeSignal(
        signal=direction, epic=market.epic, market_name=market.name,
        confidence=0.7, entry_price=current_price,
        stop_distance=round(stop_distance, 2), limit_distance=round(limit_distance, 2),
        reason=(f"Breakout: {direction.value} break of {cfg.n}-bar channel @ {level:.1f} "
                f"(stop {stop_distance:.1f}={cfg.stop_atr_mult}xATR, HTF={htf_trend})"),
        atr=round(atr, 2),
    )


def exit_channel(df, epic: str, direction: str) -> float | None:
    """Donchian-trail exit level for an open breakout position (Stage 2b — live exit).

    For a long, exit on a break of the prior M-bar LOW; for a short, the prior M-bar
    HIGH. Returns the level (price) or None if not enough data / unknown epic.
    """
    cfg = BREAKOUT_CONFIGS.get(epic)
    if cfg is None or df is None or len(df) < cfg.m + 2:
        return None
    prior = df.iloc[-(cfg.m + 1):-1]
    return float(prior["low"].min()) if direction == "BUY" else float(prior["high"].max())
