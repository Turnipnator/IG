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
    # NB: breakout runs NO take-profit — the Donchian-M trail is the only exit, so
    # fat-tail winners (the whole point) can run. limit_distance is emitted as 0.

    @property
    def m(self) -> int:
        """Exit channel lookback (Turtle exit = half the entry channel)."""
        return max(2, self.n // 2)


# Per-EPIC config. N matches each pair's live candle interval: GBP/EUR are 1h
# (N55 ≈ 55h), USD/JPY is 15m (N40 ≈ 10h). GBP is the validated lead.
BREAKOUT_CONFIGS: dict[str, BreakoutConfig] = {
    "CS.D.GBPUSD.TODAY.IP": BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # lead, cost+walk-forward validated
    "CS.D.EURUSD.TODAY.IP": BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # marginal
    # USD/JPY disabled 2026-06-22: walk-forward rejected (-5.13%/2yr, 1/4 quarters +);
    # bled two -£24 full-stop losers (-£48) in one live session. Re-enable only if re-validated.
    # "CS.D.USDJPY.TODAY.IP": BreakoutConfig(n=40, stop_atr_mult=2.0, htf_filter=True),  # weak — shadow-watch only

    # Crude Oil (2026-07-24) — exists for the DISCRETIONARY /mode toggle, NOT as a
    # validated standing edge. CL=F 730d/1h at 6bps (scripts/backtest_oil_breakout.py):
    # HTF-filtered full-period is NEGATIVE at every N (N55 PF 0.87, −17.7%) — but
    # strongly regime-dependent: the most recent quarter is +14.7% PF 1.33 (N40:
    # +22.9% PF 1.39). Oil breakout pays in trending quarters and bleeds in chop, so
    # it only makes sense driven by an external regime view (the user's tips), never
    # permanently on. Default mode is breakout-shadow (MarketConfig.default_mode);
    # /mode oil breakout is the user's deliberate, informed act. N=55 kept for
    # consistency with the validated GBP/USD shape rather than snooping N=40.
    "EN.D.CL.Month1.IP": BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),

    # 2026-07-24 (user request): every non-forex EPIC gets a breakout config so the
    # ALWAYS-ON shadow observer (main._observe_breakout_shadow) can log 1h Donchian
    # signals alongside each market's live strategy, and /mode can flip any of them
    # deliberately. N is in 1h bars — the observer/live path resamples sub-1h
    # markets to 1h (main._breakout_frame_1h), matching the validated GBP/USD shape.
    # VALIDATION STATUS (be honest at flip time):
    #   Gold    — GC=F 730d/1h: PF 1.28(N40)/1.22(N55)/1.38(N70), 3/4 quarters
    #             green, strengthening. The one PRE-validated candidate here.
    #   Indices — UNTESTED as breakout. Yahoo cash proxies can't test them (no
    #             overnight bars vs IG's ~24h sessions) — the shadow observer on
    #             the IG archive IS the test. Do not flip live without its readout.
    "CS.D.USCGC.TODAY.IP":    BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # Gold
    "IX.D.SPTRD.DAILY.IP":    BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # S&P 500
    "IX.D.NASDAQ.CASH.IP":    BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # NASDAQ 100
    "IX.D.DOW.DAILY.IP":      BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # Wall Street
    "IX.D.FTSE.DAILY.IP":     BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # FTSE 100
    "IX.D.AIIDX.DAILY.IP":    BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # AI Index
    "IX.D.RUSSELL.DAILY.IP":  BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # Russell 2000
    "IX.D.NIKKEI.DAILY.IP":   BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # Japan 225
    "IX.D.HANGSENG.DAILY.IP": BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # Hong Kong
    "CO.D.DX.Month1.IP":      BreakoutConfig(n=55, stop_atr_mult=2.0, htf_filter=True),  # Dollar Index — UNTESTED
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
    limit_distance = 0.0  # no take-profit — the Donchian-M trail is the only exit

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
        break_level=round(level, 2),
    )


def current_channel(df, epic: str) -> tuple[float, float] | None:
    """(upper, lower) of the last N CLOSED bars — the channel the NEXT bar can break.

    This is the ARMING counterpart to analyze_breakout, which is the CONFIRMING one,
    and the two deliberately slice the frame one bar apart:

        analyze_breakout, run at the close of bar T, asks "did bar T break the channel
        formed by the N bars BEFORE it?" -> df.iloc[-(n+1):-1]
        current_channel, run at the same close, asks "what must bar T+1 exceed?"
        -> df.iloc[-n:], which INCLUDES bar T

    Getting that off by one bar silently arms yesterday's level, so it is spelled out
    rather than inferred. Both callers must pass the SAME frame — for sub-1h markets
    that means main._breakout_frame_1h output, not the native stream frame (the
    frame-relative trap that misframed the Donchian trail in 0057aa9).

    Returns None when the epic has no breakout config or the frame is too short.
    """
    cfg = BREAKOUT_CONFIGS.get(epic)
    if cfg is None or df is None or len(df) < cfg.n:
        return None
    window = df.iloc[-cfg.n:]
    upper = float(window["high"].max())
    lower = float(window["low"].min())
    if not (upper > lower > 0):
        return None
    return upper, lower


@dataclass
class ArmedChannel:
    """Everything the tick-entry path needs to act on a break WITHOUT re-reading the
    frame. Computed once per closed bar; consumed by main._check_breakout_tick_trigger
    on the stream thread, which must never touch pandas."""
    upper: float
    lower: float
    stop_distance: float
    atr: float
    htf_trend: str
    htf_filter: bool
    bar_time: str


def arm_channel(df, market: MarketConfig, htf_trend: str) -> ArmedChannel | None:
    """Snapshot the channel + stop for tick-triggered entry on the NEXT bar.

    Deliberately mirrors analyze_breakout's arithmetic — same ATR period, same
    k*ATR-vs-min_stop clamp, same htf_filter flag — so a tick entry and a close
    entry differ ONLY in WHEN they fire, never in what they risk. If the two ever
    drift apart, the slip measurement they exist to produce becomes meaningless.

    Returns None when the epic has no config, the frame is short, or ATR is unusable.
    """
    cfg = BREAKOUT_CONFIGS.get(market.epic)
    if cfg is None:
        return None
    ch = current_channel(df, market.epic)
    if ch is None:
        return None
    atr_series = calculate_atr(df["high"], df["low"], df["close"], 14)
    atr = float(atr_series.iloc[-1])
    if pd.isna(atr) or atr <= 0:
        return None
    # Identity of the bar that produced this channel. Used as the arming latch key,
    # so it must change exactly once per closed bar and never within one.
    if "date" in getattr(df, "columns", []):
        bar_time = str(df["date"].iloc[-1])
    else:
        bar_time = str(df.index[-1])
    return ArmedChannel(
        upper=ch[0], lower=ch[1],
        stop_distance=round(max(atr * cfg.stop_atr_mult, market.min_stop_distance), 2),
        atr=round(atr, 2), htf_trend=htf_trend, htf_filter=cfg.htf_filter,
        bar_time=bar_time,
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
