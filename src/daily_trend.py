"""Daily-bar, LONG-ONLY Donchian trend following (2026-09-09).

The rule, fixed BEFORE it was tested (research_notes.md, "Optimum strategy per
instrument", pre-registration of 2026-09-09):

    enter long   when the daily CLOSE exceeds the highest HIGH of the prior
                 n_entry (55) completed bars;
    hard stop    stop_mult (2.0) x ATR(atr_n=20, Wilder) below the fill;
    exit         when the daily CLOSE falls below the lowest LOW of the prior
                 n_exit (20) completed bars;
    one position at a time; no shorts (every long/short index variant tested
    negative: the short side pays the drift).

Evidence: Gold long-only positive in 30/30 parameter cells and in both 11-year
halves of 2004-2026 after IG spread + nightly financing (+0.69R/trade, +1.76R/yr,
PF 2.06 at the MEASURED 5.8%/yr Gold financing rate, re-charged 2026-09-09 on the
golden trades; the study's +0.87R/+2.3R used 2.9%, a rows-counted-as-nights
error — research_notes.md "Gold financing rate RE-MEASURED"); NASDAQ 30/30
(weaker). What it IS: crash-protected beta — about 40% of buy-and-hold with
about 15% of its drawdown. What it is NOT: a directional edge. Hold is weeks,
trade rate about 2.6/yr on Gold. Financing is the dominant cost: about 0.35R
per trade at the measured rate on a 47-night mean hold.

Bar hygiene (prepare_daily): IG's DAY bars for spot Gold roll at 00:00 London
and include a one-HOUR Sunday stub (Sun 23:00-23:59) that the Yahoo-based
backtest never saw. Sunday rows are dropped so the 55/20/20 windows keep the
5-bars-per-week structure they were validated on. Duplicate dates keep the
LAST row (a re-fetched bar supersedes its forming snapshot).

`replay` is the backtest engine and the shadow resolver's reference; the
golden test in tests/test_daily_trend.py pins it to the trade list produced by
the study's independent engine, so the live code cannot drift from the rule
that was validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyTrendConfig:
    n_entry: int = 55
    n_exit: int = 20
    atr_n: int = 20
    stop_mult: float = 2.0

    @property
    def min_bars(self) -> int:
        """Bars needed before the first valid signal (entry channel + 1 for the shift)."""
        return max(self.n_entry, self.n_exit, self.atr_n) + 1


# Per-EPIC configs. Only markets that PASSED the 22-year study belong here.
# NASDAQ (IX.D.NASDAQ.CASH.IP) also passed and may be added by a later decision.
DAILY_TREND_CONFIGS: dict[str, DailyTrendConfig] = {
    "CS.D.USCGC.TODAY.IP": DailyTrendConfig(),    # Gold — 30/30 cells, both halves, z≈2 (LIVE)
    "IX.D.NIKKEI.DAILY.IP": DailyTrendConfig(),   # Japan 225 — 30/30 cells, both halves, lumpy years (SHADOW)
}

VALID_DAILY_TREND_MODES = ("off", "shadow", "live")


def has_daily_trend_config(epic: str) -> bool:
    return epic in DAILY_TREND_CONFIGS


def get_daily_trend_config(epic: str) -> Optional[DailyTrendConfig]:
    return DAILY_TREND_CONFIGS.get(epic)


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a daily OHLC frame: normalise dates, drop Sunday stubs, dedupe (last
    wins), sort. Accepts a `date` column or a DatetimeIndex."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    d = df.copy()
    if "date" not in d.columns:
        d = d.reset_index().rename(columns={d.index.name or "index": "date"})
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d[d["date"].dt.weekday != 6]                 # Sunday stub (IG spot Gold)
    d = d.drop_duplicates(subset="date", keep="last").sort_values("date")
    if "volume" not in d.columns:
        d["volume"] = 0
    d = d[["date", "open", "high", "low", "close", "volume"]]
    d = d[(d["close"] > 0) & (d["high"] >= d["low"])]
    return d.reset_index(drop=True)


def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_levels(df: pd.DataFrame, cfg: DailyTrendConfig) -> pd.DataFrame:
    """ATR plus the two channels. Channels are shifted ONE bar so bar i is judged
    against the prior n bars only — the same convention as src/breakout.py."""
    d = df.copy()
    d["atr"] = wilder_atr(d, cfg.atr_n)
    d["entry_hi"] = d["high"].rolling(cfg.n_entry).max().shift(1)
    d["exit_lo"] = d["low"].rolling(cfg.n_exit).min().shift(1)
    return d


@dataclass
class DailySignal:
    action: str                 # ENTER_LONG | EXIT | HOLD | WAIT
    bar_date: Optional[pd.Timestamp]
    close: float = float("nan")
    atr: float = float("nan")
    stop_distance: float = float("nan")   # stop_mult x ATR, in price units
    entry_hi: float = float("nan")
    exit_lo: float = float("nan")
    reason: str = ""


def evaluate(df: pd.DataFrame, cfg: DailyTrendConfig, in_position: bool) -> DailySignal:
    """Decision on the LAST completed bar of `df`. Must agree with `replay`:
    not in position -> ENTER_LONG iff close > entry_hi; in position -> EXIT iff
    close < exit_lo. Any non-finite level -> WAIT (never trade on a short frame)."""
    d = add_levels(prepare_daily(df), cfg)
    if len(d) < cfg.min_bars:
        return DailySignal("WAIT", None, reason=f"{len(d)} bars < {cfg.min_bars} needed")
    last = d.iloc[-1]
    vals = (last["close"], last["atr"], last["entry_hi"], last["exit_lo"])
    if not all(np.isfinite(v) for v in vals):
        return DailySignal("WAIT", last["date"], reason="non-finite level")
    close, atr, hi, lo = (float(v) for v in vals)
    base = dict(bar_date=last["date"], close=close, atr=atr, stop_distance=cfg.stop_mult * atr,
                entry_hi=hi, exit_lo=lo)
    if not in_position:
        if close > hi:
            return DailySignal("ENTER_LONG", reason=f"close {close:.2f} > {cfg.n_entry}d high {hi:.2f}", **base)
        return DailySignal("HOLD", reason=f"flat: close {close:.2f} <= {cfg.n_entry}d high {hi:.2f}", **base)
    if close < lo:
        return DailySignal("EXIT", reason=f"close {close:.2f} < {cfg.n_exit}d low {lo:.2f}", **base)
    return DailySignal("HOLD", reason=f"long: close {close:.2f} >= {cfg.n_exit}d low {lo:.2f}", **base)


def replay(df: pd.DataFrame, cfg: DailyTrendConfig, start: Optional[str] = None) -> list[dict]:
    """Backtest engine. Semantics (identical to the study's engine):
      * signal on bar i close; fill at bar i+1 OPEN; ATR for the stop = ATR(i);
      * the hard stop is checked on EVERY bar including the fill bar; a gap
        through the stop fills at the open (min(open, stop) for a long);
      * channel exit signalled on bar i close fills at bar i+1 open;
      * after a stop-out the same bar's close may re-arm an entry for i+1.
    Returns closed trades only (an open position at the end is NOT returned)."""
    d = add_levels(prepare_daily(df), cfg)
    dates = d["date"].values
    o, h, l, c, atr = (d[x].values.astype(float) for x in ("open", "high", "low", "close", "atr"))
    hi, lo = d["entry_hi"].values.astype(float), d["exit_lo"].values.astype(float)
    trades: list[dict] = []
    pos = 0; pend = 0; e_px = e_atr = stop = None; e_i = None
    start_ts = pd.Timestamp(start) if start else None
    for i in range(1, len(d)):
        if pend != pos:
            if pos:
                trades.append(_trade(dates, e_i, i, e_px, o[i], e_atr, "channel"))
                pos = 0
            if pend:
                pos = 1; e_px = o[i]; e_atr = atr[i - 1]; e_i = i; stop = e_px - cfg.stop_mult * e_atr
        if pos and l[i] <= stop:
            trades.append(_trade(dates, e_i, i, e_px, min(o[i], stop), e_atr, "stop"))
            pos = 0; pend = 0
        if not np.isfinite(hi[i]) or not np.isfinite(lo[i]) or not np.isfinite(atr[i]):
            pend = pos; continue
        if pos == 0:
            pend = 1 if (c[i] > hi[i] and (start_ts is None or dates[i] >= start_ts)) else 0
        else:
            pend = 0 if c[i] < lo[i] else 1
    return trades


def _trade(dates, e_i, x_i, e_px, x_px, e_atr, reason) -> dict:
    return dict(entry_date=pd.Timestamp(dates[e_i]), exit_date=pd.Timestamp(dates[x_i]),
                entry=float(e_px), exit=float(x_px), pts=float(x_px - e_px),
                atr_at_entry=float(e_atr), r_gross=float((x_px - e_px) / (2.0 * e_atr)) if e_atr else float("nan"),
                hold_days=int((pd.Timestamp(dates[x_i]) - pd.Timestamp(dates[e_i])).days), reason=reason)


def resolve_open_episode(df: pd.DataFrame, cfg: DailyTrendConfig, signal_date, entry_price: float,
                         stop_distance: float) -> Optional[dict]:
    """Shadow resolver. Walk the bars AFTER the fill bar (the first bar dated after
    `signal_date`) with the live exit rules; return the closing trade dict or None
    while the episode is still open. Mirrors `replay` bar-for-bar."""
    d = add_levels(prepare_daily(df), cfg)
    after = d[d["date"] > pd.Timestamp(signal_date)].reset_index(drop=True)
    if after.empty:
        return None
    o, h, l, c = (after[x].values.astype(float) for x in ("open", "high", "low", "close"))
    lo = after["exit_lo"].values.astype(float)
    stop = entry_price - stop_distance
    pend_exit = False
    for i in range(len(after)):
        if pend_exit:   # channel exit signalled on the previous close -> fill at this open
            return dict(exit_date=after["date"].iloc[i], exit=float(o[i]), reason="channel", bars=i + 1)
        if l[i] <= stop:
            return dict(exit_date=after["date"].iloc[i], exit=float(min(o[i], stop)), reason="stop", bars=i + 1)
        if np.isfinite(lo[i]) and c[i] < lo[i]:
            pend_exit = True
    return None
