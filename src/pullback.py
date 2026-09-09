"""Pullback-in-uptrend, LONG-ONLY, daily cash-session bars (2026-09-09).

Rule, fixed BEFORE testing (research_notes.md, "Second sweep", pre-registration
of 2026-09-09 14:40 BST):

    regime   close > SMA(sma_n=200) of daily closes;
    enter    at the close, when the close < lowest LOW of the prior n_in (5)
             sessions;
    exit     at the close, when the close > highest HIGH of the prior n_out (5)
             sessions, OR after max_hold (10) sessions;
    stop     stop_mult (3.0) x ATR(atr_n=20) below the fill — the safety-net
             variant that tested identically to the stopless classic form;
    one position per market; no shorts.

Evidence (22y Yahoo cash bars, IG spread + nightly financing): S&P 500 +0.22R/
trade, z 2.4-2.8, PF 1.57, 73% of years positive; NASDAQ 100 +0.22R, z 2.7, 65%;
both positive in 72/72 parameter cells and in both 11-year halves; 45%/38% of
buy-and-hold captured in ~24% of calendar time. Mechanism: multi-day reversal in
US equity indices. ~8 trades/yr/instrument, hit rate ~70%, hold ~4 sessions.

Bars: the channels MUST be computed on CASH-SESSION bars (14:30-21:00 London for
US indices; see src/session_bars.py) — IG's 24h bars have lows 25-35% wider and
would suppress the tested signals. The SMA200 may use IG DAY closes (23:55 London,
within 0.1% of the cash close). `replay` is the backtest engine; the golden test
pins it to the trade lists of the sweep's independent engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PullbackConfig:
    n_in: int = 5
    n_out: int = 5
    max_hold: int = 10
    sma_n: int = 200
    atr_n: int = 20
    stop_mult: float = 3.0

    @property
    def min_session_bars(self) -> int:
        return max(self.n_in, self.n_out, self.atr_n) + 1


# Markets that PASSED the pre-registered gate AND the 72-cell plateau (sweep 2).
PULLBACK_CONFIGS: dict[str, PullbackConfig] = {
    "IX.D.SPTRD.DAILY.IP": PullbackConfig(),    # S&P 500  — 72/72, z 2.4-2.8, 73% yrs
    "IX.D.NASDAQ.CASH.IP": PullbackConfig(),    # NASDAQ 100 — 72/72, z 2.7, 65% yrs
}

VALID_PULLBACK_MODES = ("off", "shadow", "live")


def has_pullback_config(epic: str) -> bool:
    return epic in PULLBACK_CONFIGS


def get_pullback_config(epic: str) -> Optional[PullbackConfig]:
    return PULLBACK_CONFIGS.get(epic)


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a daily frame: date column, weekdays, dedupe (last wins), sorted."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    d = df.copy()
    if "date" not in d.columns:
        d = d.reset_index().rename(columns={d.index.name or "index": "date"})
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d[d["date"].dt.weekday < 5]
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


def add_levels(df: pd.DataFrame, cfg: PullbackConfig, sma_closes: Optional[pd.Series] = None) -> pd.DataFrame:
    """ATR, entry low-channel, exit high-channel (both shifted one bar), SMA.

    sma_closes: optional LONGER close series (e.g. IG DAY closes) indexed by date,
    used for the SMA when the session frame is too short for sma_n bars. Today's
    session close overrides the same-date entry so the regime is judged on the
    price the decision is made at."""
    d = df.copy()
    d["atr"] = wilder_atr(d, cfg.atr_n)
    d["entry_lo"] = d["low"].rolling(cfg.n_in).min().shift(1)
    d["exit_hi"] = d["high"].rolling(cfg.n_out).max().shift(1)
    if sma_closes is None:
        d["sma"] = d["close"].rolling(cfg.sma_n).mean()
    else:
        s = pd.Series(sma_closes).copy()
        s.index = pd.to_datetime(s.index).normalize()
        sess = pd.Series(d["close"].values, index=d["date"].values)
        merged = pd.concat([s[~s.index.isin(sess.index)], sess]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        sma = merged.rolling(cfg.sma_n).mean()
        d["sma"] = sma.reindex(d["date"].values).values
    return d


@dataclass
class PullbackSignal:
    action: str                     # ENTER_LONG | EXIT | HOLD | WAIT
    bar_date: Optional[pd.Timestamp]
    close: float = float("nan")
    atr: float = float("nan")
    stop_distance: float = float("nan")   # stop_mult x ATR (price units)
    risk_unit: float = float("nan")       # 2 x ATR — the R the study is measured in
    entry_lo: float = float("nan")
    exit_hi: float = float("nan")
    sma: float = float("nan")
    reason: str = ""


def evaluate(session_df: pd.DataFrame, cfg: PullbackConfig, in_position: bool, bars_held: int = 0,
             sma_closes: Optional[pd.Series] = None, exited_today: bool = False) -> PullbackSignal:
    """Decision on the LAST session bar of `session_df` (the bar that just closed).
    Must agree with `replay`: flat -> ENTER_LONG iff close > sma and close < entry_lo;
    long -> EXIT iff close > exit_hi or bars_held >= max_hold. Non-finite -> WAIT.
    exited_today: a position on this market was closed during THIS session (broker
    stop) — the engine never re-enters on an exit bar, so neither does live."""
    d = add_levels(prepare_daily(session_df), cfg, sma_closes)
    if len(d) < cfg.min_session_bars:
        return PullbackSignal("WAIT", None, reason=f"{len(d)} session bars < {cfg.min_session_bars}")
    last = d.iloc[-1]
    vals = (last["close"], last["atr"], last["entry_lo"], last["exit_hi"], last["sma"])
    if not all(np.isfinite(float(v)) for v in vals):
        return PullbackSignal("WAIT", last["date"], reason="non-finite level (SMA needs 200 closes)")
    close, atr, lo, hi, sma = (float(v) for v in vals)
    base = dict(bar_date=last["date"], close=close, atr=atr, stop_distance=cfg.stop_mult * atr,
                risk_unit=2.0 * atr, entry_lo=lo, exit_hi=hi, sma=sma)
    if in_position:
        if close > hi:
            return PullbackSignal("EXIT", reason=f"close {close:.1f} > {cfg.n_out}d high {hi:.1f}", **base)
        if bars_held >= cfg.max_hold:
            return PullbackSignal("EXIT", reason=f"held {bars_held} sessions >= {cfg.max_hold}", **base)
        return PullbackSignal("HOLD", reason=f"long {bars_held}d: close {close:.1f} <= {cfg.n_out}d high {hi:.1f}", **base)
    if exited_today:
        return PullbackSignal("HOLD", reason="exited this session — no same-session re-entry", **base)
    if close > sma and close < lo:
        return PullbackSignal("ENTER_LONG", reason=f"close {close:.1f} < {cfg.n_in}d low {lo:.1f}, above SMA{cfg.sma_n} {sma:.1f}", **base)
    why = "below SMA" if close <= sma else f"close {close:.1f} >= {cfg.n_in}d low {lo:.1f}"
    return PullbackSignal("HOLD", reason=f"flat: {why}", **base)


def replay(df: pd.DataFrame, cfg: PullbackConfig, start: Optional[str] = None) -> list[dict]:
    """Backtest engine, identical to the sweep engine: on each bar, if long — stop
    first (gap fills min(open, stop)), then channel exit at the close, then the
    time exit at the close; a bar that exits never re-enters; if flat, enter AT
    the close when close > sma and close < entry_lo. Loop starts flat at `start`."""
    d = add_levels(prepare_daily(df), cfg)
    if start is not None:
        d = d[d["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    d = d.dropna(subset=["atr", "sma", "entry_lo", "exit_hi"]).reset_index(drop=True)
    dates = d["date"].values
    o, h, l, c, atr, sma, lo, hi = (d[x].values.astype(float) for x in ("open", "high", "low", "close", "atr", "sma", "entry_lo", "exit_hi"))
    T: list[dict] = []; pos = 0; e_px = e_atr = None; e_i = None
    for i in range(len(d)):
        if pos:
            stop = e_px - cfg.stop_mult * e_atr if cfg.stop_mult else -1e18
            if l[i] <= stop:
                T.append(_trade(dates, e_i, i, e_px, min(o[i], stop), e_atr, "stop")); pos = 0; continue
            if c[i] > hi[i]:
                T.append(_trade(dates, e_i, i, e_px, c[i], e_atr, "channel")); pos = 0; continue
            if (i - e_i) >= cfg.max_hold:
                T.append(_trade(dates, e_i, i, e_px, c[i], e_atr, "time")); pos = 0; continue
        if pos == 0 and c[i] > sma[i] and c[i] < lo[i]:
            pos = 1; e_px = c[i]; e_atr = atr[i]; e_i = i
    return T


def _trade(dates, e_i, x_i, e_px, x_px, e_atr, reason) -> dict:
    return dict(entry_date=pd.Timestamp(dates[e_i]), exit_date=pd.Timestamp(dates[x_i]), entry=float(e_px),
                exit=float(x_px), pts=float(x_px - e_px), atr_at_entry=float(e_atr),
                r_gross=float((x_px - e_px) / (2.0 * e_atr)) if e_atr else float("nan"),
                bars_held=int(x_i - e_i), reason=reason)


def resolve_open_episode(session_df: pd.DataFrame, cfg: PullbackConfig, entry_date, entry_price: float,
                         stop_distance: float) -> Optional[dict]:
    """Shadow resolver: walk the session bars AFTER the entry bar with the live exit
    rules (stop, channel, time). None while still open."""
    d = add_levels(prepare_daily(session_df), cfg)
    after = d[d["date"] > pd.Timestamp(entry_date)].reset_index(drop=True)
    if after.empty:
        return None
    o, l, c, hi = (after[x].values.astype(float) for x in ("open", "low", "close", "exit_hi"))
    stop = entry_price - stop_distance
    for i in range(len(after)):
        if l[i] <= stop:
            return dict(exit_date=after["date"].iloc[i], exit=float(min(o[i], stop)), reason="stop", bars=i + 1)
        if np.isfinite(hi[i]) and c[i] > hi[i]:
            return dict(exit_date=after["date"].iloc[i], exit=float(c[i]), reason="channel", bars=i + 1)
        if (i + 1) >= cfg.max_hold:
            return dict(exit_date=after["date"].iloc[i], exit=float(c[i]), reason="time", bars=i + 1)
    return None
