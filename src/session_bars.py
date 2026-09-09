"""Cash-session daily bars from the 5m stream + durable archive (2026-09-09).

Why: the pullback rule was validated on cash-index bars (NYSE 09:30-16:00 ET).
IG's US index DFBs trade ~23h, so their DAY bars carry lows 25-35% wider than the
cash session's and would suppress the tested signals. This module slices the
bot's own 5m candles (archive + live stream deque) to the session window and
aggregates one bar per date.

Window is given in LONDON time and holds across DST for US markets because London
and New York shift within a fortnight of each other (the screener relies on the
same property). A session needs at least MIN_BARS 5m candles to count — partial
days (half-sessions, outages, holidays) are dropped rather than mis-measured.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.archive_loader import load_archive, ARCHIVE_DIR

MIN_BARS = 60        # of 78 five-minute bars in a 6.5h session
US_SESSION = ("14:30", "21:00")


def _easter(y: int):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31; day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, month, day)


def _nth_weekday(y: int, month: int, weekday: int, n: int):
    d = pd.Timestamp(y, month, 1)
    d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return d + pd.Timedelta(days=7 * (n - 1))


def _last_weekday(y: int, month: int, weekday: int):
    d = pd.Timestamp(y, month, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    if d.weekday() == 5: return d - pd.Timedelta(days=1)
    if d.weekday() == 6: return d + pd.Timedelta(days=1)
    return d


def nyse_holidays(y: int) -> set:
    """NYSE full-day closures. A DFB still trades on these days, so a session bar
    would exist with no cash market behind it — a bar the backtest never saw."""
    hs = set()
    ny = pd.Timestamp(y, 1, 1)
    if ny.weekday() == 6: hs.add(ny + pd.Timedelta(days=1))     # NYSE: Sat New Year is NOT observed on Friday
    elif ny.weekday() < 5: hs.add(ny)
    hs.add(_nth_weekday(y, 1, 0, 3))                              # MLK
    hs.add(_nth_weekday(y, 2, 0, 3))                              # Presidents
    hs.add(_easter(y) - pd.Timedelta(days=2))                     # Good Friday
    hs.add(_last_weekday(y, 5, 0))                                # Memorial
    if y >= 2022: hs.add(_observed(pd.Timestamp(y, 6, 19)))       # Juneteenth
    hs.add(_observed(pd.Timestamp(y, 7, 4)))                      # Independence
    hs.add(_nth_weekday(y, 9, 0, 1))                              # Labor
    hs.add(_nth_weekday(y, 11, 3, 4))                             # Thanksgiving
    hs.add(_observed(pd.Timestamp(y, 12, 25)))                    # Christmas
    return {h.normalize() for h in hs}


def is_nyse_holiday(d) -> bool:
    d = pd.Timestamp(d).normalize()
    return d in nyse_holidays(d.year)


def _hm(s: str) -> tuple[int, int]:
    return int(s[:2]), int(s[3:])


def build_session_bars(epic: str, stream_df: Optional[pd.DataFrame] = None,
                       window: tuple[str, str] = US_SESSION, archive_dir: Path = ARCHIVE_DIR,
                       min_bars: int = MIN_BARS) -> pd.DataFrame:
    """One OHLC bar per weekday for the session window, from archive + stream candles."""
    frames = []
    a = load_archive(epic, archive_dir)
    if a is not None and len(a):
        frames.append(a[["date", "open", "high", "low", "close"]])
    if stream_df is not None and len(stream_df) and "date" in stream_df.columns:
        s = stream_df[["date", "open", "high", "low", "close"]].copy()
        s["date"] = pd.to_datetime(s["date"])
        frames.append(s)
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "n"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    df = df[df["date"].dt.weekday < 5]
    (h0, m0), (h1, m1) = _hm(window[0]), _hm(window[1])
    t = df["date"].dt.hour * 60 + df["date"].dt.minute
    df = df[(t >= h0 * 60 + m0) & (t < h1 * 60 + m1)]
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "n"])
    g = df.groupby(df["date"].dt.normalize()).agg(open=("open", "first"), high=("high", "max"),
                                                     low=("low", "min"), close=("close", "last"), n=("close", "size"))
    g = g[g["n"] >= min_bars]
    if g.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "n"])
    g = g.reset_index().rename(columns={g.index.name or "index": "date"})
    g = g[~g["date"].map(is_nyse_holiday)]
    g["volume"] = 0
    return g[["date", "open", "high", "low", "close", "volume", "n"]].reset_index(drop=True)
