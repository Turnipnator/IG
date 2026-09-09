"""Durable per-EPIC store of IG DAY bars for the daily-trend strategy (2026-09-09).

Why its own store and not the shared price cache: get_historical_prices caches
by `epic|DAY`, the HTF refresh writes 30 bars into that key once a day, and the
55-bar entry channel needs ~80 bars of history. Reading through the shared cache
would hand the strategy a truncated or forming frame without any error. This
store is append/merge-only JSONL under data/ (gitignored, survives rebuilds),
seeded ONCE (~130 REST points) and then topped up with a handful of points a day.

Rows: {"date": "YYYY-MM-DD", "open", "high", "low", "close", "volume"} — mid
prices as the client computes them. A re-fetched date REPLACES the stored row, so
the forming-bar snapshot taken at 22:30 is superseded by the final bar on the
next refresh. Sunday stubs are kept on disk (faithful record) and dropped by
daily_trend.prepare_daily at read time.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DAILY_BARS_DIR = (Path("/app/data") if os.path.exists("/app") else Path("data")) / "daily_bars"
SEED_POINTS = 130          # one-off: 55-bar channel + 20 ATR + headroom for Sunday stubs
REFRESH_POINTS = 5         # daily top-up: covers a weekend gap and re-finalises yesterday
COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _path(epic: str, base: Optional[Path] = None) -> Path:
    return (base or DAILY_BARS_DIR) / f"{epic}.jsonl"


def load_daily_bars(epic: str, base: Optional[Path] = None) -> pd.DataFrame:
    p = _path(epic, base)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)[COLUMNS]


def merge_daily_bars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union by date, NEW rows win, sorted. Both frames carry a `date` column."""
    if new is None or len(new) == 0:
        return existing.reset_index(drop=True)
    n = new.copy()
    n["date"] = pd.to_datetime(n["date"]).dt.normalize()
    if "volume" not in n.columns:
        n["volume"] = 0
    n = n[COLUMNS]
    both = pd.concat([existing[COLUMNS], n], ignore_index=True) if len(existing) else n
    return both.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def store_daily_bars(epic: str, df: pd.DataFrame, base: Optional[Path] = None) -> None:
    p = _path(epic, base)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, r in df.iterrows():
        lines.append(json.dumps({"date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                                 "open": float(r["open"]), "high": float(r["high"]),
                                 "low": float(r["low"]), "close": float(r["close"]),
                                 "volume": float(r.get("volume", 0) or 0)}))
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    tmp.replace(p)


def refresh_daily_bars(client, epic: str, min_bars: int, base: Optional[Path] = None) -> pd.DataFrame:
    """Fetch DAY bars from IG (seed-size if the store is thin, else a small top-up),
    merge into the store and return the merged frame. Bypasses the shared price
    cache (use_cache=False) — see module docstring. Any failure returns the store
    unchanged; the caller decides whether the frame is usable."""
    existing = load_daily_bars(epic, base)
    n_points = SEED_POINTS if len(existing) < min_bars else REFRESH_POINTS
    try:
        fetched = client.get_historical_prices(epic, resolution="DAY", num_points=n_points, use_cache=False)
    except Exception as e:
        logger.warning(f"[DAILY-TREND] {epic}: DAY fetch failed ({e}); store unchanged ({len(existing)} bars)")
        return existing
    if fetched is None or len(fetched) == 0:
        logger.warning(f"[DAILY-TREND] {epic}: DAY fetch returned nothing; store unchanged ({len(existing)} bars)")
        return existing
    merged = merge_daily_bars(existing, fetched)
    store_daily_bars(epic, merged, base)
    logger.info(f"[DAILY-TREND] {epic}: fetched {len(fetched)} DAY bars ({n_points} requested) -> "
                f"store {len(merged)} bars, last {merged['date'].iloc[-1].date()}")
    return merged
