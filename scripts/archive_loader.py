#!/usr/bin/env python3
"""Load a durable candle-archive JSONL (harvested from the free stream — see
src/streaming.py archive_candles_to_disk) into a clean OHLC DataFrame.

This is the Yahoo-independent, IG-native backtest source: real-instrument candles
with no proxy error and zero historical-allowance usage. Use it in place of
Backtester.fetch_data for IG-only EPICs (e.g. AI Index) once enough history has
accumulated. Sorts by timestamp and de-dupes (the seed + the live harvester can
overlap at the seam).
"""
import json
import os
from pathlib import Path

import pandas as pd

ARCHIVE_DIR = Path("/app/data/candle_archive") if os.path.exists("/app") \
    else Path("data/candle_archive")


def load_archive(epic: str, archive_dir: Path = ARCHIVE_DIR) -> pd.DataFrame:
    """Return a sorted, de-duped OHLC DataFrame (columns: date, open, high, low,
    close, volume) for an EPIC, or an empty frame if no archive exists yet."""
    path = archive_dir / f"{epic}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn line from a concurrent append
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"])
    df = (df.drop_duplicates(subset="date")
            .sort_values("date")
            .reset_index(drop=True))
    return df[["date", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    import sys
    epic = sys.argv[1] if len(sys.argv) > 1 else "IX.D.AIIDX.DAILY.IP"
    df = load_archive(epic)
    if df.empty:
        print(f"{epic}: no archive yet")
    else:
        span_h = (df["date"].iloc[-1] - df["date"].iloc[0]).total_seconds() / 3600
        print(f"{epic}: {len(df)} candles  "
              f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]}  (~{span_h:.0f}h)")
