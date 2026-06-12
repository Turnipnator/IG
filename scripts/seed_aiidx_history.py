#!/usr/bin/env python3
"""ONE-OFF REST seed of AI Index history into the durable candle archive.

AI Index (IX.D.AIIDX.DAILY.IP) is IG-proprietary — no Yahoo equivalent — so it
can't be backtested the usual way. The free stream-harvester (src/streaming.py
archive_candles_to_disk) builds history going FORWARD; this script gives it a
head-start by pulling recent 5m history via the IG REST endpoint ONCE and writing
it into the same archive format the harvester appends to.

COST: ~num_points of the 10,000/week historical allowance, charged ONCE. Default
500 (~5% of the weekly budget). Do NOT run this on a schedule — it's a one-off
seed; the harvester is free and takes over from here.

RUN IN-CONTAINER ON THE VPS (live creds live there, never committed):
    docker exec ig-trading-bot python3 scripts/seed_aiidx_history.py [num_points]

Caveat: IG snapshotTime is in the account timezone; the live harvester stamps
candles off the stream clock. A small tz seam between seed and live history is
possible — archive_loader de-dupes by timestamp and the harvester is the
authoritative long-term source, so treat the seed as a rough jump-start.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app" if os.path.exists("/app") else
                  os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_ig_config
from src.client import IGClient

EPIC = "IX.D.AIIDX.DAILY.IP"
ARCHIVE_DIR = Path("/app/data/candle_archive") if os.path.exists("/app") \
    else Path("data/candle_archive")


def main():
    num_points = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    cfg = load_ig_config()
    client = IGClient(cfg)
    if not client.login():
        print("LOGIN FAILED — aborting (no allowance spent).")
        return

    print(f"Seeding {EPIC} — requesting {num_points} MINUTE_5 points "
          f"(~{num_points} allowance pts, ONE-OFF)...")
    df = client.get_historical_prices(EPIC, resolution="MINUTE_5",
                                      num_points=num_points, use_cache=False)
    if df is None or df.empty:
        print("No data returned (allowance exceeded, market unavailable, or "
              "weekend skip). Nothing written.")
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE_DIR / f"{EPIC}.jsonl"

    # De-dupe against anything the live harvester already wrote.
    existing = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.add(json.loads(line)["timestamp"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    written = 0
    with open(path, "a") as f:
        for _, r in df.iterrows():
            ts = r["date"].isoformat()
            if ts in existing:
                continue
            f.write(json.dumps({
                "timestamp": ts,
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": int(r.get("volume", 0) or 0),
            }) + "\n")
            written += 1

    span_h = (df["date"].iloc[-1] - df["date"].iloc[0]).total_seconds() / 3600
    print(f"Fetched {len(df)} candles ({df['date'].iloc[0]} -> "
          f"{df['date'].iloc[-1]}, ~{span_h:.0f}h). Wrote {written} new "
          f"(skipped {len(df) - written} already present).")
    print(f"Archive: {path}")


if __name__ == "__main__":
    main()
