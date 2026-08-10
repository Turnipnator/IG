"""One-off backfill: turn already-logged breakout-shadow SIGNALS into resolved
EPISODES, so the v3 review has history instead of starting from the deploy date.

The always-on observer (126300d) has been logging 'Breakout-shadow:' rows to
rejected_signals since 2026-07-24 with no outcome attached — hundreds of signals,
zero P&L. Outcome tracking went live 2026-08-10 (0057aa9); this replays everything
before it.

Source of truth is rejected_signals, NOT a fresh sweep of the archive: those rows
are what the observer ACTUALLY emitted, with the live HTF trend already applied at
the time. The HTF trend at an arbitrary past hour cannot be reconstructed, so a
from-scratch archive sweep would silently invent signals the observer never made.

Rows are written with bench_type='breakout-shadow-bf' (not 'breakout-shadow') so
they can never collide with live episode dedup — has_open_breakout_shadow only
looks at the live bench_type. Union the two for the review; the split also lets
you sanity-check backfilled against forward-collected results.

Dry run (prints proposed episodes, writes nothing):
    docker exec -i ig-trading-bot python3 - < scripts/backfill_breakout_shadow.py
Apply:
    docker exec -i ig-trading-bot python3 - --apply < scripts/backfill_breakout_shadow.py
"""
import json
import re
import sqlite3
import sys
from datetime import datetime

import pandas as pd

from src import breakout

APPLY = "--apply" in sys.argv
BENCH_TYPE = "breakout-shadow-bf"
MAX_BARS = 240  # ~10 trading days, matching main._resolve_breakout_shadow
ARCHIVE = "/app/data/candle_archive"

db = sqlite3.connect("/app/data/trade_journal.db")
db.row_factory = sqlite3.Row

existing = db.execute(
    "SELECT COUNT(*) n FROM benched_outcomes WHERE bench_type=?", (BENCH_TYPE,)
).fetchone()["n"]
if existing:
    print(f"{existing} '{BENCH_TYPE}' rows already present — delete them first to re-run:")
    print(f"  DELETE FROM benched_outcomes WHERE bench_type='{BENCH_TYPE}';")
    raise SystemExit(1)


def hourly(epic: str):
    """1h frame from the durable archive (identical to _breakout_frame_1h's output:
    verified 0.0 max OHLC difference against archive+stream on 2026-08-10)."""
    rows = []
    try:
        with open(f"{ARCHIVE}/{epic}.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return None
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["timestamp"])
    d = (d[["date", "open", "high", "low", "close"]]
         .drop_duplicates(subset="date").sort_values("date").set_index("date"))
    h = d.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return h.iloc[:-1].reset_index()


def replay(frame, epic, direction, entry, risk, start):
    """Byte-for-byte main._resolve_breakout_shadow: k×ATR stop ratcheted to the
    Donchian-M trail, checked against each subsequent CLOSED 1h bar."""
    is_buy = direction == "BUY"
    fwd = frame[frame["date"] > pd.to_datetime(start)].reset_index(drop=True)
    if fwd.empty:
        return None
    base = len(frame) - len(fwd)
    stop = entry - risk if is_buy else entry + risk
    initial = stop
    ncdl = 0
    for i in range(len(fwd)):
        ncdl = i + 1
        bar = fwd.iloc[i]
        if (is_buy and float(bar["low"]) <= stop) or \
           ((not is_buy) and float(bar["high"]) >= stop):
            r = ((stop - entry) if is_buy else (entry - stop)) / risk
            return ("WIN" if r > 0 else "LOSS",
                    "stop" if stop == initial else "trail", stop, r, ncdl)
        lvl = breakout.exit_channel(frame.iloc[:base + i + 1], epic, direction)
        if lvl is not None:
            stop = max(stop, lvl) if is_buy else min(stop, lvl)
        if ncdl >= MAX_BARS:
            break
    px = float(fwd.iloc[-1]["close"])
    r = ((px - entry) if is_buy else (entry - px)) / risk
    if ncdl >= MAX_BARS:
        return ("EXPIRED", "expired", px, r, ncdl)
    return ("OPEN", None, None, None, ncdl)  # still running at the archive edge


signals = db.execute(
    """SELECT epic, market_name, direction, timestamp, adx AS atr, reject_reason
       FROM rejected_signals
       WHERE reject_reason LIKE 'Breakout-shadow:%'
       ORDER BY timestamp"""
).fetchall()
print(f"{len(signals)} logged breakout-shadow signals "
      f"({'APPLY' if APPLY else 'DRY RUN'})\n")

frames: dict = {}
open_until: dict = {}   # epic -> exit timestamp of the episode currently running
episodes = []
skipped_dup = skipped_parse = skipped_frame = 0

for s in signals:
    epic = s["epic"]
    ts = pd.to_datetime(s["timestamp"])

    # Dedup exactly as live does: a standing break re-signals every hour, but it is
    # ONE position held to the trail exit with no re-entry while open.
    if epic in open_until and (open_until[epic] is None or ts <= open_until[epic]):
        skipped_dup += 1
        continue

    m = re.search(r"stop ([\d.]+)=", s["reject_reason"] or "")
    if not m:
        skipped_parse += 1
        continue
    risk = float(m.group(1))
    if risk <= 0:
        skipped_parse += 1
        continue

    if epic not in frames:
        frames[epic] = hourly(epic)
    frame = frames[epic]
    if frame is None or frame.empty:
        skipped_frame += 1
        continue

    # Entry = close of the last CLOSED hour at/before the signal, the same bar the
    # observer signalled on. (market.mid_price at the time is not recorded.)
    prior = frame[frame["date"] <= ts]
    if prior.empty:
        skipped_frame += 1
        continue
    entry = float(prior.iloc[-1]["close"])
    bench_at = prior.iloc[-1]["date"]

    out = replay(frame, epic, s["direction"], entry, risk, bench_at)
    if out is None:
        skipped_frame += 1
        continue
    status, outcome, exit_px, r, ncdl = out

    fwd = frame[frame["date"] > bench_at].reset_index(drop=True)
    open_until[epic] = (fwd.iloc[ncdl - 1]["date"]
                        if status != "OPEN" and ncdl <= len(fwd) else None)

    episodes.append({
        "epic": epic, "market": s["market_name"], "direction": s["direction"],
        "benched_at": bench_at, "entry": entry, "risk": risk,
        "status": status, "outcome": outcome, "exit_px": exit_px,
        "r": r, "bars": ncdl,
    })

print(f"{len(episodes)} distinct episodes "
      f"({skipped_dup} duplicate signals of a standing break collapsed, "
      f"{skipped_parse} unparseable, {skipped_frame} no usable frame)\n")

by_market: dict = {}
for e in episodes:
    b = by_market.setdefault(e["market"], {"n": 0, "w": 0, "l": 0, "r": 0.0, "open": 0})
    b["n"] += 1
    if e["status"] == "OPEN":
        b["open"] += 1
        continue
    b["r"] += e["r"]
    if e["status"] == "WIN":
        b["w"] += 1
    elif e["status"] == "LOSS":
        b["l"] += 1

print(f"{'market':<22} {'eps':>4} {'W':>3} {'L':>3} {'open':>5} {'sum R':>8} {'avg R':>7}")
print("-" * 60)
for name, b in sorted(by_market.items(), key=lambda x: -x[1]["r"]):
    closed = b["n"] - b["open"]
    avg = (b["r"] / closed) if closed else 0.0
    print(f"{name:<22} {b['n']:>4} {b['w']:>3} {b['l']:>3} {b['open']:>5} "
          f"{b['r']:>+8.2f} {avg:>+7.2f}")

if APPLY:
    for e in episodes:
        db.execute(
            """INSERT INTO benched_outcomes
               (epic, market_name, direction, benched_at, entry_price,
                stop_distance, limit_distance, score, bench_type, status,
                outcome, resolved_at, candles_to_resolve, r_multiple, exit_price)
               VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?)""",
            (e["epic"], e["market"], e["direction"], e["benched_at"].isoformat(),
             e["entry"], e["risk"], BENCH_TYPE, e["status"], e["outcome"],
             datetime.now().isoformat(), e["bars"],
             None if e["r"] is None else round(e["r"], 3),
             None if e["exit_px"] is None else round(e["exit_px"], 2)),
        )
    db.commit()
    print(f"\n{len(episodes)} episodes written as bench_type='{BENCH_TYPE}'")
else:
    print("\nDRY RUN — nothing written. Re-run with --apply to persist.")
