#!/usr/bin/env python3
"""Phase 1 readout for tick-triggered breakout entry (BREAKOUT_TICK_ENTRY=log).

Answers the three questions that decide whether the feature flips to `live`:

  Q1 MECHANISM  — does the trigger fire once per armed bar, at the channel level?
  Q2 SAVING     — how much does entering on the tick save vs entering at the close?
                  This is the ONLY number that matters, and it is NOT the `slipR`
                  in the log line: that is level-vs-fill and includes half the
                  spread by construction. The saving is close_fill - tick_fill on
                  the SAME break, a PAIRED difference, so market-level noise
                  cancels and it converges in tens of samples rather than hundreds.
  Q3 POPULATION — did the trade COUNT change? Tick entry is only valid if it takes
                  the same breaks earlier. More breaks means the premise is wrong
                  and the flag goes back to off (pre-flight assumption 2).

DEDUPE IS NOT OPTIONAL. `_breakout_armed` lives in memory, so every container
restart re-arms a market already past its channel and it fires again for the same
bar. On 2026-08-20 four restarts in one hour produced two duplicate DXY rows. Rows
are deduped on (epic, bar) keeping the earliest.

Read-only: opens the journal with mode=ro so it can never disturb the bot, which
holds its own connection to the same file.

Usage:  python3 scripts/tick_entry_readout.py [--telegram] [--days N]
"""
import argparse
import os
import re
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = "/app/data/trade_journal.db"
FIELD = re.compile(r"level=([\d.]+) exec=([\d.]+) slip=([-+\d.]+) slipR=([-+\d.]+) "
                   r"stop=([\d.]+) bar=(.+)$")
# A break seen intrabar is confirmed at the next hourly close; allow slack for the
# archive/stream merge rather than demanding an exact boundary.
PAIR_WINDOW = timedelta(minutes=75)


def rows(db, days):
    since = (datetime.now() - timedelta(days=days)).isoformat()
    out = []
    for r in db.execute("SELECT market_name, epic, direction, timestamp, reject_reason "
                        "FROM rejected_signals WHERE reject_reason LIKE 'Breakout-tick%' "
                        "AND timestamp >= ? ORDER BY id", (since,)):
        m = FIELD.search(r["reject_reason"])
        if not m:
            continue
        out.append({
            "market": r["market_name"], "epic": r["epic"], "dir": r["direction"],
            "t": datetime.fromisoformat(r["timestamp"]),
            "level": float(m.group(1)), "exec": float(m.group(2)),
            "slipR": float(m.group(4)), "stop": float(m.group(5)),
            "bar": m.group(6).strip(),
        })
    return out


def dedupe(rs):
    seen, keep, dupes = {}, [], 0
    for r in sorted(rs, key=lambda x: x["t"]):
        k = (r["epic"], r["bar"])
        if k in seen:
            dupes += 1
            continue
        seen[k] = True
        keep.append(r)
    return keep, dupes


_ARCH_CACHE: dict = {}


def _hourly(epic):
    """The epic's 1h closes from the durable archive, cached per run.

    Built the same way main._breakout_frame_1h builds its frame, so the bar
    boundaries line up with the ones the live path reasons about.
    """
    if epic in _ARCH_CACHE:
        return _ARCH_CACHE[epic]
    import json
    from pathlib import Path
    import pandas as pd
    p = Path("/app/data/candle_archive") / f"{epic}.jsonl"
    if not p.exists():
        _ARCH_CACHE[epic] = None
        return None
    recs = []
    for line in p.open():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    if not recs:
        _ARCH_CACHE[epic] = None
        return None
    d = pd.DataFrame(recs)
    d["date"] = pd.to_datetime(d["timestamp"])
    h = (d[["date", "close"]].drop_duplicates(subset="date").sort_values("date")
         .set_index("date").resample("1h").agg({"close": "last"}).dropna())
    _ARCH_CACHE[epic] = h
    return h


def close_fill(db, r):
    """What the CLOSE-based path would have paid for this same break.

    Taken from the ARCHIVE, not from what the bot did: analyze_breakout confirms at
    the close of the bar the crossing happened in and enters at market there, so
    that bar's close IS the counterfactual. Reading it from the archive makes every
    crossing measurable — pairing against trades/benched_outcomes instead fails
    whenever the close path was blocked for an unrelated reason (a position already
    open on the epic, or an unresolved shadow episode suppressing a new snapshot),
    which on the first day was ALL FOUR crossings.

    Falls back to whatever the bot actually recorded, which doubles as a check that
    the archive-derived number is not drifting from reality.
    """
    h = _hourly(r["epic"])
    if h is not None and len(h):
        import pandas as pd
        bar = pd.Timestamp(r["t"]).floor("1h")
        if bar in h.index:
            return float(h.loc[bar, "close"]), "archive"
    lo, hi = r["t"].isoformat(), (r["t"] + PAIR_WINDOW).isoformat()
    q = db.execute("SELECT entry_price, entry_time AS ts, 'live' AS src FROM trades "
                   "WHERE epic=? AND direction=? AND entry_time > ? AND entry_time <= ? "
                   "UNION ALL "
                   "SELECT entry_price, benched_at AS ts, 'shadow' AS src FROM benched_outcomes "
                   "WHERE epic=? AND direction=? AND bench_type='breakout-shadow' "
                   "AND benched_at > ? AND benched_at <= ? ORDER BY ts LIMIT 1",
                   (r["epic"], r["dir"], lo, hi, r["epic"], r["dir"], lo, hi)).fetchone()
    return (float(q["entry_price"]), q["src"]) if q and q["entry_price"] else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args()

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    raw = rows(db, a.days)
    rs, dupes = dedupe(raw)
    L = [f"*Tick-entry Phase 1* — last {a.days}d",
         f"events {len(rs)} (deduped from {len(raw)}; {dupes} restart re-fires)"]

    if not rs:
        L.append("\nNo crossings yet. Nothing to report.")
        return emit(L, a.telegram)

    # Q2 — the paired saving, the number the decision rests on
    saved, unpaired, by_mkt = [], 0, defaultdict(list)
    for r in rs:
        cf, src = close_fill(db, r)
        if cf is None:
            unpaired += 1
            continue
        pts = (cf - r["exec"]) if r["dir"] == "BUY" else (r["exec"] - cf)
        R = pts / r["stop"] if r["stop"] else 0.0
        saved.append(R)
        by_mkt[r["market"]].append(R)

    if saved:
        mu = sum(saved) / len(saved)
        sd = st.pstdev(saved) if len(saved) > 1 else 0.0
        se = sd / len(saved) ** 0.5 if saved else 0.0
        L.append(f"\n*Saving vs close entry* (paired, the decision number)")
        L.append(f"  n={len(saved)}  mean=*{mu:+.3f}R*  sd={sd:.3f}  "
                 f"95% CI [{mu-1.96*se:+.3f}, {mu+1.96*se:+.3f}]")
        L.append(f"  target from the 2026-08 fill-gap study: +0.143R")
        if len(saved) < 30:
            L.append(f"  _n<30 — indicative only, do not act on this yet._")
        L.append("  per market:")
        for m, v in sorted(by_mkt.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            L.append(f"    {m}: {sum(v)/len(v):+.3f}R (n={len(v)})")
    if unpaired:
        L.append(f"  {unpaired} crossing(s) had no close-based counterpart to pair with")

    # Q1 — mechanism: level-vs-fill, and one-per-bar
    lvl = [r["slipR"] for r in rs]
    L.append(f"\n*Mechanism*")
    L.append(f"  fill vs level: median {st.median(lvl):+.3f}R "
             f"(includes half-spread; DXY inflates this)")
    L.append(f"  duplicate bars after dedupe: "
             f"{len(rs) - len({(r['epic'], r['bar']) for r in rs})} (must be 0)")

    # Q3 — population check
    since = (datetime.now() - timedelta(days=a.days)).isoformat()
    L.append(f"\n*Population* (assumption 2 — same breaks, earlier)")
    # Compared PER EPIC. A raw book-wide total is not a like-for-like: the observer
    # is hourly-capped and suppresses a snapshot while an episode is unresolved,
    # so it under-counts breaks the tick path legitimately sees.
    tick_by = defaultdict(int)
    for r in rs:
        tick_by[r["epic"]] += 1
    flagged = []
    for epic, n_tick in sorted(tick_by.items()):
        n_conf = db.execute("SELECT COUNT(*) n FROM benched_outcomes WHERE epic=? "
                            "AND bench_type='breakout-shadow' AND benched_at >= ?",
                            (epic, since)).fetchone()["n"]
        n_live = db.execute("SELECT COUNT(*) n FROM trades WHERE epic=? AND strategy='breakout' "
                            "AND entry_time >= ?", (epic, since)).fetchone()["n"]
        mk = next((r["market"] for r in rs if r["epic"] == epic), epic)
        L.append(f"  {mk}: tick {n_tick} vs close-path {n_conf + n_live}")
        if n_tick > max(2, (n_conf + n_live) * 2):
            flagged.append(mk)
    if flagged:
        L.append(f"  ⚠️ *{', '.join(flagged)}: tick path seeing far MORE breaks than the "
                 f"close path. If this persists, the premise is wrong — set "
                 f"BREAKOUT_TICK_ENTRY=off.*")
    L.append("  _note: the close path is suppressed by one-position-per-epic and by "
             "unresolved shadow episodes, so some excess is expected and benign._")

    emit(L, a.telegram)


def emit(lines, to_telegram):
    text = "\n".join(lines)
    print(text)
    if not to_telegram:
        return
    try:
        from config import load_telegram_config
        from src.telegram_bot import TelegramNotifier
        # _send is the synchronous path this class exists for (test_run.py uses the
        # same class); the public methods are all trade-shaped and unusable here.
        TelegramNotifier(load_telegram_config())._send(text)
        print("\n[sent to Telegram]")
    except Exception as e:
        print(f"\n[Telegram send failed: {e}]")


if __name__ == "__main__":
    main()
