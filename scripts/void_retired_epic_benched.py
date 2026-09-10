"""One-off: VOID benched_outcomes rows left OPEN on an epic no longer in config.py.

Row #143 (Crude Oil breakout-shadow, benched 2026-08-19T15:00 on EN.D.CL.Month1.IP)
was snapshotted at the exact moment that Month1 slot re-pointed to an OFFLINE
contract (project-month1-epic-rollover-2026-09). Every resolver runs per CONFIGURED
epic -- _resolve_breakout_shadow(epic, df) is only ever called for markets in
MARKETS -- so once Crude moved to CC.D.CL.USS.IP (2026-09-04, a5f0604) nothing could
pick #143 up again. It cannot be replayed honestly either: the archive after 15:00
that day is the OTHER contract's ticks (a two-contract splice), which is exactly the
cross-instrument contamination the resolver's frame-ownership guard exists to refuse.
So it gets a terminal status that says precisely that:

    status='VOID', outcome='epic_retired', r_multiple NULL, exit_price NULL

NULL r_multiple keeps it OUT of every pooled-R readout (they filter or COALESCE on
r_multiple); the row itself stays, so the audited row count is unchanged and the
record that a signal fired survives. EXPIRED was deliberately NOT reused: in this
table EXPIRED means "ran the full MAX_BARS horizon" -- a real model outcome with an R.

Dry run (lists candidates, writes nothing):
    docker exec -i -w /app ig-trading-bot python3 - < scripts/void_retired_epic_benched.py
Apply, addressed by id so nothing else can be touched:
    docker exec -i -w /app ig-trading-bot python3 - --apply 143 < scripts/void_retired_epic_benched.py
"""
import sqlite3
import sys
from datetime import datetime

from config import MARKETS

DB = "/app/data/trade_journal.db"
args = sys.argv[1:]
APPLY = "--apply" in args
ids = {int(a) for a in args if a.isdigit()}

configured = {m.epic for m in MARKETS}
db = sqlite3.connect(DB, timeout=30)
db.execute("PRAGMA busy_timeout = 30000")   # the live bot holds this file open
db.row_factory = sqlite3.Row

rows = db.execute(
    "SELECT id, epic, market_name, direction, bench_type, benched_at, entry_price, stop_distance "
    "FROM benched_outcomes WHERE status='OPEN' ORDER BY benched_at").fetchall()
stale = [r for r in rows if r["epic"] not in configured]

print(f"{len(rows)} OPEN rows, {len(stale)} on epics no longer configured "
      f"({'APPLY' if APPLY else 'DRY RUN'})")
for r in stale:
    print(f"  #{r['id']:<4} {r['market_name']:<16} {r['bench_type']:<18} {r['direction']:<4} "
          f"{r['epic']:<24} benched {r['benched_at']} entry={r['entry_price']} "
          f"stop={r['stop_distance']}")

if not APPLY:
    print("\nDRY RUN -- nothing written. Re-run with --apply <id> [<id>...] to void listed rows.")
    sys.exit(0)

targets = [r for r in stale if r["id"] in ids]
missing = ids - {r["id"] for r in stale}
if missing:
    print(f"\nREFUSING: id(s) {sorted(missing)} are not OPEN rows on a retired epic.")
    sys.exit(1)
if not targets:
    print("\nNo ids given -- nothing written.")
    sys.exit(1)

with db:   # one transaction; the WHERE re-checks status so a racing resolver is never overwritten
    cur = db.executemany(
        "UPDATE benched_outcomes SET status='VOID', outcome='epic_retired', resolved_at=?, "
        "candles_to_resolve=NULL, r_multiple=NULL, exit_price=NULL "
        "WHERE id=? AND status='OPEN'",
        [(datetime.now().isoformat(), r["id"]) for r in targets])
print(f"\n{cur.rowcount} row(s) voided.")
for r in db.execute("SELECT id, status, outcome, resolved_at FROM benched_outcomes WHERE id IN (%s)"
                    % ",".join(str(r["id"]) for r in targets)):
    print(f"  #{r['id']} {r['status']} {r['outcome']} {r['resolved_at']}")
