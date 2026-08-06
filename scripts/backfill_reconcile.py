"""One-off backfill: reconcile stale PROVISIONAL/UNMATCHED journal rows.

These rows are victims of the entry-time age-out bug (fixed 2026-08-06): trades
held >3h were marked UNMATCHED the minute they closed, before the reconciler
ever searched for their close transaction. IG still holds the transactions, so
match each row against its own exit-date window and promote to CLOSED with the
broker-confirmed pnl/exit_price.

Journal-only correction: deliberately does NOT touch the live daily P&L
counters — every affected row belongs to an already-summarised session.

Run inside the container (dry-run by default, prints proposed updates):
    docker exec -i ig-trading-bot python3 - < scripts/backfill_reconcile.py
Apply:
    docker exec -i ig-trading-bot python3 - --apply < scripts/backfill_reconcile.py
"""
import sqlite3
import sys
import time
from datetime import datetime, timedelta

from src.client import IGClient
from config import load_ig_config

APPLY = "--apply" in sys.argv

client = IGClient(load_ig_config())
if not client.login():
    print("IG login failed")
    raise SystemExit(1)

db = sqlite3.connect("/app/data/trade_journal.db")
db.row_factory = sqlite3.Row

rows = db.execute(
    """SELECT id, deal_id, market_name, direction, size, entry_price,
              entry_time, exit_time, pnl, status
       FROM trades
       WHERE status IN ('PROVISIONAL', 'UNMATCHED')
       ORDER BY id"""
).fetchall()
print(f"{len(rows)} stale rows to reconcile ({'APPLY' if APPLY else 'DRY RUN'})\n")


def fetch_window(center: datetime) -> list[dict]:
    """Transactions in a ±24h window around a row's exit (1 free REST call)."""
    frm = (center - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    to = (center + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    resp = client.session.get(
        f"{client.config.base_url}/history/transactions",
        params={"type": "ALL", "from": frm, "to": to, "pageSize": 50},
        headers=client._get_headers(version="2"),
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  txn fetch failed ({resp.status_code}) for window {frm}..{to}")
        return []
    return resp.json().get("transactions", [])


matched = unmatched = 0
for row in rows:
    ref = row["exit_time"] or row["entry_time"]
    try:
        center = datetime.fromisoformat(ref)
    except (TypeError, ValueError):
        print(f"#{row['id']} {row['market_name']}: bad exit_time {ref!r} — skipped")
        continue

    txns = fetch_window(center)
    time.sleep(2)  # stay well under the trading-history rate limit

    hit = None
    for txn in txns:
        if txn.get("transactionType") != "DEAL":
            continue
        try:
            txn_open = float(txn.get("openLevel", 0))
            txn_size = float(str(txn.get("size", "0")).replace("+", ""))
        except (ValueError, TypeError):
            continue
        txn_dir = "BUY" if txn_size > 0 else "SELL"
        if (
            abs(txn_open - row["entry_price"]) < 1.0
            and txn_dir == row["direction"]
            and abs(abs(txn_size) - (row["size"] or 0)) < 0.01
        ):
            hit = txn
            break

    if hit is None:
        unmatched += 1
        print(
            f"#{row['id']} {row['market_name']:20s} {row['direction']:4s} "
            f"entry={row['entry_price']} exit_t={ref[:16]} — NO MATCH "
            f"(stays {row['status']}, cached pnl {row['pnl']})"
        )
        continue

    actual_pnl = IGClient._parse_pnl(hit.get("profitAndLoss", "0"))
    exit_price = float(hit.get("closeLevel") or 0.0)
    matched += 1
    print(
        f"#{row['id']} {row['market_name']:20s} {row['direction']:4s} "
        f"entry={row['entry_price']} exit_t={ref[:16]} — MATCH "
        f"{hit.get('instrumentName','?')!r} pnl {row['pnl']} -> {actual_pnl} "
        f"exit_price -> {exit_price}"
    )
    if APPLY:
        db.execute(
            """UPDATE trades SET pnl = ?, exit_price = ?, status = 'CLOSED'
               WHERE id = ? AND status IN ('PROVISIONAL', 'UNMATCHED')""",
            (actual_pnl, exit_price, row["id"]),
        )
        db.commit()

print(f"\n{matched} matched, {unmatched} no-match ({'written' if APPLY else 'dry run — nothing written'})")
