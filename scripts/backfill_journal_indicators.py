#!/usr/bin/env python3
"""One-off: backfill trades.adx / trades.rsi from the reason string for rows
that were written 0.0 before the indicator-snapshot fix. ATR/EMAs are NOT in
the reason text so they remain 0.0 for historical rows (recoverable only going
forward). Idempotent: only touches rows where adx=0 or rsi=0 AND reason carries
the values. Read-only-safe: parses then UPDATEs by id, nothing destructive."""
import sqlite3, re

con = sqlite3.connect("/app/data/trade_journal.db")
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT id, reason, adx, rsi FROM trades "
    "WHERE (adx=0 OR adx IS NULL OR rsi=0 OR rsi IS NULL) AND reason LIKE '%ADX=%'"
).fetchall()

updated = 0
for r in rows:
    reason = r["reason"] or ""
    ma = re.search(r"ADX=([\d.]+)", reason)
    mr = re.search(r"RSI=([\d.]+)", reason)
    adx = float(ma.group(1)) if ma else (r["adx"] or 0.0)
    rsi = float(mr.group(1)) if mr else (r["rsi"] or 0.0)
    if (ma and adx != (r["adx"] or 0.0)) or (mr and rsi != (r["rsi"] or 0.0)):
        con.execute("UPDATE trades SET adx=?, rsi=? WHERE id=?", (adx, rsi, r["id"]))
        updated += 1

con.commit()
print(f"Rows scanned: {len(rows)}  |  rows backfilled (adx/rsi from reason): {updated}")
# Verify
tot = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
zero = con.execute("SELECT COUNT(*) FROM trades WHERE adx=0 AND reason LIKE '%ADX=%'").fetchone()[0]
print(f"Trades total: {tot}  |  still adx=0 despite ADX in reason: {zero}")
