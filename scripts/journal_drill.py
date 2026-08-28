#!/usr/bin/env python3
"""Drill into the confidence cliff + hour bleed. ADX is parsed from the reason
string because the trades.adx column is stored as 0.0 (never backfilled)."""
import sqlite3, collections, datetime as dt, re

con = sqlite3.connect("/app/data/trade_journal.db")
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute("SELECT * FROM trades WHERE pnl IS NOT NULL ORDER BY id")]


def adx_of(t):
    m = re.search(r"ADX=([\d.]+)", t.get("reason") or "")
    return float(m.group(1)) if m else None


def hour(t):
    try:
        return dt.datetime.fromisoformat(t["entry_time"]).hour
    except Exception:
        return None


print("=== A. The 0.80+ confidence trades — ADX (from reason) + outcome ===")
hi = [t for t in rows if (t["confidence"] or 0) >= 0.80]
for t in sorted(hi, key=lambda x: x["pnl"]):
    a = adx_of(t)
    print(f"  {t['market_name']:14s} {t['direction']:4s} conf={t['confidence']:.2f} "
          f"ADX={a if a is not None else '?':>5} pnl=GBP{t['pnl']:+.2f}")
hav = [adx_of(t) for t in hi if adx_of(t) is not None]
allav = [adx_of(t) for t in rows if adx_of(t) is not None]
print(f"  mean ADX of 0.80+ entries: {sum(hav)/len(hav):.1f}" if hav else "")
print(f"  mean ADX of ALL entries:   {sum(allav)/len(allav):.1f}" if allav else "")
# winners vs losers ADX across the whole book
win_adx = [adx_of(t) for t in rows if t["pnl"] > 0.5 and adx_of(t) is not None]
los_adx = [adx_of(t) for t in rows if t["pnl"] < -0.5 and adx_of(t) is not None]
print(f"  mean ADX of WINNERS: {sum(win_adx)/len(win_adx):.1f}  |  LOSERS: {sum(los_adx)/len(los_adx):.1f}")

print("\n=== B. Hour 16 UTC by market (is -81 broad or one market?) ===")
h16 = [t for t in rows if hour(t) == 16]
bm = collections.defaultdict(lambda: [0, 0.0])
for t in h16:
    bm[t["market_name"]][0] += 1
    bm[t["market_name"]][1] += t["pnl"]
for m, (c, p) in sorted(bm.items(), key=lambda x: x[1][1]):
    print(f"  {m:16s} {c:2d}t  GBP{p:+.2f}")

print("\n=== C. Hours 14/16/19 combined vs rest ===")
bad = {14, 16, 19}


def stat(g):
    w = sum(1 for t in g if t["pnl"] > 0.5)
    l = sum(1 for t in g if t["pnl"] < -0.5)
    wr = w / (w + l) * 100 if (w + l) else 0
    return f"{len(g)}t {w}W/{l}L WR={wr:.0f}% GBP{sum(t['pnl'] for t in g):+.2f}"


print(f"  hrs 14/16/19: {stat([t for t in rows if hour(t) in bad])}")
print(f"  all others:   {stat([t for t in rows if hour(t) not in bad])}")
