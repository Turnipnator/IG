#!/usr/bin/env python3
"""One-off journal forensics — runs inside the container against the live DB.

Three cuts (zero API / zero Yahoo cost):
  1. Break-even scratches — stop-outs at ~breakeven imply BE moved the stop to
     entry, i.e. the trade ran >= breakeven_trigger (0.7R) in profit then gave
     it all back. Counts how often we cut a would-be winner.
  2. P&L by entry-confidence bucket — does higher confidence actually win?
  3. P&L by entry hour (UTC) — is there a dead zone to exclude?

R-multiple = pnl / (stop_distance * size): a full stop = -1R, a BE scratch ~0R.
"""
import sqlite3, collections, datetime as dt

con = sqlite3.connect("/app/data/trade_journal.db")
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(
    "SELECT * FROM trades WHERE pnl IS NOT NULL AND exit_reason IS NOT NULL ORDER BY id")]
print(f"Closed trades analysed: {len(rows)}")


def risk(t):
    return (t.get("stop_distance") or 0) * (t.get("size") or 0)


def Rmult(t):
    rk = risk(t)
    return (t["pnl"] / rk) if rk else None


# 1. BE-SCRATCH ANALYSIS
print("\n" + "=" * 70)
print("1. BREAK-EVEN SCRATCHES (reached >=0.7R then gave it back)")
print("=" * 70)
stops = [t for t in rows if "stop" in (t["exit_reason"] or "").lower()]
be_scratch, full_stop = [], []
for t in stops:
    R = Rmult(t)
    if R is None:
        continue
    if -0.20 < R < 0.20:
        be_scratch.append(t)
    elif R <= -0.6:
        full_stop.append(t)
print(f"Stop/limit-hit trades: {len(stops)}")
print(f"  BE scratches (|R|<0.20 -> hit BE then reversed): {len(be_scratch)}")
print(f"  Full stop-outs (R<=-0.6):                        {len(full_stop)}")
print(f"  BE-scratch realised P&L: GBP{sum(t['pnl'] for t in be_scratch):+.2f}")
bm = collections.defaultdict(lambda: [0, 0.0])
for t in be_scratch:
    bm[t["market_name"]][0] += 1
    bm[t["market_name"]][1] += t["pnl"]
for m, (c, p) in sorted(bm.items(), key=lambda x: -x[1][0]):
    print(f"    {m:20s} {c:2d} scratches  GBP{p:+.2f}")

# 2. P&L BY CONFIDENCE BUCKET
print("\n" + "=" * 70)
print("2. P&L BY ENTRY CONFIDENCE")
print("=" * 70)
print(f"{'Bucket':<12}{'Trades':>7}{'WR':>7}{'TotP&L':>10}{'AvgR':>7}")
for lo, hi in [(0.0, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]:
    grp = [t for t in rows if lo <= (t["confidence"] or 0) < hi]
    if not grp:
        print(f"{f'{lo:.2f}-{hi:.2f}':<12}{0:>7}")
        continue
    w = sum(1 for t in grp if t["pnl"] > 0.5)
    l = sum(1 for t in grp if t["pnl"] < -0.5)
    wr = w / (w + l) * 100 if (w + l) else 0
    tot = sum(t["pnl"] for t in grp)
    Rs = [Rmult(t) for t in grp if Rmult(t) is not None]
    avgR = sum(Rs) / len(Rs) if Rs else 0
    print(f"{f'{lo:.2f}-{hi:.2f}':<12}{len(grp):>7}{wr:>6.0f}%{tot:>+10.2f}{avgR:>+7.2f}")

# 3. P&L BY HOUR-OF-DAY (UTC)
print("\n" + "=" * 70)
print("3. P&L BY ENTRY HOUR (UTC)")
print("=" * 70)
bh = collections.defaultdict(lambda: [0, 0, 0, 0.0])
for t in rows:
    try:
        h = dt.datetime.fromisoformat(t["entry_time"]).hour
    except Exception:
        continue
    b = bh[h]
    b[0] += 1
    if t["pnl"] > 0.5:
        b[1] += 1
    elif t["pnl"] < -0.5:
        b[2] += 1
    b[3] += t["pnl"]
print(f"{'Hr':>3}{'Trades':>7}{'W':>4}{'L':>4}{'TotP&L':>10}")
for h in sorted(bh):
    n, w, l, p = bh[h]
    flag = " <== bleed" if p < -10 else (" <== strong" if p > 15 else "")
    print(f"{h:>3}{n:>7}{w:>4}{l:>4}{p:>+10.2f}{flag}")
