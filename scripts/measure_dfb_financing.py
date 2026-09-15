#!/usr/bin/env python3
"""Measure IG's ACTUAL overnight financing from the account's transaction history.

Why this exists. Every daily-horizon number in this repo charges financing from
IG's PUBLISHED formula — index long pays (benchmark + 2.5%)/365 of notional per
night — never from a reading. That matters more than it sounds: financing as a
fraction of R is price / (2 x ATR) x rate / 365, i.e. inversely proportional to
the instrument's ATR as a percent of price, and scale-invariant, so sizing cannot
escape it. Over a 40-night hold the sweep's table puts S&P at 0.41R against Gold's
0.05R — which is why daily trend-following died on S&P/WS/FTSE and survived on
Gold/NASDAQ. It was the financing, not the signal.

It is also the one cost that has been measured wrong before: the 2026-08-19 Gold
table counted transaction ROWS as nights and so halved the true charge, corrected
on 2026-09-09 to ~GBP 0.70/night (about 5.8%/yr). Index DFB financing has never
been observed on this account at all, because the bot had never held an index
position overnight — momentum was intraday. The S&P pullback arm changed that on
2026-09-10 (trade #349), and it holds ~4-6 sessions with ~100% overnight exposure.

Read-only: one login, one or more history reads, one logout. Placing no orders and
touching no state. Run it INSIDE the container so it picks up the bot's .env:

    docker exec ig-trading-bot python3 /app/scripts/measure_dfb_financing.py --days 30

Caveat worth keeping: this opens a SECOND IG session alongside the running bot's.
Verify the bot's stream afterwards (`docker logs ... | grep Lightstreamer`).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_ig_config  # noqa: E402
from src.client import IGClient  # noqa: E402

# IG labels the nightly adjustment differently across product families; match loosely
# on the transaction type AND the human-readable reason so nothing is missed.
FIN_HINTS = ("interest", "financ", "funding", "overnight", "swap", "carry", "adjust")


def fetch(client: IGClient, days: int, page_size: int = 500) -> list[dict]:
    """Raw transaction rows over `days`. Wider window and page size than
    IGClient.get_recent_transactions (24h / 50), which exists for trade
    reconciliation rather than cost archaeology."""
    now = datetime.utcnow()
    resp = client.session.get(
        f"{client.config.base_url}/history/transactions",
        params={
            "type": "ALL",
            "from": (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S"),
            "to": now.strftime("%Y-%m-%dT%H:%M:%S"),
            "pageSize": page_size,
        },
        headers=client._get_headers(version="2"),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("transactions", [])


def is_financing(txn: dict) -> bool:
    blob = " ".join(str(txn.get(k, "")) for k in
                    ("transactionType", "instrumentName", "reference")).lower()
    return any(h in blob for h in FIN_HINTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--show-all-types", action="store_true",
                    help="print the transactionType census, to spot a label this misses")
    args = ap.parse_args()

    client = IGClient(load_ig_config())
    if not client.login():
        print("LOGIN FAILED — aborting", file=sys.stderr)
        sys.exit(1)
    try:
        txns = fetch(client, args.days)
    finally:
        client.logout()   # release the session promptly; the bot has its own

    print(f"window: last {args.days} days · {len(txns)} transactions\n")
    if not txns:
        print("No transactions in the window.")
        return

    census: dict[str, int] = defaultdict(int)
    for t in txns:
        census[str(t.get("transactionType"))] += 1
    print("transactionType census: " + ", ".join(f"{k}={v}" for k, v in sorted(census.items())))

    fin = [t for t in txns if is_financing(t)]
    print(f"rows matching a financing hint: {len(fin)}\n")

    if args.show_all_types:
        print("--- every DISTINCT (type, instrument) pair, to catch a missed label ---")
        seen = sorted({(str(t.get("transactionType")), str(t.get("instrumentName"))) for t in txns})
        for ttype, inst in seen:
            print(f"  {ttype:<12} {inst}")
        print()

    if not fin:
        print("NO financing rows found. Either the window predates any overnight hold,\n"
              "or IG labels them with a word not in FIN_HINTS — rerun with\n"
              "--show-all-types and read the census before concluding anything.")
        return

    print("--- financing rows ---")
    by_inst: dict[str, list[float]] = defaultdict(list)
    for t in sorted(fin, key=lambda x: str(x.get("date"))):
        amt = IGClient._parse_pnl(str(t.get("profitAndLoss", "0")))
        inst = str(t.get("instrumentName", "?"))
        by_inst[inst].append(amt)
        print(f"  {str(t.get('date')):<12} {str(t.get('transactionType')):<10} "
              f"{inst:<34} {amt:+8.2f}  {str(t.get('reference',''))[:22]}")

    print("\n--- per instrument ---")
    print(f"  {'instrument':<34} {'rows':>5} {'total £':>10} {'mean/row':>10}")
    for inst, amts in sorted(by_inst.items()):
        print(f"  {inst:<34} {len(amts):>5} {sum(amts):>+10.2f} {sum(amts)/len(amts):>+10.2f}")
    print(f"\n  NET across all financing rows: £{sum(sum(v) for v in by_inst.values()):+.2f}")
    print("\n⚠ Rows are NOT nights. IG can post several rows per position per night, and\n"
          "  the 2026-08-19 Gold error was exactly this conflation — it halved the charge.\n"
          "  Convert to £/night using the position's real open→close date span before\n"
          "  quoting a rate.")


if __name__ == "__main__":
    main()
