"""
Discover ALL spread bet markets available on IG, gather metadata, rank by
basic tradeability.

Output: JSON file (data/spreadbet_universe.json) + console summary.

Usage:
    docker exec ig-trading-bot python3 scripts/discover_spreadbet_markets.py

Each search call costs 1 IG API request. ~80 search terms total. Then a
get_market_info call per unique discovered market — typically caps around
~150-300 calls. Stays well under IG's 10k/week data budget (which only
tracks historical-prices calls anyway).

Tradeability score (0-100):
  - Spread/min_stop ratio (40pts) — a tight spread relative to the minimum
    allowed stop means the bot can size precisely without throwing money at
    the broker
  - Min deal size feasibility (20pts) — smaller min_deal = better for our
    £200-account scale; >5/pt is hard to risk-size at 1%
  - Status TRADEABLE right now (20pts)
  - Has tradeable hours data (10pts)
  - Spread bet expiry type (10pts) — DFB = daily funded, prefer over month
    contracts (rollover hassle)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_ig_config, MARKETS
from src.client import IGClient

OUT_PATH = Path("/app/data/spreadbet_universe.json") if Path("/app").exists() else Path("data/spreadbet_universe.json")

# Broad search terms by category
SEARCH_TERMS = {
    "Indices": [
        "FTSE 100", "FTSE 250", "S&P 500", "NASDAQ 100", "Wall Street", "Russell 2000",
        "Germany 40", "DAX", "France 40", "CAC 40", "Italy 40", "Spain 35", "IBEX",
        "Netherlands 25", "Switzerland 20", "SMI", "Sweden 30", "OMX", "Norway 25",
        "Japan 225", "Nikkei", "Hong Kong HS50", "Hang Seng", "China A50", "Singapore",
        "Australia 200", "ASX", "India 50", "Sensex", "South Africa 40", "TR/CC CRB",
        "VIX", "AI Index", "MIB",
    ],
    "Forex Majors": [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    ],
    "Forex Crosses": [
        "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD",
        "GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD",
        "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
        "CAD/JPY", "CHF/JPY", "NZD/JPY", "NZD/CAD",
    ],
    "Forex Exotic": [
        "USD/SEK", "USD/NOK", "USD/MXN", "USD/ZAR", "USD/SGD", "USD/HKD",
        "USD/TRY", "USD/PLN", "USD/CNH", "USD/ILS",
    ],
    "Energy": [
        "Crude Oil", "Brent", "WTI", "Natural Gas", "Heating Oil", "Gasoline",
        "Carbon Emissions", "London Gas Oil",
    ],
    "Metals": [
        "Gold", "Silver", "Platinum", "Palladium", "Copper",
        "Aluminium", "Nickel", "Zinc", "Lead", "Tin",
    ],
    "Soft Commodities": [
        "Cocoa", "Coffee", "Cotton", "Sugar", "Soybeans", "Soybean Oil", "Soybean Meal",
        "Wheat", "Corn", "Oats", "Rough Rice", "Orange Juice", "Lumber",
        "Live Cattle", "Lean Hogs", "Feeder Cattle",
    ],
    "Rates": [
        "US 2-Year", "US 5-Year", "US 10-Year", "US 30-Year", "T-Note", "T-Bond",
        "Long Gilt", "Short Sterling", "Bund", "Bobl", "Schatz", "BTP",
        "Euribor", "SOFR", "Eurodollar",
    ],
    "Crypto": [
        "Bitcoin", "Ethereum", "Litecoin", "Ripple", "Bitcoin Cash", "Cardano",
        "Solana", "Polkadot", "Dogecoin",
    ],
}

EXCLUDED_PREFIXES = ("CC.D.", "IN.D.VIX")  # CFD-only or unavailable on demo
EXCLUDED_TYPES = {"OPT_COMMODITIES", "OPT_CURRENCIES", "OPT_EQUITIES", "SHARES"}


@dataclass
class MarketEntry:
    epic: str
    name: str
    category: str
    instrument_type: str
    expiry: str
    bid: float
    offer: float
    spread: float
    min_stop_distance: float
    min_deal_size: float
    market_status: str
    score: float = 0.0
    score_breakdown: str = ""
    currently_traded: bool = False


def score_entry(e: MarketEntry) -> tuple[float, str]:
    pts = 0.0
    parts = []

    # 1. Spread / min_stop ratio (40 pts)
    if e.min_stop_distance > 0 and e.spread > 0:
        ratio = e.spread / e.min_stop_distance
        if ratio <= 0.20:
            pts += 40; parts.append("spread<=20%minStop:40")
        elif ratio <= 0.40:
            pts += 30; parts.append("spread<=40%minStop:30")
        elif ratio <= 0.70:
            pts += 20; parts.append("spread<=70%minStop:20")
        elif ratio <= 1.00:
            pts += 10; parts.append("spread<=100%minStop:10")
        else:
            parts.append("spread>minStop:0")
    elif e.spread == 0:
        pts += 30; parts.append("zero-spread:30")
    else:
        parts.append("nodata:0")

    # 2. Min deal size feasibility (20 pts) — smaller is better
    if e.min_deal_size <= 0.5:
        pts += 20; parts.append("minDeal<=0.5:20")
    elif e.min_deal_size <= 1.0:
        pts += 15; parts.append("minDeal<=1.0:15")
    elif e.min_deal_size <= 5.0:
        pts += 8; parts.append("minDeal<=5.0:8")
    else:
        parts.append(f"minDeal={e.min_deal_size}:0")

    # 3. Currently TRADEABLE (20 pts)
    if e.market_status == "TRADEABLE":
        pts += 20; parts.append("tradeable:20")
    elif e.market_status == "EDITS_ONLY":
        pts += 5; parts.append("edits:5")
    else:
        parts.append(f"{e.market_status}:0")

    # 4. Has bid/offer prices (10 pts)
    if e.bid > 0 and e.offer > 0:
        pts += 10; parts.append("priced:10")

    # 5. DFB = daily funded (preferred), other expiries get less (10 pts)
    if e.expiry == "DFB":
        pts += 10; parts.append("DFB:10")
    elif e.expiry in ("-",):
        pts += 5; parts.append("nostatic:5")
    else:
        parts.append(f"expiry={e.expiry}:0")

    return pts, "|".join(parts)


def main():
    cfg = load_ig_config()
    c = IGClient(cfg)
    if not c.login():
        print("Login failed.")
        return
    spreadbet_id = c.get_spreadbet_account_id()
    if spreadbet_id and spreadbet_id != c.account_id:
        c.switch_account(spreadbet_id)

    current_epics = {m.epic for m in MARKETS}

    print("\n=== DISCOVERY ===")
    seen: dict[str, str] = {}  # epic -> first matching category
    discovered: list[MarketEntry] = []

    for category, terms in SEARCH_TERMS.items():
        for term in terms:
            try:
                results = c.search_markets(term)
            except Exception as e:
                print(f"  search '{term}' failed: {e}")
                continue
            for m in results or []:
                epic = m.get("epic", "")
                if not epic or epic in seen:
                    continue
                if any(epic.startswith(p) for p in EXCLUDED_PREFIXES):
                    continue
                inst_type = m.get("instrumentType", "")
                if inst_type in EXCLUDED_TYPES:
                    continue
                # Only spread-bet style EPICs (skip equities/options entirely)
                if not (epic.startswith(("IX.D.", "CS.D.", "EN.D.", "CO.D.", "IR.D.", "IN.D.", "MT.D."))):
                    continue
                seen[epic] = category
                discovered.append(MarketEntry(
                    epic=epic,
                    name=m.get("instrumentName", ""),
                    category=category,
                    instrument_type=inst_type,
                    expiry=m.get("expiry", "") or "-",
                    bid=m.get("bid") or 0,
                    offer=m.get("offer") or 0,
                    spread=0,
                    min_stop_distance=0,
                    min_deal_size=0,
                    market_status=m.get("marketStatus", "UNKNOWN"),
                ))
            # IG limit: 60 non-trading calls/min. Stay well under: 1.2s/call = 50/min.
            time.sleep(1.2)
        print(f"  {category:20s}: {len(terms):2d} terms searched, {sum(1 for d in discovered if d.category == category):3d} markets so far", flush=True)

    print(f"\nDiscovered {len(discovered)} unique spread bet EPICs")

    # Hydrate metadata for each
    print("\n=== HYDRATION (get_market_info per epic) ===", flush=True)
    for i, e in enumerate(discovered):
        if i % 25 == 0:
            print(f"  {i}/{len(discovered)}...", flush=True)
        try:
            info = c.get_market_info(e.epic)
            if info:
                e.bid = info.bid or 0
                e.offer = info.offer or 0
                e.spread = (info.offer - info.bid) if (info.bid and info.offer) else 0
                e.min_stop_distance = info.min_stop_distance or 0
                e.min_deal_size = info.min_deal_size or 0
                e.market_status = info.market_status or e.market_status
        except Exception as ex:
            print(f"  {e.epic}: hydrate failed: {ex}", flush=True)
        # Same 50/min cap as discovery
        time.sleep(1.2)

    # Score
    for e in discovered:
        e.score, e.score_breakdown = score_entry(e)
        e.currently_traded = e.epic in current_epics

    # Sort by category then score desc
    discovered.sort(key=lambda x: (x.category, -x.score))

    # Save full result
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump([asdict(e) for e in discovered], f, indent=2)
    print(f"\nSaved {len(discovered)} entries to {OUT_PATH}")

    # Print summary by category — top entries per group
    print("\n" + "=" * 80)
    print("RANKED BY CATEGORY (score | epic | name | spread | minStop | minDeal | status)")
    print("=" * 80)
    by_cat = defaultdict(list)
    for e in discovered:
        by_cat[e.category].append(e)
    for cat, entries in by_cat.items():
        print(f"\n--- {cat} ({len(entries)} markets) ---")
        for e in entries[:15]:  # top 15 per category
            star = "*" if e.currently_traded else " "
            print(f"  {star}{e.score:5.0f} | {e.epic:32s} | {e.name[:28]:28s} | "
                  f"sp={e.spread:7.2f} | st={e.min_stop_distance:6.1f} | "
                  f"d={e.min_deal_size:5.2f} | {e.market_status}")

    # Currently-traded markets summary
    print("\n" + "=" * 80)
    print(f"CURRENTLY TRADED ({len(current_epics)} markets) — score check")
    print("=" * 80)
    for e in [d for d in discovered if d.currently_traded]:
        print(f"  {e.score:5.0f} | {e.epic:32s} | {e.name[:30]:30s} | {e.market_status}")
    missing = current_epics - {e.epic for e in discovered}
    if missing:
        print(f"\nIn config but NOT discovered (likely renamed/delisted): {missing}")

    # Top candidates the bot does NOT currently trade
    print("\n" + "=" * 80)
    print("TOP UN-TRADED CANDIDATES (not in current 17, score >= 60, TRADEABLE)")
    print("=" * 80)
    candidates = [
        e for e in discovered
        if not e.currently_traded
        and e.score >= 60
        and e.market_status == "TRADEABLE"
    ]
    candidates.sort(key=lambda x: -x.score)
    for e in candidates[:30]:
        print(f"  {e.score:5.0f} | {e.epic:32s} | {e.name[:30]:30s} | "
              f"{e.category:18s} | sp={e.spread:6.2f} | st={e.min_stop_distance:5.1f} | "
              f"d={e.min_deal_size:5.2f}")


if __name__ == "__main__":
    main()
