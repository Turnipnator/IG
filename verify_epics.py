"""
Script to search and verify market EPICs on IG.
Run this to find the correct EPICs for your markets.
"""

from config import load_ig_config
from src.client import IGClient


def main():
    config = load_ig_config()

    if not config.api_key or not config.username or not config.password:
        print("Please fill in your IG credentials in .env file first:")
        print("  IG_API_KEY=your_api_key")
        print("  IG_USERNAME=your_username")
        print("  IG_PASSWORD=your_password")
        return

    client = IGClient(config)

    print("Logging in to IG...")
    if not client.login():
        print("Login failed. Check your credentials.")
        return

    print("Login successful!\n")

    # Markets to search for
    searches = [
        ("S&P 500", ["US 500", "S&P 500", "US500"]),
        ("NASDAQ 100", ["NASDAQ", "US Tech 100", "NASDAQ 100"]),
        ("Crude Oil", ["Crude Oil", "US Crude", "Oil"]),
        ("Dollar Index", ["Dollar Index", "DXY", "US Dollar Index"]),
        ("EUR/USD", ["EURUSD", "EUR/USD"]),
        ("Gold", ["Gold", "Spot Gold", "XAUUSD"]),
    ]

    print("=" * 70)
    print("SEARCHING FOR MARKETS")
    print("=" * 70)

    for market_name, search_terms in searches:
        print(f"\n{market_name}:")
        print("-" * 40)

        found = set()
        for term in search_terms:
            results = client.search_markets(term)
            for r in results[:5]:  # Top 5 results per search
                epic = r.get("epic", "")
                name = r.get("instrumentName", "")
                instrument_type = r.get("instrumentType", "")

                # Filter for spread betting (CURRENCIES, INDICES, COMMODITIES)
                if epic and epic not in found:
                    found.add(epic)
                    print(f"  {epic}")
                    print(f"    Name: {name}")
                    print(f"    Type: {instrument_type}")
                    print()

        if not found:
            print("  No results found")

    # Also check market info for our configured EPICs
    print("\n" + "=" * 70)
    print("CHECKING CONFIGURED EPICS")
    print("=" * 70)

    configured_epics = [
        ("S&P 500", "IX.D.SPTRD.DAILY.IP"),
        ("NASDAQ 100", "IX.D.NASDAQ.CASH.IP"),
        ("Crude Oil", "CC.D.CL.UNC.IP"),
        ("Dollar Index", "CC.D.DX.UMP.IP"),
        ("EUR/USD", "CS.D.EURUSD.TODAY.IP"),
        ("Gold", "CS.D.USCGC.TODAY.IP"),
    ]

    for name, epic in configured_epics:
        print(f"\n{name} ({epic}):")
        info = client.get_market_info(epic)
        if info:
            print(f"  Status: {info.market_status}")
            print(f"  Name: {info.instrument_name}")
            print(f"  Bid/Offer: {info.bid:.2f} / {info.offer:.2f}")
            print(f"  Min deal size: {info.min_deal_size}")
            print(f"  Min stop: {info.min_stop_distance}")
        else:
            print("  EPIC NOT FOUND - needs updating")

    client.logout()
    print("\nDone!")


if __name__ == "__main__":
    main()
