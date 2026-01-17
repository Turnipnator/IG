"""
Test run script - analyzes markets without executing trades.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)

from config import (
    load_ig_config,
    load_telegram_config,
    load_trading_config,
    MARKETS,
    STRATEGY_PARAMS,
)
from src.client import IGClient
from src.strategy import TradingStrategy, Signal
from src.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)


def main():
    # Load configs
    ig_config = load_ig_config()
    telegram_config = load_telegram_config()
    trading_config = load_trading_config()

    # Initialize
    client = IGClient(ig_config)
    strategy = TradingStrategy(STRATEGY_PARAMS)
    telegram = TelegramNotifier(telegram_config)

    print("=" * 60)
    print("IG TRADING BOT - TEST RUN")
    print("=" * 60)
    print(f"Trading Enabled: {trading_config.trading_enabled}")
    print(f"Risk per trade: {trading_config.risk_per_trade:.1%}")
    print(f"Max positions: {trading_config.max_positions}")
    print()

    # Login
    logger.info("Logging in to IG...")
    if not client.login():
        print("Login failed!")
        return

    # Get balance
    balance = client.get_balance()
    print(f"Account Balance: £{balance:,.2f}")
    print()

    # Send startup notification
    telegram.send_startup_message(balance, [m.name for m in MARKETS])

    # Check positions
    positions = client.get_positions()
    print(f"Open Positions: {len(positions)}")
    for pos in positions:
        print(f"  - {pos.epic}: {pos.direction} {pos.size} @ {pos.open_level}")
    print()

    # Analyze markets
    print("=" * 60)
    print("MARKET ANALYSIS")
    print("=" * 60)

    signals_found = []

    for market in MARKETS:
        print(f"\n{market.name} ({market.epic})")
        print("-" * 40)

        # Get market info
        info = client.get_market_info(market.epic)
        if not info:
            print("  Could not get market info")
            continue

        print(f"  Status: {info.market_status}")
        print(f"  Price: {info.bid:.2f} / {info.offer:.2f}")

        if info.market_status != "TRADEABLE":
            print("  Market not tradeable - skipping analysis")
            continue

        # Get historical data
        df = client.get_historical_prices(
            market.epic, resolution="MINUTE_5", num_points=100
        )
        if df is None or df.empty:
            print("  No price data available")
            continue

        print(f"  Data points: {len(df)}")

        # Analyze
        current_price = (info.bid + info.offer) / 2
        signal = strategy.analyze(df, market, current_price)

        print(f"  Signal: {signal.signal.value}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Reason: {signal.reason}")

        if signal.signal != Signal.HOLD:
            signals_found.append(signal)
            print(f"  Stop: {signal.stop_distance} pts")
            print(f"  Target: {signal.limit_distance} pts")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Markets analyzed: {len(MARKETS)}")
    print(f"Signals found: {len(signals_found)}")

    for sig in signals_found:
        print(f"  - {sig.market_name}: {sig.signal.value} ({sig.confidence:.0%})")
        telegram.send_trade_signal(sig)

    if not signals_found:
        print("  No trade signals at this time")

    print()
    print("Test run complete!")
    client.logout()


if __name__ == "__main__":
    main()
