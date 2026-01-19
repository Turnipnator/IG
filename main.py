"""
IG Trading Bot - Main Entry Point

Automated spread betting platform using IG Markets API.
Runs on a schedule, analyzing markets and executing trades.
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import datetime

import schedule

from config import (
    load_ig_config,
    load_telegram_config,
    load_trading_config,
    MARKETS,
    STRATEGY_PARAMS,
)
from src.client import IGClient
from src.strategy import TradingStrategy, Signal, should_close_position
from src.risk_manager import RiskManager
from src.telegram_bot import TelegramBot
from src.utils import setup_logging, is_market_open, get_market_type, RateLimiter

# Globals
logger = logging.getLogger(__name__)
client: IGClient = None
strategy: TradingStrategy = None
risk_manager: RiskManager = None
telegram: TelegramBot = None
rate_limiter: RateLimiter = None
running = True


def initialize() -> bool:
    """Initialize all components."""
    global client, strategy, risk_manager, telegram, rate_limiter

    # Load configs
    ig_config = load_ig_config()
    telegram_config = load_telegram_config()
    trading_config = load_trading_config()

    # Initialize components (pass cache TTL to client)
    client = IGClient(ig_config, cache_ttl_minutes=trading_config.cache_ttl_minutes)
    strategy = TradingStrategy(STRATEGY_PARAMS)
    risk_manager = RiskManager(trading_config)
    telegram = TelegramBot(telegram_config)
    rate_limiter = RateLimiter(requests_per_minute=25)

    # Login to IG
    if not client.login():
        logger.error("Failed to login to IG")
        return False

    # Set IG client reference in telegram bot
    telegram.set_ig_client(client)

    # Get account balance
    balance = client.get_balance()
    if balance:
        logger.info(f"Account balance: £{balance:,.2f}")
    else:
        logger.warning("Could not retrieve account balance")

    return True


def check_and_close_positions() -> None:
    """Check if any open positions should be closed based on strategy."""
    if not telegram.trading_enabled:
        return

    trading_config = load_trading_config()
    positions = client.get_positions()

    for position in positions:
        market = next((m for m in MARKETS if m.epic == position.epic), None)
        if not market:
            continue

        rate_limiter.wait_if_needed()

        df = client.get_historical_prices(
            position.epic,
            resolution="MINUTE_5",
            num_points=trading_config.price_data_points,
        )

        if df is None or df.empty:
            continue

        should_close, reason = should_close_position(
            df, position.direction, STRATEGY_PARAMS
        )

        if should_close:
            logger.info(f"Closing position {position.deal_id}: {reason}")

            result = client.close_position(
                position.deal_id,
                position.direction,
                position.size,
            )

            if result:
                # Update telegram stats
                telegram.daily_pnl += position.profit_loss
                risk_manager.update_daily_pnl(position.profit_loss)

                # Send notification via async
                asyncio.run_coroutine_threadsafe(
                    telegram.notify_trade_closed(
                        market.name,
                        position.direction,
                        position.profit_loss,
                        reason,
                    ),
                    telegram_loop,
                )


def analyze_markets() -> None:
    """Analyze all configured markets and generate signals."""
    if not telegram.trading_enabled:
        logger.info("Trading paused - skipping market analysis")
        return

    balance = client.get_balance()
    if not balance:
        logger.error("Could not get account balance")
        return

    positions = client.get_positions()
    trading_config = load_trading_config()

    for market in MARKETS:
        market_type = get_market_type(market.epic)
        if not is_market_open(market_type):
            logger.debug(f"{market.name} market appears closed")
            continue

        rate_limiter.wait_if_needed()

        market_info = client.get_market_info(market.epic)
        if not market_info:
            logger.warning(f"Could not get market info for {market.epic}")
            continue

        if market_info.market_status != "TRADEABLE":
            logger.debug(f"{market.name} not tradeable: {market_info.market_status}")
            continue

        rate_limiter.wait_if_needed()

        df = client.get_historical_prices(
            market.epic,
            resolution="MINUTE_5",
            num_points=trading_config.price_data_points,
        )

        if df is None or df.empty:
            logger.warning(f"No price data for {market.epic}")
            continue

        current_price = (market_info.bid + market_info.offer) / 2
        trade_signal = strategy.analyze(df, market, current_price)

        logger.info(
            f"{market.name}: {trade_signal.signal.value} "
            f"(confidence: {trade_signal.confidence:.0%}) - {trade_signal.reason}"
        )

        if trade_signal.signal == Signal.HOLD:
            continue

        is_valid, reason = risk_manager.validate_trade(
            positions,
            market.epic,
            trade_signal.signal.value,
            balance,
        )

        if not is_valid:
            logger.info(f"Trade not validated: {reason}")
            continue

        min_confidence = 0.5
        if trade_signal.confidence < min_confidence:
            logger.info(
                f"Confidence too low: {trade_signal.confidence:.0%} < {min_confidence:.0%}"
            )
            continue

        position_size = risk_manager.calculate_position_size(
            balance,
            trade_signal.stop_distance,
            market,
        )

        if not position_size.approved:
            logger.warning(f"Position size not approved: {position_size.reason}")
            continue

        # Send signal notification
        asyncio.run_coroutine_threadsafe(
            telegram.notify_signal(
                market.name,
                trade_signal.signal.value,
                trade_signal.confidence,
                trade_signal.reason,
            ),
            telegram_loop,
        )

        logger.info(
            f"Opening {trade_signal.signal.value} position on {market.name}: "
            f"size={position_size.size}, stop={trade_signal.stop_distance}, "
            f"limit={trade_signal.limit_distance}"
        )

        result = client.open_position(
            epic=market.epic,
            direction=trade_signal.signal.value,
            size=position_size.size,
            stop_distance=trade_signal.stop_distance,
            limit_distance=trade_signal.limit_distance,
        )

        if result:
            telegram.trades_today += 1
            asyncio.run_coroutine_threadsafe(
                telegram.notify_trade_opened(
                    market.name,
                    trade_signal.signal.value,
                    position_size.size,
                    trade_signal.entry_price,
                    trade_signal.stop_distance,
                    trade_signal.limit_distance,
                ),
                telegram_loop,
            )
            positions = client.get_positions()
        else:
            asyncio.run_coroutine_threadsafe(
                telegram.notify_error(f"Failed to open position on {market.name}"),
                telegram_loop,
            )


def run_trading_cycle() -> None:
    """Run a complete trading cycle."""
    global running

    if not running:
        return

    logger.info("=" * 50)
    logger.info(f"Trading cycle started at {datetime.now()}")

    try:
        check_and_close_positions()
        analyze_markets()
        logger.info("Trading cycle completed")

    except Exception as e:
        logger.exception(f"Error in trading cycle: {e}")
        if telegram and telegram.app:
            asyncio.run_coroutine_threadsafe(
                telegram.notify_error(f"Trading cycle error: {str(e)}"),
                telegram_loop,
            )


def send_daily_summary() -> None:
    """Send daily summary at end of trading day."""
    balance = client.get_balance()
    positions = client.get_positions()

    if balance and telegram and telegram.app:
        asyncio.run_coroutine_threadsafe(
            telegram.notify_daily_summary(
                balance,
                telegram.daily_pnl,
                telegram.trades_today,
                positions,
            ),
            telegram_loop,
        )

    risk_manager.reset_daily_pnl()
    telegram.reset_daily_stats()


def refresh_session() -> None:
    """Refresh IG session to prevent timeout."""
    logger.info("Refreshing IG session...")
    if not client.login():
        logger.error("Failed to refresh session")
        if telegram and telegram.app:
            asyncio.run_coroutine_threadsafe(
                telegram.notify_error("Failed to refresh IG session"),
                telegram_loop,
            )


def run_scheduler():
    """Run the schedule in a separate thread."""
    while running:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.exception(f"Error in scheduler: {e}")
            time.sleep(60)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False


# Global event loop for telegram
telegram_loop = None


async def main_async():
    """Async main entry point."""
    global running, telegram_loop

    # Store the event loop
    telegram_loop = asyncio.get_event_loop()

    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file="logs/ig_bot.log",
    )

    logger.info("IG Trading Bot starting...")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize
    if not initialize():
        logger.error("Initialization failed")
        sys.exit(1)

    # Start Telegram bot
    await telegram.start()

    # Send startup notification
    balance = client.get_balance() or 0
    market_names = [m.name for m in MARKETS]
    await telegram.notify_startup(balance, market_names)

    # Load trading config for interval
    trading_config = load_trading_config()
    check_interval = trading_config.check_interval

    # Schedule jobs
    schedule.every(check_interval).minutes.do(run_trading_cycle)
    schedule.every(6).hours.do(refresh_session)
    schedule.every().day.at("21:00").do(send_daily_summary)

    # Run first cycle
    run_trading_cycle()

    logger.info(f"Bot running. Checking markets every {check_interval} minutes.")

    # Run scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Keep async loop running for Telegram
    while running:
        await asyncio.sleep(1)

    # Cleanup
    logger.info("Shutting down...")
    await telegram.stop()
    client.logout()
    logger.info("Goodbye!")


def main():
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
