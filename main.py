"""
IG Trading Bot - Main Entry Point

Automated spread betting platform using IG Markets API.
Runs on a schedule, analyzing markets and executing trades.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timedelta

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
from src.telegram_bot import TelegramNotifier
from src.utils import setup_logging, is_market_open, get_market_type, RateLimiter

# Globals
logger = logging.getLogger(__name__)
client: IGClient = None
strategy: TradingStrategy = None
risk_manager: RiskManager = None
telegram: TelegramNotifier = None
rate_limiter: RateLimiter = None
running = True


def initialize() -> bool:
    """Initialize all components."""
    global client, strategy, risk_manager, telegram, rate_limiter

    # Load configs
    ig_config = load_ig_config()
    telegram_config = load_telegram_config()
    trading_config = load_trading_config()

    # Initialize components
    client = IGClient(ig_config)
    strategy = TradingStrategy(STRATEGY_PARAMS)
    risk_manager = RiskManager(trading_config)
    telegram = TelegramNotifier(telegram_config)
    rate_limiter = RateLimiter(requests_per_minute=25)  # Stay under limit

    # Login to IG
    if not client.login():
        logger.error("Failed to login to IG")
        telegram.send_error("Failed to login to IG API")
        return False

    # Get account balance
    balance = client.get_balance()
    if balance:
        logger.info(f"Account balance: £{balance:,.2f}")
        market_names = [m.name for m in MARKETS]
        telegram.send_startup_message(balance, market_names)
    else:
        logger.warning("Could not retrieve account balance")

    return True


def check_and_close_positions() -> None:
    """Check if any open positions should be closed based on strategy."""
    positions = client.get_positions()

    for position in positions:
        # Get market config
        market = next((m for m in MARKETS if m.epic == position.epic), None)
        if not market:
            continue

        rate_limiter.wait_if_needed()

        # Get latest price data
        df = client.get_historical_prices(
            position.epic,
            resolution="MINUTE_5",
            num_points=100,
        )

        if df is None or df.empty:
            continue

        # Check if should close
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
                telegram.send_trade_closed(
                    market.name,
                    position.direction,
                    position.profit_loss,
                    reason,
                )
                risk_manager.update_daily_pnl(position.profit_loss)


def analyze_markets() -> None:
    """Analyze all configured markets and generate signals."""
    balance = client.get_balance()
    if not balance:
        logger.error("Could not get account balance")
        return

    positions = client.get_positions()
    trading_config = load_trading_config()

    for market in MARKETS:
        # Check if market is open
        market_type = get_market_type(market.epic)
        if not is_market_open(market_type):
            logger.debug(f"{market.name} market appears closed")
            continue

        rate_limiter.wait_if_needed()

        # Get market info
        market_info = client.get_market_info(market.epic)
        if not market_info:
            logger.warning(f"Could not get market info for {market.epic}")
            continue

        if market_info.market_status != "TRADEABLE":
            logger.debug(f"{market.name} not tradeable: {market_info.market_status}")
            continue

        rate_limiter.wait_if_needed()

        # Get historical prices
        df = client.get_historical_prices(
            market.epic,
            resolution="MINUTE_5",
            num_points=100,
        )

        if df is None or df.empty:
            logger.warning(f"No price data for {market.epic}")
            continue

        # Analyze and get signal
        current_price = (market_info.bid + market_info.offer) / 2
        trade_signal = strategy.analyze(df, market, current_price)

        logger.info(
            f"{market.name}: {trade_signal.signal.value} "
            f"(confidence: {trade_signal.confidence:.0%}) - {trade_signal.reason}"
        )

        # Skip HOLD signals
        if trade_signal.signal == Signal.HOLD:
            continue

        # Validate trade
        is_valid, reason = risk_manager.validate_trade(
            positions,
            market.epic,
            trade_signal.signal.value,
            balance,
        )

        if not is_valid:
            logger.info(f"Trade not validated: {reason}")
            continue

        # Only trade with sufficient confidence
        min_confidence = 0.5
        if trade_signal.confidence < min_confidence:
            logger.info(
                f"Confidence too low: {trade_signal.confidence:.0%} < {min_confidence:.0%}"
            )
            continue

        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            balance,
            trade_signal.stop_distance,
            market,
        )

        if not position_size.approved:
            logger.warning(f"Position size not approved: {position_size.reason}")
            continue

        # Send signal notification
        telegram.send_trade_signal(trade_signal)

        # Execute trade
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
            telegram.send_trade_opened(
                market.name,
                trade_signal.signal.value,
                position_size.size,
                trade_signal.entry_price,
                trade_signal.stop_distance,
                trade_signal.limit_distance,
            )

            # Update positions list
            positions = client.get_positions()
        else:
            telegram.send_error(f"Failed to open position on {market.name}")


def run_trading_cycle() -> None:
    """Run a complete trading cycle: check positions, analyze markets."""
    global running

    if not running:
        return

    logger.info("=" * 50)
    logger.info(f"Trading cycle started at {datetime.now()}")

    try:
        # First, check existing positions for exit signals
        check_and_close_positions()

        # Then analyze markets for new opportunities
        analyze_markets()

        logger.info("Trading cycle completed")

    except Exception as e:
        logger.exception(f"Error in trading cycle: {e}")
        telegram.send_error(f"Trading cycle error: {str(e)}")


def send_daily_summary() -> None:
    """Send daily summary at end of trading day."""
    balance = client.get_balance()
    positions = client.get_positions()

    if balance:
        # Note: Proper trade counting would need tracking throughout the day
        telegram.send_daily_summary(
            account_balance=balance,
            daily_pnl=risk_manager.daily_pnl,
            trades_count=0,  # Would need proper tracking
            winning_trades=0,
            open_positions=positions,
        )

    # Reset daily P&L
    risk_manager.reset_daily_pnl()


def refresh_session() -> None:
    """Refresh IG session to prevent timeout."""
    logger.info("Refreshing IG session...")
    if not client.login():
        logger.error("Failed to refresh session")
        telegram.send_error("Failed to refresh IG session")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False


def main():
    """Main entry point."""
    global running

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

    # Load trading config for interval
    trading_config = load_trading_config()
    check_interval = trading_config.check_interval

    # Schedule jobs
    schedule.every(check_interval).minutes.do(run_trading_cycle)
    schedule.every(6).hours.do(refresh_session)  # Refresh session periodically
    schedule.every().day.at("21:00").do(send_daily_summary)  # Daily summary at 9pm

    # Run first cycle immediately
    run_trading_cycle()

    logger.info(f"Bot running. Checking markets every {check_interval} minutes.")

    # Main loop
    while running:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            time.sleep(60)  # Wait a minute before retrying

    # Cleanup
    logger.info("Shutting down...")
    telegram.send_shutdown_message("Bot stopped")
    client.logout()
    logger.info("Goodbye!")


if __name__ == "__main__":
    main()
