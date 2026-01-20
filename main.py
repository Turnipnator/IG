"""
IG Trading Bot - Main Entry Point

Automated spread betting platform using IG Markets API.
Uses Lightstreamer for real-time price streaming to avoid API rate limits.
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import datetime

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
from src.streaming import IGStreamService, MarketStream, LIGHTSTREAMER_AVAILABLE
from src.utils import setup_logging, RateLimiter

# Globals
logger = logging.getLogger(__name__)
client: IGClient = None
strategy: TradingStrategy = None
risk_manager: RiskManager = None
telegram: TelegramBot = None
stream_service: IGStreamService = None
rate_limiter: RateLimiter = None
running = True
telegram_loop = None

# Track last analysis time per market to avoid duplicate signals
last_analysis: dict[str, datetime] = {}


def initialize() -> bool:
    """Initialize all components."""
    global client, strategy, risk_manager, telegram, rate_limiter

    # Load configs
    ig_config = load_ig_config()
    telegram_config = load_telegram_config()
    trading_config = load_trading_config()

    # Initialize components
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


def initialize_streaming() -> bool:
    """Initialize Lightstreamer streaming connection."""
    global stream_service

    if not LIGHTSTREAMER_AVAILABLE:
        logger.warning("Lightstreamer not available - falling back to polling mode")
        return False

    if not client.is_logged_in:
        logger.error("Must login before initializing streaming")
        return False

    try:
        ig_config = load_ig_config()
        is_demo = ig_config.acc_type.upper() == "DEMO"

        # Switch to SPREADBET account for streaming (CFD accounts don't support streaming)
        # ALWAYS call switch_account to ensure tokens are properly associated with SPREADBET context
        spreadbet_id = client.get_spreadbet_account_id()
        if spreadbet_id:
            logger.info(f"Ensuring SPREADBET account ({spreadbet_id}) for streaming (current: {client.account_id})...")
            if not client.switch_account(spreadbet_id):
                logger.warning("Failed to switch to SPREADBET account")
            else:
                logger.info(f"SPREADBET account confirmed - tokens refreshed")
        else:
            logger.warning("No SPREADBET account found - streaming may not work")

        stream_service = IGStreamService(
            cst=client.cst,
            security_token=client.security_token,
            account_id=client.account_id,
            is_demo=is_demo,
            on_price_update=on_price_update,
            on_candle_complete=on_candle_complete,
        )

        # Connect to Lightstreamer
        if not stream_service.connect():
            logger.error("Failed to connect to Lightstreamer")
            return False

        # Subscribe to markets
        epics = [m.epic for m in MARKETS]
        names = [m.name for m in MARKETS]

        if not stream_service.subscribe_markets(epics, names):
            logger.error("Failed to subscribe to markets")
            return False

        # Initialize candles with historical data (one-time API usage)
        logger.info("Initializing candle history from REST API (one-time)...")
        trading_config = load_trading_config()

        for market in MARKETS:
            rate_limiter.wait_if_needed()
            df = client.get_historical_prices(
                market.epic,
                resolution="MINUTE_5",
                num_points=trading_config.price_data_points,
                use_cache=False,  # Force fresh data at startup
            )
            if df is not None and not df.empty:
                stream_service.initialize_candles(market.epic, df)
            else:
                logger.warning(f"Could not initialize candles for {market.name}")

        logger.info("Streaming initialized successfully")
        return True

    except Exception as e:
        logger.exception(f"Error initializing streaming: {e}")
        return False


def on_price_update(epic: str, market: MarketStream) -> None:
    """Callback for real-time price updates."""
    # This fires frequently - use for monitoring/logging only
    pass


def on_candle_complete(epic: str, market: MarketStream) -> None:
    """
    Callback when a 5-minute candle completes.

    This is where we analyze the market and check for trade signals.
    """
    global last_analysis

    if not telegram.trading_enabled:
        return

    # Avoid analyzing too frequently (debounce)
    now = datetime.now()
    if epic in last_analysis:
        if (now - last_analysis[epic]).seconds < 60:
            return

    last_analysis[epic] = now

    # Run analysis in background to not block streaming
    threading.Thread(
        target=analyze_market_from_stream,
        args=(epic, market),
        daemon=True,
    ).start()


def analyze_market_from_stream(epic: str, market: MarketStream) -> None:
    """Analyze a market using streaming data."""
    try:
        market_config = next((m for m in MARKETS if m.epic == epic), None)
        if not market_config:
            return

        # Skip if market not tradeable
        if market.market_state != "TRADEABLE":
            logger.debug(f"{market.name} not tradeable: {market.market_state}")
            return

        # Get candle DataFrame
        df = market.to_dataframe()
        if df is None or len(df) < 50:
            logger.debug(f"Insufficient candles for {market.name}: {len(df) if df is not None else 0}")
            return

        # Calculate current price from stream
        current_price = market.mid_price

        # Analyze
        trade_signal = strategy.analyze(df, market_config, current_price)

        logger.info(
            f"[STREAM] {market.name}: {trade_signal.signal.value} "
            f"(confidence: {trade_signal.confidence:.0%}) - {trade_signal.reason}"
        )

        if trade_signal.signal == Signal.HOLD:
            return

        # Get balance and positions (REST API calls)
        balance = client.get_balance()
        if not balance:
            return

        positions = client.get_positions()
        trading_config = load_trading_config()

        # Validate trade
        is_valid, reason = risk_manager.validate_trade(
            positions,
            epic,
            trade_signal.signal.value,
            balance,
        )

        if not is_valid:
            logger.info(f"Trade not validated: {reason}")
            return

        # Check confidence
        min_confidence = 0.5
        if trade_signal.confidence < min_confidence:
            logger.info(
                f"Confidence too low: {trade_signal.confidence:.0%} < {min_confidence:.0%}"
            )
            return

        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            balance,
            trade_signal.stop_distance,
            market_config,
        )

        if not position_size.approved:
            logger.warning(f"Position size not approved: {position_size.reason}")
            return

        # Send signal notification
        if telegram_loop:
            asyncio.run_coroutine_threadsafe(
                telegram.notify_signal(
                    market.name,
                    trade_signal.signal.value,
                    trade_signal.confidence,
                    trade_signal.reason,
                ),
                telegram_loop,
            )

        # Open position
        logger.info(
            f"Opening {trade_signal.signal.value} position on {market.name}: "
            f"size={position_size.size}, stop={trade_signal.stop_distance}, "
            f"limit={trade_signal.limit_distance}"
        )

        result = client.open_position(
            epic=epic,
            direction=trade_signal.signal.value,
            size=position_size.size,
            stop_distance=trade_signal.stop_distance,
            limit_distance=trade_signal.limit_distance,
        )

        if result:
            telegram.trades_today += 1
            if telegram_loop:
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
        else:
            if telegram_loop:
                asyncio.run_coroutine_threadsafe(
                    telegram.notify_error(f"Failed to open position on {market.name}"),
                    telegram_loop,
                )

    except Exception as e:
        logger.exception(f"Error analyzing {epic}: {e}")


def check_positions_from_stream() -> None:
    """Check open positions using streaming data for exit signals."""
    if not telegram.trading_enabled:
        return

    if not stream_service or not stream_service.connected:
        return

    positions = client.get_positions()

    for position in positions:
        market = stream_service.get_market_data(position.epic)
        if not market:
            continue

        df = market.to_dataframe()
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
                market_config = next((m for m in MARKETS if m.epic == position.epic), None)
                market_name = market_config.name if market_config else position.epic

                telegram.daily_pnl += position.profit_loss
                risk_manager.update_daily_pnl(position.profit_loss)

                if telegram_loop:
                    asyncio.run_coroutine_threadsafe(
                        telegram.notify_trade_closed(
                            market_name,
                            position.direction,
                            position.profit_loss,
                            reason,
                        ),
                        telegram_loop,
                    )


def periodic_tasks() -> None:
    """Run periodic tasks (position checks, etc.)."""
    while running:
        try:
            # Check positions every minute
            check_positions_from_stream()

            # Log streaming status every 5 minutes
            if stream_service and stream_service.connected:
                status = stream_service.get_status()
                active_markets = sum(
                    1 for m in status["markets"].values()
                    if m["state"] == "TRADEABLE"
                )
                logger.debug(
                    f"Streaming: {status['connection_status']}, "
                    f"{active_markets}/{len(status['markets'])} markets active"
                )

        except Exception as e:
            logger.exception(f"Error in periodic tasks: {e}")

        time.sleep(60)


def send_daily_summary() -> None:
    """Send daily summary."""
    balance = client.get_balance()
    positions = client.get_positions()

    if balance and telegram and telegram_loop:
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
    """Refresh IG session and reconnect streaming."""
    global stream_service

    logger.info("Refreshing IG session...")

    # Disconnect streaming
    if stream_service:
        stream_service.disconnect()

    # Re-login
    if not client.login():
        logger.error("Failed to refresh session")
        if telegram and telegram_loop:
            asyncio.run_coroutine_threadsafe(
                telegram.notify_error("Failed to refresh IG session"),
                telegram_loop,
            )
        return

    # Reconnect streaming with new tokens
    if LIGHTSTREAMER_AVAILABLE:
        initialize_streaming()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False


async def main_async():
    """Async main entry point."""
    global running, telegram_loop

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

    # Initialize REST client and Telegram
    if not initialize():
        logger.error("Initialization failed")
        sys.exit(1)

    # Start Telegram bot
    await telegram.start()

    # Initialize streaming (real-time prices)
    streaming_enabled = False
    if LIGHTSTREAMER_AVAILABLE:
        streaming_enabled = initialize_streaming()
    else:
        logger.warning(
            "Lightstreamer not installed. Install with: pip install lightstreamer-client-lib"
        )

    # Send startup notification
    balance = client.get_balance() or 0
    market_names = [m.name for m in MARKETS]

    mode = "STREAMING" if streaming_enabled else "POLLING"
    await telegram.send_notification(
        f"🚀 *IG Trading Bot Started*\n\n"
        f"Mode: {mode}\n"
        f"Balance: £{balance:,.2f}\n"
        f"Markets: {', '.join(market_names)}\n\n"
        f"{'Real-time price streaming active!' if streaming_enabled else 'Using scheduled polling (API limited)'}"
    )

    if streaming_enabled:
        logger.info("Bot running with real-time streaming. Analyzing on candle completion.")

        # Start periodic task thread
        periodic_thread = threading.Thread(target=periodic_tasks, daemon=True)
        periodic_thread.start()

        # Schedule session refresh and daily summary
        import schedule

        schedule.every(6).hours.do(refresh_session)
        schedule.every().day.at("21:00").do(send_daily_summary)

        # Run scheduler in background
        def run_schedule():
            while running:
                schedule.run_pending()
                time.sleep(30)

        schedule_thread = threading.Thread(target=run_schedule, daemon=True)
        schedule_thread.start()

    else:
        # Fallback to polling mode (old behavior)
        logger.warning("Running in polling mode - API rate limits apply")

        import schedule
        from src.utils import is_market_open, get_market_type

        trading_config = load_trading_config()

        def polling_cycle():
            if not running or not telegram.trading_enabled:
                return

            logger.info("=" * 50)
            logger.info(f"Polling cycle started at {datetime.now()}")

            # Analysis logic from old main.py
            balance = client.get_balance()
            if not balance:
                return

            positions = client.get_positions()

            for market in MARKETS:
                market_type = get_market_type(market.epic)
                if not is_market_open(market_type):
                    continue

                rate_limiter.wait_if_needed()

                df = client.get_historical_prices(
                    market.epic,
                    resolution="MINUTE_5",
                    num_points=trading_config.price_data_points,
                )

                if df is None or df.empty:
                    continue

                market_info = client.get_market_info(market.epic)
                if not market_info or market_info.market_status != "TRADEABLE":
                    continue

                current_price = (market_info.bid + market_info.offer) / 2
                trade_signal = strategy.analyze(df, market, current_price)

                logger.info(
                    f"{market.name}: {trade_signal.signal.value} "
                    f"(confidence: {trade_signal.confidence:.0%})"
                )

            logger.info("Polling cycle completed")

        schedule.every(trading_config.check_interval).minutes.do(polling_cycle)
        schedule.every(6).hours.do(refresh_session)
        schedule.every().day.at("21:00").do(send_daily_summary)

        # Run first cycle
        polling_cycle()

        def run_schedule():
            while running:
                schedule.run_pending()
                time.sleep(1)

        schedule_thread = threading.Thread(target=run_schedule, daemon=True)
        schedule_thread.start()

    # Keep running
    while running:
        await asyncio.sleep(1)

    # Cleanup
    logger.info("Shutting down...")
    if stream_service:
        stream_service.disconnect()
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
