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
from datetime import datetime, timedelta

from config import (
    load_ig_config,
    load_telegram_config,
    load_trading_config,
    MARKETS,
    STRATEGY_PARAMS,
)
from src.client import IGClient, Position
from src.strategy import TradingStrategy, Signal, should_close_position
from src.risk_manager import RiskManager
from src.telegram_bot import TelegramBot
from src.streaming import IGStreamService, MarketStream, LIGHTSTREAMER_AVAILABLE
from src.calendar import EconomicCalendar
from src.utils import setup_logging, RateLimiter
from src.regime import (
    MarketRegime,
    classify_regime,
    get_regime_params,
    format_regime_status,
)

# Globals
logger = logging.getLogger(__name__)
client: IGClient = None
strategy: TradingStrategy = None
risk_manager: RiskManager = None
telegram: TelegramBot = None
stream_service: IGStreamService = None
rate_limiter: RateLimiter = None
calendar: EconomicCalendar = None
running = True
telegram_loop = None

# Track last analysis time per market to avoid duplicate signals
last_analysis: dict[str, datetime] = {}

# Track when positions were last closed - cooldown prevents immediate re-entry
last_close_time: dict[str, datetime] = {}

# Track known open positions to detect external closes (stop/limit hit by IG)
known_positions: dict[str, Position] = {}  # deal_id -> Position

# Higher timeframe trend per market (updated hourly from 1H candles)
htf_trends: dict[str, str] = {}  # epic -> "BULLISH"/"BEARISH"/"NEUTRAL"

# Market regime based on S&P 500 - determines allowed trade direction
# BULLISH = longs only, BEARISH = shorts only, NEUTRAL = no trades
market_regime: str = "BULLISH"  # Default to BULLISH until we have real data
market_regime_confirmed: bool = False  # True when we've successfully fetched S&P 500 data
SP500_EPIC = "IX.D.SPTRD.DAILY.IP"

# Per-market regime classification (trend strength + volatility)
# Used for position sizing and strategy selection
market_regimes: dict[str, MarketRegime] = {}  # epic -> MarketRegime

# Track loss cooldowns separately (1hr after loss vs 15min after any close)
loss_cooldown_until: dict[str, datetime] = {}  # epic -> datetime when cooldown ends
LOSS_COOLDOWN_MINUTES = 60


def initialize() -> bool:
    """Initialize all components."""
    global client, strategy, risk_manager, telegram, rate_limiter, calendar

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
    calendar = EconomicCalendar(buffer_minutes=30)

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


def initialize_streaming(preserved_candles: dict = None) -> bool:
    """
    Initialize Lightstreamer streaming connection.

    Args:
        preserved_candles: Optional dict of epic -> DataFrame with candle data
                          to restore from a previous session (avoids data loss on refresh)
    """
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
        candle_intervals = [m.candle_interval for m in MARKETS]

        if not stream_service.subscribe_markets(epics, names, candle_intervals):
            logger.error("Failed to subscribe to markets")
            return False

        # Initialize candles - prefer preserved data, then API, then start fresh
        logger.info("Initializing candle history...")
        trading_config = load_trading_config()

        for market in MARKETS:
            # First try preserved candles from previous session
            if preserved_candles and market.epic in preserved_candles:
                df = preserved_candles[market.epic]
                if df is not None and not df.empty:
                    stream_service.initialize_candles(market.epic, df)
                    logger.info(f"  {market.name}: Restored {len(df)} candles from previous session")
                    continue

            # Fall back to API
            rate_limiter.wait_if_needed()
            resolution = f"MINUTE_{market.candle_interval}"
            df = client.get_historical_prices(
                market.epic,
                resolution=resolution,
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


def update_htf_trends() -> None:
    """
    Fetch 1H candles and determine higher timeframe trend for each market.
    Called at startup and every 4 hours thereafter.
    Also updates the market regime based on S&P 500 trend.
    """
    global market_regime, market_regime_confirmed
    from src.indicators import calculate_ema, add_all_indicators

    logger.info("Updating higher timeframe trends (1H candles)...")

    for market in MARKETS:
        try:
            rate_limiter.wait_if_needed()
            df = client.get_historical_prices(
                market.epic,
                resolution="HOUR",
                num_points=25,  # Reduced from 50 to conserve API allowance (enough for EMA 9/21)
                use_cache=False,
            )

            if df is None or len(df) < 21:
                # Don't set a value - leave it unset so we know fetch failed
                logger.warning(f"  {market.name}: Insufficient data for HTF trend")
                continue

            # Calculate EMAs on hourly data
            ema_9 = calculate_ema(df["close"], 9)
            ema_21 = calculate_ema(df["close"], 21)

            latest_ema_9 = ema_9.iloc[-1]
            latest_ema_21 = ema_21.iloc[-1]
            latest_close = df["close"].iloc[-1]

            # Determine trend
            if latest_ema_9 > latest_ema_21 and latest_close > latest_ema_21:
                htf_trends[market.epic] = "BULLISH"
            elif latest_ema_9 < latest_ema_21 and latest_close < latest_ema_21:
                htf_trends[market.epic] = "BEARISH"
            else:
                htf_trends[market.epic] = "NEUTRAL"

            logger.info(f"  {market.name}: HTF trend = {htf_trends[market.epic]}")

            # Classify market regime using hourly data (has ADX and ATR)
            try:
                df_with_indicators = add_all_indicators(df, STRATEGY_PARAMS)
                regime = classify_regime(df_with_indicators)
                market_regimes[market.epic] = regime
                logger.info(f"  {market.name}: Regime = {format_regime_status(regime)}")
            except Exception as re:
                logger.warning(f"  {market.name}: Could not classify regime: {re}")

        except Exception as e:
            logger.warning(f"Failed to get HTF trend for {market.name}: {e}")
            # Don't set a value - leave it unset so we know fetch failed

    # Update market regime based on S&P 500 trend
    sp500_trend = htf_trends.get(SP500_EPIC, None)

    if sp500_trend is not None:
        # Successfully fetched S&P 500 data - use real trend
        market_regime = sp500_trend
        market_regime_confirmed = True
        logger.info(f"Market regime (S&P 500): {market_regime} - {'Longs only' if market_regime == 'BULLISH' else 'Shorts only' if market_regime == 'BEARISH' else 'No trades'}")
    else:
        # Couldn't fetch S&P data - default to BULLISH
        # This allows longs while blocking riskier shorts until we have real data
        if not market_regime_confirmed:
            market_regime = "BULLISH"
            if client and client.is_weekend():
                logger.info("Market regime: Defaulting to BULLISH (weekend - markets closed)")
            else:
                logger.info("Market regime: Defaulting to BULLISH (S&P 500 data unavailable - possible API issue)")


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

        # Get higher timeframe trend for this market
        htf_trend = htf_trends.get(epic, "NEUTRAL")

        # Analyze with multi-timeframe context
        trade_signal = strategy.analyze(df, market_config, current_price, htf_trend)

        logger.info(
            f"[STREAM] {market.name}: {trade_signal.signal.value} "
            f"(confidence: {trade_signal.confidence:.0%}) - {trade_signal.reason}"
        )

        if trade_signal.signal == Signal.HOLD:
            return

        # Market regime filter: align trade direction with S&P 500 trend
        if market_regime == "NEUTRAL":
            logger.info(f"Market regime NEUTRAL (S&P sideways) - no trades allowed")
            return
        elif market_regime == "BULLISH" and trade_signal.signal == Signal.SELL:
            logger.info(f"Market regime BULLISH - blocking SELL on {market.name}")
            return
        elif market_regime == "BEARISH" and trade_signal.signal == Signal.BUY:
            logger.info(f"Market regime BEARISH - blocking BUY on {market.name}")
            return

        # Per-market regime filter: check if regime allows trading
        per_market_regime = market_regimes.get(epic)
        if per_market_regime:
            if not per_market_regime.is_tradeable:
                logger.info(
                    f"{market.name}: Regime {per_market_regime.code} not tradeable - skipping"
                )
                return

            # Get regime-adjusted parameters
            regime_params = get_regime_params(per_market_regime)

            # Check if strategy type is allowed in this regime
            if trade_signal.signal == Signal.BUY and not regime_params.allow_trend_follow:
                logger.info(
                    f"{market.name}: Trend-follow (BUY) blocked in {per_market_regime.code} regime"
                )
                return
            if trade_signal.signal == Signal.SELL and not regime_params.allow_trend_follow:
                logger.info(
                    f"{market.name}: Trend-follow (SELL) blocked in {per_market_regime.code} regime"
                )
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

        # Check confidence (market-specific threshold, regime-adjusted)
        min_confidence = market_config.min_confidence
        if per_market_regime:
            regime_params = get_regime_params(per_market_regime)
            # Use the higher of market config or regime requirement
            min_confidence = max(min_confidence, regime_params.min_confidence)

        if trade_signal.confidence < min_confidence:
            logger.info(
                f"Confidence too low for {market_config.name}: {trade_signal.confidence:.0%} < {min_confidence:.0%}"
            )
            return

        # Check loss cooldown - 1 hour after a losing trade
        if epic in loss_cooldown_until:
            if datetime.now() < loss_cooldown_until[epic]:
                remaining = (loss_cooldown_until[epic] - datetime.now()).total_seconds() / 60
                logger.info(
                    f"Loss cooldown active for {market_config.name}: "
                    f"{remaining:.0f} mins remaining"
                )
                return
            else:
                # Cooldown expired, remove it
                del loss_cooldown_until[epic]

        # Check general cooldown - don't re-enter same market too quickly after closing
        cooldown_candles = 3
        cooldown_mins = market_config.candle_interval * cooldown_candles

        # Apply regime cooldown multiplier (longer cooldowns in uncertain regimes)
        if per_market_regime:
            regime_params = get_regime_params(per_market_regime)
            cooldown_mins = cooldown_mins * regime_params.cooldown_multiplier

        if epic in last_close_time:
            elapsed = (datetime.now() - last_close_time[epic]).total_seconds() / 60
            if elapsed < cooldown_mins:
                logger.info(
                    f"Cooldown active for {market_config.name}: "
                    f"{elapsed:.0f}/{cooldown_mins:.0f} mins elapsed"
                )
                return

        # Check economic calendar - avoid trading around high-impact events
        if calendar:
            is_safe, cal_reason = calendar.is_safe_to_trade(epic)
            if not is_safe:
                logger.info(f"Calendar block for {market_config.name}: {cal_reason}")
                return

        # Calculate position size (regime-adjusted)
        position_size = risk_manager.calculate_position_size(
            balance,
            trade_signal.stop_distance,
            market_config,
            regime=per_market_regime,
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
            expiry=market_config.expiry,
        )

        if result:
            telegram.trades_today += 1

            # Track in known_positions for external close detection
            deal_id = result.get("dealId", "")
            if deal_id:
                known_positions[deal_id] = Position(
                    deal_id=deal_id,
                    epic=epic,
                    direction=trade_signal.signal.value,
                    size=position_size.size,
                    open_level=result.get("level", current_price),
                    stop_level=result.get("stopLevel"),
                    limit_level=result.get("limitLevel"),
                    profit_loss=0.0,
                    created_date=datetime.now().isoformat(),
                )

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
            error_reason = client.last_error or "Unknown error"
            if telegram_loop:
                asyncio.run_coroutine_threadsafe(
                    telegram.notify_error(
                        f"Failed to open position on {market.name}: {error_reason}"
                    ),
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
    current_deal_ids = {p.deal_id for p in positions}

    # Detect positions closed externally (stop/limit hit by IG)
    for deal_id, known_pos in list(known_positions.items()):
        if deal_id not in current_deal_ids:
            # Position disappeared - closed by IG (stop or limit hit)
            market_config = next((m for m in MARKETS if m.epic == known_pos.epic), None)
            market_name = market_config.name if market_config else known_pos.epic

            logger.info(f"Position {deal_id} closed externally (stop/limit hit): {market_name}")

            # Record close time for cooldown
            last_close_time[known_pos.epic] = datetime.now()

            # If it was a loss (likely stop hit), set extended loss cooldown
            # Note: profit_loss from known_positions may be stale, but if stop hit it's negative
            if known_pos.profit_loss < 0:
                loss_cooldown_until[known_pos.epic] = datetime.now() + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
                logger.info(f"Loss cooldown set for {market_name}: {LOSS_COOLDOWN_MINUTES} mins")

            if telegram_loop:
                asyncio.run_coroutine_threadsafe(
                    telegram.notify_trade_closed(
                        market_name,
                        known_pos.direction,
                        known_pos.profit_loss,
                        "Stop/limit hit",
                    ),
                    telegram_loop,
                )

            del known_positions[deal_id]

    # Update known positions with current state
    for position in positions:
        known_positions[position.deal_id] = position

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

                # Remove from known_positions so it's not flagged as external close
                known_positions.pop(position.deal_id, None)

                # Record close time for cooldown
                last_close_time[position.epic] = datetime.now()

                # Get P&L from close confirmation (more accurate than position snapshot)
                pnl = result.get("profit", 0.0) or position.profit_loss

                # If it was a loss, set extended loss cooldown
                if pnl < 0:
                    loss_cooldown_until[position.epic] = datetime.now() + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
                    logger.info(f"Loss cooldown set for {market_name}: {LOSS_COOLDOWN_MINUTES} mins")

                telegram.daily_pnl += pnl
                risk_manager.update_daily_pnl(pnl)

                if telegram_loop:
                    asyncio.run_coroutine_threadsafe(
                        telegram.notify_trade_closed(
                            market_name,
                            position.direction,
                            pnl,
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
    """Refresh IG session and reconnect streaming, preserving candle data."""
    global stream_service

    logger.info("Refreshing IG session...")

    # Preserve existing candle data before disconnecting
    preserved_candles = {}
    if stream_service:
        for market in MARKETS:
            market_data = stream_service.get_market_data(market.epic)
            if market_data:
                df = market_data.to_dataframe()
                if df is not None and not df.empty:
                    preserved_candles[market.epic] = df
                    logger.debug(f"Preserved {len(df)} candles for {market.name}")

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

    # Reconnect streaming with new tokens, restoring preserved candles
    if LIGHTSTREAMER_AVAILABLE:
        initialize_streaming(preserved_candles)


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

    # Initialize higher timeframe trends and economic calendar
    update_htf_trends()
    if calendar:
        calendar.refresh()

    # Send startup notification
    balance = client.get_balance() or 0
    market_names = [m.name for m in MARKETS]

    # Build regime summary for notification
    regime_lines = []
    for m in MARKETS:
        regime = market_regimes.get(m.epic)
        if regime:
            regime_lines.append(f"  {m.name}: {regime.code}")

    regime_summary = "\n".join(regime_lines) if regime_lines else "  (pending)"

    mode = "STREAMING" if streaming_enabled else "POLLING"
    await telegram.send_notification(
        f"🚀 *IG Trading Bot Started*\n\n"
        f"Mode: {mode}\n"
        f"Balance: £{balance:,.2f}\n"
        f"Markets: {', '.join(market_names)}\n\n"
        f"*Regimes:*\n{regime_summary}\n\n"
        f"{'Real-time price streaming active!' if streaming_enabled else 'Using scheduled polling (API limited)'}"
    )

    if streaming_enabled:
        logger.info("Bot running with real-time streaming. Analyzing on candle completion.")

        # Start periodic task thread
        periodic_thread = threading.Thread(target=periodic_tasks, daemon=True)
        periodic_thread.start()

        # Schedule session refresh, HTF trends, and daily summary
        import schedule

        schedule.every(6).hours.do(refresh_session)
        schedule.every(4).hours.do(update_htf_trends)  # Was 1 hour - reduced to conserve API allowance
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
