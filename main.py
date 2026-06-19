"""
IG Trading Bot - Main Entry Point

Automated spread betting platform using IG Markets API.
Uses Lightstreamer for real-time price streaming to avoid API rate limits.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_DATA_DIR = Path("/app/data") if os.path.exists("/app") else Path("data")
LAST_STARTUP_FILE = _DATA_DIR / "last_startup.txt"
LAST_HTF_REFRESH_FILE = _DATA_DIR / "last_htf_refresh.txt"
HTF_TRENDS_FILE = _DATA_DIR / "htf_trends.json"
QUIET_RESTART_WINDOW = timedelta(hours=2)
HTF_REFRESH_COOLDOWN = timedelta(hours=6)  # On startup, skip HTF fetch if one ran within this window

from config import (
    load_ig_config,
    load_telegram_config,
    load_trading_config,
    MARKETS,
    STRATEGY_PARAMS,
    get_strategy_for_market,
)
from src.client import IGClient, Position
from src.strategy import TradingStrategy, Signal, should_close_position
from src.risk_manager import RiskManager
from src.telegram_bot import TelegramBot, _session_date
from src.streaming import IGStreamService, MarketStream, LIGHTSTREAMER_AVAILABLE
from src.calendar import EconomicCalendar
from src.journal import TradeJournal
from src.screener import MarketScreener
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
journal: TradeJournal = None
screener: MarketScreener = None
running = True
telegram_loop = None

# Track last analysis time per market to avoid duplicate signals
last_analysis: dict[str, datetime] = {}


def interval_to_resolution(minutes: int) -> str:
    """Map a candle interval in minutes to an IG REST API resolution string.
    IG accepts MINUTE_{1,2,3,5,10,15,30}, HOUR, HOUR_{2,3,4}, DAY, WEEK, MONTH.
    """
    minute_resolutions = {1, 2, 3, 5, 10, 15, 30}
    if minutes in minute_resolutions:
        return f"MINUTE_{minutes}"
    if minutes == 60:
        return "HOUR"
    if minutes in (120, 180, 240):
        return f"HOUR_{minutes // 60}"
    if minutes == 1440:
        return "DAY"
    raise ValueError(f"Unsupported candle_interval: {minutes} minutes")


# Track when positions were last closed - cooldown prevents immediate re-entry
last_close_time: dict[str, datetime] = {}

# Track known open positions to detect external closes (stop/limit hit by IG)
known_positions: dict[str, Position] = {}  # deal_id -> Position

# Deal IDs we've recently fired a close-externally notification for. Guards
# against IG's positions API transiently re-showing a closed position, which
# was causing duplicate Telegram alerts, double-counted daily P&L, and reset
# loss cooldowns. The 4h TTL covers slow IG-side flickers — a Gold trade on
# 2026-05-04 re-appeared 32 min after the phantom close, just past the old
# 30 min window, and triggered a second false detection.
recently_closed_deals: dict[str, datetime] = {}  # deal_id -> close detection time
RECENTLY_CLOSED_TTL_MINUTES = 240

# Streaming watchdog state. The Lightstreamer client doesn't auto-reconnect
# and doesn't surface silent "connected but no ticks" failures, so the
# watchdog tripped on 2026-05-07 when streaming died at 04:20 and the bot
# sat as a Telegram-polling zombie for 5h until manual intervention. The
# watchdog runs from periodic_tasks() once a minute, with a recovery ladder:
# trip detected → refresh_session() (in-process recovery) → if still bad
# 60s later, os._exit(1) so Docker brings us back clean.
_streaming_disconnect_since: Optional[datetime] = None
_streaming_stale_since: Optional[datetime] = None
_streaming_recovery_attempted_at: Optional[datetime] = None
STREAM_DISCONNECT_GRACE = timedelta(minutes=2)
STREAM_STALE_GRACE = timedelta(minutes=3)
STREAM_POST_RECOVERY_GRACE = timedelta(seconds=60)

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

# Track positions that have had their stop moved to break-even
breakeven_applied: set[str] = set()  # deal_ids with BE stop applied

# Track ATR trailing stop levels (after break-even, trail ratchets stop further)
trailing_stop_levels: dict[str, float] = {}  # deal_id -> current trail stop level

# Post-restart cooldown: skip opening new positions for 15 mins after startup
# to let indicators stabilise with fresh streaming data
STARTUP_COOLDOWN_MINUTES = 15

# Correlation-cluster filter — OBSERVATIONAL by default. Markets sharing a
# MarketConfig.correlation_group are one underlying bet; a 2nd group member
# opening the SAME direction within CLUSTER_FILTER_WINDOW_MIN of the first is a
# doubled position. Journal mining (2026-06-11) found same-window same-direction
# equity-index clusters at -£8.23 avg / PF 0.13 vs solo +£2.62 / PF 1.81 (n=7,
# thin — hence observational). With CLUSTER_FILTER_ENFORCE=False the would-block
# is logged + journalled (rejected_signals LIKE 'Cluster-filter%') and the trade
# proceeds; set True to actually skip the 2nd correlated entry. recent_group_entries
# records the last entry time + direction per epic for the window check.
CLUSTER_FILTER_WINDOW_MIN = 15
CLUSTER_FILTER_ENFORCE = False
recent_group_entries: dict[str, tuple[datetime, str]] = {}  # epic -> (entry_time, direction)

# MTF pullback-entry state (StrategyConfig.pullback_entry_atr_frac/window). When a
# signal arms a pullback, we hold it here until price retraces frac×ATR toward the
# EMA (then enter at the better level) or the window expires (then drop it). Keyed
# by epic -> {"signal": TradeSignal, "target": float, "deadline": datetime}.
pending_pullback: dict[str, dict] = {}
bot_start_time: datetime = datetime.now()


def initialize() -> bool:
    """Initialize all components."""
    global client, strategy, risk_manager, telegram, rate_limiter, calendar, journal, screener

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
    journal = TradeJournal()
    # Active-markets cap 8→11 (2026-06-12 veto-vs-outcome analysis,
    # scripts/screener_veto_outcomes.py). The score>=40 quality threshold earns
    # its keep — "Score too low" vetoes (band 0-34) were the only net-negative
    # band (−0.99R, correctly blocked losers). The 8-slot cap was the costly
    # part — "Below top 8" vetoes (score 45+) were net WINNERS (+3.27R) blocked
    # only for lack of a slot on a stale ≤4h-old rank. Raising to 11 admits those
    # while threshold=40 still kills the losers; real concurrency/diversification
    # now binds on MAX_POSITIONS=8 (actual capacity, not stale rank). Paper-trial:
    # re-pull the veto analysis in ~2wk. Caveat: Yahoo-futures proxy, thin sample.
    screener = MarketScreener(max_active=11)

    # Login to IG
    if not client.login():
        logger.error("Failed to login to IG")
        return False

    # Set IG client and risk manager references in telegram bot
    telegram.set_ig_client(client)
    telegram.set_risk_manager(risk_manager)
    telegram.set_journal(journal)
    telegram.set_screener(screener)
    telegram.load_daily_stats()

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
        spreadbet_id = client.get_spreadbet_account_id()
        if spreadbet_id:
            if client.account_id == spreadbet_id:
                logger.info(f"Already on SPREADBET account ({spreadbet_id})")
            else:
                logger.info(f"Switching to SPREADBET account ({spreadbet_id})...")
                if not client.switch_account(spreadbet_id):
                    logger.warning("Failed to switch to SPREADBET account")
                else:
                    logger.info(f"Switched to SPREADBET account")
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

        # Initialize candles - prefer: preserved > disk cache > API
        logger.info("Initializing candle history...")
        trading_config = load_trading_config()

        # Try loading streamed candles from disk first (0 API calls)
        disk_candles = stream_service.load_candles_from_disk()

        for market in MARKETS:
            # 1. Preserved candles from stream reconnect (in-memory)
            if preserved_candles and market.epic in preserved_candles:
                df = preserved_candles[market.epic]
                if df is not None and not df.empty:
                    stream_service.initialize_candles(market.epic, df)
                    logger.info(f"  {market.name}: Restored {len(df)} candles from previous session")
                    continue

            # 2. Disk cache from previous restart (0 API calls)
            if market.epic in disk_candles:
                df = disk_candles[market.epic]
                if df is not None and not df.empty:
                    stream_service.initialize_candles(market.epic, df)
                    continue

            # 3. Fall back to API (costs data points)
            rate_limiter.wait_if_needed()
            resolution = interval_to_resolution(market.candle_interval)
            df = client.get_historical_prices(
                market.epic,
                resolution=resolution,
                num_points=trading_config.price_data_points,
                use_cache=True,  # Use price cache if fresh
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


def update_htf_trends(force: bool = False) -> None:
    """
    Fetch 1H candles and determine higher timeframe trend for each market.
    Called at startup (force=False) and every 24h thereafter (force=True).
    Also updates the market regime based on S&P 500 trend.

    The startup call is skipped if a successful refresh ran within
    HTF_REFRESH_COOLDOWN. Without this guard, every container restart spends
    ~570 API points re-fetching HTF — disastrous during a watchdog-induced
    restart loop. The scheduled 24h refresh sets force=True to bypass the guard.
    """
    global market_regime, market_regime_confirmed
    from src.indicators import calculate_ema, add_all_indicators

    if not force:
        try:
            if LAST_HTF_REFRESH_FILE.exists() and HTF_TRENDS_FILE.exists():
                last_ts = datetime.fromisoformat(LAST_HTF_REFRESH_FILE.read_text().strip())
                since = datetime.now() - last_ts
                if since < HTF_REFRESH_COOLDOWN:
                    cached_trends = json.loads(HTF_TRENDS_FILE.read_text())
                    htf_trends.update(cached_trends)
                    # Also restore S&P 500 regime if present so trade decisions
                    # don't fall back to BULLISH default unnecessarily.
                    sp500_cached = htf_trends.get(SP500_EPIC)
                    if sp500_cached:
                        market_regime = sp500_cached
                        market_regime_confirmed = True
                    logger.info(
                        f"Skipping HTF refresh on startup — last successful refresh "
                        f"was {since.total_seconds()/3600:.1f}h ago "
                        f"(cooldown {HTF_REFRESH_COOLDOWN.total_seconds()/3600:.0f}h). "
                        f"Restored {len(cached_trends)} HTF trends from cache; saves ~570 API points."
                    )
                    return
        except Exception as e:
            logger.warning(f"Could not restore HTF trends from cache, will refresh: {e}")

    logger.info("Updating higher timeframe trends...")

    for market in MARKETS:
        try:
            rate_limiter.wait_if_needed()
            df = client.get_historical_prices(
                market.epic,
                resolution=market.htf_resolution,  # Per-market: HOUR for most, DAY for 1h-candle markets
                num_points=30,  # Need ~21 for EMA + buffer. Reduced from 50 to save API budget.
                use_cache=True,  # Use disk cache if fresh to save API calls
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

    # Record successful refresh so startup calls can skip if recent. We mark
    # the refresh as "successful" if at least half the markets returned data —
    # avoids treating a full-allowance-exhausted run as a real refresh that
    # blocks the next attempt.
    fresh_count = sum(1 for epic in (m.epic for m in MARKETS) if epic in htf_trends)
    if fresh_count >= len(MARKETS) // 2:
        try:
            LAST_HTF_REFRESH_FILE.parent.mkdir(parents=True, exist_ok=True)
            LAST_HTF_REFRESH_FILE.write_text(datetime.now().isoformat())
            HTF_TRENDS_FILE.write_text(json.dumps(dict(htf_trends)))
        except Exception as e:
            logger.warning(f"Could not write HTF refresh cache: {e}")


def _auto_roll_contract(market_config: 'MarketConfig') -> None:
    """
    Auto-detect and update expired futures contracts.

    When IG returns MARKET_ROLLED, fetch the current expiry from the API
    and update the market config in-memory. Notifies via Telegram.
    """
    try:
        info = client.get_market_info(market_config.epic)
        if not info or not info.expiry:
            logger.warning(f"Auto-roll: Could not get market info for {market_config.name}")
            return

        old_expiry = market_config.expiry
        new_expiry = info.expiry

        if new_expiry and new_expiry != old_expiry:
            market_config.expiry = new_expiry
            logger.info(
                f"Auto-rolled {market_config.name}: {old_expiry} -> {new_expiry}"
            )
            if telegram_loop:
                asyncio.run_coroutine_threadsafe(
                    telegram.send_notification(
                        f"🔄 *Contract Rolled*\n\n"
                        f"Market: {market_config.name}\n"
                        f"Old expiry: {old_expiry}\n"
                        f"New expiry: {new_expiry}\n\n"
                        f"_Updated automatically. Next signal will use new contract._"
                    ),
                    telegram_loop,
                )
        else:
            logger.warning(
                f"Auto-roll: {market_config.name} expiry unchanged ({old_expiry}). "
                f"Market may be temporarily offline."
            )

    except Exception as e:
        logger.error(f"Auto-roll failed for {market_config.name}: {e}")


def run_daily_screen(periodic: bool = False) -> None:
    """Run the market screener using streaming data (zero API cost — streaming only).

    Two cadences (2026-06-12, intra-cycle-surger fix):
      - 6 session-aligned full screens (periodic=False) → full log dump + full
        Telegram report, the usual briefings.
      - a frequent re-screen (periodic=True, every 30 min) → catches a market
        whose score crosses the top-N cutoff BETWEEN the 4-hourly screens (e.g.
        NASDAQ surging 52→70 mid-morning, missed until the next scheduled screen).
        Stays SILENT when the active set is unchanged (one log line, no Telegram —
        HDD/noise-safe); only on an actual active-set change does it dump + send a
        concise "active markets updated" alert.
    """
    if not screener or not stream_service:
        return

    prev_active = set(screener.active_epics)

    # Collect spreads from streaming data
    spreads = {}
    for epic, market in stream_service.markets.items():
        if market.bid > 0 and market.offer > 0:
            spreads[epic] = market.offer - market.bid

    scores = screener.run_screen(stream_service, htf_trends, spreads)
    new_active = set(screener.active_epics)
    changed = new_active != prev_active

    # Quiet path: a periodic re-screen that changed nothing — one line, no Telegram.
    if periodic and not changed:
        logger.info(f"Periodic screen: no active-set change ({len(new_active)} active)")
        return

    # Log results (session screen, or a periodic screen that DID change the set)
    active = [s for s in scores if s.is_active]
    inactive = [s for s in scores if not s.is_active and s.score > 0]
    label = "Periodic re-screen — ACTIVE SET CHANGED" if periodic else "Screener results"
    logger.info(f"{label}: {len(active)} active, {len(inactive)} inactive")
    for s in active:
        logger.info(f"  [ACTIVE] {s.name}: {s.score}/100 (ADX={s.adx}, ATR/Spread={s.atr_spread_ratio}x)")
    for s in inactive[:5]:
        logger.info(f"  [OFF] {s.name}: {s.score}/100 — {s.reason}")

    if not telegram_loop:
        return

    if periodic:
        # Concise change alert — only the deltas, not the full board.
        added = sorted((s for s in scores if s.epic in new_active - prev_active),
                       key=lambda s: -s.score)
        removed = sorted((s for s in scores if s.epic in prev_active - new_active),
                         key=lambda s: -s.score)
        lines = ["🔄 <b>Active markets updated</b> (intra-cycle)"]
        for s in added:
            lines.append(f"➕ {s.name} ({s.score})")
        for s in removed:
            lines.append(f"➖ {s.name} ({s.score})")
        lines.append(f"<i>{len(new_active)} active</i>")
        msg = "\n".join(lines)
    else:
        msg = screener.get_scores_text()

    asyncio.run_coroutine_threadsafe(
        telegram.send_notification(msg, parse_mode="HTML"),
        telegram_loop,
    )


def on_price_update(epic: str, market: MarketStream) -> None:
    """
    Callback for real-time price updates (fires on every tick).

    Uses streaming data to check break-even and ATR trail for open positions.
    Zero API cost for checking — only calls update_position_stop() when
    a threshold is actually crossed (throttled by 20% minimum move).
    """
    if not known_positions:
        return

    # Find any open position on this epic
    for deal_id, position in list(known_positions.items()):
        if position.epic != epic:
            continue

        market_config = next((m for m in MARKETS if m.epic == epic), None)
        if not market_config:
            continue

        strategy_cfg = get_strategy_for_market(market_config)

        if position.open_level is None or position.stop_level is None:
            continue

        # --- Break-even check (on every tick) ---
        if deal_id not in breakeven_applied:
            if position.direction == "BUY":
                stop_distance = position.open_level - position.stop_level
                current_profit = (market.bid or 0) - position.open_level
            else:
                stop_distance = position.stop_level - position.open_level
                current_profit = position.open_level - (market.offer or 0)

            if stop_distance > 0 and (market.bid or 0) > 0:
                trigger_points = stop_distance * strategy_cfg.breakeven_trigger_pct
                if current_profit >= trigger_points:
                    # Lock in a fraction of the stop as profit instead of dead-entry.
                    # lock_pct < trigger_pct guarantees the new stop stays behind price.
                    lock_offset = stop_distance * getattr(
                        strategy_cfg, "breakeven_lock_pct", 0.0
                    )
                    if position.direction == "BUY":
                        new_stop = position.open_level + lock_offset
                    else:
                        new_stop = position.open_level - lock_offset
                    lock_note = (
                        f" (+{lock_offset:.1f}pt profit locked)" if lock_offset > 0 else ""
                    )
                    logger.info(
                        f"Break-even trigger for {market_config.name}: "
                        f"profit {current_profit:.1f} pts >= {trigger_points:.1f} pts "
                        f"({strategy_cfg.breakeven_trigger_pct:.0%} of stop). "
                        f"Moving stop {position.stop_level:.1f} -> {new_stop:.1f}{lock_note}"
                    )
                    success = client.update_position_stop(
                        deal_id=deal_id,
                        new_stop_level=new_stop,
                        new_limit_level=position.limit_level,
                    )
                    if success:
                        breakeven_applied.add(deal_id)
                        trailing_stop_levels[deal_id] = new_stop
                        if telegram_loop:
                            asyncio.run_coroutine_threadsafe(
                                telegram.send_notification(
                                    f"🔒 *Break-Even Stop Set*\n\n"
                                    f"Market: {market_config.name}\n"
                                    f"Direction: {position.direction}\n"
                                    f"Entry: {position.open_level}\n"
                                    f"Stop moved: {position.stop_level:.1f} → {new_stop:.1f}\n"
                                    f"Current profit: {current_profit:.1f} pts\n\n"
                                    + (
                                        f"_Stop locks +{lock_offset:.1f}pt profit. ATR trail active._"
                                        if lock_offset > 0
                                        else "_Trade is now risk-free! ATR trail active._"
                                    )
                                ),
                                telegram_loop,
                            )
            continue  # Don't trail until BE is set

        # --- ATR trailing stop (on every tick, after break-even) ---
        df = market.to_dataframe()
        if df is None or len(df) < 20:
            continue

        from src.indicators import add_all_indicators as _add_indicators
        indicator_params = {
            "ema_fast": strategy_cfg.ema_fast,
            "ema_medium": strategy_cfg.ema_medium,
            "ema_slow": strategy_cfg.ema_slow,
            "rsi_period": strategy_cfg.rsi_period,
        }
        df = _add_indicators(df, indicator_params)
        atr = df.iloc[-1]["atr"]

        if atr != atr or atr <= 0:  # NaN check
            continue

        trail_distance = atr * strategy_cfg.atr_trail_mult
        current_stop = trailing_stop_levels.get(deal_id, position.open_level)

        if position.direction == "BUY":
            if not market.bid:
                continue
            new_trail = round(market.bid - trail_distance, 1)
            if new_trail <= current_stop:
                continue
        else:
            if not market.offer:
                continue
            new_trail = round(market.offer + trail_distance, 1)
            if new_trail >= current_stop:
                continue

        # 20% minimum move throttle — prevents excessive API calls
        if abs(new_trail - current_stop) < trail_distance * 0.2:
            continue

        logger.info(
            f"ATR trail for {market_config.name}: "
            f"stop {current_stop:.1f} -> {new_trail:.1f} "
            f"(ATR={atr:.1f}, trail={trail_distance:.1f})"
        )
        success = client.update_position_stop(
            deal_id=deal_id,
            new_stop_level=new_trail,
            new_limit_level=position.limit_level,
        )
        if success:
            trailing_stop_levels[deal_id] = new_trail


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

        # Forex gate. The momentum forex profiles are RETIRED (net-losing — 40% WR,
        # ~-£94 over the current era; see project_two_week_review). Forex now trades
        # ONLY via the breakout strategy, behind the runtime /forex toggle
        # (telegram.forex_mode: off|shadow|breakout). Default "off" = no forex
        # trading. Markets stay subscribed so candles keep archiving and the toggle
        # works live. The breakout analyzer + shadow/breakout routing land next; for
        # now forex never opens a position regardless of mode.
        if market_config.sector == "Forex":
            fx_mode = getattr(telegram, "forex_mode", "off")
            if fx_mode != "off":
                logger.debug(f"[FOREX] {market.name}: mode={fx_mode} — breakout analyzer not yet active")
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

        # Screener gate (instrumented). The strategy is now evaluated for ALL
        # markets — including screener-inactive ones — so we can measure what
        # the top-N cap is actually vetoing. Trading behaviour is unchanged:
        # inactive markets still never open a position (the return below).
        # Common case (inactive + HOLD) stays on the old quiet fast path.
        screener_inactive = bool(screener and not screener.is_active(epic))
        if screener_inactive and trade_signal.signal == Signal.HOLD:
            return

        logger.info(
            f"[STREAM] {market.name}: {trade_signal.signal.value} "
            f"(confidence: {trade_signal.confidence:.0%}) - {trade_signal.reason}"
        )

        # MTF pullback-entry resolution (StrategyConfig.pullback_entry_*). Runs
        # BEFORE the HOLD return so a pending order resolves on price action even
        # when the current candle produces no fresh signal (it's a resting limit,
        # not a re-evaluation). Gold opts in; every other profile has frac=0 and
        # skips this whole block. See scripts/backtest_gold_pullback.py.
        strategy_cfg = get_strategy_for_market(market_config)
        pb_frac = getattr(strategy_cfg, "pullback_entry_atr_frac", 0.0)
        pb_window = getattr(strategy_cfg, "pullback_entry_window", 0)
        pullback_enabled = pb_frac > 0 and pb_window > 0
        pullback_resolved = False
        if pullback_enabled and epic in pending_pullback:
            pend = pending_pullback[epic]
            if datetime.now() > pend["deadline"]:
                logger.info(
                    f"🔬 Pullback [{market.name}]: no {pb_frac}xATR retrace within "
                    f"{pb_window} candles — signal dropped (runaway avoided)"
                )
                _log_suppressed_signal(
                    market_config, df, pend["signal"],
                    f"Pullback-entry-expired ({pb_frac}xATR/{pb_window}c)",
                )
                del pending_pullback[epic]
                # fall through — a fresh signal this candle may re-arm below
            else:
                pend_dir = pend["signal"].signal.value
                reached = (current_price >= pend["target"]) if pend_dir == "SELL" \
                    else (current_price <= pend["target"])
                if reached:
                    logger.info(
                        f"✅ Pullback [{market.name}]: {pend_dir} retrace reached "
                        f"(price {current_price:.1f} vs target {pend['target']:.1f}) — entering"
                    )
                    trade_signal = pend["signal"]
                    del pending_pullback[epic]
                    pullback_resolved = True
                else:
                    logger.debug(
                        f"Pullback [{market.name}]: waiting, {current_price:.1f} "
                        f"not yet at {pend['target']:.1f}"
                    )
                    return

        if trade_signal.signal == Signal.HOLD:
            return

        # Hard direction restriction (MarketConfig.allowed_direction). For
        # markets with a one-sided edge (e.g. US 10-Year T-Note long-only,
        # 2026-06-11 cull review) — block the disallowed side outright.
        if (market_config.allowed_direction
                and trade_signal.signal.value != market_config.allowed_direction):
            logger.info(
                f"{market.name}: {trade_signal.signal.value} blocked — "
                f"market is {market_config.allowed_direction}-only"
            )
            _log_suppressed_signal(
                market_config, df, trade_signal,
                f"Direction-restricted ({market_config.allowed_direction}-only)",
            )
            return

        # MTF pullback-entry ARM. A fresh, valid signal does NOT enter immediately
        # when pullback is enabled — it arms a pending order and waits for price to
        # retrace frac×ATR toward the EMA (resolved at the top of a later call). If
        # ATR is unavailable we fall through to immediate entry (safe default).
        if pullback_enabled and not pullback_resolved:
            atr_val = getattr(trade_signal, "atr", 0.0) or 0.0
            if atr_val > 0:
                offset = pb_frac * atr_val
                if trade_signal.signal.value == "SELL":
                    target = current_price + offset   # wait for a bounce UP to short into
                else:
                    target = current_price - offset   # wait for a dip DOWN to buy
                pending_pullback[epic] = {
                    "signal": trade_signal,
                    "target": target,
                    "deadline": datetime.now() + timedelta(
                        minutes=pb_window * market_config.candle_interval
                    ),
                }
                logger.info(
                    f"🔬 Pullback [{market.name}]: {trade_signal.signal.value} armed @ "
                    f"{trade_signal.confidence:.0%} — wait ≤{pb_window} candles for price "
                    f"to reach {target:.1f} ({pb_frac}xATR={offset:.1f} from {current_price:.1f})"
                )
                return

        # Leg-size (exhaustion) filter — OBSERVATIONAL (log + journal only) for
        # markets that opt in via MarketConfig.leg_filter_lookback (currently
        # NASDAQ 100). Records entries an enforced filter WOULD block so we can
        # measure their realised P&L before turning enforcement on. The trade
        # still proceeds; query journal rejected_signals LIKE 'Leg-filter%'.
        if trade_signal.leg_would_block:
            logger.info(
                f"🔬 Leg filter [{market.name}]: {trade_signal.signal.value} @ "
                f"{trade_signal.confidence:.0%} would be BLOCKED "
                f"(legATR={trade_signal.leg_atr:.1f} > {market_config.leg_filter_threshold}, "
                f"lookback={market_config.leg_filter_lookback}) — observational, trade proceeds"
            )
            _log_suppressed_signal(
                market_config, df, trade_signal,
                f"Leg-filter-would-block (legATR={trade_signal.leg_atr:.1f})",
            )

        # ADX-ceiling (exhaustion) filter — OBSERVATIONAL (log + journal only)
        # for markets that opt in via MarketConfig.adx_ceiling (currently
        # NASDAQ 100, ceiling 55). Records entries an enforced filter WOULD block
        # so we can measure their realised P&L before turning enforcement on. The
        # trade still proceeds; query journal rejected_signals LIKE 'ADX-ceiling%'.
        if trade_signal.adx_ceiling_would_block:
            logger.info(
                f"🔬 ADX ceiling [{market.name}]: {trade_signal.signal.value} @ "
                f"{trade_signal.confidence:.0%} would be BLOCKED "
                f"(ADX={trade_signal.adx:.1f} > {market_config.adx_ceiling}) "
                f"— observational, trade proceeds"
            )
            _log_suppressed_signal(
                market_config, df, trade_signal,
                f"ADX-ceiling-would-block (ADX={trade_signal.adx:.1f})",
            )

        if screener_inactive:
            # Actionable signal on a market the screener benched. Log + journal
            # it (log-only, no trade) so the veto cost is measurable. sc.reason
            # distinguishes "Below top 8" (cap veto) from "Score too low (N)".
            sc = screener.get_score(epic)
            detail = f"score {sc.score}, {sc.reason}" if sc else "no score"
            logger.info(
                f"📋 Screener veto: {market.name} {trade_signal.signal.value} "
                f"@ {trade_signal.confidence:.0%} would have traded but inactive ({detail})"
            )
            _log_suppressed_signal(
                market_config, df, trade_signal, f"Screener-inactive ({detail})"
            )
            return

        # Market regime filter: only applied to indices (S&P, NASDAQ)
        # Commodities and forex have independent drivers and don't correlate with S&P
        if market_config.strategy == "indices":
            if market_regime == "NEUTRAL":
                logger.info(f"Market regime NEUTRAL (S&P sideways) - no trades allowed")
                _log_suppressed_signal(market_config, df, trade_signal, "Regime NEUTRAL (S&P sideways)")
                return
            elif market_regime == "BULLISH" and trade_signal.signal == Signal.SELL:
                logger.info(f"Market regime BULLISH - blocking SELL on {market.name}")
                _log_suppressed_signal(market_config, df, trade_signal, "Regime BULLISH blocks SELL")
                return
            elif market_regime == "BEARISH" and trade_signal.signal == Signal.BUY:
                logger.info(f"Market regime BEARISH - blocking BUY on {market.name}")
                _log_suppressed_signal(market_config, df, trade_signal, "Regime BEARISH blocks BUY")
                return

        # Per-market regime filter: check if regime allows trading
        per_market_regime = market_regimes.get(epic)
        if per_market_regime:
            if not per_market_regime.is_tradeable:
                logger.info(
                    f"{market.name}: Regime {per_market_regime.code} not tradeable - skipping"
                )
                _log_suppressed_signal(
                    market_config, df, trade_signal,
                    f"Regime {per_market_regime.code} not tradeable",
                )
                return

            # Note: Removed regime-based trend-follow blocking.
            # The per-market strategy profiles already have ADX thresholds built into
            # signal generation (ADX 20 for indices, ADX 25 for others).
            # The regime system was using hourly data with insufficient points,
            # causing ADX=NaN and false RANGING classification that blocked valid signals.

        # Get balance and positions (REST API calls)
        balance = client.get_balance()
        if not balance:
            return

        # Use both API positions AND local known_positions to catch duplicates
        # This fixes race condition where API doesn't return newly opened positions immediately
        api_positions = client.get_positions()
        local_positions = list(known_positions.values())

        # Merge: use API positions but add any local positions not in API response
        api_deal_ids = {p.deal_id for p in api_positions}
        positions = api_positions + [p for p in local_positions if p.deal_id not in api_deal_ids]

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
            journal.log_rejected_signal(
                epic, market_config.name, trade_signal.signal.value,
                trade_signal.confidence, trade_signal.adx,
                trade_signal.rsi, reason,
            )
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
            journal.log_rejected_signal(
                epic, market_config.name, trade_signal.signal.value,
                trade_signal.confidence, trade_signal.adx,
                trade_signal.rsi, f"Confidence {trade_signal.confidence:.0%} < {min_confidence:.0%}",
            )
            return

        # Time-of-day filter: only trade during active sessions
        # Per-market hours: indices 04-20 UTC, forex/commodities 23-21 UTC (nearly 24h)
        current_hour = datetime.now().hour
        t_start = market_config.trading_start
        t_end = market_config.trading_end
        if t_start < t_end:
            # Normal range (e.g. 04-20)
            outside_hours = current_hour < t_start or current_hour >= t_end
        else:
            # Wrap-around range (e.g. 23-21 means trade from 23:00 to 20:59)
            outside_hours = t_end <= current_hour < t_start
        if outside_hours:
            logger.info(
                f"Outside trading hours for {market_config.name}: "
                f"{current_hour:02d}:00 UTC (active: {t_start:02d}:00-{t_end:02d}:00)"
            )
            _log_suppressed_signal(
                market_config, df, trade_signal,
                f"Outside hours ({current_hour:02d} UTC, active {t_start:02d}-{t_end:02d})",
            )
            return

        # Post-restart cooldown: let indicators stabilise with fresh streaming data
        mins_since_start = (datetime.now() - bot_start_time).total_seconds() / 60
        if mins_since_start < STARTUP_COOLDOWN_MINUTES:
            logger.info(
                f"Startup cooldown for {market_config.name}: "
                f"{mins_since_start:.0f}/{STARTUP_COOLDOWN_MINUTES} mins elapsed"
            )
            _log_suppressed_signal(
                market_config, df, trade_signal,
                f"Startup cooldown ({mins_since_start:.0f}/{STARTUP_COOLDOWN_MINUTES}m)",
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
                _log_suppressed_signal(
                    market_config, df, trade_signal,
                    f"Loss cooldown ({remaining:.0f}m left)",
                )
                return
            else:
                # Cooldown expired, remove it
                del loss_cooldown_until[epic]

        # Check general cooldown - don't re-enter same market too quickly after closing
        # Increased from 3 to 6 candles to prevent chasing exhausted moves
        cooldown_candles = 6
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
                _log_suppressed_signal(
                    market_config, df, trade_signal,
                    f"Re-entry cooldown ({elapsed:.0f}/{cooldown_mins:.0f}m)",
                )
                return

        # Check economic calendar - avoid trading around high-impact events
        if calendar:
            is_safe, cal_reason = calendar.is_safe_to_trade(epic)
            if not is_safe:
                logger.info(f"Calendar block for {market_config.name}: {cal_reason}")
                _log_suppressed_signal(
                    market_config, df, trade_signal,
                    f"Calendar: {cal_reason}",
                )
                return

        # Clamp stop/limit to IG's live minNormalStopOrLimitDistance.
        # Why: config.py min_stop_distance is the strategy floor, but IG's actual
        # minimum varies by account tier / region / CFD-vs-spreadbet. If our value
        # is below IG's, the order is rejected with ATTACHED_ORDER_LEVEL_ERROR.
        live_market_info = client.get_market_info(epic)
        if live_market_info:
            if live_market_info.market_status != "TRADEABLE":
                logger.info(
                    f"[{market.name}] SKIP open: market_status={live_market_info.market_status} "
                    f"(signal={trade_signal.signal.value}, confidence={trade_signal.confidence:.0%})"
                )
                _log_suppressed_signal(
                    market_config, df, trade_signal,
                    f"market_status={live_market_info.market_status}",
                )
                return
            if live_market_info.min_stop_distance > 0:
                ig_min = live_market_info.min_stop_distance + 0.5  # 0.5pt buffer
                if trade_signal.stop_distance < ig_min:
                    logger.info(
                        f"[{market.name}] Raising stop {trade_signal.stop_distance:.2f} "
                        f"-> {ig_min:.2f} (IG min {live_market_info.min_stop_distance:.2f})"
                    )
                    trade_signal.stop_distance = ig_min
                if trade_signal.limit_distance < ig_min:
                    trade_signal.limit_distance = ig_min

        # Calculate position size (regime-adjusted)
        position_size = risk_manager.calculate_position_size(
            balance,
            trade_signal.stop_distance,
            market_config,
            regime=per_market_regime,
        )

        if not position_size.approved:
            logger.warning(f"Position size not approved: {position_size.reason}")
            _log_suppressed_signal(
                market_config, df, trade_signal,
                f"Position sizing: {position_size.reason}",
            )
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

        # Final safety check: reject absurd stop/limit distances. Price-relative
        # to match the strategy ceiling (close*0.05): the old flat min_stop*25
        # would reject legitimate full-ATR index stops once strategy.py stopped
        # truncating them (e.g. NASDAQ ~190 > 4*25=100). current_price*0.0625
        # keeps the same 20:25 ratio vs the strategy cap while still rejecting
        # genuine corruption. min_stop*25 stays as the floor for low-priced markets.
        max_safe_stop = max(market_config.min_stop_distance * 25, current_price * 0.0625)
        if trade_signal.stop_distance > max_safe_stop:
            logger.error(
                f"[{market.name}] BLOCKED: stop_distance={trade_signal.stop_distance:.2f} "
                f"exceeds safety limit {max_safe_stop:.1f}. Likely corrupted data."
            )
            _log_suppressed_signal(
                market_config, df, trade_signal,
                f"Stop {trade_signal.stop_distance:.1f} > safety {max_safe_stop:.1f} (likely bad data)",
            )
            return

        # Spread check: reject if stop distance is too close to current spread
        # (e.g. overnight sessions with wide spreads would trigger stop immediately)
        if market.bid and market.offer:
            spread = market.offer - market.bid
            if spread > 0 and trade_signal.stop_distance < spread * 1.5:
                logger.warning(
                    f"[{market.name}] BLOCKED: stop_distance={trade_signal.stop_distance:.2f} "
                    f"< 1.5x spread ({spread:.2f}). Spread too wide for safe entry."
                )
                _log_suppressed_signal(
                    market_config, df, trade_signal,
                    f"Stop {trade_signal.stop_distance:.1f} < 1.5x spread ({spread:.1f})",
                )
                return

        # Correlation-cluster filter — OBSERVATIONAL (log + journal only unless
        # CLUSTER_FILTER_ENFORCE). When a correlated group member already opened
        # the SAME direction within the window, this 2nd entry is a doubled bet
        # that historically whipsaws (cluster avg -£8.23 vs solo +£2.62, journal
        # 2026-06-11). The first member is let through; only the 2nd+ is flagged
        # — which is exactly what an enforced guard would block. Placed last so
        # only trades that would actually open are counted (matches the backtest).
        if market_config.correlation_group:
            now_t = datetime.now()
            cluster_peer = None
            for o_epic, (o_time, o_dir) in recent_group_entries.items():
                if o_epic == epic or o_dir != trade_signal.signal.value:
                    continue
                o_cfg = next((m for m in MARKETS if m.epic == o_epic), None)
                if not o_cfg or o_cfg.correlation_group != market_config.correlation_group:
                    continue
                age_min = (now_t - o_time).total_seconds() / 60
                if 0 <= age_min <= CLUSTER_FILTER_WINDOW_MIN:
                    cluster_peer = (o_cfg.name, age_min)
                    break
            if cluster_peer:
                peer_name, peer_age = cluster_peer
                verb = "BLOCKED" if CLUSTER_FILTER_ENFORCE else "would be BLOCKED"
                tail = "enforced" if CLUSTER_FILTER_ENFORCE else "observational, trade proceeds"
                logger.info(
                    f"🔬 Cluster filter [{market.name}]: {trade_signal.signal.value} @ "
                    f"{trade_signal.confidence:.0%} {verb} — correlated {peer_name} "
                    f"opened {trade_signal.signal.value} {peer_age:.0f}m ago "
                    f"(window {CLUSTER_FILTER_WINDOW_MIN}m) — {tail}"
                )
                _log_suppressed_signal(
                    market_config, df, trade_signal,
                    f"Cluster-filter-would-block (corr {peer_name} "
                    f"{trade_signal.signal.value} {peer_age:.0f}m ago)",
                )
                if CLUSTER_FILTER_ENFORCE:
                    return

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
            # trades_today incremented in notify_trade_opened()

            # Record group entry for the correlation-cluster window check so a
            # later correlated same-direction entry can detect this one.
            if market_config.correlation_group:
                recent_group_entries[epic] = (datetime.now(), trade_signal.signal.value)

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

                # Journal entry with indicator snapshot. Read indicators from
                # the signal, not from df — main.py's df has no indicator
                # columns (they're computed inside strategy.analyze() on a
                # copy), so latest.get(...) used to silently store 0.0.
                journal.log_entry(
                    deal_id=deal_id,
                    epic=epic,
                    market_name=market_config.name,
                    direction=trade_signal.signal.value,
                    size=position_size.size,
                    entry_price=result.get("level", current_price),
                    stop_distance=trade_signal.stop_distance,
                    limit_distance=trade_signal.limit_distance,
                    confidence=trade_signal.confidence,
                    reason=trade_signal.reason,
                    strategy=market_config.strategy,
                    adx=trade_signal.adx,
                    rsi=trade_signal.rsi,
                    atr=trade_signal.atr,
                    ema_fast=trade_signal.ema_fast,
                    ema_medium=trade_signal.ema_medium,
                    ema_slow=trade_signal.ema_slow,
                    htf_trend=htf_trends.get(epic, "NEUTRAL"),
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

            # Auto-roll: detect expired futures and update expiry in-memory
            if "MARKET_ROLLED" in str(error_reason).upper() or "MARKET ROLLED" in str(error_reason).upper():
                _auto_roll_contract(market_config)

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

    # Drop expired entries from the recently-closed guard
    now = datetime.now()
    ttl = timedelta(minutes=RECENTLY_CLOSED_TTL_MINUTES)
    for deal_id in list(recently_closed_deals.keys()):
        if now - recently_closed_deals[deal_id] > ttl:
            del recently_closed_deals[deal_id]

    # Detect positions closed externally (stop/limit hit by IG)
    for deal_id, known_pos in list(known_positions.items()):
        if deal_id not in current_deal_ids:
            # Already fired close for this deal recently — IG's positions API
            # is flickering. Just clean up local state and skip notification.
            if deal_id in recently_closed_deals:
                logger.debug(
                    f"Suppressed duplicate close detection for {deal_id} "
                    f"(already fired {(now - recently_closed_deals[deal_id]).total_seconds():.0f}s ago)"
                )
                breakeven_applied.discard(deal_id)
                trailing_stop_levels.pop(deal_id, None)
                del known_positions[deal_id]
                continue

            # Position disappeared - closed by IG (stop or limit hit)
            market_config = next((m for m in MARKETS if m.epic == known_pos.epic), None)
            market_name = market_config.name if market_config else known_pos.epic

            logger.info(f"Position {deal_id} closed externally (stop/limit hit): {market_name}")
            recently_closed_deals[deal_id] = now

            # Clean up break-even and trailing stop tracking
            breakeven_applied.discard(deal_id)
            trailing_stop_levels.pop(deal_id, None)

            # Record close time for cooldown
            last_close_time[known_pos.epic] = datetime.now()

            # Try to match the close against IG transaction history (1 API call).
            # IG often hasn't published the txn yet at this exact moment, so on a miss
            # we record PROVISIONAL with the cached pnl and let the reconciliation task
            # in periodic_tasks() correct it once IG settles.
            txn = client.find_close_transaction(
                known_pos.open_level, known_pos.direction
            )
            if txn is not None:
                actual_pnl = client._parse_pnl(txn.get("profitAndLoss", "0"))
                exit_price = float(txn.get("closeLevel") or 0.0)
                journal_status = "CLOSED"
                logger.info(
                    f"Actual P&L for {market_name}: £{actual_pnl:.2f} "
                    f"(cached was £{known_pos.profit_loss:.2f}), exit={exit_price}"
                )
            else:
                actual_pnl = known_pos.profit_loss
                exit_price = 0.0
                journal_status = "PROVISIONAL"
                logger.warning(
                    f"Provisional P&L for {market_name}: £{actual_pnl:.2f} (cached) "
                    f"— will reconcile against IG transactions"
                )

            # If it was a loss, set extended loss cooldown
            if actual_pnl < 0:
                loss_cooldown_until[known_pos.epic] = datetime.now() + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
                logger.info(f"Loss cooldown set for {market_name}: {LOSS_COOLDOWN_MINUTES} mins")

            risk_manager.update_daily_pnl(actual_pnl)

            # Journal exit
            journal.log_exit(
                deal_id, actual_pnl, "Stop/limit hit",
                exit_price=exit_price, status=journal_status,
            )

            if telegram_loop:
                asyncio.run_coroutine_threadsafe(
                    telegram.notify_trade_closed(
                        market_name,
                        known_pos.direction,
                        actual_pnl,
                        "Stop/limit hit",
                        provisional=(journal_status == "PROVISIONAL"),
                    ),
                    telegram_loop,
                )

            del known_positions[deal_id]

    # Update known positions with current state. Skip deals we've recently
    # fired a close for — IG sometimes briefly re-shows closed positions and
    # we don't want to start tracking them again only to fire another close.
    for position in positions:
        if position.deal_id in recently_closed_deals:
            continue
        known_positions[position.deal_id] = position

    # Break-even and ATR trailing stop now handled in on_price_update()
    # (runs on every streaming tick instead of every 60 seconds)

    for position in positions:
        market = stream_service.get_market_data(position.epic)
        if not market:
            continue

        df = market.to_dataframe()
        if df is None or df.empty:
            continue

        # Get market config for strategy-specific exit rules
        market_config = next((m for m in MARKETS if m.epic == position.epic), None)

        # Get current HTF trend for dynamic exit decisions
        current_htf_trend = htf_trends.get(position.epic, "NEUTRAL")

        should_close, reason = should_close_position(
            df, position.direction, STRATEGY_PARAMS, market=market_config,
            htf_trend=current_htf_trend
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

                # Clean up break-even and trailing stop tracking
                breakeven_applied.discard(position.deal_id)
                trailing_stop_levels.pop(position.deal_id, None)

                # Record close time for cooldown
                last_close_time[position.epic] = datetime.now()

                # Get P&L + close fill level from deal confirmation (broker-authoritative).
                # `profit` may legitimately be 0 (BE close), so distinguish "missing" via None.
                confirmed_profit = result.get("profit")
                exit_price = float(result.get("level") or 0.0)
                if confirmed_profit is not None:
                    pnl = float(confirmed_profit)
                    journal_status = "CLOSED"
                else:
                    pnl = position.profit_loss
                    journal_status = "PROVISIONAL"

                # If it was a loss, set extended loss cooldown
                if pnl < 0:
                    loss_cooldown_until[position.epic] = datetime.now() + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
                    logger.info(f"Loss cooldown set for {market_name}: {LOSS_COOLDOWN_MINUTES} mins")

                # daily_pnl incremented in notify_trade_closed()
                risk_manager.update_daily_pnl(pnl)

                # Journal exit with current ADX + actual close fill level
                exit_adx = 0.0
                if df is not None and len(df) > 0 and "adx" in df.columns:
                    exit_adx = float(df.iloc[-1]["adx"])
                journal.log_exit(
                    position.deal_id, pnl, reason,
                    exit_price=exit_price, adx_at_exit=exit_adx,
                    status=journal_status,
                )

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


def reconcile_provisional_trades(closing_books: bool = False) -> None:
    """Match PROVISIONAL journal rows against IG transaction history.

    closing_books=True is called ONCE from send_daily_summary at 21:00, before
    the reset, to make the summary reconciliation-aware: it bypasses the session
    gate (so a trade that closed in the session being summarised but whose IG
    transaction only posted around the 21:00 boundary is still pulled into the
    summary figure) and updates risk_manager.daily_pnl IN PLACE — which the
    summary reads. It deliberately skips the per-trade Telegram messages and the
    telegram.daily_pnl coroutine adjustment: a deferred coroutine could land
    after reset_daily_stats and re-introduce the cross-session leak, and the
    summary itself already reports the corrected total.

    Runs every minute. Gated by has_provisional() so we make zero API calls
    on minutes where there's nothing to reconcile (the common case).

    Rows older than 3h with still no match are flagged UNMATCHED so we stop
    retrying — their cached pnl is the best we'll get. (3h, raised from 2h on
    2026-06-10: overnight commodity bookings can post >2h late, causing false
    UNMATCHED warnings on trades that did reconcile fine — e.g. Gold £21.31.)
    """
    if not journal.has_provisional():
        return

    txns = client.get_recent_transactions(hours=24)
    if not txns:
        return

    rows = journal.get_provisional_rows(max_age_hours=24)
    stale_cutoff = datetime.now() - timedelta(hours=3)
    matched = 0
    aged_out = 0
    for row in rows:
        txn = client.find_close_transaction(
            row["entry_price"], row["direction"], transactions=txns
        )
        if txn is not None:
            actual_pnl = IGClient._parse_pnl(txn.get("profitAndLoss", "0"))
            exit_price = float(txn.get("closeLevel") or 0.0)
            provisional_pnl = row["pnl"] or 0.0
            journal.confirm_provisional(row["deal_id"], actual_pnl, exit_price)
            matched += 1
            logger.info(
                f"Reconciled {row['market_name']} ({row['deal_id']}): "
                f"pnl £{provisional_pnl:.2f} -> £{actual_pnl:.2f}, exit_price -> {exit_price}"
            )
            # Correct risk_manager's running daily P&L by the delta and notify
            # the user. notify_trade_reconciled corrects telegram.daily_pnl too.
            # BUT only touch the LIVE daily counters when this trade's close
            # belongs to the current 21:00-reset session. A trade reconciled
            # after the daily summary + reset belongs to the already-summarised
            # prior session; applying its delta leaks a phantom P&L (with 0
            # trades) into the fresh session. The journal correction above
            # (confirm_provisional) is unconditional, so history stays accurate.
            delta = actual_pnl - provisional_pnl
            exit_ref = row.get("exit_time") or row.get("entry_time")
            same_session = True
            if exit_ref:
                try:
                    same_session = (
                        _session_date(datetime.fromisoformat(exit_ref))
                        == _session_date()
                    )
                except (TypeError, ValueError):
                    same_session = True
            apply = same_session or closing_books
            if abs(delta) > 0.01 and apply:
                risk_manager.update_daily_pnl(delta)
            # In closing_books mode the summary reports the correction itself, so
            # skip the per-trade message + the deferred telegram.daily_pnl bump
            # (which could land post-reset and re-leak into the new session).
            if telegram_loop and not closing_books:
                asyncio.run_coroutine_threadsafe(
                    telegram.notify_trade_reconciled(
                        row["market_name"], provisional_pnl, actual_pnl,
                        adjust_counter=same_session,
                    ),
                    telegram_loop,
                )
        else:
            try:
                entry_dt = datetime.fromisoformat(row["entry_time"])
            except (TypeError, ValueError):
                continue
            if entry_dt < stale_cutoff:
                journal.mark_unmatched(row["deal_id"])
                aged_out += 1
                logger.warning(
                    f"Marking {row['market_name']} ({row['deal_id']}) UNMATCHED — "
                    f"no IG transaction after 3h"
                )
    if matched or aged_out:
        logger.info(f"Reconciliation: {matched} confirmed, {aged_out} aged out")


def _log_suppressed_signal(
    market_config, df, trade_signal, reason: str
) -> None:
    """Journal a non-HOLD signal blocked downstream of strategy.analyze().

    Without this, post-mortems can't tell whether a setup was filtered out
    by the strategy itself (HOLD with a stream-log explanation) or by an
    env/risk gate further down (silent return). The 2026-05-07 EUR/USD
    investigation hit this gap: BUY 76% at 03:05 with no record of why we
    didn't act.
    """
    try:
        if not journal or df is None or df.empty:
            return
        journal.log_rejected_signal(
            market_config.epic,
            market_config.name,
            trade_signal.signal.value,
            trade_signal.confidence,
            # From the signal, not df — df here has no indicator columns.
            float(getattr(trade_signal, "adx", 0) or 0),
            float(getattr(trade_signal, "rsi", 0) or 0),
            reason,
        )
    except Exception as e:
        logger.debug(f"Journal: failed to log suppressed signal: {e}")


def _maybe_rotate_daily_stats() -> None:
    """Re-save daily_stats.json when the calendar date changes.

    The 21:00 daily summary already rotates state at the operational boundary.
    This handles the calendar boundary so the on-disk file's date field
    matches today even on a no-trade day, which makes diagnostics less
    confusing (file last touched yesterday vs. today).
    """
    if not telegram:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    if getattr(_maybe_rotate_daily_stats, "_last_date", None) == today:
        return
    try:
        telegram.save_daily_stats()
        _maybe_rotate_daily_stats._last_date = today
    except Exception as e:
        logger.debug(f"Failed to rotate daily stats: {e}")


def _streaming_watchdog() -> None:
    """Detect dead/stalled streaming and recover.

    Two failure modes get tripped:
      1. status flips to DISCONNECTED for >= STREAM_DISCONNECT_GRACE
      2. last tick across all markets is older than STREAM_STALE_GRACE while
         at least one market is still TRADEABLE (suppresses weekend silence)

    Recovery ladder:
      - First trip: call refresh_session() to logout/login/resubscribe
      - Still bad >= STREAM_POST_RECOVERY_GRACE after that attempt: os._exit(1)
        so Docker restarts us clean
    """
    global _streaming_disconnect_since, _streaming_stale_since
    global _streaming_recovery_attempted_at

    if not stream_service:
        return

    now = datetime.now()

    # Signal 1: disconnected for too long
    if not stream_service.connected:
        if _streaming_disconnect_since is None:
            _streaming_disconnect_since = now
        disconnect_age = now - _streaming_disconnect_since
    else:
        _streaming_disconnect_since = None
        disconnect_age = timedelta(0)

    # Signal 2: connected but no ticks while markets are open
    tick_age = stream_service.most_recent_tick_age()
    tradeable = stream_service.tradeable_market_count()
    is_stale = (
        stream_service.connected
        and tick_age is not None
        and tradeable > 0
        and tick_age > STREAM_STALE_GRACE
    )
    if is_stale:
        if _streaming_stale_since is None:
            _streaming_stale_since = now
        stale_duration = now - _streaming_stale_since
    else:
        _streaming_stale_since = None
        stale_duration = timedelta(0)

    tripped = (
        disconnect_age > STREAM_DISCONNECT_GRACE
        or stale_duration > timedelta(0)
    )

    if not tripped:
        # Recovery succeeded — clear the attempt marker once we see a fresh tick
        if _streaming_recovery_attempted_at is not None and tick_age is not None:
            if tick_age < STREAM_STALE_GRACE:
                logger.info("Streaming watchdog: recovery confirmed, ticks flowing again")
                _streaming_recovery_attempted_at = None
        return

    reason = (
        f"disconnected for {disconnect_age}"
        if disconnect_age > STREAM_DISCONNECT_GRACE
        else f"no ticks for {tick_age} ({tradeable} markets tradeable)"
    )

    if _streaming_recovery_attempted_at is None:
        logger.warning(f"Streaming watchdog tripped: {reason} — attempting refresh_session()")
        _streaming_recovery_attempted_at = now
        try:
            refresh_session()
        except Exception as e:
            logger.error(f"refresh_session() raised during watchdog recovery: {e}")
        return

    # Recovery already attempted — wait the grace window before exiting
    since_attempt = now - _streaming_recovery_attempted_at
    if since_attempt < STREAM_POST_RECOVERY_GRACE:
        logger.warning(
            f"Streaming watchdog: still bad ({reason}) "
            f"{since_attempt.total_seconds():.0f}s after recovery — "
            f"giving it until {STREAM_POST_RECOVERY_GRACE.total_seconds():.0f}s"
        )
        return

    logger.critical(
        f"Streaming watchdog: refresh_session() did not recover ({reason}). "
        f"Exiting so Docker restarts the container."
    )
    if telegram and telegram_loop:
        try:
            asyncio.run_coroutine_threadsafe(
                telegram.notify_error(
                    f"Streaming watchdog: {reason}. In-process recovery failed. "
                    f"Restarting via Docker."
                ),
                telegram_loop,
            )
        except Exception:
            pass
    if stream_service:
        try:
            stream_service.save_candles_to_disk()
        except Exception:
            pass
    os._exit(1)


def periodic_tasks() -> None:
    """Run periodic tasks (position checks, etc.)."""
    while running:
        try:
            # Check positions every minute
            check_positions_from_stream()

            # Reconcile any provisional pnl/exit_price against IG transactions
            reconcile_provisional_trades()

            # Detect dead/stalled streaming and recover (or hard-exit)
            _streaming_watchdog()

            # Keep daily_stats.json's date field current across midnight
            _maybe_rotate_daily_stats()

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
    """Send daily summary (reconciliation-aware).

    Force a final reconciliation sweep first so any trade that closed this
    session but only got booked by IG around the 21:00 boundary is reflected in
    the headline P&L (closing_books bypasses the session gate and corrects
    risk_manager.daily_pnl in place). Whatever is still PROVISIONAL afterwards
    (IG hasn't booked it yet) is reported as pending so the figure is honestly
    labelled rather than silently provisional.
    """
    try:
        reconcile_provisional_trades(closing_books=True)
    except Exception as e:
        logger.warning(f"Closing-books reconcile before daily summary failed: {e}")

    # Trades that closed this session but IG hasn't confirmed yet. Anything from
    # a prior session would already be aged-out to UNMATCHED (3h), so these are
    # all this-session closes.
    try:
        pending = journal.get_provisional_rows(max_age_hours=24)
    except Exception:
        pending = []
    pending_count = len(pending)
    pending_pnl = sum((p.get("pnl") or 0.0) for p in pending)

    balance = client.get_balance()
    positions = client.get_positions()
    # risk_manager.daily_pnl is synchronously corrected by the sweep above and
    # tracks telegram.daily_pnl in lockstep — use it as the authoritative figure.
    summary_pnl = risk_manager.daily_pnl

    if balance and telegram and telegram_loop:
        asyncio.run_coroutine_threadsafe(
            telegram.notify_daily_summary(
                balance,
                summary_pnl,
                telegram.trades_today,
                positions,
                pending_count=pending_count,
                pending_pnl=pending_pnl,
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

    # Cleanly end the old session before creating a new one. IG returns a
    # misleading "api-key-invalid" when you POST /session while another
    # session for the same user is still active, so always logout first.
    try:
        client.logout()
    except Exception as e:
        logger.debug(f"Logout before refresh failed (ignoring): {e}")
    # Small pause — IG can take a moment to release the old session server-side.
    time.sleep(2)

    # Re-login (with a single retry in case of transient 403)
    if not client.login():
        logger.warning("Refresh login failed, retrying in 5s after explicit logout...")
        try:
            client.logout()
        except Exception:
            pass
        time.sleep(5)
        if not client.login():
            logger.error("Failed to refresh session after retry")
            if telegram and telegram_loop:
                asyncio.run_coroutine_threadsafe(
                    telegram.notify_error("Failed to refresh IG session"),
                    telegram_loop,
                )
            return

    # Need to switch back to SPREADBET account for streaming to work
    spreadbet_id = client.get_spreadbet_account_id()
    if spreadbet_id and spreadbet_id != client.account_id:
        client.switch_account(spreadbet_id)

    # Reconnect streaming with new tokens, restoring preserved candles
    if LIGHTSTREAMER_AVAILABLE:
        initialize_streaming(preserved_candles)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info(f"Received signal {signum}, shutting down...")

    # Save candle data to disk so next restart doesn't need API calls
    if stream_service:
        stream_service.save_candles_to_disk()

    running = False


# --- Scheduled candle-persistence wrappers -------------------------------
# These MUST call through the *current* global stream_service. The global is
# reassigned to a brand-new IGStreamService on every session refresh
# (refresh_session_and_reconnect -> initialize_streaming, ~every 6h when IG
# tokens expire). A bound method captured at schedule-registration time
# (schedule.do(stream_service.save_candles_to_disk)) keeps pointing at the OLD,
# now-disconnected instance whose candle deque is frozen at the moment of the
# refresh — so save/archive silently persist stale data forever while live
# trading uses the fresh instance. Referencing the global by name here resolves
# it at call time, so the jobs always act on the live stream. (Bug: 2026-06-15 —
# archive froze at Fri 17:15 after a Fri-evening refresh orphaned the jobs.)
def _scheduled_save_candles():
    if stream_service:
        stream_service.save_candles_to_disk()


def _scheduled_archive_candles():
    if stream_service:
        stream_service.archive_candles_to_disk()


def _scheduled_prune_archive():
    if stream_service:
        stream_service.prune_candle_archive()


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
            regime_lines.append(f"  {m.name}: {regime.code.replace('_', ' ')}")

    regime_summary = "\n".join(regime_lines) if regime_lines else "  (pending)"

    mode = "STREAMING" if streaming_enabled else "POLLING"

    # Quiet-restart guard: skip the Telegram banner if we restarted within
    # the last 30 minutes. IG demo flakiness causes the watchdog to fire
    # restart cycles; we don't want to spam the user with startup notifications.
    skip_banner = False
    try:
        if LAST_STARTUP_FILE.exists():
            last_ts = datetime.fromisoformat(LAST_STARTUP_FILE.read_text().strip())
            since = datetime.now() - last_ts
            if since < QUIET_RESTART_WINDOW:
                skip_banner = True
                logger.info(f"Quiet restart: last startup {since.total_seconds()/60:.1f} min ago, skipping Telegram banner")
    except Exception as e:
        logger.warning(f"Could not read last-startup timestamp: {e}")
    try:
        LAST_STARTUP_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_STARTUP_FILE.write_text(datetime.now().isoformat())
    except Exception as e:
        logger.warning(f"Could not write last-startup timestamp: {e}")

    if not skip_banner:
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

        # Run initial market screen (uses streaming data already loaded)
        run_daily_screen()

        # Start periodic task thread
        periodic_thread = threading.Thread(target=periodic_tasks, daemon=True)
        periodic_thread.start()

        # Schedule session refresh, HTF trends, and daily summary
        import schedule

        schedule.every(6).hours.do(refresh_session)
        schedule.every(24).hours.do(update_htf_trends, force=True)  # 1x/day — 19 markets × 30pts = 570pts/day, 3,990/week
        schedule.every(15).minutes.do(_scheduled_save_candles)  # Persist candles for restarts (calls CURRENT stream_service — see wrapper note)
        schedule.every(15).minutes.do(_scheduled_archive_candles)  # Durable history harvest (free, IG-native backtest source incl. AI Index)
        # Screener at each major session open — full briefings (zero API cost)
        schedule.every().day.at("23:00").do(run_daily_screen)  # Asia/forex open
        schedule.every().day.at("03:00").do(run_daily_screen)  # Pre-London
        schedule.every().day.at("07:00").do(run_daily_screen)  # London open
        schedule.every().day.at("11:00").do(run_daily_screen)  # Pre-US
        schedule.every().day.at("15:00").do(run_daily_screen)  # US open
        schedule.every().day.at("19:00").do(run_daily_screen)  # Late US/evening
        # Frequent re-screen to catch intra-cycle surgers crossing the top-N cutoff
        # between the 4-hourly briefings (2026-06-12). Silent unless the active set
        # changes — streaming-only, no API cost, no Telegram/log spam on no-change.
        schedule.every(30).minutes.do(run_daily_screen, periodic=True)
        schedule.every().day.at("21:00").do(send_daily_summary)
        schedule.every().day.at("22:00").do(_scheduled_prune_archive)  # Bound the durable archive (HDD-safe retention, default 365d)

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
