"""
Telegram Bot Integration for IG Trading Bot
Provides remote control and monitoring via Telegram
"""

import asyncio
import html
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from config import TelegramConfig, MARKETS
from src.daily_trend import VALID_DAILY_TREND_MODES, has_daily_trend_config
from src.pullback import VALID_PULLBACK_MODES, has_pullback_config

if TYPE_CHECKING:
    from src.client import IGClient
    from src.journal import TradeJournal

logger = logging.getLogger(__name__)

STATS_DIR = Path("/app/data") if os.path.exists("/app") else Path("data")
STATS_FILE = STATS_DIR / "daily_stats.json"
# Per-market strategy modes, toggled via /mode (2026-07-24; forex pairs included
# since 2026-09-09). {epic: mode}; a market absent from the dict uses its config
# default (MarketConfig.default_mode, else shadow_only-derived, else momentum) — so
# this file only ever holds deliberate user overrides. Read by main._market_mode.
MARKET_MODES_FILE = STATS_DIR / "market_modes.json"
MARKET_MODES = ("off", "momentum", "shadow", "breakout", "breakout-shadow")

# DAILY-TREND strategy modes per market (2026-09-09), toggled via /daily. Separate
# from /mode because the daily strategy runs ALONGSIDE the intraday one on the same
# epic (Gold: 1h breakout + daily trend). {epic: off|shadow|live}; an absent epic uses
# MarketConfig.daily_trend (None -> off). Read cross-thread by main._daily_trend_mode;
# the valid tuple lives in src.daily_trend so the two modules cannot diverge.
DAILY_TREND_MODES_FILE = STATS_DIR / "daily_trend_modes.json"
# Same shape for the PULLBACK strategy (2026-09-09, second sweep), toggled via /pullback.
PULLBACK_MODES_FILE = STATS_DIR / "pullback_modes.json"

# RETIRED 2026-09-09. Until then the forex pairs were governed by ONE global toggle
# (/forex, persisted here) with its own four-mode vocabulary, in which "shadow"
# meant BREAKOUT observed — a second control surface that made the /mode board
# misleading and, on 2026-09-09, let a stray `/forex momentum` put both pairs on
# the retired momentum pipeline for an hour. Folded into /mode. The file is
# translated into per-pair /mode overrides ONCE at boot (migrate_legacy_forex_mode)
# and renamed *.migrated so it can never re-apply.
LEGACY_FOREX_MODE_FILE = STATS_DIR / "forex_mode.json"
# Legacy /forex mode → /mode vocabulary. "breakout" is deliberately absent: it
# meant "as live as this pair's per-pair veto allows", which is exactly the pair's
# config default_mode now — so it translates to "no override".
LEGACY_FOREX_MODE_MAP = {"off": "off", "momentum": "momentum", "shadow": "breakout-shadow"}

# The daily stats reset at the 21:00 UTC trading-session boundary
# (send_daily_summary -> reset_daily_stats, scheduled at 21:00 UTC). Persistence
# must use the SAME boundary, not calendar midnight — otherwise a restart in the
# 00:00-21:00 window wrongly discards a session that began before midnight.
# This hour is UTC: the container runs TZ=Europe/London, so both this boundary
# and the scheduler (main.py: .at("21:00", "UTC")) must pin UTC explicitly or the
# session silently rolls at 20:00 UTC in summer (2026-07-13).
SESSION_RESET_HOUR = 21


def _to_utc(dt: datetime) -> datetime:
    """Normalise a datetime to NAIVE UTC.

    Naive timestamps across this codebase (journal entry_time/exit_time, and
    datetime.now() generally) are container-LOCAL, and the container runs
    TZ=Europe/London — so they are BST in summer. Session bucketing must happen
    in UTC, or the 21:00 boundary drifts by an hour for half the year. A naive
    input is interpreted as local (astimezone's documented behaviour) and
    converted; an aware input is just converted.
    """
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _session_date(now: Optional[datetime] = None) -> str:
    """Date label of the trading session current at `now`.

    A session runs from one 21:00 UTC boundary to the next, so before 21:00 UTC
    the live session is the one that began the previous calendar day. Used as the
    stats file's identity so persistence and the 21:00 reset agree.

    `now` may be a naive LOCAL timestamp (callers pass journal exit_time/
    entry_time straight from SQLite); it is normalised to UTC first so both the
    default path and the explicit-timestamp path bucket on the same boundary.
    """
    now = _to_utc(now) if now is not None else datetime.now(timezone.utc).replace(tzinfo=None)
    if now.hour < SESSION_RESET_HOUR:
        now = now - timedelta(days=1)
    return now.strftime("%Y-%m-%d")


def format_pnl(value: float) -> str:
    """Format P&L with sign and emoji."""
    if value >= 0:
        return f"+£{value:.2f} ✅"
    return f"-£{abs(value):.2f} 🔻"


def format_duration(seconds: int) -> str:
    """Format duration in human-readable form."""
    if seconds < 3600:
        return f"{seconds // 60}m"
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    return f"{hours}h {mins}m"


class TelegramBot:
    """
    Telegram bot for remote IG trading bot control and monitoring.

    Features:
    - Real-time notifications for trades
    - Position tracking
    - P&L reporting
    - Bot control (start/stop/emergency)
    - Status updates
    """

    def __init__(self, config: TelegramConfig, authorized_users: Optional[List[int]] = None):
        """
        Initialize Telegram bot.

        Args:
            config: Telegram configuration
            authorized_users: List of authorized user IDs
        """
        self.config = config
        self.authorized_users = set(authorized_users or [int(config.chat_id)])
        self.app: Optional[Application] = None
        self.ig_client: Optional['IGClient'] = None
        self.trading_enabled = True
        self.is_running = False
        # Per-market /mode overrides ({epic: mode}), forex pairs included.
        # Read cross-thread by main._market_mode; absent epic = config default.
        self.market_modes: dict = {}
        self.load_market_modes()
        # Per-market /daily overrides ({epic: off|shadow|live}) for the daily-trend
        # strategy. Read cross-thread by main._daily_trend_mode.
        self.daily_trend_modes: dict = {}
        self.load_daily_trend_modes()
        self.pullback_modes: dict = {}
        self.load_pullback_modes()
        # Human-readable record of the one-shot /forex → /mode translation when the
        # legacy file was found at this boot (surfaced on the startup banner and the
        # /mode board). None = nothing to migrate.
        self.mode_migration_notice: Optional[str] = self.migrate_legacy_forex_mode()

        # Statistics
        self.start_time = datetime.now()
        self.notifications_sent = 0
        self.commands_executed = 0
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.risk_manager = None  # Set via set_risk_manager() for stats persistence
        self.journal: Optional['TradeJournal'] = None  # Set via set_journal()
        self.screener = None  # Set via set_screener()

        logger.info(f"Telegram bot initialized with {len(self.authorized_users)} authorized users")

    def set_ig_client(self, client: 'IGClient') -> None:
        """Set reference to IG client."""
        self.ig_client = client

    def set_risk_manager(self, risk_manager) -> None:
        """Set reference to risk manager for stats persistence."""
        self.risk_manager = risk_manager

    def set_journal(self, journal: 'TradeJournal') -> None:
        """Set reference to trade journal for /journal command."""
        self.journal = journal

    def set_screener(self, screener) -> None:
        """Set reference to market screener for /screener command."""
        self.screener = screener

    def save_daily_stats(self) -> None:
        """Persist daily stats to disk so they survive restarts."""
        try:
            STATS_DIR.mkdir(parents=True, exist_ok=True)
            stats = {
                "date": _session_date(),
                "trades_today": self.trades_today,
                "daily_pnl": self.daily_pnl,
                "risk_daily_pnl": self.risk_manager.daily_pnl if self.risk_manager else 0.0,
                "saved_at": datetime.now().isoformat(),
            }
            STATS_FILE.write_text(json.dumps(stats))
        except Exception as e:
            logger.warning(f"Failed to save daily stats: {e}")

    def load_daily_stats(self) -> None:
        """Restore daily stats from disk if they are from today."""
        try:
            if not STATS_FILE.exists():
                return
            stats = json.loads(STATS_FILE.read_text())
            if stats.get("date") != _session_date():
                logger.info("Daily stats file is from a previous session, starting fresh")
                return
            self.trades_today = stats.get("trades_today", 0)
            self.daily_pnl = stats.get("daily_pnl", 0.0)
            if self.risk_manager:
                self.risk_manager.daily_pnl = stats.get("risk_daily_pnl", 0.0)
            logger.info(
                f"Restored daily stats: {self.trades_today} trades, "
                f"P&L £{self.daily_pnl:.2f}"
            )
        except Exception as e:
            logger.warning(f"Failed to load daily stats: {e}")

    def save_market_modes(self) -> None:
        """Persist per-market /mode overrides across restarts."""
        try:
            STATS_DIR.mkdir(parents=True, exist_ok=True)
            MARKET_MODES_FILE.write_text(json.dumps({
                "market_modes": self.market_modes,
                "saved_at": datetime.now().isoformat(),
            }))
        except Exception as e:
            logger.warning(f"Failed to save market modes: {e}")

    def load_market_modes(self) -> None:
        """Restore /mode overrides; silently drop anything malformed (config default wins)."""
        try:
            if not MARKET_MODES_FILE.exists():
                return
            raw = json.loads(MARKET_MODES_FILE.read_text()).get("market_modes", {})
            self.market_modes = {
                e: m for e, m in raw.items() if isinstance(m, str) and m in MARKET_MODES
            }
            if self.market_modes:
                logger.info(f"Restored market modes: {self.market_modes}")
        except Exception as e:
            logger.warning(f"Failed to load market modes (using config defaults): {e}")
            self.market_modes = {}

    def save_daily_trend_modes(self) -> None:
        """Persist /daily overrides across restarts."""
        try:
            STATS_DIR.mkdir(parents=True, exist_ok=True)
            DAILY_TREND_MODES_FILE.write_text(json.dumps({
                "daily_trend_modes": self.daily_trend_modes,
                "saved_at": datetime.now().isoformat(),
            }))
        except Exception as e:
            logger.warning(f"Failed to save daily-trend modes: {e}")

    def load_daily_trend_modes(self) -> None:
        """Restore /daily overrides; silently drop anything malformed (config default wins)."""
        try:
            if not DAILY_TREND_MODES_FILE.exists():
                return
            raw = json.loads(DAILY_TREND_MODES_FILE.read_text()).get("daily_trend_modes", {})
            self.daily_trend_modes = {
                e: m for e, m in raw.items() if isinstance(m, str) and m in VALID_DAILY_TREND_MODES
            }
            if self.daily_trend_modes:
                logger.info(f"Restored daily-trend modes: {self.daily_trend_modes}")
        except Exception as e:
            logger.warning(f"Failed to load daily-trend modes (using config defaults): {e}")
            self.daily_trend_modes = {}

    def save_pullback_modes(self) -> None:
        try:
            STATS_DIR.mkdir(parents=True, exist_ok=True)
            PULLBACK_MODES_FILE.write_text(json.dumps({
                "pullback_modes": self.pullback_modes, "saved_at": datetime.now().isoformat()}))
        except Exception as e:
            logger.warning(f"Failed to save pullback modes: {e}")

    def load_pullback_modes(self) -> None:
        try:
            if not PULLBACK_MODES_FILE.exists():
                return
            raw = json.loads(PULLBACK_MODES_FILE.read_text()).get("pullback_modes", {})
            self.pullback_modes = {e: m for e, m in raw.items() if isinstance(m, str) and m in VALID_PULLBACK_MODES}
            if self.pullback_modes:
                logger.info(f"Restored pullback modes: {self.pullback_modes}")
        except Exception as e:
            logger.warning(f"Failed to load pullback modes (using config defaults): {e}")
            self.pullback_modes = {}

    def _effective_pullback_mode(self, m) -> str:
        """Effective PULLBACK mode. MUST mirror main._pullback_mode: /pullback override >
        MarketConfig.pullback > off."""
        override = self.pullback_modes.get(m.epic)
        if override in VALID_PULLBACK_MODES:
            return override
        cfg = getattr(m, "pullback", None)
        return cfg if cfg in VALID_PULLBACK_MODES else "off"

    def _effective_daily_mode(self, m) -> str:
        """Effective DAILY-TREND mode for a MarketConfig.
        MUST mirror main._daily_trend_mode: /daily override > MarketConfig.daily_trend > off."""
        override = self.daily_trend_modes.get(m.epic)
        if override in VALID_DAILY_TREND_MODES:
            return override
        cfg = getattr(m, "daily_trend", None)
        return cfg if cfg in VALID_DAILY_TREND_MODES else "off"

    def migrate_legacy_forex_mode(self) -> Optional[str]:
        """One-shot translation of the retired global /forex toggle into per-pair
        /mode overrides. Returns a notice string when the legacy file was found
        (whether or not anything had to be written), else None.

        Legacy mode → per-pair /mode:
          off      → off                momentum → momentum
          shadow   → breakout-shadow    (legacy "shadow" meant BREAKOUT observed)
          breakout → the pair's config default. The global 'breakout' went live only
                     on pairs NOT vetoed by breakout_shadow_only, and that veto is
                     now expressed as default_mode="breakout-shadow" — so the
                     translation is whatever the config already says.
          anything else (corrupt file) → off, as the legacy loader did.
        A pair that already has a /mode override is left alone. An override is
        written only where the translation differs from the pair's config default,
        so deploying the fold never by itself changes what trades. The legacy file
        is then renamed *.migrated, so this cannot re-run; if the rename fails the
        re-run is a no-op because the overrides are now present."""
        try:
            if not LEGACY_FOREX_MODE_FILE.exists():
                return None
            try:
                legacy = json.loads(LEGACY_FOREX_MODE_FILE.read_text()).get("forex_mode", "off")
            except Exception as e:
                logger.warning(f"Legacy forex_mode.json unreadable ({e}); treating as 'off'")
                legacy = "off"
            written: dict = {}
            untouched: list = []
            for m in MARKETS:
                if m.sector != "Forex":
                    continue
                if m.epic in self.market_modes:
                    untouched.append(f"{m.name} (existing override `{self.market_modes[m.epic]}`)")
                    continue
                default = self._effective_mode(m)  # no override present → config default
                target = default if legacy == "breakout" else LEGACY_FOREX_MODE_MAP.get(legacy, "off")
                if target != default:
                    self.market_modes[m.epic] = target
                    written[m.epic] = target
                else:
                    untouched.append(f"{m.name} (config default `{default}`)")
            if written:
                self.save_market_modes()
            LEGACY_FOREX_MODE_FILE.replace(LEGACY_FOREX_MODE_FILE.with_suffix(".json.migrated"))
            names = {m.epic: m.name for m in MARKETS}
            parts = [f"{names.get(e, e)} → `{mode}`" for e, mode in written.items()]
            notice = (f"/forex `{legacy}` folded into /mode — "
                      + (f"override written: {', '.join(parts)}" if parts else "nothing written")
                      + (f"; unchanged: {', '.join(untouched)}" if untouched else ""))
            logger.info(f"Legacy forex mode migrated: {notice}")
            return notice
        except Exception as e:
            logger.warning(f"Legacy forex mode migration failed (left as-is): {e}")
            return None

    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        return user_id in self.authorized_users

    # ==================== COMMAND HANDLERS ====================

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id

        if not self.is_authorized(user_id):
            await update.effective_message.reply_text(
                "⛔ Unauthorized access. Your user ID has been logged."
            )
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            return

        welcome_message = (
            "🤖 *IG Spread Betting Bot Control Panel*\n\n"
            "Welcome! You can now control and monitor your trading bot.\n\n"
            "*Key Commands:*\n"
            "/status - Bot status and summary\n"
            "/balance - Account balance\n"
            "/positions - View open positions\n"
            "/markets - Market prices and status\n"
            "/help - Show all commands\n\n"
            "📊 Real-time trade notifications enabled!"
        )

        await update.effective_message.reply_text(welcome_message, parse_mode='Markdown')
        self.commands_executed += 1

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self.is_authorized(update.effective_user.id):
            return

        help_text = (
            "🤖 *IG TRADING BOT COMMANDS*\n\n"
            "*📊 Monitoring:*\n"
            "/status - Bot status and summary\n"
            "/balance - Account balance and P&L\n"
            "/positions - View open positions\n"
            "/markets - Market prices and status\n"
            "/health - Quick health check\n\n"
            "*💰 Performance:*\n"
            "/pnl - Today's P&L summary\n"
            "/journal - Trade journal stats\n\n"
            "*🎮 Control:*\n"
            "/stop - Pause trading\n"
            "/resume - Resume trading\n"
            "/mode - Per-market strategy, forex included (off|momentum|shadow|breakout|breakout-shadow)\n"
            "/daily - Daily trend-following per market (off|shadow|live)\n"
            "/pullback - Pullback-in-uptrend per market (off|shadow|live)\n"
            "/rebuild - Pull latest code & restart\n"
            "/emergency - ⚠️ Close ALL positions\n\n"
            "*🔔 Notifications:*\n"
            "Automatic alerts for trades and errors.\n\n"
            f"Your ID: `{update.effective_user.id}`"
        )

        await update.effective_message.reply_text(help_text, parse_mode='Markdown')
        self.commands_executed += 1

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.effective_message.reply_text("⚠️ IG client not connected")
                return

            # Get account info
            balance = self.ig_client.get_balance() or 0
            positions = self.ig_client.get_positions()

            # Calculate unrealized P&L
            unrealized_pnl = sum(p.profit_loss for p in positions)

            status_emoji = "✅" if self.trading_enabled else "⏸️"
            status_text = "RUNNING" if self.trading_enabled else "PAUSED"

            runtime = datetime.now() - self.start_time
            hours = int(runtime.total_seconds() // 3600)
            minutes = int((runtime.total_seconds() % 3600) // 60)

            message = (
                f"{status_emoji} *BOT STATUS: {status_text}*\n\n"
                f"*Account Summary:*\n"
                f"💰 Balance: £{balance:,.2f}\n"
                f"📈 Unrealized P&L: {format_pnl(unrealized_pnl)}\n"
                f"📍 Open Positions: {len(positions)}\n"
                f"⏱️ Uptime: {hours}h {minutes}m\n\n"
                f"*Today's Activity:*\n"
                f"📊 Trades: {self.trades_today}\n"
                f"💵 Daily P&L: {format_pnl(self.daily_pnl)}\n\n"
                f"*Markets:* {len(MARKETS)} configured"
            )

            await update.effective_message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.effective_message.reply_text(f"❌ Error getting status: {str(e)}")

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.effective_message.reply_text("⚠️ IG client not connected")
                return

            account_info = self.ig_client.get_account_info()
            balance = self.ig_client.get_balance() or 0
            positions = self.ig_client.get_positions()

            unrealized_pnl = sum(p.profit_loss for p in positions)
            total_exposure = sum(p.size * p.open_level for p in positions)

            message = (
                f"💰 *ACCOUNT BALANCE*\n\n"
                f"*Balance:* £{balance:,.2f}\n"
                f"*Unrealized P&L:* {format_pnl(unrealized_pnl)}\n"
                f"*Exposure:* £{total_exposure:,.2f}\n"
                f"*Open Positions:* {len(positions)}\n\n"
                f"*Account ID:* {self.ig_client.account_id}"
            )

            await update.effective_message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in balance command: {e}")
            await update.effective_message.reply_text(f"❌ Error: {str(e)}")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.effective_message.reply_text("⚠️ IG client not connected")
                return

            positions = self.ig_client.get_positions()

            if not positions:
                await update.effective_message.reply_text("📭 No open positions")
                return

            message = "📊 *OPEN POSITIONS*\n\n"

            for i, pos in enumerate(positions, 1):
                pnl_emoji = "🟢" if pos.profit_loss >= 0 else "🔴"
                direction_emoji = "📈" if pos.direction == "BUY" else "📉"

                # Find market name
                market_name = pos.epic
                for m in MARKETS:
                    if m.epic == pos.epic:
                        market_name = m.name
                        break

                message += (
                    f"*{i}. {market_name}*\n"
                    f"{direction_emoji} {pos.direction} @ £{pos.open_level:,.2f}\n"
                    f"Size: £{pos.size}/pt\n"
                    f"{pnl_emoji} P&L: {format_pnl(pos.profit_loss)}\n"
                )

                if pos.stop_level:
                    message += f"Stop: £{pos.stop_level:,.2f}\n"
                if pos.limit_level:
                    message += f"Limit: £{pos.limit_level:,.2f}\n"

                message += "\n"

            # Total
            total_pnl = sum(p.profit_loss for p in positions)
            message += f"*Total Unrealized P&L:* {format_pnl(total_pnl)}"

            await update.effective_message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            await update.effective_message.reply_text(f"❌ Error: {str(e)}")

    async def markets_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /markets command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.effective_message.reply_text("⚠️ IG client not connected")
                return

            message = "📈 <b>MARKET STATUS</b>\n\n"

            for market in MARKETS:
                info = self.ig_client.get_market_info(market.epic)
                if info:
                    status_emoji = "🟢" if info.market_status == "TRADEABLE" else "🔴"
                    mid_price = (info.bid + info.offer) / 2
                    spread = info.offer - info.bid

                    message += (
                        f"<b>{market.name}</b>\n"
                        f"{status_emoji} {info.market_status}\n"
                        f"Price: {mid_price:,.2f} (spread: {spread:.2f})\n\n"
                    )
                else:
                    message += f"<b>{market.name}</b>\n⚠️ Unable to fetch\n\n"

            await update.effective_message.reply_text(message, parse_mode='HTML')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in markets command: {e}")
            await update.effective_message.reply_text(f"❌ Error: {str(e)}")

    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.effective_message.reply_text("⚠️ IG client not connected")
                return

            positions = self.ig_client.get_positions()
            unrealized_pnl = sum(p.profit_loss for p in positions)

            message = (
                f"📊 *TODAY'S PERFORMANCE*\n"
                f"({datetime.now().strftime('%d %b %Y')})\n\n"
                f"💰 Realized P&L: {format_pnl(self.daily_pnl)}\n"
                f"📈 Unrealized P&L: {format_pnl(unrealized_pnl)}\n"
                f"{'─' * 20}\n"
                f"📊 Net P&L: {format_pnl(self.daily_pnl + unrealized_pnl)}\n\n"
                f"📊 Trades: {self.trades_today}\n"
                f"📍 Open Positions: {len(positions)}"
            )

            await update.effective_message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in pnl command: {e}")
            await update.effective_message.reply_text(f"❌ Error: {str(e)}")

    async def journal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /journal command — trade journal stats."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.journal:
                await update.effective_message.reply_text("⚠️ Trade journal not available")
                return

            days = 30
            overall = self.journal.get_overall_stats(days)
            by_market = self.journal.get_stats_by_market(days)
            by_exit = self.journal.get_stats_by_exit_reason(days)
            rejected = self.journal.get_rejected_count_by_market(days=7)
            rejected_breakdown = self.journal.get_rejected_reasons_by_market(days=7)

            total = overall.get("total", 0) or 0
            if total == 0:
                await update.effective_message.reply_text(
                    "📓 <b>TRADE JOURNAL</b>\n\nNo closed trades recorded yet.",
                    parse_mode="HTML",
                )
                self.commands_executed += 1
                return

            wins = overall.get("wins", 0) or 0
            losses = overall.get("losses", 0) or 0
            win_rate = (wins / total * 100) if total > 0 else 0

            msg = f"📓 <b>TRADE JOURNAL</b> ({days}d)\n\n"
            msg += f"<b>Overall:</b>\n"
            msg += f"  Trades: {total} ({wins}W / {losses}L)\n"
            msg += f"  Win rate: {win_rate:.0f}%\n"
            msg += f"  Total P&L: £{overall.get('total_pnl', 0):.2f}\n"
            msg += f"  Avg win: £{overall.get('avg_win', 0):.2f}\n"
            msg += f"  Avg loss: £{overall.get('avg_loss', 0):.2f}\n"
            msg += f"  Avg duration: {overall.get('avg_duration', 0):.0f} mins\n"
            msg += f"  Avg ADX at entry: {overall.get('avg_adx_entry', 0):.1f}\n\n"

            if by_market:
                msg += f"<b>By Market:</b>\n"
                for m in by_market:
                    w = m.get("wins", 0) or 0
                    t = m.get("total", 0) or 0
                    wr = (w / t * 100) if t > 0 else 0
                    pnl = m.get("total_pnl", 0) or 0
                    emoji = "📈" if pnl >= 0 else "📉"
                    name = html.escape(str(m['market_name']))
                    msg += f"  {emoji} {name}: {t} trades, {wr:.0f}% win, £{pnl:.2f}\n"
                msg += "\n"

            if by_exit:
                msg += f"<b>By Exit Type:</b>\n"
                for e in by_exit:
                    pnl = e.get("total_pnl", 0) or 0
                    reason = html.escape(str(e['exit_reason']))
                    msg += f"  {reason}: {e['total']} trades, £{pnl:.2f}\n"
                msg += "\n"

            if rejected:
                msg += f"<b>Rejected Signals (7d):</b>\n"
                for r in rejected[:8]:
                    name_raw = str(r['market_name'])
                    name = html.escape(name_raw)
                    msg += (
                        f"  {name}: {r['rejected']}x "
                        f"(avg ADX {r.get('avg_adx', 0):.1f})\n"
                    )
                    reasons = rejected_breakdown.get(name_raw, [])
                    if reasons:
                        top = [f"{html.escape(cat)}: {n}" for cat, n in reasons[:3]]
                        msg += f"    └ {', '.join(top)}\n"

            await update.effective_message.reply_text(msg, parse_mode="HTML")
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in journal command: {e}")
            await update.effective_message.reply_text(f"❌ Error: {str(e)}")

    async def screener_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /screener command — show market scores."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.screener:
                await update.effective_message.reply_text("⚠️ Screener not available")
                return

            text = self.screener.get_scores_text()
            if not text or "No scores" in text:
                await update.effective_message.reply_text("⚠️ No scores yet. Screener runs daily at 04:00 UTC.")
                return

            await update.effective_message.reply_text(text, parse_mode="HTML")
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in screener command: {e}")
            await update.effective_message.reply_text(f"❌ Error: {str(e)}")

    async def health_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /health command - quick health check."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            message = "🏥 *BOT HEALTH CHECK*\n\n"

            # Bot Status
            if self.trading_enabled:
                runtime = datetime.now() - self.start_time
                hours = int(runtime.total_seconds() // 3600)
                mins = int((runtime.total_seconds() % 3600) // 60)
                message += f"✅ Bot Status: Running ({hours}h {mins}m)\n"
            else:
                message += "⏸️ Bot Status: Paused\n"

            # IG Connection
            if self.ig_client and self.ig_client.is_logged_in:
                message += "✅ IG Connection: OK\n"
            else:
                message += "❌ IG Connection: Disconnected\n"

            # Balance
            if self.ig_client and self.ig_client.is_logged_in:
                balance = self.ig_client.get_balance()
                if balance and balance > 0:
                    message += f"✅ Balance: £{balance:,.2f}\n"
                else:
                    message += "⚠️ Balance: Unable to fetch\n"

            # Positions
            if self.ig_client and self.ig_client.is_logged_in:
                positions = self.ig_client.get_positions()
                if len(positions) == 0:
                    message += "✅ Positions: None (watching)\n"
                else:
                    total_pnl = sum(p.profit_loss for p in positions)
                    pnl_status = "✅" if total_pnl >= 0 else "⚠️"
                    message += f"{pnl_status} Positions: {len(positions)} ({format_pnl(total_pnl)})\n"

            # Today's P&L
            if self.daily_pnl >= 0:
                message += f"✅ Daily P&L: {format_pnl(self.daily_pnl)}\n"
            else:
                message += f"⚠️ Daily P&L: {format_pnl(self.daily_pnl)}\n"

            message += "\n👍 *Everything looks good!*" if self.ig_client and self.ig_client.is_logged_in else "\n⚠️ *Issues detected*"

            await update.effective_message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in health command: {e}")
            await update.effective_message.reply_text(f"❌ Error: {str(e)}")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - pause trading."""
        if not self.is_authorized(update.effective_user.id):
            return

        keyboard = [
            [
                InlineKeyboardButton("✅ Yes, Pause", callback_data='stop_confirm'),
                InlineKeyboardButton("❌ Cancel", callback_data='stop_cancel')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            "⚠️ *PAUSE TRADING?*\n\n"
            "This will:\n"
            "• Stop opening new positions\n"
            "• Keep existing positions open\n"
            "• Continue monitoring\n\n"
            "Confirm?",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        if not self.is_authorized(update.effective_user.id):
            return

        if not self.trading_enabled:
            self.trading_enabled = True
            await update.effective_message.reply_text("✅ *Bot resumed!* Trading will continue.")
            await self.send_notification("🟢 *Bot Resumed*\nTrading operations continuing.")
        else:
            await update.effective_message.reply_text("ℹ️ Bot is already running")

    async def forex_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /forex — RETIRED 2026-09-09. The global forex toggle was folded into
        the per-market /mode system. Kept for one release as a pointer so muscle
        memory gets a useful reply instead of silence. Changes NOTHING."""
        if not self.is_authorized(update.effective_user.id):
            return
        self.commands_executed += 1
        pairs = [m for m in MARKETS if m.sector == "Forex"]
        board = "\n".join(f"• {m.name}: `{self._effective_mode(m)}`" for m in pairs)
        await update.effective_message.reply_text(
            "ℹ️ `/forex` is retired — forex pairs live on the `/mode` board now, one "
            "mode per pair. Nothing changed.\n\n"
            f"{board}\n\n"
            "Use `/mode <pair> off|momentum|shadow|breakout|breakout-shadow`, e.g. "
            "`/mode gbp/usd breakout-shadow`.\n"
            "NB the old forex `shadow` (breakout observed) is now `breakout-shadow`; "
            "`shadow` means momentum observed, as on every other market.",
            parse_mode='Markdown')

    def _effective_mode(self, m) -> str:
        """Effective strategy mode for ANY MarketConfig (forex pairs included since 2026-09-09).
        MUST mirror main._market_mode: /mode override > default_mode > shadow_only > momentum."""
        override = self.market_modes.get(m.epic)
        if override in MARKET_MODES:
            return override
        if getattr(m, "default_mode", None) in MARKET_MODES:
            return m.default_mode
        return "shadow" if getattr(m, "shadow_only", False) else "momentum"

    async def mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mode — per-market strategy toggle for EVERY market (forex pairs
        included since 2026-09-09; the separate global /forex toggle is retired).

        /mode                     -> board of every market's effective mode
        /mode <market> <mode>     -> set override (off|momentum|shadow|breakout|breakout-shadow)
        /mode <market> default    -> clear override (config default resumes)
        """
        if not self.is_authorized(update.effective_user.id):
            return
        self.commands_executed += 1
        from config import MARKETS
        from src.breakout import has_breakout_config

        args = [a.lower() for a in (context.args or [])]
        if not args:
            emoji = {"off": "⚪", "momentum": "🔵", "shadow": "👻",
                     "breakout": "🟢", "breakout-shadow": "🟡"}
            lines = ["🎛 *Market strategy modes*\n"]
            for m in MARKETS:
                eff = self._effective_mode(m)
                tag = " _(override)_" if m.epic in self.market_modes else ""
                lines.append(f"{emoji.get(eff, '·')} {m.name}: `{eff}`{tag}")
                dmode = self._effective_daily_mode(m)
                if dmode != "off" or has_daily_trend_config(m.epic):
                    dtag = " _(override)_" if m.epic in self.daily_trend_modes else ""
                    demoji = {"live": "📅🟢", "shadow": "📅🟡", "off": "📅⚪"}[dmode]
                    lines.append(f"   {demoji} daily-trend: `{dmode}`{dtag}")
                pmode = self._effective_pullback_mode(m)
                if pmode != "off" or has_pullback_config(m.epic):
                    ptag = " _(override)_" if m.epic in self.pullback_modes else ""
                    pemoji = {"live": "📉🟢", "shadow": "📉🟡", "off": "📉⚪"}[pmode]
                    lines.append(f"   {pemoji} pullback: `{pmode}`{ptag}")
            lines.append("\nUsage: `/mode <market> off|momentum|shadow|breakout|breakout-shadow`"
                         "\n`/mode <market> default` clears the override."
                         "\n👻 shadow = momentum observed; 🟡 breakout-shadow = breakout observed.")
            if self.mode_migration_notice:
                lines.append(f"\n🔁 At this boot: {self.mode_migration_notice}")
            await update.effective_message.reply_text("\n".join(lines), parse_mode='Markdown')
            return

        if len(args) < 2:
            await update.effective_message.reply_text(
                "Usage: `/mode <market> <off|momentum|shadow|breakout|breakout-shadow|default>`",
                parse_mode='Markdown')
            return
        # Mode is the LAST token; everything before it is the market name (handles
        # "/mode hong kong shadow"). Fuzzy: case-insensitive substring on the name.
        mode_arg = args[-1]
        query = " ".join(args[:-1])
        matches = [m for m in MARKETS if query in m.name.lower()]
        if not matches:
            await update.effective_message.reply_text(f"❌ No market matches `{query}`", parse_mode='Markdown')
            return
        if len(matches) > 1:
            await update.effective_message.reply_text(
                "❌ Ambiguous: " + ", ".join(m.name for m in matches), parse_mode='Markdown')
            return
        m = matches[0]

        if mode_arg == "default":
            prev = self.market_modes.pop(m.epic, None)
            self.save_market_modes()
            await update.effective_message.reply_text(
                f"↩️ {m.name}: override cleared (was `{prev}`) — config default "
                f"`{self._effective_mode(m)}` resumes.", parse_mode='Markdown')
            return
        if mode_arg not in MARKET_MODES:
            await update.effective_message.reply_text(
                f"❌ Unknown mode `{mode_arg}`. Use: off | momentum | shadow | "
                f"breakout | breakout-shadow | default", parse_mode='Markdown')
            return
        if mode_arg in ("breakout", "breakout-shadow") and not has_breakout_config(m.epic):
            await update.effective_message.reply_text(
                f"❌ {m.name} has no breakout config (src/breakout.py BREAKOUT_CONFIGS) — "
                f"backtest and add one first.", parse_mode='Markdown')
            return

        prev = self._effective_mode(m)
        self.market_modes[m.epic] = mode_arg
        self.save_market_modes()
        logger.info(f"Market mode changed via Telegram: {m.name} {prev} -> {mode_arg}")
        warn = ""
        if mode_arg == "breakout":
            warn = "\n⚠️ Breakout trading *LIVE* — real orders on the next channel break."
            if m.epic == "CC.D.CL.USS.IP":
                warn += ("\n📊 Reminder: oil breakout is NOT a validated standing edge "
                         "(full-period PF 0.87) — it pays only in trending regimes. "
                         "Flip back to `breakout-shadow` when the trend view expires.")
        elif mode_arg == "momentum" and m.epic == "CC.D.CL.USS.IP":
            warn = "\n⚠️ Crude momentum was disabled for cause (live PF 0.38, costs eat the edge)."
        elif mode_arg == "momentum" and m.sector == "Forex":
            warn = ("\n⚠️ Forex momentum profiles were retired as net-losing (2026-06) — "
                    "this places LIVE momentum orders on the pair.")
        await update.effective_message.reply_text(
            f"🎛 *{m.name} → `{mode_arg}`* (was `{prev}`){warn}", parse_mode='Markdown')

    async def daily_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /daily — per-market toggle for the DAILY-TREND strategy (2026-09-09).

        /daily                       -> board of markets with a daily-trend config
        /daily <market> off|shadow|live -> set override (persisted)
        /daily <market> default      -> clear override (config default resumes)
        Runs alongside /mode's intraday strategy; a Gold daily position and a Gold
        1h-breakout position may coexist (user decision 2026-09-09)."""
        if not self.is_authorized(update.effective_user.id):
            return
        self.commands_executed += 1
        args = [a.lower() for a in (context.args or [])]
        if not args:
            lines = ["📅 *Daily trend-following modes* (long-only Donchian 55/20, 2×ATR20 stop)\n"]
            for m in MARKETS:
                if not has_daily_trend_config(m.epic):
                    continue
                dmode = self._effective_daily_mode(m)
                tag = " _(override)_" if m.epic in self.daily_trend_modes else ""
                emoji = {"live": "🟢", "shadow": "🟡", "off": "⚪"}[dmode]
                lines.append(f"{emoji} {m.name}: `{dmode}`{tag}")
            lines.append("\nUsage: `/daily <market> off|shadow|live` · `/daily <market> default` clears the override."
                         "\nOnly markets with a DAILY_TREND_CONFIGS entry (src/daily_trend.py) are listed.")
            await update.effective_message.reply_text("\n".join(lines), parse_mode='Markdown')
            return
        if len(args) < 2:
            await update.effective_message.reply_text(
                "Usage: `/daily <market> <off|shadow|live|default>`", parse_mode='Markdown')
            return
        mode_arg = args[-1]
        query = " ".join(args[:-1])
        matches = [m for m in MARKETS if query in m.name.lower()]
        if not matches:
            await update.effective_message.reply_text(f"❌ No market matches `{query}`", parse_mode='Markdown')
            return
        if len(matches) > 1:
            await update.effective_message.reply_text(
                "❌ Ambiguous: " + ", ".join(m.name for m in matches), parse_mode='Markdown')
            return
        m = matches[0]
        if mode_arg == "default":
            prev = self.daily_trend_modes.pop(m.epic, None)
            self.save_daily_trend_modes()
            await update.effective_message.reply_text(
                f"↩️ {m.name} daily-trend: override cleared (was `{prev}`) — config default "
                f"`{self._effective_daily_mode(m)}` resumes.", parse_mode='Markdown')
            return
        if mode_arg not in VALID_DAILY_TREND_MODES:
            await update.effective_message.reply_text(
                f"❌ Unknown mode `{mode_arg}`. Use: off | shadow | live | default", parse_mode='Markdown')
            return
        if mode_arg != "off" and not has_daily_trend_config(m.epic):
            await update.effective_message.reply_text(
                f"❌ {m.name} has no daily-trend config (src/daily_trend.py DAILY_TREND_CONFIGS) — "
                f"it must pass the 22-year study first.", parse_mode='Markdown')
            return
        prev = self._effective_daily_mode(m)
        self.daily_trend_modes[m.epic] = mode_arg
        self.save_daily_trend_modes()
        logger.info(f"Daily-trend mode changed via Telegram: {m.name} {prev} -> {mode_arg}")
        warn = ""
        if mode_arg == "live":
            warn = ("\n⚠️ LIVE — a real order on the next daily close above the 55-day high, "
                    "size = IG minimum, stop 2×ATR20 (~£160 on Gold), own cap `DAILY_TREND_MAX_RISK_GBP`.")
        await update.effective_message.reply_text(
            f"📅 *{m.name} daily-trend → `{mode_arg}`* (was `{prev}`){warn}", parse_mode='Markdown')

    async def pullback_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pullback — per-market toggle for the PULLBACK-IN-UPTREND strategy
        (2026-09-09). Same grammar as /daily. Runs alongside /mode's intraday strategy."""
        if not self.is_authorized(update.effective_user.id):
            return
        self.commands_executed += 1
        args = [a.lower() for a in (context.args or [])]
        if not args:
            lines = ["📉 *Pullback-in-uptrend modes* (long-only, close>SMA200, buy <5d low, sell >5d high / 10d, 3×ATR stop)\n"]
            for m in MARKETS:
                if not has_pullback_config(m.epic):
                    continue
                pmode = self._effective_pullback_mode(m)
                tag = " _(override)_" if m.epic in self.pullback_modes else ""
                lines.append(f"{ {'live': '🟢', 'shadow': '🟡', 'off': '⚪'}[pmode] } {m.name}: `{pmode}`{tag}")
            lines.append("\nUsage: `/pullback <market> off|shadow|live` · `/pullback <market> default` clears the override.")
            await update.effective_message.reply_text("\n".join(lines), parse_mode='Markdown')
            return
        if len(args) < 2:
            await update.effective_message.reply_text("Usage: `/pullback <market> <off|shadow|live|default>`", parse_mode='Markdown')
            return
        mode_arg = args[-1]; query = " ".join(args[:-1])
        matches = [m for m in MARKETS if query in m.name.lower()]
        if not matches:
            await update.effective_message.reply_text(f"❌ No market matches `{query}`", parse_mode='Markdown'); return
        if len(matches) > 1:
            await update.effective_message.reply_text("❌ Ambiguous: " + ", ".join(m.name for m in matches), parse_mode='Markdown'); return
        m = matches[0]
        if mode_arg == "default":
            prev = self.pullback_modes.pop(m.epic, None); self.save_pullback_modes()
            await update.effective_message.reply_text(
                f"↩️ {m.name} pullback: override cleared (was `{prev}`) — config default "
                f"`{self._effective_pullback_mode(m)}` resumes.", parse_mode='Markdown'); return
        if mode_arg not in VALID_PULLBACK_MODES:
            await update.effective_message.reply_text(f"❌ Unknown mode `{mode_arg}`. Use: off | shadow | live | default", parse_mode='Markdown'); return
        if mode_arg != "off" and not has_pullback_config(m.epic):
            await update.effective_message.reply_text(
                f"❌ {m.name} has no pullback config (src/pullback.py PULLBACK_CONFIGS) — it must pass the 22-year study first.",
                parse_mode='Markdown'); return
        prev = self._effective_pullback_mode(m)
        self.pullback_modes[m.epic] = mode_arg; self.save_pullback_modes()
        logger.info(f"Pullback mode changed via Telegram: {m.name} {prev} -> {mode_arg}")
        warn = "\n⚠️ LIVE — a real order at the next US cash close that closes below the 5-session low above SMA200; risk unit `PULLBACK_RISK_GBP`." if mode_arg == "live" else ""
        await update.effective_message.reply_text(f"📉 *{m.name} pullback → `{mode_arg}`* (was `{prev}`){warn}", parse_mode='Markdown')

    async def emergency_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /emergency command - close all and stop."""
        if not self.is_authorized(update.effective_user.id):
            return

        keyboard = [
            [
                InlineKeyboardButton("🚨 CONFIRM EMERGENCY STOP", callback_data='emergency_confirm'),
            ],
            [
                InlineKeyboardButton("❌ Cancel", callback_data='emergency_cancel')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            "🚨 *EMERGENCY STOP*\n\n"
            "⚠️ WARNING: This will:\n"
            "• Close ALL open positions immediately\n"
            "• Stop the trading bot\n"
            "• Exit at market prices\n\n"
            "*Use only in emergencies!*\n\n"
            "Are you absolutely sure?",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def rebuild_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /rebuild command - git pull, rebuild, and restart the bot."""
        if not self.is_authorized(update.effective_user.id):
            return

        keyboard = [
            [
                InlineKeyboardButton("🔄 CONFIRM REBUILD", callback_data='rebuild_confirm'),
            ],
            [
                InlineKeyboardButton("❌ Cancel", callback_data='rebuild_cancel')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text(
            "🔄 *REBUILD & RESTART*\n\n"
            "This will:\n"
            "• Pull latest code from GitHub\n"
            "• Rebuild the Docker container\n"
            "• Restart the bot\n\n"
            "⚠️ Bot will be offline for ~60 seconds.\n"
            "Open positions are NOT affected.\n\n"
            "Proceed?",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    # ==================== CALLBACK HANDLERS ====================

    async def stop_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle stop confirmation."""
        query = update.callback_query
        await query.answer()

        if not self.is_authorized(query.from_user.id):
            return

        if query.data == 'stop_confirm':
            self.trading_enabled = False
            positions = []
            if self.ig_client and self.ig_client.is_logged_in:
                positions = self.ig_client.get_positions()

            await query.edit_message_text(
                f"✅ *Bot Paused*\n\n"
                f"• New positions: Disabled\n"
                f"• Open positions: {len(positions)} (still monitored)\n"
                f"• Status: PAUSED\n\n"
                f"Use /resume to restart trading",
                parse_mode='Markdown'
            )
            await self.send_notification("⏸️ *Bot Paused*\nNo new positions will be opened.")

        elif query.data == 'stop_cancel':
            await query.edit_message_text("✅ Cancelled - bot still running")

    async def emergency_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle emergency stop confirmation."""
        query = update.callback_query
        await query.answer()

        if not self.is_authorized(query.from_user.id):
            return

        if query.data == 'emergency_confirm':
            try:
                await query.edit_message_text("🚨 *EMERGENCY STOP ACTIVATED*\n\nClosing all positions...", parse_mode='Markdown')

                closed_count = 0
                if self.ig_client and self.ig_client.is_logged_in:
                    positions = self.ig_client.get_positions()

                    for pos in positions:
                        result = self.ig_client.close_position(
                            pos.deal_id,
                            pos.direction,
                            pos.size
                        )
                        if result:
                            closed_count += 1

                self.trading_enabled = False

                await self.send_notification(
                    f"🚨 *EMERGENCY STOP COMPLETE*\n\n"
                    f"Closed {closed_count} positions\n"
                    f"Bot stopped"
                )

            except Exception as e:
                logger.error(f"Error in emergency stop: {e}")
                await query.edit_message_text(f"❌ Error: {str(e)}")

        elif query.data == 'emergency_cancel':
            await query.edit_message_text("✅ Emergency stop cancelled")

    async def rebuild_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle rebuild confirmation."""
        query = update.callback_query
        await query.answer()

        if not self.is_authorized(query.from_user.id):
            return

        if query.data == 'rebuild_confirm':
            trigger_file = STATS_DIR / "rebuild_trigger"
            try:
                trigger_file.write_text(datetime.now().isoformat())
                await query.edit_message_text(
                    "🔄 *REBUILD TRIGGERED*\n\n"
                    "Pulling latest code and rebuilding...\n"
                    "Bot will restart in ~60 seconds.\n\n"
                    "You'll receive a startup notification when ready.",
                    parse_mode='Markdown'
                )
                logger.info(f"Rebuild triggered by user {query.from_user.id}")
            except Exception as e:
                logger.error(f"Failed to trigger rebuild: {e}")
                await query.edit_message_text(f"❌ Failed to trigger rebuild: {e}")

        elif query.data == 'rebuild_cancel':
            await query.edit_message_text("✅ Rebuild cancelled")

    # ==================== NOTIFICATION METHODS ====================

    async def send_notification(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send notification to all authorized users."""
        if not self.config.enabled:
            logger.debug("Telegram notifications disabled")
            return False

        if not self.app:
            logger.warning("Telegram app not initialized")
            return False

        try:
            for user_id in self.authorized_users:
                try:
                    await self.app.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode=parse_mode
                    )
                    self.notifications_sent += 1
                except Exception as e:
                    logger.error(f"Failed to send notification to {user_id}: {e}")

            return True

        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            return False

    def send_message_sync(self, text: str) -> bool:
        """Synchronous wrapper for sending messages (for use from non-async code)."""
        if not self.config.enabled or not self.app:
            return False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_notification(text))
                return True
            else:
                return loop.run_until_complete(self.send_notification(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.send_notification(text))

    async def notify_startup(self, balance: float, markets: List[str]) -> bool:
        """Send bot startup notification."""
        message = (
            "🤖 *IG Trading Bot Started*\n\n"
            f"💰 Balance: £{balance:,.2f}\n"
            f"📊 Markets: {', '.join(markets)}\n"
            f"Status: Active\n\n"
            "Use /help to see available commands."
        )
        return await self.send_notification(message)

    async def notify_shutdown(self, reason: str = "Manual shutdown") -> bool:
        """Send bot shutdown notification."""
        message = f"🛑 *IG Trading Bot Stopped*\n\nReason: {reason}"
        return await self.send_notification(message)

    async def notify_trade_opened(
        self,
        market_name: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_distance: float,
        limit_distance: float,
    ) -> bool:
        """Notify when trade is opened."""
        emoji = "🟢" if direction == "BUY" else "🔴"
        action = "LONG" if direction == "BUY" else "SHORT"

        message = (
            f"{emoji} *POSITION OPENED*\n\n"
            f"Market: {market_name}\n"
            f"Direction: {action}\n"
            f"Size: £{size}/pt\n"
            f"Entry: {entry_price:.2f}\n"
            f"Stop: {stop_distance:.2f} pts\n"
            f"Target: {limit_distance:.2f} pts"
        )
        self.trades_today += 1
        self.save_daily_stats()
        return await self.send_notification(message)

    async def notify_trade_closed(
        self,
        market_name: str,
        direction: str,
        pnl: float,
        reason: str,
        provisional: bool = False,
    ) -> bool:
        """Notify when trade is closed.

        If provisional=True, the P&L is from the cached stream price and may
        differ from IG's actual fill. notify_trade_reconciled() will follow up
        once the IG transaction ledger settles (usually within minutes).
        """
        emoji = "✅" if pnl >= 0 else "❌"
        title = "POSITION CLOSED (provisional)" if provisional else "POSITION CLOSED"
        pnl_label = "P&L (provisional)" if provisional else "P&L"

        message = (
            f"{emoji} *{title}*\n\n"
            f"Market: {market_name}\n"
            f"{pnl_label}: {format_pnl(pnl)}\n"
            f"Reason: {reason}"
        )
        if provisional:
            message += "\n\n_Awaiting IG settlement; will confirm._"
        self.daily_pnl += pnl
        self.save_daily_stats()
        return await self.send_notification(message)

    async def notify_trade_reconciled(
        self,
        market_name: str,
        provisional_pnl: float,
        actual_pnl: float,
        adjust_counter: bool = True,
    ) -> bool:
        """Confirm a previously-provisional close with the broker-confirmed P&L.

        adjust_counter=False when the trade closed in a prior (already-summarised,
        already-reset) session — the journal/history is still corrected by the
        caller, but the live daily counter must NOT absorb the delta or a phantom
        P&L leaks into the new session (shown with 0 trades).
        """
        delta = actual_pnl - provisional_pnl
        emoji = "✅" if actual_pnl >= 0 else "❌"
        message = (
            f"{emoji} *P&L CONFIRMED*\n\n"
            f"Market: {market_name}\n"
            f"Provisional: {format_pnl(provisional_pnl)}\n"
            f"Actual: {format_pnl(actual_pnl)}\n"
            f"Adjustment: {format_pnl(delta)}"
        )
        # Correct the running daily total by the delta (provisional was already
        # added) — but only if the trade belongs to the current session.
        if adjust_counter:
            self.daily_pnl += delta
            self.save_daily_stats()
        else:
            message += "\n\n_Prior session — today's total unchanged._"
        return await self.send_notification(message)

    async def notify_position_readopted(
        self,
        market_name: str,
        reversed_pnl: float,
        adjust_counter: bool = True,
    ) -> bool:
        """Alert that a falsely-closed position was re-adopted (still live at IG).

        Close-detection fired on a transient positions-API flicker; the deal was
        actually still open, so the provisional close is being undone. reversed_pnl
        is the provisional P&L booked at the false close — back it out of the live
        daily counter (unless it belonged to a prior, already-reset session) so the
        running total matches reality. Mirrors notify_trade_reconciled's lockstep
        handling of self.daily_pnl.
        """
        if adjust_counter:
            self.daily_pnl -= reversed_pnl
            self.save_daily_stats()
        message = (
            f"♻️ *POSITION RE-ADOPTED*\n\n"
            f"Market: {market_name}\n"
            f"A close was detected on a positions-API flicker, but the deal is "
            f"still open at IG — resuming management.\n"
            f"Reversed provisional P&L: {format_pnl(reversed_pnl)}"
        )
        if not adjust_counter:
            message += "\n\n_Prior session — today's total unchanged._"
        return await self.send_notification(message)

    async def notify_signal(
        self,
        market_name: str,
        direction: str,
        confidence: float,
        reason: str,
    ) -> bool:
        """Notify about a trade signal."""
        emoji = "🟢" if direction == "BUY" else "🔴"

        message = (
            f"{emoji} *TRADE SIGNAL*\n\n"
            f"Market: {market_name}\n"
            f"Signal: {direction}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Reason: {reason}"
        )
        return await self.send_notification(message)

    async def notify_error(self, error_message: str) -> bool:
        """Send error notification."""
        safe_msg = error_message.replace("_", " ")
        message = f"⚠️ *ERROR*\n\n{safe_msg}"
        return await self.send_notification(message)

    async def notify_daily_summary(
        self,
        balance: float,
        daily_pnl: float,
        trades_count: int,
        positions: List,
        pending_count: int = 0,
        pending_pnl: float = 0.0,
    ) -> bool:
        """Send daily summary.

        pending_count/pending_pnl describe trades that closed this session but
        IG hasn't booked yet — included in daily_pnl at their provisional value
        and flagged so the headline figure is honestly labelled.
        """
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

        positions_text = ""
        if positions:
            positions_text = "\n\n*Open Positions:*\n"
            for pos in positions:
                positions_text += f"• {pos.epic}: {pos.direction} ({format_pnl(pos.profit_loss)})\n"

        pending_text = ""
        if pending_count > 0:
            pending_text = (
                f"\n\n⏳ {pending_count} trade(s) ({format_pnl(pending_pnl)}) "
                f"awaiting broker confirmation — final total may differ."
            )

        message = (
            f"{pnl_emoji} *DAILY SUMMARY*\n\n"
            f"Balance: £{balance:,.2f}\n"
            f"Daily P&L: {format_pnl(daily_pnl)}\n"
            f"Trades: {trades_count}"
            f"{positions_text}"
            f"{pending_text}"
        )
        return await self.send_notification(message)

    # ==================== BOT LIFECYCLE ====================

    async def start(self) -> None:
        """Start the Telegram bot."""
        if not self.config.enabled or not self.config.bot_token:
            logger.info("Telegram bot disabled or not configured")
            return

        try:
            self.app = Application.builder().token(self.config.bot_token).build()

            # Add command handlers
            self.app.add_handler(CommandHandler("start", self.start_command))
            self.app.add_handler(CommandHandler("help", self.help_command))
            self.app.add_handler(CommandHandler("status", self.status_command))
            self.app.add_handler(CommandHandler("balance", self.balance_command))
            self.app.add_handler(CommandHandler("positions", self.positions_command))
            self.app.add_handler(CommandHandler("markets", self.markets_command))
            self.app.add_handler(CommandHandler("pnl", self.pnl_command))
            self.app.add_handler(CommandHandler("journal", self.journal_command))
            self.app.add_handler(CommandHandler("screener", self.screener_command))
            self.app.add_handler(CommandHandler("health", self.health_command))
            self.app.add_handler(CommandHandler("stop", self.stop_command))
            self.app.add_handler(CommandHandler("pause", self.stop_command))  # Alias
            self.app.add_handler(CommandHandler("resume", self.resume_command))
            self.app.add_handler(CommandHandler("forex", self.forex_command))
            self.app.add_handler(CommandHandler("mode", self.mode_command))
            self.app.add_handler(CommandHandler("daily", self.daily_command))
            self.app.add_handler(CommandHandler("pullback", self.pullback_command))
            self.app.add_handler(CommandHandler("emergency", self.emergency_command))
            self.app.add_handler(CommandHandler("rebuild", self.rebuild_command))

            # Add callback handlers
            self.app.add_handler(CallbackQueryHandler(self.stop_callback, pattern='^stop_'))
            self.app.add_handler(CallbackQueryHandler(self.emergency_callback, pattern='^emergency_'))
            self.app.add_handler(CallbackQueryHandler(self.rebuild_callback, pattern='^rebuild_'))

            # Start bot
            logger.info("Starting Telegram bot...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)

            self.is_running = True
            logger.info("Telegram bot started successfully!")

        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self.app:
            return

        try:
            await self.notify_shutdown("Bot shutting down")
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self.is_running = False
            logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of trading day)."""
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.save_daily_stats()
        logger.info("Daily Telegram stats reset")


class TelegramNotifier:
    """
    Lightweight synchronous Telegram notifier for test_run.py.

    Unlike TelegramBot (async, full command handler), this class uses
    simple HTTP requests to send messages. No event loop required.
    """

    def __init__(self, config: TelegramConfig):
        self.config = config
        self.enabled = config.enabled and bool(config.bot_token) and bool(config.chat_id)

    def _send(self, text: str) -> None:
        """Send a message via Telegram API."""
        if not self.enabled:
            return

        import requests

        try:
            requests.post(
                f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage",
                json={
                    "chat_id": self.config.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")

    def send_startup_message(self, balance: float, markets: list) -> None:
        """Send test run startup notification."""
        self._send(
            f"🧪 *IG Bot - Test Run*\n\n"
            f"Balance: £{balance:,.2f}\n"
            f"Markets: {', '.join(markets)}"
        )

    def send_trade_signal(self, signal) -> None:
        """Send a trade signal notification."""
        self._send(
            f"📊 *Signal: {signal.signal.value}*\n\n"
            f"Market: {signal.market_name}\n"
            f"Confidence: {signal.confidence:.0%}\n"
            f"Reason: {signal.reason}"
        )
