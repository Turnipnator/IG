"""
Telegram Bot Integration for IG Trading Bot
Provides remote control and monitoring via Telegram
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, TYPE_CHECKING

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from config import TelegramConfig, MARKETS

if TYPE_CHECKING:
    from src.client import IGClient

logger = logging.getLogger(__name__)


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

        # Statistics
        self.start_time = datetime.now()
        self.notifications_sent = 0
        self.commands_executed = 0
        self.trades_today = 0
        self.daily_pnl = 0.0

        logger.info(f"Telegram bot initialized with {len(self.authorized_users)} authorized users")

    def set_ig_client(self, client: 'IGClient') -> None:
        """Set reference to IG client."""
        self.ig_client = client

    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        return user_id in self.authorized_users

    # ==================== COMMAND HANDLERS ====================

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id

        if not self.is_authorized(user_id):
            await update.message.reply_text(
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

        await update.message.reply_text(welcome_message, parse_mode='Markdown')
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
            "/pnl - Today's P&L summary\n\n"
            "*🎮 Control:*\n"
            "/stop - Pause trading\n"
            "/resume - Resume trading\n"
            "/emergency - ⚠️ Close ALL positions\n\n"
            "*🔔 Notifications:*\n"
            "Automatic alerts for trades and errors.\n\n"
            f"Your ID: `{update.effective_user.id}`"
        )

        await update.message.reply_text(help_text, parse_mode='Markdown')
        self.commands_executed += 1

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.message.reply_text("⚠️ IG client not connected")
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

            await update.message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text(f"❌ Error getting status: {str(e)}")

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.message.reply_text("⚠️ IG client not connected")
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

            await update.message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in balance command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.message.reply_text("⚠️ IG client not connected")
                return

            positions = self.ig_client.get_positions()

            if not positions:
                await update.message.reply_text("📭 No open positions")
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

            await update.message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def markets_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /markets command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.message.reply_text("⚠️ IG client not connected")
                return

            message = "📈 *MARKET STATUS*\n\n"

            for market in MARKETS:
                info = self.ig_client.get_market_info(market.epic)
                if info:
                    status_emoji = "🟢" if info.market_status == "TRADEABLE" else "🔴"
                    mid_price = (info.bid + info.offer) / 2
                    spread = info.offer - info.bid

                    message += (
                        f"*{market.name}*\n"
                        f"{status_emoji} {info.market_status}\n"
                        f"Price: {mid_price:,.2f} (spread: {spread:.2f})\n\n"
                    )
                else:
                    message += f"*{market.name}*\n⚠️ Unable to fetch\n\n"

            await update.message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in markets command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command."""
        if not self.is_authorized(update.effective_user.id):
            return

        try:
            if not self.ig_client or not self.ig_client.is_logged_in:
                await update.message.reply_text("⚠️ IG client not connected")
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

            await update.message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in pnl command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

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

            await update.message.reply_text(message, parse_mode='Markdown')
            self.commands_executed += 1

        except Exception as e:
            logger.error(f"Error in health command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

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

        await update.message.reply_text(
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
            await update.message.reply_text("✅ *Bot resumed!* Trading will continue.")
            await self.send_notification("🟢 *Bot Resumed*\nTrading operations continuing.")
        else:
            await update.message.reply_text("ℹ️ Bot is already running")

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

        await update.message.reply_text(
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

    # ==================== NOTIFICATION METHODS ====================

    async def send_notification(self, message: str) -> bool:
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
                        parse_mode='Markdown'
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
        return await self.send_notification(message)

    async def notify_trade_closed(
        self,
        market_name: str,
        direction: str,
        pnl: float,
        reason: str,
    ) -> bool:
        """Notify when trade is closed."""
        emoji = "✅" if pnl >= 0 else "❌"

        message = (
            f"{emoji} *POSITION CLOSED*\n\n"
            f"Market: {market_name}\n"
            f"P&L: {format_pnl(pnl)}\n"
            f"Reason: {reason}"
        )
        self.daily_pnl += pnl
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
        message = f"⚠️ *ERROR*\n\n{error_message}"
        return await self.send_notification(message)

    async def notify_daily_summary(
        self,
        balance: float,
        daily_pnl: float,
        trades_count: int,
        positions: List,
    ) -> bool:
        """Send daily summary."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

        positions_text = ""
        if positions:
            positions_text = "\n\n*Open Positions:*\n"
            for pos in positions:
                positions_text += f"• {pos.epic}: {pos.direction} ({format_pnl(pos.profit_loss)})\n"

        message = (
            f"{pnl_emoji} *DAILY SUMMARY*\n\n"
            f"Balance: £{balance:,.2f}\n"
            f"Daily P&L: {format_pnl(daily_pnl)}\n"
            f"Trades: {trades_count}"
            f"{positions_text}"
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
            self.app.add_handler(CommandHandler("health", self.health_command))
            self.app.add_handler(CommandHandler("stop", self.stop_command))
            self.app.add_handler(CommandHandler("pause", self.stop_command))  # Alias
            self.app.add_handler(CommandHandler("resume", self.resume_command))
            self.app.add_handler(CommandHandler("emergency", self.emergency_command))

            # Add callback handlers
            self.app.add_handler(CallbackQueryHandler(self.stop_callback, pattern='^stop_'))
            self.app.add_handler(CallbackQueryHandler(self.emergency_callback, pattern='^emergency_'))

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
