"""
Telegram notifications for trade alerts and status updates.
"""

import logging
from typing import Optional
import asyncio

import aiohttp

from config import TelegramConfig
from src.strategy import TradeSignal, Signal
from src.client import Position

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Send notifications to Telegram.

    Uses direct HTTP API calls for simplicity.
    """

    def __init__(self, config: TelegramConfig):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"

    async def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat."""
        if not self.config.enabled:
            logger.debug("Telegram notifications disabled")
            return False

        if not self.config.bot_token or not self.config.chat_id:
            logger.warning("Telegram not configured")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        logger.debug("Telegram message sent")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Telegram API error: {error}")
                        return False

        except asyncio.TimeoutError:
            logger.error("Telegram request timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message(self, text: str) -> bool:
        """Synchronous wrapper for sending messages."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._send_message(text))

    def send_startup_message(self, account_balance: float, markets: list[str]) -> bool:
        """Send bot startup notification."""
        message = (
            "<b>IG Trading Bot Started</b>\n\n"
            f"Account Balance: £{account_balance:,.2f}\n"
            f"Markets: {', '.join(markets)}\n"
            f"Status: Active"
        )
        return self.send_message(message)

    def send_shutdown_message(self, reason: str = "Manual shutdown") -> bool:
        """Send bot shutdown notification."""
        message = (
            "<b>IG Trading Bot Stopped</b>\n\n"
            f"Reason: {reason}"
        )
        return self.send_message(message)

    def send_trade_signal(self, signal: TradeSignal) -> bool:
        """Send trade signal notification."""
        if signal.signal == Signal.HOLD:
            return True  # Don't notify for HOLD signals

        emoji = "🟢" if signal.signal == Signal.BUY else "🔴"
        direction = "LONG" if signal.signal == Signal.BUY else "SHORT"

        message = (
            f"{emoji} <b>Trade Signal: {direction}</b>\n\n"
            f"Market: {signal.market_name}\n"
            f"Entry: {signal.entry_price:.2f}\n"
            f"Stop: {signal.stop_distance:.2f} pts\n"
            f"Target: {signal.limit_distance:.2f} pts\n"
            f"Confidence: {signal.confidence:.0%}\n"
            f"Reason: {signal.reason}"
        )
        return self.send_message(message)

    def send_trade_opened(
        self,
        market_name: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_distance: float,
        limit_distance: float,
    ) -> bool:
        """Send trade opened notification."""
        emoji = "🟢" if direction == "BUY" else "🔴"
        action = "LONG" if direction == "BUY" else "SHORT"

        message = (
            f"{emoji} <b>Position Opened</b>\n\n"
            f"Market: {market_name}\n"
            f"Direction: {action}\n"
            f"Size: £{size}/pt\n"
            f"Entry: {entry_price:.2f}\n"
            f"Stop: {stop_distance:.2f} pts\n"
            f"Target: {limit_distance:.2f} pts"
        )
        return self.send_message(message)

    def send_trade_closed(
        self,
        market_name: str,
        direction: str,
        pnl: float,
        reason: str,
    ) -> bool:
        """Send trade closed notification."""
        emoji = "✅" if pnl >= 0 else "❌"
        pnl_text = f"+£{pnl:.2f}" if pnl >= 0 else f"-£{abs(pnl):.2f}"

        message = (
            f"{emoji} <b>Position Closed</b>\n\n"
            f"Market: {market_name}\n"
            f"P&L: {pnl_text}\n"
            f"Reason: {reason}"
        )
        return self.send_message(message)

    def send_error(self, error_message: str) -> bool:
        """Send error notification."""
        message = (
            "⚠️ <b>Error</b>\n\n"
            f"{error_message}"
        )
        return self.send_message(message)

    def send_daily_summary(
        self,
        account_balance: float,
        daily_pnl: float,
        trades_count: int,
        winning_trades: int,
        open_positions: list[Position],
    ) -> bool:
        """Send daily trading summary."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        pnl_text = f"+£{daily_pnl:.2f}" if daily_pnl >= 0 else f"-£{abs(daily_pnl):.2f}"
        win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0

        positions_text = ""
        if open_positions:
            positions_text = "\n\n<b>Open Positions:</b>\n"
            for pos in open_positions:
                pos_pnl = f"+£{pos.profit_loss:.2f}" if pos.profit_loss >= 0 else f"-£{abs(pos.profit_loss):.2f}"
                positions_text += f"• {pos.epic}: {pos.direction} {pos.size}@{pos.open_level:.2f} ({pos_pnl})\n"

        message = (
            f"{pnl_emoji} <b>Daily Summary</b>\n\n"
            f"Balance: £{account_balance:,.2f}\n"
            f"Daily P&L: {pnl_text}\n"
            f"Trades: {trades_count}\n"
            f"Win Rate: {win_rate:.1f}%"
            f"{positions_text}"
        )
        return self.send_message(message)

    def send_market_status(self, market_name: str, status: str, reason: str = "") -> bool:
        """Send market status change notification."""
        emoji = "🟢" if status == "TRADEABLE" else "🔴"
        message = f"{emoji} <b>{market_name}</b>: {status}"
        if reason:
            message += f"\n{reason}"
        return self.send_message(message)
