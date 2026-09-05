"""Telegram command handlers must survive an EDITED command message.

python-telegram-bot routes edited messages to CommandHandler by default
(``filters.UpdateType.MESSAGES`` = MESSAGE | EDITED_MESSAGE). On such an update
``update.message`` is None and only ``update.edited_message`` is set, so every
``update.message.reply_text(...)`` raised AttributeError. On 2026-09-04 21:49 that
crashed /mode twice: once on the "unknown mode" reply, and once AFTER a DXY mode
change had been applied and persisted — the change went through with no
confirmation. ``update.effective_message`` resolves to whichever one is set.
"""
import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src" / "telegram_bot.py"
GOLD = "CS.D.USCGC.TODAY.IP"


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def edited_update(user_id=1):
    """An Update shaped like an edited command: message=None, edited_message set."""
    msg = FakeMessage()
    update = SimpleNamespace(message=None, edited_message=msg, effective_message=msg,
                             effective_user=SimpleNamespace(id=user_id))
    return update, msg


class TestNoDirectMessageAccess(unittest.TestCase):
    def test_handlers_never_touch_update_message_directly(self):
        hits = [i + 1 for i, line in enumerate(SRC.read_text().splitlines())
                if re.search(r"update\.message\b", line)]
        self.assertEqual(hits, [],
                         f"update.message used at lines {hits} — it is None for an edited "
                         "command; use update.effective_message")


class TestModeCommandOnEditedMessage(unittest.TestCase):
    def setUp(self):
        try:
            import src.telegram_bot as tb
        except ImportError as e:  # pragma: no cover
            self.skipTest(f"telegram not importable: {e}")
        self.tb = tb
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.modes_file = tmp / "market_modes.json"
        # Never let the test read or write the real data/market_modes.json.
        self.patches = [mock.patch.object(tb, "STATS_DIR", tmp),
                        mock.patch.object(tb, "MARKET_MODES_FILE", self.modes_file)]
        for p in self.patches:
            p.start()
        from config import TelegramConfig
        self.bot = tb.TelegramBot(TelegramConfig(bot_token="x", chat_id="1"))
        self.assertEqual(self.bot.market_modes, {})

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_mode(self, *args):
        update, msg = edited_update()
        asyncio.run(self.bot.mode_command(update, SimpleNamespace(args=list(args))))
        return msg

    def test_valid_change_is_applied_persisted_and_confirmed(self):
        msg = self.run_mode("gold", "breakout")
        self.assertEqual(self.bot.market_modes.get(GOLD), "breakout")
        self.assertEqual(json.loads(self.modes_file.read_text())["market_modes"][GOLD],
                         "breakout")
        self.assertEqual(len(msg.replies), 1, msg.replies)
        self.assertIn("breakout", msg.replies[0])

    def test_unknown_mode_replies_and_changes_nothing(self):
        msg = self.run_mode("gold", "bogus")
        self.assertEqual(self.bot.market_modes, {})
        self.assertFalse(self.modes_file.exists())
        self.assertEqual(len(msg.replies), 1, msg.replies)
        self.assertIn("Unknown mode", msg.replies[0])

    def test_bare_mode_prints_the_board(self):
        msg = self.run_mode()
        self.assertEqual(len(msg.replies), 1, msg.replies)
        self.assertIn("Gold", msg.replies[0])

    def test_unauthorised_user_is_ignored(self):
        update, msg = edited_update(user_id=999)
        asyncio.run(self.bot.mode_command(update, SimpleNamespace(args=["gold", "breakout"])))
        self.assertEqual(msg.replies, [])
        self.assertEqual(self.bot.market_modes, {})
