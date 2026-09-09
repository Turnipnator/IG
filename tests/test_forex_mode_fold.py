"""The global /forex toggle is folded into per-market /mode (2026-09-09).

Until then forex pairs were governed by ONE global mode with its own vocabulary
("shadow" meant BREAKOUT observed) plus a per-pair config veto
(breakout_shadow_only). Two control surfaces made the /mode board misleading
and, on 2026-09-09, let a stray `/forex momentum` put both pairs on the retired
momentum pipeline for an hour. This file pins:

  * the config defaults that replace the veto (EUR/USD breakout-shadow, GBP/USD
    breakout), and that both pairs have a breakout config so those modes can fire;
  * the one-shot boot migration of data/forex_mode.json into per-pair overrides —
    written ONLY where the translation differs from the config default, so the
    deploy itself changes nothing about what trades;
  * that main._market_mode and the screener exemption agree with the Telegram
    board for forex pairs after migration;
  * the board and toggles: every market on one page, forex pairs toggled like any
    other, and /forex reduced to a pointer that changes nothing;
  * that no runtime code still keys on the retired toggle.
"""
import asyncio
import dataclasses
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from config import MARKETS, MarketConfig, TelegramConfig
from src.breakout import has_breakout_config
from tests.test_telegram_edited_message import edited_update

REPO = Path(__file__).resolve().parents[1]
EUR = "CS.D.EURUSD.TODAY.IP"
GBP = "CS.D.GBPUSD.TODAY.IP"
FOREX = [m for m in MARKETS if m.sector == "Forex"]
BY_EPIC = {m.epic: m for m in MARKETS}


class TestConfigDefaultsReplaceTheVeto(unittest.TestCase):
    def test_forex_pairs_present(self):
        self.assertEqual({m.epic for m in FOREX}, {EUR, GBP})

    def test_eurusd_observes_and_gbpusd_trades(self):
        self.assertEqual(BY_EPIC[EUR].default_mode, "breakout-shadow")
        self.assertEqual(BY_EPIC[GBP].default_mode, "breakout")

    def test_both_pairs_have_a_breakout_config(self):
        for epic in (EUR, GBP):
            self.assertTrue(has_breakout_config(epic), epic)

    def test_veto_field_is_gone(self):
        names = {f.name for f in dataclasses.fields(MarketConfig)}
        self.assertNotIn("breakout_shadow_only", names)


class _BotHarness(unittest.TestCase):
    """TelegramBot with STATS_DIR and both mode files redirected to a temp dir, so
    the real data/ is never read or written."""

    def setUp(self):
        try:
            import src.telegram_bot as tb
        except ImportError as e:  # pragma: no cover
            self.skipTest(f"telegram not importable: {e}")
        self.tb = tb
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.modes_file = tmp / "market_modes.json"
        self.legacy = tmp / "forex_mode.json"
        self.migrated = tmp / "forex_mode.json.migrated"
        self.patches = [mock.patch.object(tb, "STATS_DIR", tmp),
                        mock.patch.object(tb, "MARKET_MODES_FILE", self.modes_file),
                        mock.patch.object(tb, "LEGACY_FOREX_MODE_FILE", self.legacy)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def make_bot(self):
        return self.tb.TelegramBot(TelegramConfig(bot_token="x", chat_id="1"))

    def write_legacy(self, mode):
        self.legacy.write_text(json.dumps({"forex_mode": mode, "saved_at": "t"}))

    def write_overrides(self, modes):
        self.modes_file.write_text(json.dumps({"market_modes": modes, "saved_at": "t"}))

    def reset_files(self):
        for f in (self.modes_file, self.legacy, self.migrated):
            if f.exists():
                f.unlink()


class TestLegacyForexMigration(_BotHarness):
    # legacy global mode -> overrides that must be written, given the defaults
    # EUR/USD=breakout-shadow and GBP/USD=breakout. An entry is absent wherever the
    # translation already equals the config default.
    TABLE = [
        ("off", {EUR: "off", GBP: "off"}),
        ("momentum", {EUR: "momentum", GBP: "momentum"}),
        ("shadow", {GBP: "breakout-shadow"}),          # legacy shadow = breakout observed
        ("breakout", {}),                               # = each pair's config default
        ("bogus", {EUR: "off", GBP: "off"}),            # corrupt -> off, as the old loader did
    ]

    def test_translation_table(self):
        for legacy, expected in self.TABLE:
            with self.subTest(legacy=legacy):
                self.reset_files()
                self.write_legacy(legacy)
                bot = self.make_bot()
                self.assertEqual(bot.market_modes, expected)
                self.assertFalse(self.legacy.exists(), "legacy file must be renamed away")
                self.assertTrue(self.migrated.exists())
                self.assertIsNotNone(bot.mode_migration_notice)
                if expected:
                    self.assertEqual(json.loads(self.modes_file.read_text())["market_modes"], expected)
                else:
                    self.assertFalse(self.modes_file.exists(), "nothing to write -> no file")
                    self.assertIn("nothing written", bot.mode_migration_notice)

    def test_live_breakout_deploy_is_a_noop_for_the_book(self):
        """The state the VPS is in at deploy time: /forex breakout. Both pairs
        must resolve exactly as before — GBP/USD live, EUR/USD observe-only —
        with no override written."""
        self.write_legacy("breakout")
        bot = self.make_bot()
        self.assertEqual(bot._effective_mode(BY_EPIC[GBP]), "breakout")
        self.assertEqual(bot._effective_mode(BY_EPIC[EUR]), "breakout-shadow")
        self.assertEqual(bot.market_modes, {})

    def test_existing_per_pair_override_is_left_alone(self):
        self.write_overrides({GBP: "momentum"})
        self.write_legacy("off")
        bot = self.make_bot()
        self.assertEqual(bot.market_modes, {GBP: "momentum", EUR: "off"})
        self.assertIn("existing override", bot.mode_migration_notice)

    def test_no_legacy_file_means_no_migration(self):
        bot = self.make_bot()
        self.assertIsNone(bot.mode_migration_notice)
        self.assertEqual(bot.market_modes, {})
        self.assertFalse(self.migrated.exists())

    def test_runs_exactly_once(self):
        self.write_legacy("off")
        first = self.make_bot()
        self.assertEqual(first.market_modes, {EUR: "off", GBP: "off"})
        second = self.make_bot()  # next boot: file renamed, overrides persisted
        self.assertIsNone(second.mode_migration_notice)
        self.assertEqual(second.market_modes, {EUR: "off", GBP: "off"})

    def test_main_resolves_forex_like_the_board(self):
        """main._market_mode (trading) and the Telegram board must agree on forex
        pairs, and the screener exemption must follow the same source — the
        forex special-case that read the global toggle is gone."""
        import main
        original = getattr(main, "telegram", None)
        try:
            for legacy, _ in self.TABLE:
                with self.subTest(legacy=legacy):
                    self.reset_files()
                    self.write_legacy(legacy)
                    bot = self.make_bot()
                    main.telegram = bot
                    for m in FOREX:
                        self.assertEqual(main._market_mode(m), bot._effective_mode(m), m.epic)
                    exempt = main._screener_exempt_epics()
                    for m in FOREX:
                        live = bot._effective_mode(m) in ("momentum", "breakout")
                        self.assertEqual(m.epic not in exempt, live, (legacy, m.epic))
        finally:
            main.telegram = original


class TestBoardAndToggles(_BotHarness):
    def setUp(self):
        super().setUp()
        self.bot = self.make_bot()

    def run_mode(self, *args):
        update, msg = edited_update()
        asyncio.run(self.bot.mode_command(update, SimpleNamespace(args=list(args))))
        return msg

    def test_board_shows_every_market_on_one_page(self):
        board = self.run_mode().replies[0]
        self.assertNotIn("/forex", board)
        for m in MARKETS:
            self.assertIn(f" {m.name}: `", board, m.name)
        self.assertIn("EUR/USD: `breakout-shadow`", board)
        self.assertIn("GBP/USD: `breakout`", board)

    def test_forex_pair_toggles_like_any_other_market(self):
        msg = self.run_mode("gbp/usd", "breakout-shadow")
        self.assertEqual(self.bot.market_modes.get(GBP), "breakout-shadow")
        self.assertEqual(json.loads(self.modes_file.read_text())["market_modes"][GBP],
                         "breakout-shadow")
        self.assertIn("breakout-shadow", msg.replies[0])
        board = self.run_mode().replies[0]
        self.assertIn("GBP/USD: `breakout-shadow` _(override)_", board)

    def test_forex_shadow_is_momentum_observed_and_accepted(self):
        self.run_mode("eur/usd", "shadow")
        self.assertEqual(self.bot.market_modes.get(EUR), "shadow")

    def test_forex_momentum_warns_it_is_retired(self):
        msg = self.run_mode("gbp/usd", "momentum")
        self.assertEqual(self.bot.market_modes.get(GBP), "momentum")
        self.assertIn("retired", msg.replies[0])

    def test_default_clears_a_forex_override(self):
        self.run_mode("eur/usd", "off")
        msg = self.run_mode("eur/usd", "default")
        self.assertNotIn(EUR, self.bot.market_modes)
        self.assertIn("breakout-shadow", msg.replies[0])

    def test_forex_command_is_a_pointer_that_changes_nothing(self):
        update, msg = edited_update()
        asyncio.run(self.bot.forex_command(update, SimpleNamespace(args=["momentum"])))
        self.assertEqual(self.bot.market_modes, {})
        self.assertFalse(self.modes_file.exists())
        self.assertEqual(len(msg.replies), 1)
        self.assertIn("retired", msg.replies[0])
        self.assertIn("/mode", msg.replies[0])
        self.assertNotIn("forex_mode", dir(self.bot))


class TestNoRuntimeReferenceToTheRetiredToggle(unittest.TestCase):
    """Static guards. The forex gate, the screener's forex branch and the re-adopt
    routing all read telegram.forex_mode; the shadow branch excluded forex. Any
    one of those coming back re-creates the second control surface."""

    def test_main_no_longer_reads_the_global_toggle(self):
        text = (REPO / "main.py").read_text()
        self.assertIsNone(re.search(r"\bforex_mode\b", text))
        self.assertNotIn("breakout_shadow_only", text)

    def test_main_shadow_branch_no_longer_excludes_forex(self):
        text = (REPO / "main.py").read_text()
        self.assertNotIn('mkt_mode == "shadow" and market_config.sector != "Forex"', text)
        self.assertIn('if mkt_mode == "shadow":', text)

    def test_telegram_bot_has_no_global_forex_state(self):
        text = (REPO / "src" / "telegram_bot.py").read_text()
        self.assertNotIn("self.forex_mode", text)
        self.assertNotIn("FOREX_MODES", text)
        self.assertNotIn("save_forex_mode", text)

    def test_config_no_longer_sets_the_veto(self):
        text = (REPO / "config.py").read_text()
        self.assertIsNone(re.search(r"^\s*breakout_shadow_only\s*[:=]", text, re.M))


if __name__ == "__main__":
    unittest.main()
