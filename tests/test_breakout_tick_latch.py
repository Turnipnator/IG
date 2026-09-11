"""Two breakout-path guarantees added 2026-09-11 (research_notes.md, "Gold #348").

1. The tick-entry latch survives a restart. `_breakout_armed` is in-memory; a restart
   re-armed the same closed bar fresh and the crossing fired a second time — 16 of the
   17 double-fired bars between 2026-08-20 and 09-10 coincide with a restart. The
   consumed bar is now persisted per epic and a re-arm of that bar starts consumed.

2. The three silent returns in `_execute_breakout_entry` (position already open on the
   epic, loss cooldown, startup cooldown) are visible: one throttled log line per epic
   per reason per hour, and a `Breakout-blocked: entry-gate` journal row for the two
   cooldown cases. The open-position case is log-only — journalling it would
   double-count trades the bot is already in, the distortion the shadow observer was
   silenced for in 46b1351. The hour-close path re-calls the entry function on every
   5-minute candle for the whole confirmation hour, so the throttle is what keeps this
   from writing twelve rows an hour.
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import main
from src.breakout import ArmedChannel
from src.client import Position

EPIC = "CS.D.USCGC.TODAY.IP"   # Gold: live breakout by config
BAR = "2026-09-10 12:00:00"
NEXT_BAR = "2026-09-10 13:00:00"


def _channel(bar_time=BAR):
    return ArmedChannel(upper=4443.0, lower=4341.4, stop_distance=34.57, atr=17.29,
                        htf_trend="BEARISH", htf_filter=True, bar_time=bar_time)


def _frame(n=80):
    idx = pd.date_range("2026-09-07", periods=n, freq="h")
    return pd.DataFrame({"date": idx, "open": 4400.0, "high": 4410.0, "low": 4390.0,
                         "close": 4405.0, "volume": 0})


class _Market:
    market_state = "TRADEABLE"
    name = "Gold"
    mid_price = 4340.0        # below the armed lower band -> SELL cross
    bid = 4339.8
    offer = 4340.2

    def to_dataframe(self):
        return _frame()


def _pos(deal_id, direction="SELL"):
    return Position(deal_id=deal_id, epic=EPIC, direction=direction, size=1.0, open_level=4351.37,
                    stop_level=4395.6, limit_level=None, profit_loss=0.0,
                    created_date="2026-09-10T14:05:00")


class _Globals(unittest.TestCase):
    """Save/restore every piece of module state these tests touch."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (dict(main.known_positions), set(main.breakout_deals), main.journal,
                       dict(main.htf_trends), dict(main._breakout_armed),
                       dict(main._breakout_tick_consumed), dict(main._breakout_entry_refusal_last),
                       dict(main.loss_cooldown_until), main.bot_start_time,
                       main.BREAKOUT_TICK_ENTRY, main.BREAKOUT_TICK_LATCH_FILE, main.client,
                       main.risk_manager)
        main.known_positions.clear()
        main.breakout_deals.clear()
        main._breakout_armed.clear()
        main._breakout_tick_consumed.clear()
        main._breakout_entry_refusal_last.clear()
        main.loss_cooldown_until.clear()
        main.journal = MagicMock()
        main.client = MagicMock()
        main.htf_trends[EPIC] = "BEARISH"
        main.BREAKOUT_TICK_ENTRY = "log"
        main.BREAKOUT_TICK_LATCH_FILE = Path(self._tmp.name) / "breakout_tick_latch.json"
        main.bot_start_time = datetime.now() - timedelta(hours=3)   # well past startup cooldown
        self.cfg = next(m for m in main.MARKETS if m.epic == EPIC)

    def tearDown(self):
        (kp, bd, journal, htf, armed, consumed, refusals, cooldowns, start,
         tick, latch_file, client, risk_manager) = self._saved
        main.risk_manager = risk_manager
        main.known_positions.clear(); main.known_positions.update(kp)
        main.breakout_deals.clear(); main.breakout_deals.update(bd)
        main.journal = journal
        main.htf_trends.clear(); main.htf_trends.update(htf)
        main._breakout_armed.clear(); main._breakout_armed.update(armed)
        main._breakout_tick_consumed.clear(); main._breakout_tick_consumed.update(consumed)
        main._breakout_entry_refusal_last.clear(); main._breakout_entry_refusal_last.update(refusals)
        main.loss_cooldown_until.clear(); main.loss_cooldown_until.update(cooldowns)
        main.bot_start_time = start
        main.BREAKOUT_TICK_ENTRY = tick
        main.BREAKOUT_TICK_LATCH_FILE = latch_file
        main.client = client
        self._tmp.cleanup()

    def _arm(self, bar_time=BAR, live=True):
        with patch.object(main.breakout, "arm_channel", return_value=_channel(bar_time)):
            main._arm_breakout_levels(EPIC, self.cfg, _frame(), "BEARISH", live=live)


class LatchSurvivesRestart(_Globals):
    def test_consuming_the_latch_persists_the_bar(self):
        self._arm()
        main._check_breakout_tick_trigger(EPIC, _Market())
        self.assertTrue(main._breakout_armed[EPIC]["consumed"])
        main.journal.log_rejected_signal.assert_called_once()
        self.assertEqual(json.loads(main.BREAKOUT_TICK_LATCH_FILE.read_text()), {EPIC: BAR})

    def test_restart_then_same_bar_arms_consumed_and_does_not_refire(self):
        self._arm()
        main._check_breakout_tick_trigger(EPIC, _Market())
        main.journal.log_rejected_signal.reset_mock()
        # --- simulated restart: in-memory state gone, disk file remains ---
        main._breakout_armed.clear()
        main._breakout_tick_consumed.clear()
        main._load_tick_latch()
        self._arm()                                   # first analysis pass re-arms the SAME bar
        self.assertTrue(main._breakout_armed[EPIC]["consumed"])
        main._check_breakout_tick_trigger(EPIC, _Market())
        main.journal.log_rejected_signal.assert_not_called()

    def test_restart_then_newer_bar_arms_fresh(self):
        self._arm()
        main._check_breakout_tick_trigger(EPIC, _Market())
        main.journal.log_rejected_signal.reset_mock()
        main._breakout_armed.clear()
        main._breakout_tick_consumed.clear()
        main._load_tick_latch()
        self._arm(bar_time=NEXT_BAR)
        self.assertFalse(main._breakout_armed[EPIC]["consumed"])
        main._check_breakout_tick_trigger(EPIC, _Market())
        main.journal.log_rejected_signal.assert_called_once()

    def test_missing_or_corrupt_file_is_harmless(self):
        main._load_tick_latch()                       # no file
        self.assertEqual(main._breakout_tick_consumed, {})
        main.BREAKOUT_TICK_LATCH_FILE.write_text("{not json")
        main._load_tick_latch()                       # corrupt file
        self.assertEqual(main._breakout_tick_consumed, {})
        self._arm()
        self.assertFalse(main._breakout_armed[EPIC]["consumed"])

    def test_latch_file_is_written_outside_the_lock(self):
        # If _save_tick_latch ran under _breakout_tick_lock, a slow disk would stall every
        # arming thread. Prove the lock is free when the write happens.
        self._arm()
        seen = []
        real_save = main._save_tick_latch

        def probe():
            seen.append(main._breakout_tick_lock.acquire(blocking=False))
            if seen[-1]:
                main._breakout_tick_lock.release()
            real_save()

        with patch.object(main, "_save_tick_latch", side_effect=probe):
            main._check_breakout_tick_trigger(EPIC, _Market())
        self.assertEqual(seen, [True])


class EntryRefusalsAreVisible(_Globals):
    def setUp(self):
        super().setUp()
        self.sig = main.TradeSignal(
            signal=main.Signal.SELL, epic=EPIC, market_name="Gold", confidence=0.7,
            entry_price=4351.37, stop_distance=44.25, limit_distance=0.0,
            reason="Breakout: SELL break of 55-bar channel @ 4341.4", atr=22.1, break_level=4341.4)

    def _enter(self):
        with self.assertLogs(main.logger, level="INFO") as cm:
            main._execute_breakout_entry(EPIC, _Market(), self.cfg, self.sig, None)
        return cm.output

    def _enter_quietly(self):
        # assertLogs raises if nothing is logged; this variant asserts silence.
        with self.assertNoLogs(main.logger, level="INFO"):
            main._execute_breakout_entry(EPIC, _Market(), self.cfg, self.sig, None)

    def _journal_reasons(self):
        return [c.kwargs["reject_reason"] for c in main.journal.log_rejected_signal.call_args_list]

    def test_open_position_refusal_logs_once_and_never_journals(self):
        main.known_positions["DIAAAAYFRW2LLAU"] = _pos("DIAAAAYFRW2LLAU")
        out = self._enter()
        self.assertEqual(len([l for l in out if "Breakout entry refused" in l]), 1)
        self.assertIn("position open", out[0])
        self.assertIn("DIAAAAYFRW2LLAU", out[0])
        main.journal.log_rejected_signal.assert_not_called()
        main.client.get_market_info.assert_not_called()
        main.client.open_position.assert_not_called()
        self._enter_quietly()                          # second pass inside the hour: throttled

    def test_daily_trend_position_does_not_block(self):
        # A daily-trend hold on Gold must not refuse the 1h breakout (user decision 2026-09-09).
        main.known_positions["DAILY1"] = _pos("DAILY1", direction="BUY")
        with patch.object(main, "_daily_managed", return_value=True):
            main.client.get_market_info.return_value = None
            main.risk_manager = MagicMock()
            main.risk_manager.calculate_position_size.return_value = MagicMock(approved=False, reason="test stop")
            main._execute_breakout_entry(EPIC, _Market(), self.cfg, self.sig, None)
        main.client.get_market_info.assert_called_once()   # got past the one-per-epic gate

    def test_loss_cooldown_refusal_logs_and_journals_once(self):
        main.loss_cooldown_until[EPIC] = datetime.now() + timedelta(minutes=37)
        out = self._enter()
        self.assertEqual(len([l for l in out if "Breakout entry refused" in l]), 1)
        self.assertIn("loss cooldown", out[0])
        reasons = self._journal_reasons()
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("Breakout-blocked: entry-gate loss cooldown"), reasons[0])
        main.client.open_position.assert_not_called()
        self._enter_quietly()
        self.assertEqual(len(self._journal_reasons()), 1)   # still one row

    def test_startup_cooldown_refusal_logs_and_journals_once(self):
        main.bot_start_time = datetime.now() - timedelta(minutes=4)
        out = self._enter()
        self.assertIn("startup cooldown", out[0])
        reasons = self._journal_reasons()
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("Breakout-blocked: entry-gate startup cooldown"), reasons[0])
        main.client.open_position.assert_not_called()
        self._enter_quietly()

    def test_throttle_is_per_reason(self):
        # A loss-cooldown refusal must not silence a later startup-cooldown refusal.
        main.loss_cooldown_until[EPIC] = datetime.now() + timedelta(minutes=5)
        self._enter()
        main.loss_cooldown_until.clear()
        main.bot_start_time = datetime.now() - timedelta(minutes=2)
        out = self._enter()
        self.assertIn("startup cooldown", out[0])
        self.assertEqual(len(self._journal_reasons()), 2)

    def test_expired_loss_cooldown_does_not_refuse(self):
        main.loss_cooldown_until[EPIC] = datetime.now() - timedelta(minutes=1)
        main.client.get_market_info.return_value = None
        main.risk_manager = MagicMock()
        main.risk_manager.calculate_position_size.return_value = MagicMock(approved=False, reason="test stop")
        main._execute_breakout_entry(EPIC, _Market(), self.cfg, self.sig, None)
        main.client.get_market_info.assert_called_once()
        self.assertNotIn(EPIC, main._breakout_entry_refusal_last)


if __name__ == "__main__":
    unittest.main()
