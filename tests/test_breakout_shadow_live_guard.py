"""The breakout SHADOW observer must stay silent while a LIVE breakout position is
open on the same epic.

Seen 2026-09-09: Crude entered live at 01:00, /mode flipped it to breakout-shadow at
09:19, and every hourly close after that logged "Breakout [Crude Oil] (shadow): would
BUY" plus a Breakout-shadow rejected_signals row, and the tick path logged "[log]"
crosses — for a trade the bot was already in. The live trade IS the outcome; the
shadow rows would double-count it. Momentum and daily-trend positions on the epic
must NOT silence the observer, or the breakout shadow record would be biased by
other strategies' positions. The guard is the breakout_deals tag, nothing else.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import main
from src.client import Position

EPIC = "CC.D.CL.USS.IP"


def _pos(deal_id, epic=EPIC):
    return Position(deal_id=deal_id, epic=epic, direction="BUY", size=0.16, open_level=9260.1,
                    stop_level=9125.6, limit_level=None, profit_loss=0.0,
                    created_date="2026-09-09T01:00:00")


def _frame(n=80):
    idx = pd.date_range("2026-09-01", periods=n, freq="h")
    return pd.DataFrame({"date": idx, "open": 9000.0, "high": 9010.0, "low": 8990.0,
                         "close": 9005.0, "volume": 0})


class _Market:
    market_state = "TRADEABLE"
    name = "Crude Oil"
    mid_price = 9450.0
    bid = 9449.0
    offer = 9451.0

    def to_dataframe(self):
        return _frame()


class _Globals(unittest.TestCase):
    """Save/restore the module state every test touches."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (dict(main.known_positions), set(main.breakout_deals), main.journal,
                       dict(main.htf_trends), dict(main._breakout_armed), main.BREAKOUT_TICK_ENTRY,
                       dict(main._breakout_tick_consumed), main.BREAKOUT_TICK_LATCH_FILE)
        main.known_positions.clear()
        main.breakout_deals.clear()
        main._breakout_armed.clear()
        main._breakout_tick_consumed.clear()
        main.journal = MagicMock()
        main.htf_trends[EPIC] = "BULLISH"
        # Consuming the tick latch persists it (9b4bc77). Redirect the file, or a run
        # of this suite inside the live container overwrites the bot's real latch
        # with the Crude fixture below (seen 2026-09-11, twice).
        main.BREAKOUT_TICK_LATCH_FILE = Path(self._tmp.name) / "breakout_tick_latch.json"

    def tearDown(self):
        kp, bd, journal, htf, armed, tick, consumed, latch_file = self._saved
        main.known_positions.clear(); main.known_positions.update(kp)
        main.breakout_deals.clear(); main.breakout_deals.update(bd)
        main.journal = journal
        main.htf_trends.clear(); main.htf_trends.update(htf)
        main._breakout_armed.clear(); main._breakout_armed.update(armed)
        main._breakout_tick_consumed.clear(); main._breakout_tick_consumed.update(consumed)
        main.BREAKOUT_TICK_ENTRY = tick
        main.BREAKOUT_TICK_LATCH_FILE = latch_file
        self._tmp.cleanup()


class LiveBreakoutPositionHelper(_Globals):
    def test_tagged_breakout_position_counts(self):
        main.known_positions["D1"] = _pos("D1"); main.breakout_deals.add("D1")
        self.assertTrue(main._live_breakout_position(EPIC))

    def test_untagged_momentum_position_does_not_count(self):
        main.known_positions["D2"] = _pos("D2")
        self.assertFalse(main._live_breakout_position(EPIC))

    def test_other_epic_does_not_count(self):
        main.known_positions["D3"] = _pos("D3", epic="CS.D.USCGC.TODAY.IP"); main.breakout_deals.add("D3")
        self.assertFalse(main._live_breakout_position(EPIC))


class ShadowBranchGuard(_Globals):
    def setUp(self):
        super().setUp()
        self.cfg = next(m for m in main.MARKETS if m.epic == EPIC)
        self.sig = main.TradeSignal(
            signal=main.Signal.BUY, epic=EPIC, market_name="Crude Oil", confidence=0.7,
            entry_price=9450.0, stop_distance=120.0, limit_distance=0.0,
            reason="Breakout: BUY break of 55-bar channel @ 9446.0", atr=60.0, break_level=9446.0)

    def _run(self):
        with patch.object(main.breakout, "analyze_breakout", return_value=self.sig), \
             patch.object(main, "_arm_breakout_levels"), \
             patch.object(main, "_resolve_breakout_shadow"), \
             patch.object(main, "_snapshot_breakout_shadow") as snap:
            main.analyze_forex_breakout(EPIC, _Market(), self.cfg, "breakout-shadow")
        return snap

    def test_silent_while_live_breakout_position_open(self):
        main.known_positions["D1"] = _pos("D1"); main.breakout_deals.add("D1")
        snap = self._run()
        snap.assert_not_called()
        main.journal.log_rejected_signal.assert_not_called()

    def test_observes_when_flat(self):
        snap = self._run()
        snap.assert_called_once()
        main.journal.log_rejected_signal.assert_called_once()

    def test_observes_over_a_momentum_position(self):
        main.known_positions["D2"] = _pos("D2")   # open on the epic but NOT a breakout deal
        snap = self._run()
        snap.assert_called_once()


class TickCrossLogGuard(_Globals):
    def setUp(self):
        super().setUp()
        main.BREAKOUT_TICK_ENTRY = "log"
        ch = MagicMock(upper=9446.0, lower=9200.0, htf_filter=True, htf_trend="BULLISH",
                       stop_distance=120.0, atr=60.0, bar_time="2026-09-09T20:00:00")
        main._breakout_armed[EPIC] = {"channel": ch, "live": False, "consumed": False, "armed_at": None}

    def test_log_only_cross_suppressed_and_latch_consumed(self):
        main.known_positions["D1"] = _pos("D1"); main.breakout_deals.add("D1")
        main._check_breakout_tick_trigger(EPIC, _Market())
        main.journal.log_rejected_signal.assert_not_called()
        self.assertTrue(main._breakout_armed[EPIC]["consumed"])   # one cross per armed bar, still

    def test_log_only_cross_recorded_when_flat(self):
        main._check_breakout_tick_trigger(EPIC, _Market())
        main.journal.log_rejected_signal.assert_called_once()


if __name__ == "__main__":
    unittest.main()
