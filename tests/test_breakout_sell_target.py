"""Gold breakout SHORTS carry a 50-point take-profit; nothing else does.

2026-11-27, user decision, DEMO trial (research_notes "Forward scoring of breakout
exits"). The target is config (BreakoutConfig.sell_limit_points) read through ONE
helper, breakout.limit_distance_for, by both entry routes — the hour-close signal
(analyze_breakout) and the tick-cross signal (_check_breakout_tick_trigger).

The trap this file exists for: the Donchian trail amends the broker stop with a PUT
that must RE-SEND the limit. If it ever went out without one, the target would be
dropped at the broker and the trade would silently revert to trail-only.
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import main
from src import breakout
from src.client import Position

GOLD = "CS.D.USCGC.TODAY.IP"


def _frame(last_high=4305.0, last_low=4295.0, n=80):
    """Flat 4290–4310 channel; the last bar sits inside it unless a break is asked for."""
    idx = pd.date_range("2026-09-01", periods=n, freq="h")
    df = pd.DataFrame({"date": idx, "open": 4300.0, "high": 4310.0, "low": 4290.0,
                       "close": 4300.0, "volume": 0})
    df.loc[n - 1, ["high", "low"]] = [last_high, last_low]
    return df


def _gold():
    return next(m for m in main.MARKETS if m.epic == GOLD)


class Config(unittest.TestCase):
    def test_gold_short_target_is_50_points(self):
        self.assertEqual(breakout.limit_distance_for(GOLD, "SELL"), 50.0)

    def test_gold_long_has_no_target(self):
        self.assertEqual(breakout.limit_distance_for(GOLD, "BUY"), 0.0)

    def test_no_other_market_has_a_target(self):
        for epic in breakout.BREAKOUT_CONFIGS:
            if epic == GOLD:
                continue
            for d in ("BUY", "SELL"):
                self.assertEqual(breakout.limit_distance_for(epic, d), 0.0, f"{epic} {d}")

    def test_unknown_epic_has_no_target(self):
        self.assertEqual(breakout.limit_distance_for("NOT.AN.EPIC", "SELL"), 0.0)


class HourCloseSignal(unittest.TestCase):
    def test_short_break_carries_the_target(self):
        sig = breakout.analyze_breakout(_frame(last_low=4200.0), _gold(), 4205.0, "BEARISH")
        self.assertEqual(sig.signal, main.Signal.SELL)
        self.assertEqual(sig.limit_distance, 50.0)

    def test_long_break_carries_none(self):
        sig = breakout.analyze_breakout(_frame(last_high=4400.0), _gold(), 4395.0, "BULLISH")
        self.assertEqual(sig.signal, main.Signal.BUY)
        self.assertEqual(sig.limit_distance, 0.0)

    def test_stop_is_unchanged_by_the_target(self):
        short = breakout.analyze_breakout(_frame(last_low=4200.0), _gold(), 4205.0, "BEARISH")
        ch = breakout.arm_channel(_frame(last_low=4200.0), _gold(), "BEARISH")
        self.assertEqual(short.stop_distance, ch.stop_distance)


class TickCrossSignal(unittest.TestCase):
    def setUp(self):
        self._armed = dict(main._breakout_armed)
        self._consumed = dict(main._breakout_tick_consumed)
        main._breakout_armed.clear()

    def tearDown(self):
        main._breakout_armed.clear(); main._breakout_armed.update(self._armed)
        main._breakout_tick_consumed.clear(); main._breakout_tick_consumed.update(self._consumed)

    def _fire(self, mid):
        ch = breakout.ArmedChannel(upper=4310.0, lower=4290.0, stop_distance=36.0, atr=18.0,
                                   htf_trend="BEARISH" if mid < 4300 else "BULLISH",
                                   htf_filter=True, bar_time="2026-11-27 10:00:00")
        main._breakout_armed[GOLD] = {"channel": ch, "live": True, "consumed": False,
                                      "armed_at": datetime.now()}
        market = MagicMock(mid_price=mid, bid=mid - 0.3, offer=mid + 0.3)
        with patch.object(main, "BREAKOUT_TICK_ENTRY", "live"), \
             patch.object(main, "_save_tick_latch"), \
             patch.object(main, "journal", MagicMock()), \
             patch.object(main, "_live_breakout_position", return_value=False), \
             patch.object(main, "_execute_breakout_entry") as execute:
            main._check_breakout_tick_trigger(GOLD, market)
        execute.assert_called_once()
        return execute.call_args.args[3]

    def test_short_tick_entry_carries_the_target(self):
        sig = self._fire(4289.0)
        self.assertEqual(sig.signal, main.Signal.SELL)
        self.assertEqual(sig.limit_distance, 50.0)

    def test_long_tick_entry_carries_none(self):
        sig = self._fire(4311.0)
        self.assertEqual(sig.signal, main.Signal.BUY)
        self.assertEqual(sig.limit_distance, 0.0)


class TrailKeepsTheTarget(unittest.TestCase):
    """The stop amendment must re-send the position's limit level."""

    def _trail(self, limit_level):
        pos = Position(deal_id="D1", epic=GOLD, direction="SELL", size=1.0, open_level=3524.5,
                       stop_level=3567.9, limit_level=limit_level, profit_loss=0.0,
                       created_date="2026-11-27T10:00:00")
        client = MagicMock()
        client.update_position_stop.return_value = True
        saved = dict(main.trailing_stop_levels)
        try:
            with patch.object(main, "client", client), \
                 patch.object(main, "stream_service", None), \
                 patch.object(main, "_trail_frame", return_value=_frame()), \
                 patch.object(main.breakout, "exit_channel", return_value=3550.0):
                main._update_breakout_trail(pos, None)
        finally:
            main.trailing_stop_levels.clear(); main.trailing_stop_levels.update(saved)
        client.update_position_stop.assert_called_once()
        return client.update_position_stop.call_args

    def test_amendment_resends_the_limit(self):
        call = self._trail(3474.5)
        self.assertEqual(call.args[:3], ("D1", 3550.0, 3474.5))

    def test_position_without_a_limit_stays_without_one(self):
        call = self._trail(None)
        self.assertEqual(call.args[:3], ("D1", 3550.0, None))


if __name__ == "__main__":
    unittest.main()
