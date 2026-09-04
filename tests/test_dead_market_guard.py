"""_dead_market_guard: alert when an analysed market is stuck non-TRADEABLE
while ticks keep flowing inside its trading window.

Regression for 2026-08-19 -> 09-04: Crude's MonthN futures slot re-pointed to
an OFFLINE contract, indicative ticks kept arriving, so archive/screener/HTF all
looked alive while analyze_forex_breakout returned silently on every bar. No
log line, no alert, 16 days, eight consecutive channel breaks skipped.
"""
import logging
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import main


@dataclass
class _Stream:
    market_state: str = "TRADEABLE"
    last_update: Optional[datetime] = None
    name: str = "X"


@dataclass
class _Cfg:
    epic: str
    name: str
    trading_start: int = 23
    trading_end: int = 21
    default_mode: Optional[str] = "breakout"
    shadow_only: bool = False
    sector: str = "Commodities"


class _Svc:
    def __init__(self, markets):
        self.markets = markets


class DeadMarketGuardTests(unittest.TestCase):
    EPIC = "CC.D.CL.USS.IP"

    def setUp(self):
        self.t0 = datetime(2026, 9, 4, 14, 0, 0)
        self.stream = _Stream(market_state="OFFLINE", last_update=self.t0 - timedelta(minutes=1), name="Crude Oil")
        self.cfg = _Cfg(epic=self.EPIC, name="Crude Oil")
        self._saved = (main.stream_service, main.MARKETS, main.telegram, main.telegram_loop, main.utc_hour)
        main.stream_service = _Svc({self.EPIC: self.stream})
        main.MARKETS = [self.cfg]
        main.telegram = None          # _market_mode -> default_mode; notify is a no-op
        main.telegram_loop = None
        main.utc_hour = lambda: 13    # inside the 23..21 wrap-around window
        main._dead_market_since.clear()
        main._dead_market_alerted_at.clear()
        self.sent = []
        self._orig_notify = main._dead_market_notify
        main._dead_market_notify = lambda msg: self.sent.append(msg)

    def tearDown(self):
        (main.stream_service, main.MARKETS, main.telegram, main.telegram_loop, main.utc_hour) = self._saved
        main._dead_market_notify = self._orig_notify
        main._dead_market_since.clear()
        main._dead_market_alerted_at.clear()

    def _run(self, minutes, state=None):
        """Advance the injected clock to t0+minutes, keep ticks fresh, run once."""
        now = self.t0 + timedelta(minutes=minutes)
        if state is not None:
            self.stream.market_state = state
        self.stream.last_update = now - timedelta(minutes=1)
        main._dead_market_guard(now=now)
        return now

    def test_offline_with_ticks_alerts_once_after_grace(self):
        self._run(0)
        self._run(30)
        self.assertEqual(self.sent, [], "must not alert inside the grace period")
        self._run(61)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("OFFLINE", self.sent[0])
        self.assertIn("Crude Oil", self.sent[0])
        self.assertIn("no entries can fire", self.sent[0])
        for m in (62, 90, 600):
            self._run(m)
        self.assertEqual(len(self.sent), 1, "one alert per episode, not one per minute")

    def test_daily_reminder_while_it_persists(self):
        self._run(0); self._run(61)
        self.assertEqual(len(self.sent), 1)
        self._run(61 + 23 * 60)
        self.assertEqual(len(self.sent), 1)
        self._run(61 + 24 * 60 + 1)
        self.assertEqual(len(self.sent), 2)

    def test_recovery_clears_state_and_notifies(self):
        self._run(0); self._run(61)
        self.assertEqual(len(self.sent), 1)
        self._run(120, state="TRADEABLE")
        self.assertEqual(len(self.sent), 2)
        self.assertIn("TRADEABLE again", self.sent[1])
        self.assertNotIn(self.EPIC, main._dead_market_since)
        self.assertNotIn(self.EPIC, main._dead_market_alerted_at)
        # A fresh episode alerts again from scratch.
        self._run(130, state="OFFLINE"); self._run(200)
        self.assertEqual(len(self.sent), 3)

    def test_edits_only_is_not_dead(self):
        self.stream.market_state = "EDITS_ONLY"
        self._run(0); self._run(120)
        self.assertEqual(self.sent, [])
        self.assertNotIn(self.EPIC, main._dead_market_since)

    def test_no_ticks_means_closed_not_dead(self):
        # OFFLINE but the quote is frozen (no update for hours): a closed market / weekend.
        self.stream.market_state = "OFFLINE"
        for m in (0, 61, 120):
            now = self.t0 + timedelta(minutes=m)
            self.stream.last_update = now - timedelta(hours=3)
            main._dead_market_guard(now=now)
        self.assertEqual(self.sent, [])

    def test_outside_trading_window_is_ignored(self):
        main.utc_hour = lambda: 22   # 21..23 UTC is outside the 23..21 window
        self._run(0); self._run(120)
        self.assertEqual(self.sent, [])

    def test_mode_off_is_ignored(self):
        self.cfg.default_mode = "off"
        self._run(0); self._run(120)
        self.assertEqual(self.sent, [])

    def test_observer_wording(self):
        self.cfg.default_mode = "breakout-shadow"
        self._run(0); self._run(61)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("observer is recording nothing", self.sent[0])

    def test_no_stream_service_is_a_noop(self):
        main.stream_service = None
        main._dead_market_guard(now=self.t0)
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
