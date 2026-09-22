"""Economic calendar: parse the feed ForexFactory actually sends, and stay log-only.

Until 2026-09-22 the parser required a separate `time` field the feed no longer has,
so every event was dropped and the block never fired — while the log said "0
high-impact events this week" as if that were a fact. The fixture is the real feed
captured that day. The main.py tests pin the log-only contract: near an event the
momentum path logs and carries on unless CALENDAR_ENFORCE is set, and the breakout
path never blocks and never touches the network.
"""

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calendar import CURRENCY_EPIC_MAP, EconomicCalendar, EconomicEvent  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "ff_calendar_2026-09-22.json"
GBPUSD, EURUSD, GOLD = "CS.D.GBPUSD.TODAY.IP", "CS.D.EURUSD.TODAY.IP", "CS.D.USCGC.TODAY.IP"


def _loaded(rows) -> EconomicCalendar:
    cal = EconomicCalendar(buffer_minutes=30)
    resp = MagicMock(status_code=200)
    resp.json.return_value = rows
    with patch("src.calendar.requests.get", return_value=resp):
        assert cal.refresh()
    return cal


def _near(currency: str, minutes: float = 10) -> EconomicCalendar:
    cal = EconomicCalendar(buffer_minutes=30)
    cal.events = [EconomicEvent("Test", currency, currency, "High",
                                datetime.now(timezone.utc) + timedelta(minutes=minutes))]
    cal.last_fetch = datetime.now()          # fresh, so no refresh is attempted
    return cal


class FeedParsing(unittest.TestCase):

    def test_real_feed_parses_every_high_impact_row(self):
        rows = json.loads(FIXTURE.read_text())
        high = [r for r in rows if r["impact"] == "High"]
        cal = _loaded(rows)
        self.assertEqual(len(high), 7)
        self.assertEqual(len(cal.events), 7)
        for e in cal.events:
            self.assertEqual(e.event_time.utcoffset(), timedelta(0))

    def test_offset_is_honoured(self):
        cal = _loaded(json.loads(FIXTURE.read_text()))
        aud = next(e for e in cal.events if e.title == "Employment Change")
        # 2026-09-23T21:30:00-04:00
        self.assertEqual(aud.event_time, datetime(2026, 9, 24, 1, 30, tzinfo=timezone.utc))

    def test_winter_offset(self):
        cal = EconomicCalendar()
        t = cal._parse_event_time("2026-01-09T08:30:00-05:00")
        self.assertEqual(t, datetime(2026, 1, 9, 13, 30, tzinfo=timezone.utc))

    def test_legacy_split_format_uses_real_eastern_offset(self):
        cal = EconomicCalendar()
        # July: ET is UTC-4, which the old fixed +5h got wrong by an hour
        self.assertEqual(cal._parse_event_time("2026-07-02", "8:30am"),
                         datetime(2026, 7, 2, 12, 30, tzinfo=timezone.utc))
        self.assertIsNone(cal._parse_event_time("2026-07-02", "All Day"))
        self.assertIsNone(cal._parse_event_time("2026-07-02", ""))

    def test_zero_parsed_is_a_warning_not_a_quiet_week(self):
        rows = [{"title": "NFP", "country": "USD", "date": "garbage", "impact": "High"}]
        with self.assertLogs("src.calendar", level="WARNING") as cm:
            _loaded(rows)
        self.assertIn("parsed 0 of 1", " ".join(cm.output))

    def test_failed_fetch_keeps_events_and_backs_off(self):
        cal = _near("USD", 5)
        cal.last_fetch = datetime.now() - timedelta(hours=7)   # stale
        with patch("src.calendar.requests.get", return_value=MagicMock(status_code=429)) as g:
            self.assertIsNotNone(cal.would_block(GOLD))
            self.assertIsNotNone(cal.would_block(GOLD))
        self.assertEqual(g.call_count, 1)                       # retry backoff
        self.assertEqual(len(cal.events), 1)                    # old events kept


class Mapping(unittest.TestCase):

    def test_gbpusd_is_mapped_to_both_sides(self):
        self.assertIn(GBPUSD, CURRENCY_EPIC_MAP["USD"])
        self.assertIn(GBPUSD, CURRENCY_EPIC_MAP["GBP"])
        self.assertIsNotNone(_near("GBP").would_block(GBPUSD))
        self.assertIsNotNone(_near("USD").would_block(GBPUSD))
        self.assertIsNone(_near("GBP").would_block(EURUSD))

    def test_window(self):
        self.assertIsNotNone(_near("USD", 29).would_block(GOLD))
        self.assertIsNone(_near("USD", 31).would_block(GOLD))
        self.assertIsNotNone(_near("USD", -29).would_block(GOLD))

    def test_no_network_when_refresh_disallowed(self):
        cal = EconomicCalendar()
        with patch("src.calendar.requests.get") as g:
            self.assertIsNone(cal.would_block(GOLD, allow_refresh=False))
        g.assert_not_called()


class LogOnlyContract(unittest.TestCase):
    """main.py: the calendar never blocks unless CALENDAR_ENFORCE is on, and the
    breakout path never blocks at all."""

    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main

    def setUp(self):
        self._saved = (self.main.calendar, self.main.CALENDAR_ENFORCE)
        self.mc = MagicMock(); self.mc.name = "Gold"

    def tearDown(self):
        self.main.calendar, self.main.CALENDAR_ENFORCE = self._saved

    def test_momentum_logs_and_carries_on_by_default(self):
        self.main.calendar, self.main.CALENDAR_ENFORCE = _near("USD"), False
        with patch.object(self.main, "_log_suppressed_signal") as sup, \
                self.assertLogs(self.main.logger, level="INFO") as cm:
            self.assertFalse(self.main._calendar_blocks_momentum(GOLD, self.mc, None, None))
        sup.assert_not_called()
        self.assertIn("Calendar (log-only): would block Gold", " ".join(cm.output))

    def test_momentum_blocks_only_when_enforced(self):
        self.main.calendar, self.main.CALENDAR_ENFORCE = _near("USD"), True
        with patch.object(self.main, "_log_suppressed_signal") as sup:
            self.assertTrue(self.main._calendar_blocks_momentum(GOLD, self.mc, None, None))
        sup.assert_called_once()

    def test_momentum_no_event_no_block(self):
        self.main.calendar, self.main.CALENDAR_ENFORCE = _near("USD", 120), True
        self.assertFalse(self.main._calendar_blocks_momentum(GOLD, self.mc, None, None))

    def test_breakout_note_never_raises_or_fetches(self):
        broken = MagicMock()
        broken.would_block.side_effect = RuntimeError("boom")
        self.main.calendar = broken
        self.main._calendar_note_breakout(GOLD, self.mc)          # must not raise
        self.main.calendar = _near("USD")
        with patch("src.calendar.requests.get") as g, \
                self.assertLogs(self.main.logger, level="INFO") as cm:
            self.main._calendar_note_breakout(GOLD, self.mc)
        g.assert_not_called()
        self.assertIn("would block breakout Gold", " ".join(cm.output))


if __name__ == "__main__":
    unittest.main()
