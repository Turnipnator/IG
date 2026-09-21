"""The container healthcheck must notice a frozen feed — and nothing else.

Replaces a check that fetched demo-api.ig.com and therefore reported `healthy`
for all 62 hours of the 2026-09-18 outage. The risk in fixing it is the
opposite failure: a check that cries unhealthy on a quiet Friday night gets
muted, and then the next real outage is invisible again. So most of what
follows is about staying quiet.
"""

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import docker_healthcheck as hc


class TestMarketWindow(unittest.TestCase):
    """Coarse on purpose — the edges decide whether a human gets woken."""

    def _at(self, iso):
        return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)

    def test_midweek_is_open(self):
        self.assertTrue(hc.markets_should_be_open(self._at("2026-09-23T14:00")))

    def test_saturday_is_closed(self):
        self.assertFalse(hc.markets_should_be_open(self._at("2026-09-19T14:00")))

    def test_friday_evening_is_closed(self):
        # The 09-18 outage began 19:31 BST on a Friday; the window must not
        # fire then, which is exactly why the in-process watchdog is primary.
        self.assertFalse(hc.markets_should_be_open(self._at("2026-09-18T21:30")))

    def test_friday_afternoon_is_open(self):
        self.assertTrue(hc.markets_should_be_open(self._at("2026-09-18T15:00")))

    def test_sunday_daytime_is_closed(self):
        self.assertFalse(hc.markets_should_be_open(self._at("2026-09-20T12:00")))

    def test_sunday_late_evening_is_open(self):
        self.assertTrue(hc.markets_should_be_open(self._at("2026-09-20T23:30")))


class TestCheck(unittest.TestCase):

    def setUp(self):
        self.tmp = mock.patch.object(hc, "CANDLE_CACHE").start()
        self.log = mock.patch.object(hc, "LOG_FILE").start()
        # Default to a long-running process; the boot-grace tests override it.
        mock.patch.object(hc, "process_uptime",
                          return_value=timedelta(hours=5)).start()
        self.addCleanup(mock.patch.stopall)
        self.log.exists.return_value = False  # log age tested separately

    def _cache(self, newest: datetime | None):
        self.tmp.exists.return_value = newest is not None
        if newest is not None:
            self.tmp.read_text.return_value = json.dumps(
                {"IX.D.FTSE.DAILY.IP": {"candles": [
                    {"timestamp": newest.isoformat(), "open": 1, "high": 1,
                     "low": 1, "close": 1, "volume": 0}
                ]}}
            )

    def test_frozen_feed_during_a_session_is_unhealthy(self):
        """THE regression: 09-21 09:37, newest candle from 09-18 19:25."""
        self._cache(datetime.now() - timedelta(days=2, hours=14))
        with mock.patch.object(hc, "markets_should_be_open", return_value=True):
            ok, detail = hc.check()
        self.assertFalse(ok, detail)
        self.assertIn("feed frozen", detail)

    def test_frozen_feed_at_the_weekend_is_healthy(self):
        self._cache(datetime.now() - timedelta(days=2, hours=14))
        with mock.patch.object(hc, "markets_should_be_open", return_value=False):
            ok, _ = hc.check()
        self.assertTrue(ok, "a closed market is silent; that is not a fault")

    def test_live_feed_is_healthy(self):
        self._cache(datetime.now() - timedelta(minutes=3))
        with mock.patch.object(hc, "markets_should_be_open", return_value=True):
            ok, _ = hc.check()
        self.assertTrue(ok)

    def test_cold_start_without_a_cache_is_healthy(self):
        self._cache(None)
        with mock.patch.object(hc, "markets_should_be_open", return_value=True):
            ok, _ = hc.check()
        self.assertTrue(ok, "a missing cache right after boot is not an outage")

    def test_silent_log_is_unhealthy(self):
        import time
        self.log.exists.return_value = True
        self.log.stat.return_value = mock.Mock(st_mtime=time.time() - 3600)
        ok, detail = hc.check()
        self.assertFalse(ok)
        self.assertIn("no log output", detail)

    def test_a_broken_cache_file_never_fails_the_container(self):
        """A bug in the healthcheck must not restart a working bot."""
        self.tmp.exists.return_value = True
        self.tmp.read_text.return_value = "{not json"
        with mock.patch.object(hc, "markets_should_be_open", return_value=True):
            with self.assertRaises(Exception):
                hc.check()
            self.assertEqual(hc.main(), 0, "main() must swallow what check() raises")

    def test_stale_cache_just_after_a_restart_is_not_an_outage(self):
        """Observed live 2026-09-21: 3 false failures after a healthy restart.

        Persistence is `every(15).minutes` and uptime-relative, so the cache
        still holds the previous run's candles for up to ~15 min. Treating that
        as a frozen feed alarms on EVERY restart during market hours.
        """
        self._cache(datetime.now() - timedelta(days=2, hours=15))
        with mock.patch.object(hc, "markets_should_be_open", return_value=True), \
             mock.patch.object(hc, "process_uptime", return_value=timedelta(minutes=3)):
            ok, detail = hc.check()
        self.assertTrue(ok, detail)
        self.assertIn("boot grace", detail)

    def test_the_same_stale_cache_IS_an_outage_once_settled(self):
        """The grace must expire — otherwise it just hides the real thing."""
        self._cache(datetime.now() - timedelta(days=2, hours=15))
        with mock.patch.object(hc, "markets_should_be_open", return_value=True), \
             mock.patch.object(hc, "process_uptime", return_value=timedelta(minutes=45)):
            ok, detail = hc.check()
        self.assertFalse(ok)
        self.assertIn("feed frozen", detail)

    def test_unreadable_uptime_does_not_disable_the_check(self):
        """Failing open here would silently switch off the whole point."""
        self._cache(datetime.now() - timedelta(days=2))
        with mock.patch.object(hc, "markets_should_be_open", return_value=True), \
             mock.patch.object(hc, "process_uptime", return_value=None):
            ok, _ = hc.check()
        self.assertFalse(ok)

    def test_a_dead_process_still_fails_inside_the_boot_grace(self):
        """Boot grace covers candle age only — not a bot that stopped logging."""
        import time as _t
        self.log.exists.return_value = True
        self.log.stat.return_value = mock.Mock(st_mtime=_t.time() - 3600)
        with mock.patch.object(hc, "process_uptime", return_value=timedelta(minutes=3)):
            ok, detail = hc.check()
        self.assertFalse(ok)
        self.assertIn("no log output", detail)

    def test_main_returns_one_on_a_real_outage(self):
        self._cache(datetime.now() - timedelta(days=2))
        with mock.patch.object(hc, "markets_should_be_open", return_value=True):
            self.assertEqual(hc.main(), 1)


if __name__ == "__main__":
    unittest.main()
