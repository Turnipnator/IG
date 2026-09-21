"""Disk-cache staleness, judged on missed trading rather than wall-clock.

The old test asked how long ago the FILE was written. That number stayed
permanently fresh, because save_candles_to_disk kept rewriting the same frozen
candles every 15 minutes for the whole 62-hour outage of 2026-09-18 — so the bot
restored nearly-three-day-old candles while logging "age: 17m", and spliced
Monday's prices onto Friday's. The +0.80% step that created pegged RSI(7) near
100 for hours.

The replacement asks a different question: did the market TRADE during the gap?

Both directions cost something real, which is why both are pinned here:

  * too lax  — stale candles seed every indicator, and the distortion is silent
  * too eager — each unnecessary refetch is ~680 API points against a 10,000
                point weekly allowance, and over a weekend it re-downloads bars
                identical to the ones already cached

So the headline property is not "does it catch the outage" (easy) but "does a
normal weekend still cost nothing" (the one that protects the budget).

Window values below are the real ones from config.py: S&P 4->20, FTSE 8->17,
Japan 0->8, Gold 23->21 (wraps midnight), GBP/USD 7->21.
"""

import unittest
from datetime import datetime, timedelta

from src.streaming import (
    CACHE_MAX_AGE_DAYS, CACHE_MAX_MISSING_BARS, IGStreamService, MarketStream,
    expected_bars_between, in_session,
)

# 2026-09-18 is a Friday; 09-19 Sat, 09-20 Sun, 09-21 Mon.
FRI_1925 = datetime(2026, 9, 18, 19, 25)
MON_1100 = datetime(2026, 9, 21, 11, 0)


class TestInSession(unittest.TestCase):

    def test_plain_window(self):
        self.assertTrue(in_session(4, 20, datetime(2026, 9, 21, 10)))   # Mon 10:00
        self.assertFalse(in_session(4, 20, datetime(2026, 9, 21, 3)))   # before open
        self.assertFalse(in_session(4, 20, datetime(2026, 9, 21, 20)))  # end exclusive

    def test_wrapped_window_spans_midnight(self):
        """Gold/Crude/DXY/EUR-USD are configured 23 -> 21."""
        self.assertTrue(in_session(23, 21, datetime(2026, 9, 21, 23)))  # Mon 23:00
        self.assertTrue(in_session(23, 21, datetime(2026, 9, 21, 2)))   # Mon 02:00
        self.assertFalse(in_session(23, 21, datetime(2026, 9, 21, 22)))  # the gap hour

    def test_saturday_is_shut_for_everything(self):
        for start, end in ((4, 20), (23, 21), (0, 8)):
            for hour in range(24):
                self.assertFalse(
                    in_session(start, end, datetime(2026, 9, 19, hour)),
                    f"Saturday {hour:02d}:00 counted as open for window {start}-{end}",
                )

    def test_sunday_only_the_late_reopen_counts(self):
        # Forex/commodities reopen 22:00 UTC Sunday; a 23->21 window reaches it.
        self.assertTrue(in_session(23, 21, datetime(2026, 9, 20, 23)))
        self.assertFalse(in_session(23, 21, datetime(2026, 9, 20, 12)))
        # An equity window does not reach 22:00, so Sunday stays shut.
        self.assertFalse(in_session(4, 20, datetime(2026, 9, 20, 23)))


class TestExpectedBars(unittest.TestCase):

    def test_weekend_gap_costs_nothing(self):
        """THE budget property: Friday close -> Monday open, no bars missed."""
        fri_close = datetime(2026, 9, 18, 20, 0)   # S&P window ends 20:00
        mon_open = datetime(2026, 9, 21, 4, 0)     # and reopens 04:00
        self.assertEqual(expected_bars_between(4, 20, 5, fri_close, mon_open), 0)

    def test_outage_across_a_live_session_is_counted(self):
        bars = expected_bars_between(4, 20, 5, FRI_1925, MON_1100)
        # Mon 04:00-11:00 = 7h of 5m bars, plus Fri 19:25-20:00.
        self.assertGreater(bars, 80)

    def test_a_short_gap_inside_one_session_is_small(self):
        t0 = datetime(2026, 9, 21, 10, 0)
        self.assertEqual(expected_bars_between(4, 20, 5, t0, t0 + timedelta(minutes=30)), 6)

    def test_interval_scales_the_count(self):
        t0 = datetime(2026, 9, 21, 4, 0)
        t1 = datetime(2026, 9, 21, 10, 0)
        self.assertEqual(expected_bars_between(4, 20, 5, t0, t1), 72)
        self.assertEqual(expected_bars_between(4, 20, 60, t0, t1), 6)

    def test_end_before_start_is_zero(self):
        self.assertEqual(expected_bars_between(4, 20, 5, MON_1100, FRI_1925), 0)

    def test_closed_market_gap_is_zero_even_over_days(self):
        """Japan (0->8) over a weekend: nothing missed."""
        fri = datetime(2026, 9, 18, 8, 0)
        sun = datetime(2026, 9, 20, 20, 0)
        self.assertEqual(expected_bars_between(0, 8, 5, fri, sun), 0)


def _service(market: MarketStream = None):
    svc = IGStreamService.__new__(IGStreamService)
    svc.markets = {"E": market} if market else {}
    return svc


def _item(newest: datetime, n: int = 100, interval: int = 5, saved_at=None):
    step = timedelta(minutes=interval)
    candles = [
        {"timestamp": (newest - step * (n - 1 - i)).isoformat(),
         "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0}
        for i in range(n)
    ]
    return {
        "name": "Test", "candle_interval": interval,
        "saved_at": (saved_at or datetime.now()).isoformat(),
        "candles": candles,
    }


class TestCacheDecision(unittest.TestCase):

    def _market(self, start=4, end=20, interval=5):
        return MarketStream(epic="E", name="Test", candle_interval=interval,
                            trading_start=start, trading_end=end)

    def test_the_2026_09_18_outage_is_rejected(self):
        """THE regression. Old code accepted this, logging 'age: 17m'."""
        svc = _service(self._market())
        item = _item(FRI_1925, saved_at=MON_1100 - timedelta(minutes=17))
        with _frozen_now(MON_1100):
            usable, reason = svc._cache_is_usable(item, svc.markets["E"])
        self.assertFalse(usable, reason)
        self.assertIn("in-session bars missing", reason)

    def test_a_fresh_saved_at_cannot_rescue_stale_candles(self):
        """The writer kept running for 62h; saved_at must carry no weight."""
        svc = _service(self._market())
        item = _item(FRI_1925, saved_at=MON_1100)   # written seconds ago
        with _frozen_now(MON_1100):
            usable, _ = svc._cache_is_usable(item, svc.markets["E"])
        self.assertFalse(usable)

    def test_normal_weekend_restart_keeps_the_cache(self):
        """The budget property: no refetch, so no ~680 points spent."""
        svc = _service(self._market())
        fri_close = datetime(2026, 9, 18, 19, 55)
        with _frozen_now(datetime(2026, 9, 21, 4, 5)):   # Monday, just after open
            usable, reason = svc._cache_is_usable(_item(fri_close), svc.markets["E"])
        self.assertTrue(usable, reason)

    def test_a_brief_restart_mid_session_keeps_the_cache(self):
        svc = _service(self._market())
        now = datetime(2026, 9, 21, 11, 0)
        with _frozen_now(now):
            usable, _ = svc._cache_is_usable(_item(now - timedelta(minutes=15)),
                                             svc.markets["E"])
        self.assertTrue(usable)

    def test_the_threshold_is_where_it_says_it_is(self):
        svc = _service(self._market())
        now = datetime(2026, 9, 21, 11, 0)
        just_under = now - timedelta(minutes=5 * CACHE_MAX_MISSING_BARS)
        well_over = now - timedelta(minutes=5 * (CACHE_MAX_MISSING_BARS + 20))
        with _frozen_now(now):
            self.assertTrue(svc._cache_is_usable(_item(just_under), svc.markets["E"])[0])
            self.assertFalse(svc._cache_is_usable(_item(well_over), svc.markets["E"])[0])

    def test_abandoned_file_backstop(self):
        """Even if session maths says zero, an ancient file is junk."""
        svc = _service(self._market(start=0, end=8))   # Japan: shut most of the time
        now = datetime(2026, 9, 21, 11, 0)
        ancient = now - timedelta(days=CACHE_MAX_AGE_DAYS + 3)
        with _frozen_now(now):
            usable, reason = svc._cache_is_usable(_item(ancient), svc.markets["E"])
        self.assertFalse(usable)
        self.assertIn("backstop", reason)

    def test_unreadable_timestamp_is_rejected_not_crashed(self):
        svc = _service(self._market())
        item = _item(MON_1100)
        item["candles"][-1]["timestamp"] = "not-a-date"
        usable, reason = svc._cache_is_usable(item, svc.markets["E"])
        self.assertFalse(usable)
        self.assertIn("unreadable", reason)

    def test_empty_cache_is_rejected(self):
        svc = _service(self._market())
        self.assertFalse(svc._cache_is_usable({"candles": []}, svc.markets["E"])[0])

    def test_routine_restarts_stay_free(self):
        """Cost control: the shapes that happen weekly must not refetch.

        Measured against the live config when the threshold was chosen. If a
        change makes one of these refetch, it is spending ~50 points per market
        on data the cache already holds — check the table in src/streaming.py
        before adjusting the threshold to make this pass.
        """
        svc = _service(self._market())
        cases = {
            "10-minute deploy": (datetime(2026, 9, 21, 10, 54), datetime(2026, 9, 21, 11, 4)),
            "1h mid-session":   (datetime(2026, 9, 21, 10, 0), datetime(2026, 9, 21, 11, 0)),
            "weekend -> Sunday": (datetime(2026, 9, 18, 20, 0), datetime(2026, 9, 20, 12, 0)),
        }
        for label, (cache_time, restart) in cases.items():
            with _frozen_now(restart):
                usable, reason = svc._cache_is_usable(_item(cache_time), svc.markets["E"])
            self.assertTrue(usable, f"{label} would refetch unnecessarily: {reason}")

    def test_multi_hour_session_gaps_still_refetch(self):
        """The other side of the dial — accuracy must not be traded away."""
        svc = _service(self._market())
        with _frozen_now(datetime(2026, 9, 21, 10, 0)):
            usable, _ = svc._cache_is_usable(
                _item(datetime(2026, 9, 21, 4, 0)), svc.markets["E"])
        self.assertFalse(usable, "a 6h hole in a live session must not seed indicators")

    def test_unknown_market_falls_back_to_age_only(self):
        """No window to reason with — accept unless the backstop trips."""
        svc = _service()
        with _frozen_now(MON_1100):
            self.assertTrue(svc._cache_is_usable(_item(FRI_1925), None)[0])


class _frozen_now:
    """Pin datetime.now() inside src.streaming for a deterministic test."""

    def __init__(self, when):
        self.when = when

    def __enter__(self):
        import src.streaming as s
        self._real = s.datetime
        when = self.when

        class _DT(s.datetime):
            @classmethod
            def now(cls, tz=None):
                return when

        s.datetime = _DT
        return self

    def __exit__(self, *exc):
        import src.streaming as s
        s.datetime = self._real
        return False


if __name__ == "__main__":
    unittest.main()
