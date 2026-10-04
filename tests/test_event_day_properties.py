"""Tagging properties for scripts/replay_event_day.py.

Pre-registration §8: an entry's event-day tag depends only on its decision time
and the event table (never a price), a rotated control day is never a real event
day, and a decision at 23:59:59 vs 00:00:00 UTC lands on the right date.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from tests.pbt import settings, st  # noqa: E402  (Hypothesis profiles; skips in container)
from hypothesis import given  # noqa: E402

import replay_event_day as red  # noqa: E402

BASE = np.datetime64("2010-01-04", "D")


def _days(offsets) -> np.ndarray:
    return np.unique(BASE + np.array(sorted(set(offsets)), dtype="timedelta64[D]"))


def _decisions(seconds) -> pd.Series:
    return pd.Series(pd.to_datetime(np.datetime64("2010-01-04T00:00:00", "s")
                                    + np.array(seconds, dtype="timedelta64[s]")).tz_localize("UTC"))


class TestEventDayTagging(unittest.TestCase):

    @given(st.lists(st.integers(0, 400), min_size=1, max_size=60),
           st.lists(st.integers(0, 400 * 86400), min_size=1, max_size=80))
    @settings()
    def test_tag_matches_utc_date_membership(self, offsets, seconds):
        days = _days(offsets)
        dec = _decisions(seconds)
        got = red.tag_event_day(dec, days)
        want = np.array([np.datetime64(ts.strftime("%Y-%m-%d"), "D") in set(days) for ts in dec])
        np.testing.assert_array_equal(got, want)

    @given(st.lists(st.integers(0, 400), min_size=1, max_size=60), st.sampled_from(red.SHIFTS))
    @settings()
    def test_rotated_day_is_never_a_real_event_day(self, offsets, k):
        real = _days(offsets)
        sh = red.shifted_days(real, k)
        self.assertFalse(np.isin(sh, real).any())
        self.assertTrue(np.isin(sh - np.timedelta64(7 * k, "D"), real).all())

    def test_midnight_boundary(self):
        days = np.array([np.datetime64("2010-01-05", "D")])
        dec = pd.Series(pd.to_datetime(["2010-01-04 23:59:59", "2010-01-05 00:00:00",
                                        "2010-01-05 23:59:59", "2010-01-06 00:00:00"]).tz_localize("UTC"))
        np.testing.assert_array_equal(red.tag_event_day(dec, days), [False, True, True, False])

    def test_bst_decision_uses_utc_date(self):
        # 00:30 London in summer is 23:30 UTC the previous day.
        days = np.array([np.datetime64("2020-07-01", "D")])
        dec = pd.Series(pd.to_datetime(["2020-07-02 00:30"]).tz_localize("Europe/London"))
        np.testing.assert_array_equal(red.tag_event_day(dec, days), [True])


if __name__ == "__main__":
    unittest.main()
