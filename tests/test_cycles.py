"""Tests for the PFO cycle-window lookup (src/cycles.py).

These guard a MEASUREMENT, not a trade: if the lookup mislabels days, every
in/out-of-window comparison built on it is wrong in a way that looks plausible.
The tolerance and coverage tests exist because both are easy to get subtly
wrong and neither would raise.
"""
import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from src.cycles import (CycleState, INSTRUMENT_EPICS, cycle_state, load_cycles,
                        covered_range)

SPX = "IX.D.SPTRD.DAILY.IP"


def _sheet(tmp: Path, name: str, crosses, weeks, vol=None, month="2026-06"):
    (tmp / name).write_text(json.dumps({
        "month": month,
        "instruments": {"US Equities": {
            "crosses": crosses, "weeks": weeks, "volatility": vol or []}}}))


class TestCycleLookup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cross_applies_plus_minus_one_day(self):
        """The report says a cross 'can happen day before or after'."""
        _sheet(self.dir, "a.json", ["2026-06-10"], [])
        c = load_cycles(self.dir, cross_tolerance_days=1)[SPX]
        for d in ("2026-06-09", "2026-06-10", "2026-06-11"):
            self.assertTrue(c[date.fromisoformat(d)].cross, d)
        self.assertNotIn(date(2026, 6, 8), c)
        self.assertNotIn(date(2026, 6, 12), c)

    def test_tolerance_zero_marks_only_the_stated_day(self):
        _sheet(self.dir, "a.json", ["2026-06-10"], [])
        c = load_cycles(self.dir, cross_tolerance_days=0)[SPX]
        self.assertEqual(set(c), {date(2026, 6, 10)})

    def test_weeks_are_not_widened_by_tolerance(self):
        """Bands are already multi-day; widening both would swallow the month."""
        _sheet(self.dir, "a.json", [], [["2026-06-15", "2026-06-19"]])
        c = load_cycles(self.dir, cross_tolerance_days=1)[SPX]
        self.assertEqual(min(c), date(2026, 6, 15))
        self.assertEqual(max(c), date(2026, 6, 19))
        self.assertEqual(len(c), 5)

    def test_strong_requires_both_cross_and_week(self):
        _sheet(self.dir, "a.json", ["2026-06-16"], [["2026-06-15", "2026-06-19"]])
        c = load_cycles(self.dir, cross_tolerance_days=1)[SPX]
        self.assertTrue(c[date(2026, 6, 16)].strong)
        self.assertEqual(c[date(2026, 6, 16)].label, "strong")
        # in the band but no cross -> week only
        self.assertFalse(c[date(2026, 6, 19)].strong)
        self.assertEqual(c[date(2026, 6, 19)].label, "week")
        # cross spilling outside the band -> cross only
        self.assertEqual(c[date(2026, 6, 15)].label, "strong")

    def test_sheets_merge_rather_than_collide(self):
        _sheet(self.dir, "a.json", ["2026-06-10"], [], month="2026-06")
        _sheet(self.dir, "b.json", [], [["2026-06-10", "2026-06-10"]], month="2026-07")
        c = load_cycles(self.dir, cross_tolerance_days=0)[SPX]
        self.assertTrue(c[date(2026, 6, 10)].strong, "merge lost a flag")

    def test_unknown_epic_and_date_are_quiet(self):
        st = cycle_state("NO.SUCH.EPIC", datetime(2026, 6, 10))
        self.assertFalse(st.any)
        self.assertEqual(st.label, "none")

    def test_datetime_and_string_accepted(self):
        _sheet(self.dir, "a.json", ["2026-06-10"], [])
        c = load_cycles(self.dir, cross_tolerance_days=0)[SPX]
        self.assertIn(date(2026, 6, 10), c)
        self.assertEqual(CycleState(), CycleState())

    def test_torn_sheet_is_skipped_not_fatal(self):
        """Transcriptions of photos; one bad file must not kill the run."""
        (self.dir / "bad.json").write_text("{not json")
        _sheet(self.dir, "good.json", ["2026-06-10"], [])
        c = load_cycles(self.dir, cross_tolerance_days=0)
        self.assertIn(SPX, c)

    def test_unmapped_instrument_ignored(self):
        (self.dir / "a.json").write_text(json.dumps({
            "month": "2026-06",
            "instruments": {"Soybeans": {"crosses": ["2026-06-10"], "weeks": []}}}))
        self.assertEqual(load_cycles(self.dir), {})

    def test_covered_range_spans_whole_months(self):
        _sheet(self.dir, "a.json", [], [], month="2026-06")
        _sheet(self.dir, "b.json", [], [], month="2026-08")
        self.assertEqual(covered_range(self.dir),
                         (date(2026, 6, 1), date(2026, 8, 31)))

    def test_us_equities_maps_to_sp500_only(self):
        """Confirmed 2026-09-01. Mapping it to the other index EPICs would
        silently multiply the sample with markets the sheet does not cover."""
        self.assertEqual(INSTRUMENT_EPICS["US Equities"], (SPX,))


class TestRealSheets(unittest.TestCase):
    """The committed sheets must parse and cover the period we claim to test."""

    def test_real_sheets_load_and_cover_jun_to_sep(self):
        c = load_cycles()
        self.assertIn(SPX, c, "S&P missing from the real sheets")
        rng = covered_range()
        self.assertIsNotNone(rng)
        self.assertEqual(rng[0], date(2026, 6, 1))
        self.assertGreaterEqual(rng[1], date(2026, 9, 30))

    def test_every_mapped_epic_present(self):
        c = load_cycles()
        for inst, epics in INSTRUMENT_EPICS.items():
            for e in epics:
                self.assertIn(e, c, f"{inst} -> {e} produced no days")


if __name__ == "__main__":
    unittest.main()
