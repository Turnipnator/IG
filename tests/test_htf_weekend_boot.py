"""update_htf_trends / _htf_staleness_guard: weekend boot with a stale marker.

Regression for 2026-09-05: a Saturday rebuild found last_htf_refresh.txt 16.5h old,
fell past the 6h cooldown branch into the fetch loop, every REST call was
weekend-skipped, the in-memory dict stayed EMPTY (Friday's labels sat unread on
disk), the regime defaulted BULLISH-unconfirmed against a real BEARISH S&P, and
because nothing was labelled the marker was never rewritten -- so from marker+25h
_htf_staleness_guard forced the 14-market loop every 60s until Monday (474 fires).

Three properties are pinned here:
  1. a pass that labels nothing restores the persisted labels (weekend or weekday);
  2. only a WEEKEND restore rewrites the marker -- a weekday API failure must leave
     it stale so the guard keeps retrying;
  3. the guard is throttled to one forced attempt per HTF_GUARD_RETRY.
"""
import json
import logging
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import main


@dataclass
class _Cfg:
    epic: str
    name: str
    htf_resolution: str = "HOUR"


class _Client:
    """Stub IGClient: returns a pre-baked frame per epic, None otherwise."""

    def __init__(self, weekend: bool, frames=None):
        self.weekend = weekend
        self.frames = frames or {}
        self.calls = 0

    def is_weekend(self) -> bool:
        return self.weekend

    def get_historical_prices(self, epic, resolution="HOUR", num_points=30, use_cache=True):
        self.calls += 1
        return self.frames.get(epic)


class _Limiter:
    def wait_if_needed(self):
        pass


MARKETS = [
    _Cfg("IX.D.SPTRD.DAILY.IP", "S&P 500"),
    _Cfg("IX.D.NASDAQ.CASH.IP", "NASDAQ 100"),
    _Cfg("CS.D.USCGC.TODAY.IP", "Gold"),
    _Cfg("CC.D.CL.USS.IP", "Crude Oil"),
]
FRIDAY_LABELS = {
    "IX.D.SPTRD.DAILY.IP": "BEARISH",
    "IX.D.NASDAQ.CASH.IP": "BEARISH",
    "CS.D.USCGC.TODAY.IP": "NEUTRAL",
    "CC.D.CL.USS.IP": "BULLISH",
}


def _rising_frame(n: int = 30) -> pd.DataFrame:
    close = [100.0 + i for i in range(n)]
    return pd.DataFrame({
        "date": pd.date_range("2026-09-01", periods=n, freq="h"),
        "open": close, "high": [c + 1 for c in close], "low": [c - 1 for c in close],
        "close": close, "volume": [1] * n,
    })


class HtfWeekendBootTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self._saved = {k: getattr(main, k) for k in (
            "client", "rate_limiter", "MARKETS", "LAST_HTF_REFRESH_FILE", "HTF_TRENDS_FILE",
            "market_regime", "market_regime_confirmed", "_observe_archive_htf",
            "_htf_guard_last_attempt", "update_htf_trends")}
        self._saved_trends = dict(main.htf_trends)
        main.htf_trends.clear()
        main.MARKETS = MARKETS
        main.rate_limiter = _Limiter()
        main.LAST_HTF_REFRESH_FILE = d / "last_htf_refresh.txt"
        main.HTF_TRENDS_FILE = d / "htf_trends.json"
        main.market_regime = "BULLISH"
        main.market_regime_confirmed = False
        main._htf_guard_last_attempt = None
        self.xchecks = []
        main._observe_archive_htf = lambda force=False, tag="drift": self.xchecks.append(tag)
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        for k, v in self._saved.items():
            setattr(main, k, v)
        main.htf_trends.clear()
        main.htf_trends.update(self._saved_trends)
        self.tmp.cleanup()

    def _persist(self, marker_age_hours: float):
        main.LAST_HTF_REFRESH_FILE.write_text(
            (datetime.now() - timedelta(hours=marker_age_hours)).isoformat())
        main.HTF_TRENDS_FILE.write_text(json.dumps(FRIDAY_LABELS))

    def _marker_age_hours(self) -> float:
        last = datetime.fromisoformat(main.LAST_HTF_REFRESH_FILE.read_text().strip())
        return (datetime.now() - last).total_seconds() / 3600

    # 1. The 2026-09-05 scenario, exactly.
    def test_weekend_boot_with_stale_marker_restores_friday_labels(self):
        main.client = _Client(weekend=True)
        self._persist(marker_age_hours=30)

        main.update_htf_trends()  # force=False: 30h > 6h cooldown -> fetch loop

        self.assertEqual(dict(main.htf_trends), FRIDAY_LABELS)
        self.assertEqual(main.market_regime, "BEARISH")
        self.assertTrue(main.market_regime_confirmed)
        self.assertLess(self._marker_age_hours(), 0.01, "weekend restore must rewrite the marker")
        self.assertEqual(self.xchecks, [], "nothing fetched -> no 'at-refresh' sample")

    # 2. Weekend guard: one forced pass heals the marker, the next call is a no-op.
    def test_weekend_guard_heals_marker_and_stops(self):
        main.client = _Client(weekend=True)
        self._persist(marker_age_hours=30)

        main._htf_staleness_guard()
        self.assertEqual(main.client.calls, len(MARKETS))
        self.assertLess(self._marker_age_hours(), 0.01)

        main._htf_staleness_guard()
        self.assertEqual(main.client.calls, len(MARKETS), "fresh marker -> no second refresh")

    # 3. Weekday failure: labels restored, but the marker stays stale so retries continue.
    def test_weekday_fetch_failure_restores_labels_but_keeps_marker_stale(self):
        main.client = _Client(weekend=False)
        self._persist(marker_age_hours=30)

        main.update_htf_trends(force=True)

        self.assertEqual(dict(main.htf_trends), FRIDAY_LABELS)
        self.assertEqual(main.market_regime, "BEARISH")
        self.assertTrue(main.market_regime_confirmed)
        self.assertGreater(self._marker_age_hours(), 29.9, "weekday restore must NOT count as a refresh")
        self.assertEqual(self.xchecks, [])

    # 4. Partial fetch: fresh labels win, gaps are back-filled, no marker, x-check fires.
    def test_partial_fetch_keeps_fresh_labels_and_fills_gaps(self):
        main.client = _Client(weekend=False, frames={"IX.D.SPTRD.DAILY.IP": _rising_frame()})
        self._persist(marker_age_hours=30)

        main.update_htf_trends(force=True)

        self.assertEqual(main.htf_trends["IX.D.SPTRD.DAILY.IP"], "BULLISH", "fresh label must not be overwritten")
        for epic in ("IX.D.NASDAQ.CASH.IP", "CS.D.USCGC.TODAY.IP", "CC.D.CL.USS.IP"):
            self.assertEqual(main.htf_trends[epic], FRIDAY_LABELS[epic])
        self.assertEqual(main.market_regime, "BULLISH")
        self.assertGreater(self._marker_age_hours(), 29.9, "1/4 labelled is below half -> no marker")
        self.assertEqual(self.xchecks, ["at-refresh"])

    # 5. Full weekday fetch: marker written, as before.
    def test_full_weekday_fetch_writes_marker(self):
        main.client = _Client(weekend=False, frames={m.epic: _rising_frame() for m in MARKETS})
        self._persist(marker_age_hours=30)

        main.update_htf_trends(force=True)

        self.assertTrue(all(v == "BULLISH" for v in main.htf_trends.values()))
        self.assertLess(self._marker_age_hours(), 0.01)
        self.assertEqual(json.loads(main.HTF_TRENDS_FILE.read_text()), dict(main.htf_trends))

    # 6. Guard throttle: bounded to one forced attempt per HTF_GUARD_RETRY.
    def test_guard_throttles_forced_refresh(self):
        calls = []
        main.update_htf_trends = lambda force=False: calls.append(force)  # never heals the marker
        self._persist(marker_age_hours=30)

        for _ in range(3):
            main._htf_staleness_guard()
        self.assertEqual(calls, [True], "stale marker -> exactly one forced attempt per window")

        main._htf_guard_last_attempt = datetime.now() - main.HTF_GUARD_RETRY - timedelta(seconds=1)
        main._htf_staleness_guard()
        self.assertEqual(calls, [True, True], "window elapsed -> one more attempt")

    # 7. No persisted labels at all (first boot): nothing restored, no marker, no crash.
    def test_no_disk_labels_is_a_clean_noop(self):
        main.client = _Client(weekend=True)

        main.update_htf_trends(force=True)

        self.assertEqual(dict(main.htf_trends), {})
        self.assertFalse(main.LAST_HTF_REFRESH_FILE.exists())
        self.assertFalse(main.market_regime_confirmed)
        main._htf_staleness_guard()  # no marker file -> returns early

    # 8. A market removed from config.py must not be resurrected from the persisted
    #    file. Bitcoin lingered as a 14th key after its 2026-09-09 removal because the
    #    restore copied every key and the save wrote the whole dict back.
    def test_weekend_restore_prunes_epics_no_longer_configured(self):
        main.client = _Client(weekend=True)
        main.LAST_HTF_REFRESH_FILE.write_text(
            (datetime.now() - timedelta(hours=30)).isoformat())
        main.HTF_TRENDS_FILE.write_text(
            json.dumps(dict(FRIDAY_LABELS, **{"CS.D.BITCOIN.TODAY.IP": "NEUTRAL"})))

        main.update_htf_trends()  # 30h > cooldown -> fetch loop -> weekend restore + save

        self.assertEqual(dict(main.htf_trends), FRIDAY_LABELS)
        self.assertNotIn("CS.D.BITCOIN.TODAY.IP",
                         json.loads(main.HTF_TRENDS_FILE.read_text()),
                         "save must drop the retired epic, not persist it forever")

    def test_cooldown_restore_prunes_epics_no_longer_configured(self):
        main.client = _Client(weekend=False)
        main.LAST_HTF_REFRESH_FILE.write_text(
            (datetime.now() - timedelta(hours=1)).isoformat())
        main.HTF_TRENDS_FILE.write_text(
            json.dumps(dict(FRIDAY_LABELS, **{"CS.D.BITCOIN.TODAY.IP": "NEUTRAL"})))

        main.update_htf_trends()  # 1h < 6h cooldown -> disk-restore branch, no fetch

        self.assertEqual(dict(main.htf_trends), FRIDAY_LABELS)
        self.assertEqual(main.client.calls, 0)


if __name__ == "__main__":
    unittest.main()
