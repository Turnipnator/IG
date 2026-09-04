"""The REST price cache must be keyed by (epic, resolution).

Regression for 2026-09-04: keyed by epic alone, a DAY request made within the
TTL of a HOUR fetch got the HOUR bars back. At boot the candle seed fetched 50
HOUR bars for Crude, the daily HTF refresh asked for DAY 24s later, received the
hourly bars, and computed HTF=BEARISH on intraday closes while the real daily
series was BULLISH — a breakout gate pointing the wrong way for a live market.
"""
import json
import pathlib
import tempfile
import unittest
from datetime import datetime, timedelta

import pandas as pd

from config import IGConfig
from src import client as client_module
from src.client import IGClient


def _bars(n, start_close=100.0):
    return pd.DataFrame({
        "date": pd.date_range("2026-09-01", periods=n, freq="h"),
        "open": start_close, "high": start_close + 1, "low": start_close - 1,
        "close": [start_close + i for i in range(n)], "volume": 0,
    })


class PriceCacheKeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = pathlib.Path(self.tmp.name)
        # Never let a test touch data/price_cache.json.
        self._orig = (client_module.CACHE_DIR, client_module.PRICE_CACHE_FILE)
        client_module.CACHE_DIR = tmp
        client_module.PRICE_CACHE_FILE = tmp / "price_cache.json"
        self.client = IGClient(
            IGConfig(api_key="k", username="u", password="p", acc_type="DEMO"),
            cache_ttl_minutes=55,
        )

    def tearDown(self):
        client_module.CACHE_DIR, client_module.PRICE_CACHE_FILE = self._orig
        self.tmp.cleanup()

    def test_hour_bars_do_not_satisfy_a_day_request(self):
        epic = "CC.D.CL.USS.IP"
        self.client._cache_prices(epic, _bars(50), "HOUR")
        self.assertIsNone(self.client._get_cached_prices(epic, "DAY"),
                          "a DAY request must never be served from HOUR bars")
        got = self.client._get_cached_prices(epic, "HOUR")
        self.assertIsNotNone(got)
        self.assertEqual(len(got), 50)

    def test_resolutions_coexist_for_one_epic(self):
        epic = "CC.D.CL.USS.IP"
        self.client._cache_prices(epic, _bars(50), "HOUR")
        self.client._cache_prices(epic, _bars(30, 500.0), "DAY")
        self.assertEqual(len(self.client._get_cached_prices(epic, "HOUR")), 50)
        self.assertEqual(len(self.client._get_cached_prices(epic, "DAY")), 30)
        self.assertEqual(self.client._get_cached_prices(epic, "DAY")["close"].iloc[0], 500.0)

    def test_clear_cache_drops_every_resolution_of_the_epic(self):
        epic = "CC.D.CL.USS.IP"
        self.client._cache_prices(epic, _bars(5), "HOUR")
        self.client._cache_prices(epic, _bars(5), "DAY")
        self.client._cache_prices("CS.D.USCGC.TODAY.IP", _bars(5), "DAY")
        self.client.clear_cache(epic)
        self.assertIsNone(self.client._get_cached_prices(epic, "HOUR"))
        self.assertIsNone(self.client._get_cached_prices(epic, "DAY"))
        self.assertIsNotNone(self.client._get_cached_prices("CS.D.USCGC.TODAY.IP", "DAY"))

    def test_disk_round_trip_and_legacy_keys_ignored(self):
        epic = "CC.D.CL.USS.IP"
        self.client._cache_prices(epic, _bars(7), "DAY")  # writes the file
        # Inject a legacy epic-only entry alongside it, as a pre-fix file would have.
        raw = json.loads(client_module.PRICE_CACHE_FILE.read_text())
        raw["CS.D.USCGC.TODAY.IP"] = {
            "fetched_at": datetime.now().isoformat(),
            "data": _bars(3).assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records"),
        }
        client_module.PRICE_CACHE_FILE.write_text(json.dumps(raw))
        fresh = IGClient(IGConfig(api_key="k", username="u", password="p", acc_type="DEMO"),
                         cache_ttl_minutes=55)
        self.assertEqual(len(fresh._get_cached_prices(epic, "DAY")), 7)
        self.assertIsNone(fresh._get_cached_prices("CS.D.USCGC.TODAY.IP", "DAY"),
                          "a legacy epic-only entry has no known resolution and must not be served")
        self.assertIsNone(fresh._get_cached_prices("CS.D.USCGC.TODAY.IP", "MINUTE_5"))

    def test_ttl_still_applies(self):
        epic = "CC.D.CL.USS.IP"
        self.client._cache_prices(epic, _bars(5), "DAY")
        key = self.client._cache_key(epic, "DAY")
        self.client._price_cache[key].fetched_at = datetime.now() - timedelta(minutes=56)
        self.assertIsNone(self.client._get_cached_prices(epic, "DAY"))


if __name__ == "__main__":
    unittest.main()
