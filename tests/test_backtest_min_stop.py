"""src.backtest._min_stop_for: the backtester's stop floor must be the LIVE one.

Until 2026-09-01 src/backtest.py floored every stop with a hand-written "~0.5% of
price" table the live bot never applied (S&P 30.0 vs config 2.0), which made the
ATR risk model inert on S&P and manufactured the "never reaches 2R" finding.
Three properties pinned here:
  1. a configured market resolves to MarketConfig.min_stop_distance, not the table;
  2. markets IG quotes in scaled points (Crude, DXY, forex) are converted to the
     Yahoo price units the backtester trades in: 12 IG points on Crude is $0.12,
     not $12;
  3. a study script's explicit MIN_STOP_DISTANCE_MAP entry still wins, so the four
     scripts that patch it keep their published behaviour.
"""
import unittest

from config import MARKETS
from src import backtest


def _cfg(name):
    return next(m for m in MARKETS if m.name == name)


class MinStopForTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(backtest.MIN_STOP_DISTANCE_MAP)
        backtest.MIN_STOP_DISTANCE_MAP.clear()

    def tearDown(self):
        backtest.MIN_STOP_DISTANCE_MAP.clear()
        backtest.MIN_STOP_DISTANCE_MAP.update(self._saved)

    def test_index_floor_is_config_not_legacy_table(self):
        self.assertEqual(backtest._min_stop_for("S&P 500"), _cfg("S&P 500").min_stop_distance)
        self.assertNotEqual(backtest._min_stop_for("S&P 500"), 30.0)
        self.assertEqual(backtest._min_stop_for("Gold"), _cfg("Gold").min_stop_distance)

    def test_scaled_point_markets_land_in_price_units(self):
        self.assertAlmostEqual(backtest._min_stop_for("Crude Oil"),
                               _cfg("Crude Oil").min_stop_distance / 100.0)
        self.assertAlmostEqual(backtest._min_stop_for("EUR/USD"),
                               _cfg("EUR/USD").min_stop_distance / 10000.0)
        # TICKER_MAP calls it "Dollar Index"; config.py "Dollar Index (DXY)".
        self.assertAlmostEqual(backtest._min_stop_for("Dollar Index"),
                               _cfg("Dollar Index (DXY)").min_stop_distance / 100.0)

    def test_every_ticker_map_floor_is_tighter_than_the_legacy_table(self):
        for name in backtest.TICKER_MAP:
            self.assertLess(backtest._min_stop_for(name),
                            backtest._LEGACY_MIN_STOP_DISTANCE_MAP[name], name)

    def test_script_override_wins(self):
        backtest.MIN_STOP_DISTANCE_MAP["S&P 500"] = 30.0
        self.assertEqual(backtest._min_stop_for("S&P 500"), 30.0)

    def test_unknown_market_has_no_floor(self):
        self.assertEqual(backtest._min_stop_for("Copper"), 0.0)


if __name__ == "__main__":
    unittest.main()
