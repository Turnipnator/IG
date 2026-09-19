"""The root backtest.py HTF gate must only see higher-timeframe bars that have CLOSED.

lookup_htf_trend matched on `date <= timestamp`. Rows are stamped at bar START,
so that selected the hour still FORMING at the 5m bar being traded — its close
is up to 55 minutes in the future. On the backtest cache it mislabelled a median
3.8% of bars (up to 10% on the cash indices), concentrated at trend flips, which
is exactly where entries and the HTF-flip exits fire. Live never does this: it
fetches COMPLETED bars.

Third copy of the same bug: htf_series (5cba8b3) and Backtester.calculate_htf_trend
(117a4e8) carried it independently. Found 2026-09-19 by a property-based spike
stating the rule below once, for every copy of the join.

Two properties, because either alone can be satisfied by a wrong fix:
  1. rewriting bars that have not closed by t never changes the trend at t
     (a fix that lags too little fails this);
  2. a bar IS used from the moment it closes (a fix that lags too much —
     e.g. shifting two bars — passes (1) and fails this).
"""

import unittest
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import backtest
from tests.helpers import load_candles

HOUR = pd.Timedelta(hours=1)


def trend_at(htf: pd.DataFrame, t) -> str:
    return backtest.lookup_htf_trend(backtest.calculate_htf_trend(htf), t)


def rewrite_from(htf: pd.DataFrame, j: int, factor: float) -> pd.DataFrame:
    """Scale bar j and every later bar — a different future, same past."""
    out = htf.copy()
    out.loc[j:, ["open", "high", "low", "close"]] *= factor
    return out


class RootBacktestHTFLookahead(unittest.TestCase):
    def setUp(self):
        self.htf = load_candles("gbpusd_1h.jsonl")
        # Past the EMA-21 warm-up; every 7th bar keeps the loop quick while
        # still crossing many trend states.
        self.bars = range(30, len(self.htf) - 1, 7)

    def test_forming_bar_never_changes_the_trend(self):
        for j in self.bars:
            start = self.htf["date"].iloc[j]
            for minute in (0, 5, 30, 55):
                t = start + pd.Timedelta(minutes=minute)      # bar j still forming
                base = trend_at(self.htf, t)
                for factor in (0.9, 1.1):
                    self.assertEqual(
                        trend_at(rewrite_from(self.htf, j, factor), t), base,
                        f"trend at {t} moved when only the unclosed bar {start} and later changed",
                    )

    def test_bar_is_used_once_it_has_closed(self):
        """A crash in bar j must reach the gate at start_j + 1h. If this never
        fires the lookup is lagging more than one bar."""
        seen = 0
        for j in self.bars:
            closed = self.htf["date"].iloc[j] + HOUR
            before = trend_at(self.htf, closed)
            crashed = self.htf.copy()
            crashed.loc[j, ["open", "high", "low", "close"]] *= 0.9
            if trend_at(crashed, closed) != before:
                seen += 1
        self.assertGreater(seen, 0, "no closed bar ever influenced the gate at its own close")

    def test_nothing_closed_yet_is_neutral(self):
        self.assertEqual(trend_at(self.htf, self.htf["date"].iloc[0]), "NEUTRAL")

    def test_closed_at_is_start_plus_one_bar(self):
        out = backtest.calculate_htf_trend(self.htf)
        self.assertTrue(((out["closed_at"] - out["date"]) == HOUR).all())


if __name__ == "__main__":
    unittest.main()
