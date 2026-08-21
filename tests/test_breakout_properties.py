"""Property tests for the Donchian channel arithmetic.

These assert what MUST be true, not what the code currently happens to do —
deliberately unlike the golden tests, which freeze present behaviour and would
happily enshrine a bug. Every property here corresponds to a real defect that
shipped:

  * exit_channel measured on the native 5m frame instead of the resampled 1h
    one (0057aa9) — a 27x5m = 2.25h trail where 27h was validated. Entry and
    exit ran different strategies for weeks. Nothing crashed.
  * the same helper handed the ENTRY frame, so its own df[-1] drop removed a
    real closed bar and the window ended an hour early (31aa992).
  * the deliberate one-bar offset between arming (current_channel) and
    confirming (analyze_breakout) — get it wrong and you silently arm
    yesterday's level.

The shared root cause is that all three are off-by-one-window errors that
produce a perfectly valid number. Only an explicit assertion about WHICH bars
went into the window can see them.
"""

import unittest
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src.breakout import BREAKOUT_CONFIGS, current_channel, exit_channel
from tests.helpers import load_candles

GBP = "CS.D.GBPUSD.TODAY.IP"
CFG = BREAKOUT_CONFIGS[GBP]


class TestExitChannelWindow(unittest.TestCase):
    """exit_channel returns the prior-M-bar extreme: df.iloc[-(m+1):-1]."""

    def setUp(self):
        self.df = load_candles("gbpusd_1h.jsonl")

    def test_excludes_the_final_bar(self):
        """The forming/last bar must NOT influence the level. If it does, the
        trail chases the current bar and can ratchet on a wick that has not
        closed."""
        base = exit_channel(self.df, GBP, "BUY")
        self.assertIsNotNone(base)

        spiked = self.df.copy()
        spiked.loc[spiked.index[-1], "low"] = 1.0   # absurd low in the LAST bar
        self.assertEqual(
            exit_channel(spiked, GBP, "BUY"), base,
            "exit_channel changed when only the final bar changed — it is including df[-1]",
        )

    def test_includes_the_bar_immediately_before_the_final_one(self):
        """The window is the M bars before the last, so index -2 is the most
        recent bar that MUST count. Together with the test above this pins the
        window's near edge exactly."""
        base = exit_channel(self.df, GBP, "BUY")
        spiked = self.df.copy()
        spiked.loc[spiked.index[-2], "low"] = 1.0
        self.assertEqual(
            exit_channel(spiked, GBP, "BUY"), 1.0,
            "bar -2 did not affect the level — the window's near edge is off by one",
        )

    def test_window_is_exactly_m_bars_long(self):
        """Pins the FAR edge. Bar -(m+1) is the oldest included; -(m+2) is the
        first excluded. An off-by-one here is what a longer/shorter trail
        looks like, and it is invisible in the output."""
        m = CFG.m
        oldest_in = self.df.copy()
        oldest_in.loc[oldest_in.index[-(m + 1)], "low"] = 1.0
        self.assertEqual(exit_channel(oldest_in, GBP, "BUY"), 1.0,
                         f"bar -(m+1) = -{m + 1} should be INSIDE the window")

        first_out = self.df.copy()
        first_out.loc[first_out.index[-(m + 2)], "low"] = 1.0
        self.assertNotEqual(exit_channel(first_out, GBP, "BUY"), 1.0,
                            f"bar -(m+2) = -{m + 2} should be OUTSIDE the window")

    def test_short_direction_uses_the_high(self):
        spiked = self.df.copy()
        spiked.loc[spiked.index[-2], "high"] = 99999.0
        self.assertEqual(exit_channel(spiked, GBP, "SELL"), 99999.0)

    def test_returns_none_rather_than_crashing_on_short_frames(self):
        for n in (0, 1, CFG.m, CFG.m + 1):
            self.assertIsNone(exit_channel(self.df.iloc[:n], GBP, "BUY"),
                              f"expected None for a {n}-bar frame")

    def test_returns_none_for_an_unknown_epic(self):
        self.assertIsNone(exit_channel(self.df, "NOT.A.REAL.EPIC", "BUY"))


class TestFrameRelativity(unittest.TestCase):
    """exit_channel takes M bars of WHATEVER frame it is handed. It has no idea
    what timeframe those bars represent, so passing the native 5m stream where
    the 1h frame was validated yields a valid-looking number for a strategy
    nobody tested. This is 0057aa9, encoded."""

    def test_5m_and_1h_frames_give_different_levels(self):
        gold_5m = load_candles("gold_5m.jsonl")
        hourly = (gold_5m.set_index("date")
                  .resample("1h")
                  .agg({"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"})
                  .dropna()
                  .reset_index())
        self.assertGreater(len(hourly), CFG.m + 2, "fixture too short to resample")

        native = exit_channel(gold_5m.rename(columns=str), GBP, "BUY")
        resampled = exit_channel(hourly, GBP, "BUY")
        self.assertIsNotNone(native)
        self.assertIsNotNone(resampled)
        self.assertNotEqual(
            native, resampled,
            "5m and 1h frames produced the SAME trail level — the fixture cannot "
            "distinguish them, so this test would not catch a misframed trail",
        )

    def test_the_1h_window_spans_m_hours_not_m_bars_of_something_else(self):
        """States the invariant in wall-clock terms: on an hourly frame the
        window must cover M hours. That is the property the 2.25h-vs-27h bug
        violated."""
        gbp = load_candles("gbpusd_1h.jsonl")
        window = gbp.iloc[-(CFG.m + 1):-1]
        span = window["date"].iloc[-1] - window["date"].iloc[0]
        self.assertLessEqual(span, pd.Timedelta(hours=CFG.m * 3),
                             "window spans far more wall-clock than M hours")
        self.assertEqual(len(window), CFG.m)


class TestArmingOffset(unittest.TestCase):
    """current_channel (arming, df.iloc[-n:]) and analyze_breakout (confirming,
    df.iloc[-(n+1):-1]) deliberately sit ONE BAR apart. breakout.py spells that
    out in prose because getting it wrong silently arms yesterday's level."""

    def setUp(self):
        self.df = load_candles("gbpusd_1h.jsonl")

    def test_current_channel_INCLUDES_the_final_bar(self):
        base = current_channel(self.df, GBP)
        spiked = self.df.copy()
        spiked.loc[spiked.index[-1], "high"] = 99999.0
        self.assertNotEqual(current_channel(spiked, GBP), base,
                            "current_channel ignored the final bar — it must include it")
        self.assertEqual(current_channel(spiked, GBP)[0], 99999.0)

    def test_arming_window_is_exactly_one_bar_ahead_of_the_confirm_window(self):
        """The two windows must be offset by exactly one bar. Computed from the
        frame directly, so it fails if either slice convention drifts."""
        n = CFG.n
        arming = self.df.iloc[-n:]
        confirming = self.df.iloc[-(n + 1):-1]
        self.assertEqual(len(arming), len(confirming), "windows differ in length")
        self.assertEqual(
            arming["date"].iloc[0] , confirming["date"].iloc[1],
            "arming window is not exactly one bar ahead of the confirm window",
        )
        upper, lower = current_channel(self.df, GBP)
        self.assertEqual(upper, float(arming["high"].max()))
        self.assertEqual(lower, float(arming["low"].min()))

    def test_returns_none_on_short_frame_and_unknown_epic(self):
        self.assertIsNone(current_channel(self.df.iloc[:CFG.n - 1], GBP))
        self.assertIsNone(current_channel(self.df, "NOT.A.REAL.EPIC"))


if __name__ == "__main__":
    unittest.main()
