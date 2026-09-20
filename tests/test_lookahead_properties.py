"""Generated look-ahead properties: nothing may use a price from its own future.

Every silent bug in this repo's backtest-vs-live gap has had one shape. A
computation at time t quietly reads a price from after t and still returns a
perfectly valid number. Three separate copies of the HTF join did it (htf_series
5cba8b3, the engine 117a4e8, root backtest.py 57ba785), and each was found by a
person, separately, after it had already fed decisions.

These tests state the rule once and let Hypothesis hunt for a counterexample
across generated frames (irregular spacing, weekend gaps, any price scale)
rather than one fixture. The rule takes three forms:

  * a decision at t must not change when bars that have not CLOSED by t change
    (the HTF gates in both backtesters);
  * a bar, once published, is final: more data may add bars, never revise one
    (the live 1h breakout frame and the cash-session bars);
  * a level at bar i uses bars <= i, and a channel uses bars < i only
    (indicators, daily-trend and pullback levels).

Replayed deterministically in CI and explored randomly every week; see tests/pbt.py.
"""

import tempfile
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tests.pbt import OHLC, ohlc_frames, price_moves, st
from hypothesis import given

import backtest as root_backtest
import main
from src import daily_trend, pullback, session_bars
from src.backtest import Backtester
from src.indicators import add_all_indicators, calculate_ema
from src.streaming import CANDLE_ARCHIVE_DIR

HOUR = pd.Timedelta(hours=1)
FIVE_MIN = pd.Timedelta(minutes=5)
NO_ARCHIVE_EPIC = "PBT.NO.ARCHIVE"   # no archive file, so the live builders read only the frame given


def same(a, b) -> bool:
    """Elementwise equal, NaN == NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-12, atol=0)))


def rescale(df: pd.DataFrame, rows, factor: float) -> pd.DataFrame:
    """A different future: the same bars' prices scaled by `factor`."""
    out = df.copy()
    out.loc[rows, OHLC] = out.loc[rows, OHLC] * factor
    return out


factors = st.floats(0.5, 2.0).filter(lambda f: abs(f - 1) > 1e-3)


# ---------------------------------------------------------------- HTF gates

@st.composite
def htf_decision(draw):
    """An hourly HTF frame and a 5m bar inside it. Both backtesters stamp the 5m
    bar at its START, t, and decide at its close, t + 5m."""
    htf = draw(ohlc_frames(min_bars=30, max_bars=90, freq="1h"))
    j = draw(st.integers(22, len(htf) - 1))
    t = htf["date"].iloc[j] + pd.Timedelta(minutes=draw(st.sampled_from(range(0, 60, 5))))
    return htf, t, draw(factors)


class HTFGateSeesOnlyClosedBars(unittest.TestCase):
    """Rewriting every HTF bar that has not closed by decision time (its start +
    1h is after t + 5m) must leave the trend the decision sees unchanged."""

    @staticmethod
    def unclosed(htf, t):
        return htf["date"] + HOUR > t + FIVE_MIN

    @given(htf_decision())
    def test_root_backtest(self, case):
        htf, t, factor = case
        trend = lambda f: root_backtest.lookup_htf_trend(root_backtest.calculate_htf_trend(f), t)
        self.assertEqual(trend(rescale(htf, self.unclosed(htf, t), factor)), trend(htf),
                         f"root backtest.py: the HTF trend at {t} moved when only unclosed bars changed")

    @given(htf_decision())
    def test_engine(self, case):
        htf, t, factor = case
        engine = Backtester()

        def trend(f):
            f = f.copy()
            f["ema_9"] = calculate_ema(f["close"], 9)
            f["ema_21"] = calculate_ema(f["close"], 21)
            return engine.calculate_htf_trend("X", t, f)

        self.assertEqual(trend(rescale(htf, self.unclosed(htf, t), factor)), trend(htf),
                         f"src/backtest.py: the HTF trend at {t} moved when only unclosed bars changed")


# ---------------------------------------------------------------- live bar builders

class LiveBarsAreFinal(unittest.TestCase):

    def setUp(self):
        self.assertFalse((CANDLE_ARCHIVE_DIR / f"{NO_ARCHIVE_EPIC}.jsonl").exists())

    @given(ohlc_frames(min_bars=24, max_bars=150, freq="5min"), st.data())
    def test_breakout_1h_frame_never_revises_a_published_bar(self, candles, data):
        """The entry path's 1h frame (drop_forming=True) is published every hour.
        A bar it has published must never change when later candles arrive. If
        it did, the frame included an hour that was still forming.

        The TRAIL's drop_forming=False frame ends with the forming hour on
        purpose (see _breakout_frame_1h), so it is deliberately not held to this."""
        k = data.draw(st.integers(2, len(candles)), label="candles seen")
        early = main._breakout_frame_1h(NO_ARCHIVE_EPIC, candles.iloc[:k])
        if early is None:
            return
        full = main._breakout_frame_1h(NO_ARCHIVE_EPIC, candles).set_index("date")
        early = early.set_index("date")
        self.assertTrue(early.index.isin(full.index).all(), "a published hour disappeared later")
        self.assertTrue(same(early[OHLC], full.loc[early.index, OHLC]),
                        f"an hour published after {k} candles was revised by later ones")

    @given(ohlc_frames(min_bars=24, max_bars=150, freq="5min"))
    def test_breakout_1h_bar_holds_only_its_own_hour(self, candles):
        """A bar stamped H is built from candles in [H, H+1h) and nothing else.
        A right-closed or right-labelled resample would pull the NEXT hour's first
        candle into it: a price from the future, one hour early."""
        out = main._breakout_frame_1h(NO_ARCHIVE_EPIC, candles, drop_forming=False)
        if out is None:
            return
        own_hour = candles.groupby(candles["date"].dt.floor("h")).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        self.assertTrue(out["date"].isin(own_hour.index).all(),
                        "a 1h bar is stamped at an hour none of its candles belong to")
        self.assertTrue(same(out[OHLC], own_hour.loc[out["date"], OHLC]),
                        "a 1h bar holds candles from outside its own hour")

    @given(st.data())
    def test_session_bar_is_final_once_the_window_closes(self, data):
        """run_pullback decides at 21:05 on the bar the 14:30-21:00 window built.
        Nothing arriving after the window has closed (post-session candles, later
        days) may change that bar or any earlier one."""
        days = data.draw(st.integers(2, 6), label="days")
        first = pd.Timestamp("2026-03-02") + pd.Timedelta(days=data.draw(st.integers(0, 20), label="start"))
        grid = [first + pd.Timedelta(days=d) + pd.Timedelta(minutes=30 * s)
                for d in range(days) for s in range(26, 48)]           # 13:00-23:30, 30-minute candles
        keep = data.draw(st.lists(st.sampled_from((True,) * 5 + (False,)),
                                  min_size=len(grid), max_size=len(grid)), label="present")
        moves = data.draw(price_moves(len(grid), 0.02), label="moves")
        p, rows = 5000.0, []
        for t, present, r in zip(grid, keep, moves):
            o, c = p, p * (1 + r)
            p = c
            if present:
                rows.append({"date": t, "open": o, "high": max(o, c), "low": min(o, c), "close": c})
        candles = pd.DataFrame(rows, columns=["date"] + OHLC)

        day = first + pd.Timedelta(days=data.draw(st.integers(0, days - 1), label="decision day"))
        now = day + pd.Timedelta(hours=21) + FIVE_MIN * data.draw(st.integers(0, 30), label="minutes after close")
        seen = candles[candles["date"] + pd.Timedelta(minutes=30) <= now]   # candles CLOSED by then

        with tempfile.TemporaryDirectory() as empty:
            build = lambda c: session_bars.build_session_bars(NO_ARCHIVE_EPIC, c, archive_dir=Path(empty), min_bars=3)
            then, later = build(seen), build(candles)
        then = then[then["date"] <= day].reset_index(drop=True)
        later = later[later["date"] <= day].reset_index(drop=True)
        self.assertEqual(list(then["date"]), list(later["date"]), "a closed session bar appeared or vanished")
        self.assertTrue(same(then[OHLC], later[OHLC]), f"the {day.date()} session bar changed after 21:00")


# ---------------------------------------------------------------- levels

def assert_rows_before_unchanged(tc, before, after, cols, upto, what):
    for col in cols:
        tc.assertTrue(same(before[col].iloc[:upto], after[col].iloc[:upto]),
                      f"{what}: {col} changed at a bar before the rewritten ones")


class LevelsUseOnlyThePast(unittest.TestCase):
    """Rewrite bar i and everything after it. Anything computed AT a bar may
    change from i onward. A channel is built from the bars BEFORE i, so
    it must not change at i either. That one-bar shift is what the daily-trend
    and pullback docstrings promise: 'bar i is judged against the prior n bars only'."""

    @given(ohlc_frames(min_bars=40, max_bars=140, freq="1D"), st.data(), factors)
    def test_daily_trend(self, bars, data, factor):
        cfg = daily_trend.get_daily_trend_config("CS.D.USCGC.TODAY.IP")
        i = data.draw(st.integers(1, len(bars) - 1), label="first rewritten bar")
        a = daily_trend.add_levels(bars, cfg)
        b = daily_trend.add_levels(rescale(bars, bars.index >= i, factor), cfg)
        assert_rows_before_unchanged(self, a, b, ["atr"], i, "daily-trend")
        assert_rows_before_unchanged(self, a, b, ["entry_hi", "exit_lo"], i + 1, "daily-trend channel")

    @given(ohlc_frames(min_bars=25, max_bars=80, freq="1D", gaps=False), st.data(), factors)
    def test_pullback(self, bars, data, factor):
        """Includes the SMA path the live bot uses: IG DAY closes merged with the
        session closes. A DAY close dated at or after bar i is part of the
        rewritten future too.

        The DAY closes reach back far enough for the SMA200 to be defined from the
        first session bar. With less history it was NaN almost everywhere, NaN ==
        NaN, and a mutant reading tomorrow's close passed. The notna assertion keeps
        that from recurring silently."""
        cfg = pullback.get_pullback_config("IX.D.SPTRD.DAILY.IP")
        dates = pd.date_range(bars["date"].iloc[0] - pd.Timedelta(days=320),
                              bars["date"].iloc[-1] + pd.Timedelta(days=3), freq="D")
        closes = data.draw(st.lists(st.floats(1_000, 20_000), min_size=len(dates), max_size=len(dates)),
                           label="DAY closes")
        day_closes = pd.Series(closes, index=dates)
        i = data.draw(st.integers(1, len(bars) - 1), label="first rewritten bar")
        cut = bars["date"].iloc[i]
        a = pullback.add_levels(bars, cfg, day_closes)
        self.assertTrue(a["sma"].notna().all(), "SMA undefined on some bars, so the check below would be vacuous there")
        b = pullback.add_levels(rescale(bars, bars.index >= i, factor), cfg,
                                day_closes.where(day_closes.index < cut, day_closes * factor))
        assert_rows_before_unchanged(self, a, b, ["atr", "sma"], i, "pullback")
        assert_rows_before_unchanged(self, a, b, ["entry_lo", "exit_hi"], i + 1, "pullback channel")

    @given(ohlc_frames(min_bars=60, max_bars=160, freq="5min"), st.data(), factors)
    def test_momentum_indicators(self, candles, data, factor):
        """Every column add_all_indicators writes (EMAs, RSI, MACD, Bollinger,
        ATR, ADX, stochastic), the whole momentum/regime input set."""
        i = data.draw(st.integers(1, len(candles) - 1), label="first rewritten bar")
        a = add_all_indicators(candles, {})
        b = add_all_indicators(rescale(candles, candles.index >= i, factor), {})
        added = [c for c in a.columns if c not in candles.columns]
        self.assertGreater(len(added), 10)
        assert_rows_before_unchanged(self, a, b, added, i, "indicators")


if __name__ == "__main__":
    unittest.main()
