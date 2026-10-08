"""Properties for scripts/replay_dxy_frames.py (daily frame).

Pre-registration §10 / D4 / D6: the symmetric engine's long side equals
daily_trend.replay; the short side is the exact mirror of the long side (negate
the price series and every long becomes a short with the same gross R); the
random-timing null never opens a trade inside another.
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

import replay_dxy_frames as rdx  # noqa: E402
from src import daily_trend as dt  # noqa: E402

CFG = dt.DailyTrendConfig()


def _walk(seed: int, n: int = 700, base: float = 10000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 40, n) + np.where(rng.random(n) < 0.03, rng.normal(0, 200, n), 0))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 10, n)
    wick = np.abs(rng.normal(0, 30, (2, n)))
    dates = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame({"date": dates, "open": open_, "close": close,
                         "high": np.maximum(open_, close) + wick[0],
                         "low": np.minimum(open_, close) - wick[1]})


def _mirror(df: pd.DataFrame, k: float = 30000.0) -> pd.DataFrame:
    m = df.copy()
    m["open"], m["close"] = k - df["open"], k - df["close"]
    m["high"], m["low"] = k - df["low"], k - df["high"]
    return m


class TestSymmetricDaily(unittest.TestCase):

    @given(st.integers(0, 10_000))
    @settings()
    def test_long_side_equals_daily_trend_replay(self, seed):
        df = _walk(seed)
        ref = dt.replay(df, CFG)
        mine = rdx.sym_replay(df, CFG, shorts=False)
        self.assertEqual([(a["entry_date"], a["exit_date"], a["entry"], a["exit"], a["reason"]) for a in ref],
                         [(b["entry_date"], b["exit_date"], b["entry"], b["exit"], b["reason"]) for b in mine])

    @given(st.integers(0, 10_000))
    @settings()
    def test_short_side_is_the_mirror_of_the_long_side(self, seed):
        """Negate the series: a SHORT-only run on the mirror must equal the
        LONG-only run on the original, trade for trade, with the same gross R."""
        df = _walk(seed)
        lo = rdx.sym_replay(df, CFG, shorts=False)
        so = rdx.sym_replay(_mirror(df), CFG, longs=False)
        self.assertTrue(all(t["d"] == "SELL" for t in so))
        self.assertEqual([(a["entry_date"], a["exit_date"], a["reason"], round(a["r_gross"], 6)) for a in lo],
                         [(b["entry_date"], b["exit_date"], b["reason"], round(b["r_gross"], 6)) for b in so])

    @given(st.integers(0, 10_000), st.floats(0.0, 0.2), st.floats(0.0, 1.0))
    @settings()
    def test_null_never_overlaps(self, seed, p, p_buy):
        df = _walk(seed)
        n = len(dt.prepare_daily(df))
        t = rdx.sym_replay(df, CFG, rng=np.random.default_rng(seed),
                           p_enter=np.full(n, p), p_buy=p_buy)
        for a, b in zip(t, t[1:]):
            self.assertLessEqual(a["exit_date"], b["entry_date"])


if __name__ == "__main__":
    unittest.main()
