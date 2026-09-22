"""Look-ahead and tagging properties for scripts/replay_breakout_news.py.

Pre-registration §9: the BLOCKED tag of an entry must not depend on any price
after its decision time, and the rotation null must never use a real event slot as
a control. The engine and the fast HTF path are the two places a future price
could leak in, so both are pinned here.
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

import replay_breakout_news as rbn  # noqa: E402

EPIC = "CS.D.GBPUSD.TODAY.IP"
N_BARS = 1400


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    utc = pd.date_range("2021-01-04", periods=N_BARS, freq="1h", tz="UTC")
    steps = rng.normal(0, 8, N_BARS) + np.where(rng.random(N_BARS) < 0.02, rng.normal(0, 60, N_BARS), 0)
    close = 13000 + np.cumsum(steps)
    spread = np.abs(rng.normal(0, 6, N_BARS)) + 1
    df = pd.DataFrame({"utc": utc, "date": utc.tz_convert("Europe/London").tz_localize(None),
                       "open": np.r_[close[0], close[:-1]], "close": close})
    df["high"] = np.maximum(df["open"], df["close"]) + spread
    df["low"] = np.minimum(df["open"], df["close"]) - spread
    return df.drop_duplicates("date").reset_index(drop=True)


def _poison(df: pd.DataFrame, cut: int, junk: int) -> pd.DataFrame:
    p = df.copy()
    rng = np.random.default_rng(junk)
    later = p.index > cut
    base = rng.uniform(9000, 17000, int(later.sum()))
    p.loc[later, "open"] = base
    p.loc[later, "close"] = base + rng.normal(0, 50, int(later.sum()))
    p.loc[later, "high"] = p.loc[later, ["open", "close"]].max(axis=1) + 20
    p.loc[later, "low"] = p.loc[later, ["open", "close"]].min(axis=1) - 20
    return p


class NewsReplayLookahead(unittest.TestCase):

    @settings(max_examples=25)
    @given(seed=st.integers(0, 10_000), day=st.integers(25, 50), hour=st.sampled_from([22, 23, 9, 15]),
           junk=st.integers(0, 10_000))
    def test_htf_labels_ignore_later_prices(self, seed, day, hour, junk):
        # Cut in the late evening on purpose: between the 22:30 refresh and midnight,
        # the day's daily bar is still forming — the one place a same-day leak shows.
        df = _frame(seed)
        target = df["date"].iloc[0].normalize() + pd.Timedelta(days=day, hours=hour)
        cut = int(np.searchsorted(df["date"].to_numpy(), np.datetime64(target)))
        a = rbn.htf_day_fast(df)
        b = rbn.htf_day_fast(_poison(df, cut, junk))
        pd.testing.assert_series_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])

    @settings(max_examples=25)
    @given(seed=st.integers(0, 10_000), pick=st.integers(0, 10_000), junk=st.integers(0, 10_000))
    def test_entries_ignore_later_prices(self, seed, pick, junk):
        # Cut AT an entry bar: a leak from bar i+1 into the decision at bar i only
        # shows when bar i+1 is the first poisoned bar.
        df = _frame(seed)
        df["htf"] = rbn.htf_day_fast(df)
        a = rbn.run_gated(df, EPIC)
        if a.empty:
            return
        t = a["date"].iloc[pick % len(a)]
        cut = int(df.index[df["date"] == t][0])
        p = _poison(df, cut, junk)
        p["htf"] = df["htf"]            # the HTF path is pinned separately above
        b = rbn.run_gated(p, EPIC)
        cols = ["date", "d", "entry", "sd"]
        ka = a[a["date"] <= t][cols].reset_index(drop=True)
        kb = b[b["date"] <= t][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(ka, kb)

    @settings()
    @given(seed=st.integers(0, 10_000), weeks=st.sampled_from(rbn.SHIFTS))
    def test_rotation_never_uses_a_real_event_slot(self, seed, weeks):
        rng = np.random.default_rng(seed)
        base = np.datetime64("2020-01-01T00:00", "ns")
        real = base + (rng.integers(0, 60 * 24 * 900, 300) * np.timedelta64(1, "m"))
        sh = rbn.shifted_events(real, weeks)
        if len(sh):
            gap = np.abs(sh[:, None] - np.sort(real)[None, :]).min(axis=1)
            self.assertTrue((gap > np.timedelta64(60, "m")).all())

    @settings()
    @given(seed=st.integers(0, 10_000))
    def test_tag_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        base = pd.Timestamp("2022-01-03", tz="UTC")
        dec = pd.Series(base + pd.to_timedelta(rng.integers(0, 24 * 200, 150), unit="h"))
        ev = (base.tz_localize(None) + pd.to_timedelta(rng.integers(0, 60 * 24 * 200, 40), unit="m")).to_numpy("datetime64[ns]")
        got = rbn.tag_blocked(dec, ev)
        t = dec.dt.tz_localize(None).to_numpy("datetime64[ns]")
        want = (np.abs(t[:, None] - ev[None, :]) <= np.timedelta64(30, "m")).any(axis=1)
        np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
