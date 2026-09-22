"""Look-ahead properties for scripts/backtest_fx_carry.py.

The pre-registration (research_notes.md, 2026-09-22) makes the carry run INVALID
without this: the month-m signal must not move when anything dated at or after
month m's decision point is changed — rates for month m onward, prices after the
last close of month m-1. Four look-ahead leaks in this repo's backtests so far
(HTF joins, the root backtest.py forming bar) is why it is a property, not a check.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.pbt import settings, st  # noqa: E402  (Hypothesis profiles; skips in container)
from hypothesis import given  # noqa: E402

from scripts import backtest_fx_carry as fxc  # noqa: E402

N_MONTHS = 16
MONTHS = pd.period_range("2020-01", periods=N_MONTHS, freq="M")
DAYS = pd.bdate_range(MONTHS[0].start_time - pd.Timedelta(days=420), MONTHS[-1].end_time)


def _world(seed: int):
    rng = np.random.default_rng(seed)
    idx = pd.period_range(MONTHS[0] - 2, MONTHS[-1], freq="M")
    rates = pd.DataFrame(rng.uniform(-0.5, 8.0, (len(idx), len(fxc.CCYS))), index=idx, columns=fxc.CCYS)
    steps = rng.normal(0, 0.006, (len(DAYS), len(fxc.CCYS)))
    # a volatility burst somewhere, so the crash gate is actually exercised
    burst = rng.integers(300, len(DAYS) - 30)
    steps[burst:burst + 25] *= 6
    fx = pd.DataFrame(np.exp(np.cumsum(steps, axis=0)), index=DAYS, columns=fxc.CCYS)
    fx["USD"] = 1.0
    return rates, fx


class FxCarryLookahead(unittest.TestCase):

    @settings()
    @given(seed=st.integers(0, 10_000), cut=st.integers(1, N_MONTHS - 1), junk=st.integers(0, 10_000))
    def test_rank_signal_ignores_rates_from_month_m_on(self, seed, cut, junk):
        rates, _ = _world(seed)
        m = MONTHS[cut]
        base = fxc.rank_weights(rates, MONTHS)
        poisoned = rates.copy()
        poisoned.loc[m:] = np.random.default_rng(junk).uniform(-5, 30, poisoned.loc[m:].shape)
        after = fxc.rank_weights(poisoned, MONTHS)
        pd.testing.assert_series_equal(base.loc[m], after.loc[m])

    @settings(max_examples=15)
    @given(seed=st.integers(0, 10_000), cut=st.integers(1, N_MONTHS - 1), junk=st.integers(0, 10_000))
    def test_invvol_gated_signal_ignores_the_future(self, seed, cut, junk):
        rates, fx = _world(seed)
        m = MONTHS[cut]
        dec = fxc.decision_dates(fx.index, MONTHS)
        base = fxc.invvol_gated_weights(rates, fx, MONTHS, dec)
        pr, pf = rates.copy(), fx.copy()
        rng = np.random.default_rng(junk)
        pr.loc[m:] = rng.uniform(-5, 30, pr.loc[m:].shape)
        later = pf.index > dec[m]
        pf.loc[later, fxc.CCYS[1:]] = rng.uniform(0.01, 100, (int(later.sum()), len(fxc.CCYS) - 1))
        after = fxc.invvol_gated_weights(pr, pf, MONTHS, fxc.decision_dates(pf.index, MONTHS))
        pd.testing.assert_series_equal(base.loc[m], after.loc[m])

    def test_gate_fires_on_a_vol_burst(self):
        # Non-vacuity: if the gate never halves anything, the property above proves little about it.
        fired = 0
        for seed in range(12):
            rates, fx = _world(seed)
            w = fxc.invvol_gated_weights(rates, fx, MONTHS, fxc.decision_dates(fx.index, MONTHS))
            fired += int((w.abs().sum(axis=1) < 1.99).sum())
        self.assertGreater(fired, 0)

    def test_weights_are_dollar_neutral_and_gross_two(self):
        rates, fx = _world(1)
        a = fxc.rank_weights(rates, MONTHS)
        self.assertTrue(np.allclose(a.sum(axis=1), 0.0))
        self.assertTrue(np.allclose(a.abs().sum(axis=1), 2.0))


if __name__ == "__main__":
    unittest.main()
