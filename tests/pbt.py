"""Hypothesis setup and shared strategies for the generated property tests.

Two profiles, chosen by the HYPOTHESIS_PROFILE environment variable:

  ci    (the default) — derandomize=True, so every run replays the SAME examples.
        CI's `verify` job is the DEPLOY GATE: rebuild-watcher.sh refuses any commit
        without it. A random seed that happened on a new counterexample during an
        unrelated push would block an urgent fix for a reason that commit did not
        cause. The gate stays reproducible, and exploration happens in the next one.
  deep  — random, 20x the examples. health.yml runs it weekly and raises an issue
        on failure. By hand:
            HYPOTHESIS_PROFILE=deep python -m unittest tests.test_lookahead_properties
        A failure prints an @reproduce_failure(...) line that replays it exactly.

Hypothesis is test-only (requirements-test.txt) and is never in the image. tests/ IS
in the image, so the golden suite can be run against the container. Inside the
container these modules therefore skip. Anywhere else a missing Hypothesis is an
error, not a skip: a property suite that quietly stops running is worse than none.
"""

import os
import pathlib
import unittest

import pandas as pd

IN_CONTAINER = pathlib.Path(__file__).resolve().parent.parent == pathlib.Path("/app")

try:
    from hypothesis import HealthCheck, Phase, settings, strategies as st
except ImportError as e:
    if IN_CONTAINER:
        raise unittest.SkipTest("Hypothesis is test-only and not installed in the image")
    raise ImportError("hypothesis is missing: pip install -r requirements-test.txt") from e

# Frames of a few hundred generated floats are big by Hypothesis standards;
# too_slow and large_base_example flag that size, not a problem with the tests.
_QUIET = [HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.data_too_large]
# The gate still SHRINKS a failure (the minimal counterexample is the diagnosis)
# but skips the explain phase. On these frames it re-runs the failing example
# until Hypothesis's 5-minute cap, per failing test: measured at 300s for one
# look-ahead mutant, against seconds without it. The deep run keeps it.
settings.register_profile("ci", derandomize=True, max_examples=50, deadline=None,
                          database=None, print_blob=True, suppress_health_check=_QUIET,
                          phases=[p for p in Phase if p is not Phase.explain])
settings.register_profile("deep", max_examples=1000, deadline=None,
                          database=None, print_blob=True, suppress_health_check=_QUIET)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

OHLC = ["open", "high", "low", "close"]

# Mostly contiguous bars, sometimes a gap. sampled_from shrinks toward the FIRST
# element, so a failing example shrinks toward an unbroken series.
_STEPS = (1,) * 12 + (2, 3, 12, 60)


def price_moves(n: int, max_move: float = 0.03):
    """n bar-to-bar returns, every one NON-ZERO.

    Hypothesis favours 'simple' floats, and for a return the simplest is 0.0.
    Drawn from st.floats(-x, x), most examples came out as flat lines, and on a
    flat line a leaked future candle changes nothing. The session-bar test
    passed a deliberate look-ahead mutant for exactly that reason before this
    existed. A property can only see the future in a series that moves."""
    move = st.tuples(st.booleans(), st.floats(1e-4, max_move)).map(lambda t: t[1] if t[0] else -t[1])
    return st.lists(move, min_size=n, max_size=n)


@st.composite
def ohlc_frames(draw, min_bars: int, max_bars: int, freq: str = "1h",
                start: str = "2026-03-02 00:00", gaps: bool = True) -> pd.DataFrame:
    """A frame shaped like the ones the bot passes around (date, OHLC, volume).

    Internally consistent bars (low <= open, close <= high; prices > 0), any
    price scale from FX to indices, timestamps on a `freq` grid at bar START,
    with optional gaps (weekends, outages) so nothing can assume a clean range.
    """
    n = draw(st.integers(min_bars, max_bars))
    p = draw(st.floats(0.5, 20_000))
    moves = draw(price_moves(n))
    wicks = draw(st.lists(st.tuples(st.floats(0, 0.01), st.floats(0, 0.01)), min_size=n, max_size=n))
    steps = draw(st.lists(st.sampled_from(_STEPS if gaps else (1,)), min_size=n, max_size=n))
    step = pd.Timedelta(freq)
    t, rows = pd.Timestamp(start), []
    for i, (r, (up, down)) in enumerate(zip(moves, wicks)):
        if i:
            t += step * steps[i]
        o, c = p, max(p * (1 + r), 1e-6)
        rows.append({"date": t, "open": o, "high": max(o, c) * (1 + up),
                     "low": min(o, c) * (1 - down), "close": c, "volume": 0})
        p = c
    return pd.DataFrame(rows)
