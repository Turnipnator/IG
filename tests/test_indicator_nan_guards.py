"""NaN indicator values must never reach a trading decision.

Every guard in the signal path is a comparison, and every comparison against NaN
is False. So a NaN indicator does not trip a guard — it walks straight through
all of them:

    corrupted-data guard   nan > max_sane -> False    nan <= 0 -> False
    ADX ranging filter     nan < threshold -> False   (i.e. "trend is strong")
    stop ceiling           nan > max_stop -> False    (no cap applied)
    risk_manager           nan <= 0 -> False          (sizing proceeds)

The reachable source is warm-up, not corruption. calculate_adx double-smooths, so
it is NaN until ~2*period bars (first value at n=27 for period 14) — but
analyze() only gates on len(df) < strategy.ema_slow, and six profiles set that
BELOW 27: gold at 21 and all five indices_* at 26, which is the live momentum
book. 26 bars of ordinary trending S&P data therefore clears the "Insufficient
data" gate, skips the ADX filter entirely, and returns a BUY at 61% confidence
whose own reason string reads "ADX=nan". Stop and limit are finite in that case
(ATR warms up in 14 bars), so the order is well-formed and would fill.

The streaming path is protected by an external len(df) < 50 check at
main.py:1502. The polling path had none — and polling is a state this bot has
run in undetected for 2.5 days.
"""

import math
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import config as config_mod
import src.strategy as strategy_mod
from config import MARKETS, get_strategy_for_market, load_trading_config
from src.risk_manager import RiskManager
from src.strategy import Signal, TradingStrategy, should_close_position


def frame(n, seed=1, drift=0.35, vol=0.9):
    """n bars of ordinary trending OHLC. Nothing here is corrupt or degenerate."""
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(drift, vol, n))
    return pd.DataFrame({
        "date": pd.date_range("2026-08-01", periods=n, freq="5min"),
        "open": close, "high": close + 0.8, "low": close - 0.8,
        "close": close, "volume": 1000,
    })


class TestNaNCannotProduceASignal(unittest.TestCase):
    def setUp(self):
        self.strategy = TradingStrategy()
        self.indices = [m for m in MARKETS if get_strategy_for_market(m).ema_slow == 26]
        self.assertTrue(self.indices, "expected at least one ema_slow=26 market")

    def test_the_known_nan_adx_signal_is_now_held(self):
        """Exact reproduction: 26 bars, S&P profile, seed 1 -> was BUY with ADX=nan."""
        market = self.indices[0]
        df = frame(26, seed=1)
        sig = self.strategy.analyze(df, market, float(df["close"].iloc[-1]), htf_trend="BULLISH")
        self.assertEqual(
            Signal.HOLD, sig.signal,
            f"traded on a NaN indicator: {sig.reason}",
        )

    def test_no_frame_shorter_than_the_adx_warmup_can_signal(self):
        """Sweep the whole warm-up window on every profile, both HTF directions."""
        for market in MARKETS:
            slow = get_strategy_for_market(market).ema_slow
            for n in range(max(2, slow - 2), 30):
                for htf in ("BULLISH", "BEARISH"):
                    for seed in (1, 2, 3):
                        df = frame(n, seed=seed, drift=0.35 if htf == "BULLISH" else -0.35)
                        sig = self.strategy.analyze(
                            df, market, float(df["close"].iloc[-1]), htf_trend=htf
                        )
                        if sig.signal is not Signal.HOLD:
                            self.assertFalse(
                                _nan(sig.adx),
                                f"{market.name} n={n} {htf} seed={seed} signalled "
                                f"{sig.signal.value} on ADX=nan: {sig.reason}",
                            )

    def test_stop_and_limit_are_always_finite(self):
        """A NaN stop becomes a NaN size: max(nan*mult, min_stop) is nan, and
        risk/nan is nan. Nothing downstream tests for it."""
        for market in MARKETS:
            for n in (2, 5, 14, 20, 26, 30, 60):
                df = frame(n, seed=5)
                sig = self.strategy.analyze(df, market, float(df["close"].iloc[-1]))
                self.assertTrue(
                    math.isfinite(sig.stop_distance) and math.isfinite(sig.limit_distance),
                    f"{market.name} n={n}: non-finite stop/limit "
                    f"({sig.stop_distance}, {sig.limit_distance})",
                )


class TestWarmupConstantsMatchTheIndicators(unittest.TestCase):
    """The guards are only as honest as these numbers. add_all_indicators calls
    calculate_adx/calculate_atr with their DEFAULT period, so changing that default
    would silently make the constants too small and reopen the hole."""

    def _first_valid(self, series):
        idx = series.first_valid_index()
        return None if idx is None else int(idx) + 1  # bars needed, 1-based

    def test_adx_warmup_constant_is_not_optimistic(self):
        from src.indicators import calculate_adx
        df = frame(200, seed=11)
        needed = self._first_valid(calculate_adx(df["high"], df["low"], df["close"]))
        self.assertIsNotNone(needed, "ADX never became valid over 200 bars")
        self.assertGreaterEqual(
            strategy_mod.ADX_WARMUP_BARS, needed,
            f"ADX_WARMUP_BARS={strategy_mod.ADX_WARMUP_BARS} but ADX first valid at "
            f"{needed} bars — the guard would let NaN through",
        )

    def test_macd_hist_warmup_constant_is_not_optimistic(self):
        from src.indicators import add_all_indicators
        df = add_all_indicators(frame(200, seed=11), {"rsi_period": 7})
        needed = self._first_valid(df["macd_hist"].reset_index(drop=True))
        if needed is not None:
            self.assertGreaterEqual(strategy_mod.MACD_HIST_WARMUP_BARS, needed)

    def test_every_profile_gates_below_the_adx_warmup(self):
        """Documents WHY the NaN gate is needed: the length check in analyze() is
        ema_slow, and six profiles set it below the ADX warm-up. If this ever stops
        being true the NaN gate is still correct, just no longer load-bearing."""
        below = {n: p.ema_slow for n, p in config_mod.STRATEGY_PROFILES.items()
                 if p.ema_slow < strategy_mod.ADX_WARMUP_BARS}
        self.assertEqual(
            {"gold": 21, "indices": 26, "indices_adx35": 26, "indices_selective": 26,
             "indices_wide": 26, "indices_tight": 26},
            below,
            "the set of profiles gating below the ADX warm-up changed — reconfirm "
            "the NaN gate still covers them",
        )


class TestSizingRejectsNonFinite(unittest.TestCase):
    """Last gate before money. `if stop_distance <= 0` does not catch NaN."""

    def test_non_finite_stop_is_not_approved(self):
        rm = RiskManager(load_trading_config())
        market = MARKETS[0]
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(stop=bad):
                ps = rm.calculate_position_size(9200.0, bad, market)
                self.assertFalse(ps.approved, f"approved a position on stop={bad}")
                self.assertTrue(math.isfinite(ps.size), f"size was {ps.size}")
                self.assertEqual(0.0, ps.size)

    def test_ordinary_stop_still_sizes(self):
        """The guard must not reject anything that trades today."""
        rm = RiskManager(load_trading_config())
        ps = rm.calculate_position_size(9200.0, 20.0, MARKETS[0])
        self.assertTrue(ps.approved, ps.reason)
        self.assertGreater(ps.size, 0)


class TestExitPathToleratesShortFrames(unittest.TestCase):
    """NaN on the exit path means every comparison is False, i.e. "don't exit" —
    already the safe direction, with the broker stop still on the book. The bug is
    that len(df) < 3 let it run far below the indicators' warm-up and silently
    no-op. Behaviour on bad data is deliberately unchanged: forcing exits there
    would clip winners, which this book has refuted repeatedly."""

    def test_rsi_exit_still_fires_below_the_adx_and_macd_warmup(self):
        """Regression guard. Gating all three exits on one bar count set to the
        slowest indicator (35) would silently disable the RSI-extreme exit in the
        8-34 bar band, where RSI is perfectly valid. That band is reachable:
        check_positions_from_stream guards only df.empty, so a position adopted
        after a restart is managed while candles are still re-accumulating."""
        market = next(m for m in MARKETS if not get_strategy_for_market(m).use_macd_exit)
        n = 12  # RSI(7) valid; ADX(28) and MACD hist(35) both still NaN
        close = np.concatenate([np.full(3, 100.0), 100 + np.arange(1, n - 2) * 3.0])
        df = pd.DataFrame({
            "date": pd.date_range("2026-08-01", periods=n, freq="5min"),
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": 1000,
        })
        from src.indicators import add_all_indicators
        rsi = add_all_indicators(df, {"rsi_period": 7})["rsi"].iloc[-1]
        self.assertTrue(math.isfinite(rsi), "fixture must have a valid RSI")
        self.assertGreater(rsi, get_strategy_for_market(market).rsi_overbought,
                           f"fixture must be overbought, got RSI={rsi:.1f}")

        close_now, reason = should_close_position(df, "BUY", market=market)
        self.assertTrue(close_now, f"RSI exit did not fire on {n} bars (RSI={rsi:.1f})")
        self.assertIn("RSI overbought", reason)

    def test_short_frames_do_not_close_and_do_not_raise(self):
        for market in MARKETS[:4]:
            for n in (1, 2, 3, 10, 26, 40):
                for direction in ("BUY", "SELL"):
                    df = frame(n, seed=9)
                    close, reason = should_close_position(
                        df, direction, market=market, htf_trend="NEUTRAL"
                    )
                    self.assertIsInstance(close, bool)
                    if close:
                        self.assertNotIn("nan", reason.lower(),
                                         f"{market.name} n={n}: closed on a NaN reason")


class TestBacktestMacdExitMatchesLive(unittest.TestCase):
    """src/backtest.py must not fire a MACD exit that live would hold.

    It used `all(h < 0 for h in last_3 if not pd.isna(h))`. The NaN filter drops
    bars from the generator, so on [finite_neg, NaN, NaN] all() is satisfied by
    ONE element -- one bar of confirmation instead of three. Live cannot: it
    gates the same window on _finite() (402484f) and holds.

    IMPORTANT (verified 2026-08-31, after an initial claim that this was firing
    at every warm-up boundary): the idiom was vacuous but UNREACHABLE through
    calculate_macd. calculate_ema is `ewm(span=..., adjust=False)` with no
    min_periods, so macd_hist is finite from bar 0 -- and stays finite even when
    a close is NaN, because ewm skips NaNs. MACD_HIST_WARMUP_BARS (26+9) marks
    statistical convergence, NOT a NaN boundary; contrast calculate_rsi, which
    sets min_periods and genuinely does emit NaN.

    So the fix changed no backtest result. These tests pin both halves: that
    macd_hist really has no NaN warm-up (so nobody re-derives the wrong
    materiality), and that the predicate now holds rather than firing if a NaN
    ever does reach it from another source.
    """

    def test_macd_hist_has_no_nan_warmup_so_the_old_bug_was_unreachable(self):
        from src.indicators import calculate_macd, calculate_rsi

        for seed in (1, 3, 7):
            df = frame(120, seed=seed)
            _, _, hist = calculate_macd(df["close"])
            self.assertEqual(int(hist.isna().sum()), 0,
                             "macd_hist gained a NaN warm-up — the backtest "
                             "materiality note in this file must be revisited")

        # A NaN close does not propagate either (ewm skips it).
        s = frame(120, seed=2)["close"].copy()
        s.iloc[50] = float("nan")
        _, _, hist = calculate_macd(s)
        self.assertEqual(int(hist.isna().sum()), 0)

        # Contrast: RSI *does* have a real NaN warm-up (min_periods is set).
        self.assertGreater(int(calculate_rsi(frame(120, seed=1)["close"], 7).isna().sum()), 0)

    def test_predicate_holds_on_nan_and_still_exits_when_fully_aligned(self):
        """The shipped expression from src/backtest.py."""
        def fires(window):
            return (all(np.isfinite(h) for h in window)
                    and all(h < 0 for h in window))

        self.assertFalse(fires([-1.0, float("nan"), float("nan")]))
        self.assertFalse(fires([float("nan")] * 3))
        self.assertFalse(fires([-1.0, -1.0, float("nan")]))
        self.assertFalse(fires([-1.0, -1.0, 1.0]))   # not 3 consecutive
        self.assertTrue(fires([-1.0, -2.0, -0.5]))   # fully warm and aligned

        # The old idiom fired on the first of those -- that was the defect.
        self.assertTrue(all(h < 0 for h in [-1.0, float("nan"), float("nan")]
                            if not pd.isna(h)))

    def test_backtest_source_no_longer_filters_nan_out_of_the_window(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "backtest.py").read_text()
        self.assertNotIn("for h in last_3 if not pd.isna(h)", src)
        self.assertIn("all(np.isfinite(h) for h in last_3)", src)


def _nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    unittest.main()
