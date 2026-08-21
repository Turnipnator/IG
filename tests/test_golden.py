"""Golden-value regression tests.

Frozen input -> exact expected output. These exist for the class of bug that
produces no exception and no log line: a refactor, a library bump, or a
one-character slice change that quietly shifts every number the bot acts on.
The htf_series look-ahead join and the misframed Donchian trail were both
found weeks late, by hand, from P&L that looked wrong.

WHAT THESE DO NOT DO: a golden file freezes CURRENT behaviour, so it would
enshrine a bug just as faithfully as correct code. They are a change detector,
not a correctness proof. The correctness claims live in
test_breakout_properties.py, which asserts what must be true regardless.

Regenerating: run `python -m tests.test_golden --regenerate`. Do that ONLY
when a diff has been reviewed and the change is intended — a green suite after
a blind regenerate means nothing at all.
"""

import json
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src.breakout import BREAKOUT_CONFIGS, arm_channel, current_channel, exit_channel
from src.indicators import (calculate_adx, calculate_atr, calculate_bollinger_bands,
                            calculate_ema, calculate_macd, calculate_rsi,
                            calculate_sma, calculate_stochastic)
from tests.helpers import golden_path, load_candles

GBP = "CS.D.GBPUSD.TODAY.IP"
PLACES = 8


def _tail(series, k=5):
    """Last k finite values, rounded. Rounding at 8dp keeps this stable against
    last-bit float noise while still catching any real change."""
    vals = pd.Series(series).astype("float64").dropna().tolist()
    return [round(v, PLACES) for v in vals[-k:]]


def compute_indicator_goldens() -> dict:
    df = load_candles("gbpusd_1h.jsonl")
    macd = calculate_macd(df["close"])
    bb = calculate_bollinger_bands(df["close"])
    st = calculate_stochastic(df["high"], df["low"], df["close"])
    out = {
        "sma_20": _tail(calculate_sma(df["close"], 20)),
        "ema_21": _tail(calculate_ema(df["close"], 21)),
        "rsi_14": _tail(calculate_rsi(df["close"], 14)),
        "atr_14": _tail(calculate_atr(df["high"], df["low"], df["close"], 14)),
        "adx_14": _tail(calculate_adx(df["high"], df["low"], df["close"], 14)),
    }
    for i, s in enumerate(macd if isinstance(macd, tuple) else [macd]):
        out[f"macd_{i}"] = _tail(s)
    for i, s in enumerate(bb if isinstance(bb, tuple) else [bb]):
        out[f"bb_{i}"] = _tail(s)
    for i, s in enumerate(st if isinstance(st, tuple) else [st]):
        out[f"stoch_{i}"] = _tail(s)
    return out


def compute_breakout_goldens() -> dict:
    """Channel levels sampled at several points along the fixture, so a change
    to the slice convention shows up as many diffs rather than one."""
    df = load_candles("gbpusd_1h.jsonl")
    cfg = BREAKOUT_CONFIGS[GBP]
    samples = {}
    for end in (cfg.n + 5, cfg.n + 40, len(df) // 2, len(df)):
        window = df.iloc[:end]
        ch = current_channel(window, GBP)
        armed = arm_channel(window, _market(), "BULLISH")
        samples[str(end)] = {
            "channel_upper": round(ch[0], PLACES) if ch else None,
            "channel_lower": round(ch[1], PLACES) if ch else None,
            "exit_buy": round(exit_channel(window, GBP, "BUY"), PLACES),
            "exit_sell": round(exit_channel(window, GBP, "SELL"), PLACES),
            "armed_stop_distance": armed.stop_distance if armed else None,
            "armed_atr": armed.atr if armed else None,
            "armed_bar_time": armed.bar_time if armed else None,
        }
    return {"epic": GBP, "n": cfg.n, "m": cfg.m, "samples": samples}


def _market():
    from config import MARKETS
    return next(m for m in MARKETS if m.epic == GBP)


GOLDENS = {
    "golden_indicators.json": compute_indicator_goldens,
    "golden_breakout.json": compute_breakout_goldens,
}


class TestGoldens(unittest.TestCase):
    def test_indicators_match_golden(self):
        expected = json.loads(golden_path("golden_indicators.json").read_text())
        actual = compute_indicator_goldens()
        self.assertEqual(
            sorted(expected), sorted(actual),
            "indicator set changed — regenerate deliberately if that is intended",
        )
        for key in expected:
            self.assertEqual(actual[key], expected[key], f"{key} drifted from its golden value")

    def test_breakout_levels_match_golden(self):
        expected = json.loads(golden_path("golden_breakout.json").read_text())
        actual = compute_breakout_goldens()
        self.assertEqual(expected["n"], actual["n"], "BREAKOUT_CONFIGS n changed for " + GBP)
        self.assertEqual(expected["m"], actual["m"], "exit lookback m changed for " + GBP)
        for end, exp in expected["samples"].items():
            self.assertEqual(actual["samples"][end], exp, f"breakout values drifted at bar {end}")

    def test_fixtures_are_unmodified(self):
        """The goldens are only meaningful against the exact input that
        produced them. Pins the fixtures' shape and endpoints so an
        accidentally re-pulled or truncated fixture fails loudly here rather
        than silently invalidating every golden above."""
        for name, rows, first, last in (
            ("gbpusd_1h.jsonl", 400, "2026-06-08 07:00:00", "2026-06-30 11:00:00"),
            ("gold_5m.jsonl", 900, "2026-06-12 02:45:00", "2026-06-17 15:35:00"),
        ):
            df = load_candles(name)
            self.assertEqual(len(df), rows, f"{name}: row count changed")
            self.assertEqual(str(df["date"].iloc[0]), first, f"{name}: first candle changed")
            self.assertEqual(str(df["date"].iloc[-1]), last, f"{name}: last candle changed")


def regenerate():
    for name, fn in GOLDENS.items():
        golden_path(name).write_text(json.dumps(fn(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {name}")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        unittest.main()
