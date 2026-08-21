"""Fixture loading for the test suite.

Fixtures are real, contiguous slices of the live candle archive (the head of
each file, so they never shift as new candles append). Real bars matter here:
every silent bug this suite exists to catch — the misframed Donchian trail, the
look-ahead daily join — was about timestamp and frame handling, which synthetic
data on a clean RangeIndex cannot reproduce.
"""

import json
import pathlib

import pandas as pd

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_candles(name: str) -> pd.DataFrame:
    """Load a .jsonl fixture into the frame shape the bot passes around.

    Columns match what analyze_breakout/exit_channel/_trail_frame expect:
    a `date` datetime column plus open/high/low/close/volume.
    """
    rows = [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"])
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def golden_path(name: str) -> pathlib.Path:
    return FIXTURES / name


def read_golden(name: str) -> dict:
    return json.loads(golden_path(name).read_text())
