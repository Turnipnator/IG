"""The Donchian trail must only ever TIGHTEN.

_update_breakout_trail is the code that moves real money's stop loss. Its
contract — "only ever tighten, never loosen below the entry stop" — is the
single line standing between a trailing stop and a widening one, and it is
enforced by two small comparisons that a refactor could invert without any
test noticing.

Exercised through the real function with a stub broker client, rather than by
reimplementing the rule in the test (which would only assert that the test
agrees with itself).
"""

import unittest
import warnings

warnings.filterwarnings("ignore")

import main
from src.breakout import BREAKOUT_CONFIGS, exit_channel
from tests.helpers import load_candles

GBP = "CS.D.GBPUSD.TODAY.IP"


class StubClient:
    """Records stop updates instead of calling IG."""

    def __init__(self, succeed=True):
        self.calls = []
        self.succeed = succeed

    def update_position_stop(self, deal_id, new_stop, limit_level, epic=None, bid=None, offer=None):
        self.calls.append({"deal_id": deal_id, "new_stop": new_stop, "epic": epic})
        return self.succeed


class StubPosition:
    def __init__(self, stop_level, direction="BUY", epic=GBP, deal_id="TEST-DEAL-1"):
        self.epic = epic
        self.direction = direction
        self.stop_level = stop_level
        self.limit_level = None
        self.deal_id = deal_id


class TrailTestCase(unittest.TestCase):
    def setUp(self):
        self.df = load_candles("gbpusd_1h.jsonl")
        self.level = exit_channel(self.df, GBP, "BUY")
        self.assertIsNotNone(self.level, "fixture must produce a trail level")

        self._real_client = main.client
        self._real_stream = getattr(main, "stream_service", None)
        self._real_levels = dict(main.trailing_stop_levels)
        self.stub = StubClient()
        main.client = self.stub
        main.stream_service = None           # trail must not depend on a live quote
        main.trailing_stop_levels.clear()

    def tearDown(self):
        main.client = self._real_client
        main.stream_service = self._real_stream
        main.trailing_stop_levels.clear()
        main.trailing_stop_levels.update(self._real_levels)


class TestRatchetDirection(TrailTestCase):
    def test_long_stop_moves_up_when_the_channel_is_higher(self):
        pos = StubPosition(stop_level=self.level - 50, direction="BUY")
        main._update_breakout_trail(pos, self.df)
        self.assertEqual(len(self.stub.calls), 1, "expected the stop to be tightened")
        self.assertAlmostEqual(self.stub.calls[0]["new_stop"], round(self.level, 1), places=1)

    def test_long_stop_is_NOT_loosened_when_the_channel_is_lower(self):
        """The core safety property: a channel below the current stop must be
        ignored, never applied."""
        pos = StubPosition(stop_level=self.level + 50, direction="BUY")
        main._update_breakout_trail(pos, self.df)
        self.assertEqual(self.stub.calls, [], "trail LOOSENED a long stop downward")

    def test_short_stop_moves_down_only(self):
        level = exit_channel(self.df, GBP, "SELL")
        tighten = StubPosition(stop_level=level + 50, direction="SELL", deal_id="D-SHORT-1")
        main._update_breakout_trail(tighten, self.df)
        self.assertEqual(len(self.stub.calls), 1, "expected a short stop to tighten downward")

        self.stub.calls.clear()
        main.trailing_stop_levels.clear()
        loosen = StubPosition(stop_level=level - 50, direction="SELL", deal_id="D-SHORT-2")
        main._update_breakout_trail(loosen, self.df)
        self.assertEqual(self.stub.calls, [], "trail LOOSENED a short stop upward")

    def test_equal_level_is_a_no_op(self):
        """Guards against `<` drifting to `<=`: re-running on an unchanged
        channel must not spend an API call or re-log a move."""
        pos = StubPosition(stop_level=round(self.level, 1), direction="BUY")
        main._update_breakout_trail(pos, self.df)
        self.assertEqual(self.stub.calls, [], "re-applied an identical stop level")


class TestTrailRefusals(TrailTestCase):
    def test_missing_stop_level_is_ignored(self):
        main._update_breakout_trail(StubPosition(stop_level=None), self.df)
        self.assertEqual(self.stub.calls, [])

    def test_short_frame_never_widens_the_stop(self):
        """A frame too short for exit_channel must leave the broker stop alone.
        Returning early is correct; falling through to a None level and
        writing it would remove the stop entirely."""
        pos = StubPosition(stop_level=self.level - 50, direction="BUY")
        main._update_breakout_trail(pos, self.df.iloc[:5])
        self.assertEqual(self.stub.calls, [], "acted on a frame too short to measure")

    def test_a_sequence_of_frames_never_loosens(self):
        """End-to-end ratchet: walk the fixture forward bar by bar and assert
        the recorded stop is monotonically non-decreasing for a long."""
        pos = StubPosition(stop_level=0.0, direction="BUY", deal_id="D-WALK")
        cfg = BREAKOUT_CONFIGS[GBP]
        applied = []
        for end in range(cfg.m + 2, len(self.df)):
            main._update_breakout_trail(pos, self.df.iloc[:end])
            if self.stub.calls:
                applied.append(self.stub.calls[-1]["new_stop"])
        self.assertGreater(len(applied), 3, "fixture produced too few ratchets to be a real test")
        self.assertEqual(applied, sorted(applied), "long trail moved DOWN at some point")

    def test_failed_broker_call_does_not_record_the_level(self):
        """If IG rejects the update, trailing_stop_levels must not claim it
        succeeded — otherwise the next cycle compares against a stop the
        broker never accepted and skips the retry."""
        main.client = StubClient(succeed=False)
        pos = StubPosition(stop_level=self.level - 50, direction="BUY", deal_id="D-FAIL")
        main._update_breakout_trail(pos, self.df)
        self.assertNotIn("D-FAIL", main.trailing_stop_levels,
                         "recorded a stop level the broker rejected")


if __name__ == "__main__":
    unittest.main()
