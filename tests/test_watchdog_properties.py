"""Generated properties for the streaming watchdog's liveness decision.

The 2026-09-18 outage was not a missing check — the watchdog HAD a staleness
check and it fired, correctly, at 19:35:41. The failure was that its inputs were
derived from the feed it was judging, so its own recovery attempt reset them to
values it read as healthy. It then stayed silent for 62 hours.

One example test pins the exact shape that happened. These properties state the
invariant that shape violated, and let Hypothesis look for any OTHER arrangement
of ticks, market states and timings that reaches the same silent state:

    if the socket is connected, and no market has produced a tick within the
    staleness grace, and at least one market is really open, the watchdog must
    trip — no matter how the individual markets' last_update / market_state
    happen to be arranged.

The converse matters just as much, because a watchdog that cries wolf gets
muted and then ignored: with every market genuinely closed, it must stay silent.

Replayed deterministically in CI and explored randomly every week; see tests/pbt.py.
"""

import unittest
import warnings
from datetime import datetime, timedelta
from unittest import mock

warnings.filterwarnings("ignore")

from tests.pbt import st
from hypothesis import given, settings

import main
from src.streaming import IGStreamService, MarketStream


EPICS = ["EPIC.A", "EPIC.B", "EPIC.C", "EPIC.D"]

# Any per-market state the stream could be left in. "CLOSED" is the constructor
# default a resubscribe resets every market to, and is the value that made the
# weekend suppressor misfire on a Friday evening.
MARKET_STATES = st.sampled_from(["TRADEABLE", "CLOSED", "EDITS_ONLY", "SUSPENDED"])


def _service(states, tick_ages_s, subscribed_mins_ago):
    """Build a service in an arbitrary — possibly incoherent — stream state."""
    svc = IGStreamService.__new__(IGStreamService)
    svc.markets = {}
    now = datetime.now()
    for epic, state, age in zip(EPICS, states, tick_ages_s):
        m = MarketStream(epic=epic, name=epic)
        m.market_state = state
        m.last_update = None if age is None else now - timedelta(seconds=age)
        svc.markets[epic] = m
    svc.connected = True
    svc.connection_status = "CONNECTED:WS-STREAMING"
    svc.subscription = None
    svc.subscription_group = "MARKET"
    svc.subscription_failed = False
    svc.subscribed_at = now - timedelta(minutes=subscribed_mins_ago)
    return svc


def _drive(svc, rest_status):
    """Two watchdog passes (the loop is per-minute; staleness needs two)."""
    fake_client = mock.Mock()
    fake_client.get_market_info.return_value = mock.Mock(market_status=rest_status)
    refresh = mock.Mock()
    saved = (
        main._streaming_disconnect_since,
        main._streaming_stale_since,
        main._streaming_recovery_attempted_at,
        main._watchdog_rest_probed_at,
        main._subscription_alert_last,
    )
    main._streaming_disconnect_since = None
    main._streaming_stale_since = None
    main._streaming_recovery_attempted_at = None
    main._watchdog_rest_probed_at = None
    main._subscription_alert_last = None
    try:
        with mock.patch.object(main, "stream_service", svc), \
             mock.patch.object(main, "client", fake_client), \
             mock.patch.object(main, "refresh_session", refresh), \
             mock.patch.object(main, "telegram", None):
            main._streaming_watchdog()
            main._streaming_watchdog()
    finally:
        (
            main._streaming_disconnect_since,
            main._streaming_stale_since,
            main._streaming_recovery_attempted_at,
            main._watchdog_rest_probed_at,
            main._subscription_alert_last,
        ) = saved
    return refresh.called


class TestSilentFeedAlwaysTrips(unittest.TestCase):

    @given(
        states=st.lists(MARKET_STATES, min_size=len(EPICS), max_size=len(EPICS)),
        subscribed_mins_ago=st.integers(min_value=5, max_value=5000),
    )
    def test_no_tick_ever_while_a_market_is_open_must_trip(
        self, states, subscribed_mins_ago
    ):
        """The 62-hour state, generalised over every stream-state arrangement.

        No market has ticked since the subscribe. Whatever MARKET_STATE the
        stream happens to be holding is stale by definition, so the broker is
        asked instead — and the broker says a market is open.
        """
        svc = _service(states, [None] * len(EPICS), subscribed_mins_ago)
        self.assertTrue(
            _drive(svc, rest_status="TRADEABLE"),
            f"silent feed for {subscribed_mins_ago}m with markets open did not trip "
            f"(stream states: {states})",
        )

    @given(
        states=st.lists(MARKET_STATES, min_size=len(EPICS), max_size=len(EPICS)),
        tick_ages=st.lists(
            st.integers(min_value=301, max_value=500000),
            min_size=len(EPICS), max_size=len(EPICS),
        ),
    )
    def test_all_ticks_older_than_the_grace_must_trip(self, states, tick_ages):
        """Ticks exist but all are stale; here the stream's own state is usable."""
        states = ["TRADEABLE"] + list(states)[1:]  # at least one market open
        svc = _service(states, tick_ages, subscribed_mins_ago=10_000)
        self.assertTrue(
            _drive(svc, rest_status="TRADEABLE"),
            f"every tick older than the grace did not trip (ages: {tick_ages})",
        )


class TestQuietWhenNothingIsWrong(unittest.TestCase):
    """A watchdog that false-alarms gets muted, and then it protects nothing."""

    @given(subscribed_mins_ago=st.integers(min_value=5, max_value=5000))
    def test_genuine_weekend_never_trips(self, subscribed_mins_ago):
        svc = _service(["CLOSED"] * len(EPICS), [None] * len(EPICS), subscribed_mins_ago)
        self.assertFalse(
            _drive(svc, rest_status="CLOSED"),
            "no ticks with every market closed at the BROKER is a weekend",
        )

    @given(
        states=st.lists(MARKET_STATES, min_size=len(EPICS), max_size=len(EPICS)),
        fresh_age=st.integers(min_value=0, max_value=170),
    )
    def test_a_single_fresh_tick_keeps_it_quiet(self, states, fresh_age):
        """One market ticking inside the grace proves the feed is alive."""
        ages = [fresh_age] + [None] * (len(EPICS) - 1)
        svc = _service(states, ages, subscribed_mins_ago=5000)
        self.assertFalse(
            _drive(svc, rest_status="TRADEABLE"),
            f"a tick {fresh_age}s old means the feed works; must not trip",
        )


class TestRefusalIsAlwaysAFault(unittest.TestCase):

    @given(
        states=st.lists(MARKET_STATES, min_size=len(EPICS), max_size=len(EPICS)),
        tick_ages=st.lists(
            st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
            min_size=len(EPICS), max_size=len(EPICS),
        ),
    )
    @settings(max_examples=25)
    def test_refused_subscription_trips_regardless_of_tick_state(self, states, tick_ages):
        """A refusal is a fact, not an inference — no tick pattern excuses it."""
        svc = _service(states, tick_ages, subscribed_mins_ago=1)
        svc.subscription_failed = True
        self.assertTrue(
            _drive(svc, rest_status="TRADEABLE"),
            "every group refused must trip even while stale ticks linger in memory",
        )


if __name__ == "__main__":
    unittest.main()
