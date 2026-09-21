"""Regression tests for the 2026-09-18 silent streaming outage.

The feed died at 19:31 on a Friday and nothing noticed until Monday morning —
62 hours, through 12 six-hourly reconnects, with the container reporting
`healthy` and the screener cheerfully re-scoring frozen candles the whole time.

Nothing here tests "can the watchdog spot a dead feed" in the abstract; the old
watchdog could, and did, at 19:35:41. What it could not do was spot it a SECOND
time, because its own recovery attempt wiped the state it used as evidence. So
these tests are about the disarm paths specifically:

  1. resubscribing resets every MarketStream, so `last_update` returns to None
     and the old `most_recent_tick_age() is not None` guard went permanently
     False.
  2. the same reset puts `market_state` back to its "CLOSED" default, so
     `tradeable_market_count()` returned 0 and the weekend suppressor engaged
     on a Friday evening.
  3. a refused subscription was logged and then read by nothing at all, while
     the socket stayed CONNECTED so the disconnect guard never fired either.

Each is a way of asking the feed whether the feed is alive. The fix is that
silence is timed from the subscribe, and "are markets open?" is answered by the
broker over REST when no tick has ever arrived to answer it.
"""

import unittest
from datetime import datetime, timedelta
from unittest import mock

from src.streaming import IGStreamService, MarketStream


def _service(markets=("EPIC.A", "EPIC.B")):
    """An IGStreamService with the Lightstreamer import requirement bypassed."""
    svc = IGStreamService.__new__(IGStreamService)
    svc.markets = {e: MarketStream(epic=e, name=e) for e in markets}
    svc.connected = True
    svc.connection_status = "CONNECTED:WS-STREAMING"
    svc.subscription = None
    svc.subscription_group = "MARKET"
    svc.subscription_failed = False
    svc.subscribed_at = datetime.now()
    return svc


class TestFeedSilence(unittest.TestCase):
    """feed_silence() is the input that replaced most_recent_tick_age()."""

    def test_never_ticked_since_subscribe_is_not_treated_as_benign(self):
        # The exact 09-18 state: subscribed 10 minutes ago, not one tick since.
        svc = _service()
        svc.subscribed_at = datetime.now() - timedelta(minutes=10)

        self.assertIsNone(
            svc.most_recent_tick_age(),
            "precondition: the old input really does go None after a resubscribe",
        )
        silence = svc.feed_silence()
        self.assertIsNotNone(silence, "silence must be measurable without any tick")
        self.assertGreater(silence, timedelta(minutes=9))

    def test_silence_tracks_the_newest_tick_once_ticks_exist(self):
        svc = _service()
        svc.markets["EPIC.A"].last_update = datetime.now() - timedelta(minutes=30)
        svc.markets["EPIC.B"].last_update = datetime.now() - timedelta(seconds=20)
        self.assertLess(svc.feed_silence(), timedelta(minutes=1))

    def test_none_only_before_any_subscribe(self):
        svc = _service()
        svc.subscribed_at = None
        self.assertIsNone(svc.feed_silence())

    def test_has_ever_ticked_distinguishes_the_two_zero_states(self):
        svc = _service()
        self.assertFalse(svc.has_ever_ticked())
        svc.markets["EPIC.A"].last_update = datetime.now()
        self.assertTrue(svc.has_ever_ticked())


class TestTradeableCountIsNotSelfReferential(unittest.TestCase):
    """The weekend suppressor must not be fed by the feed it suppresses for."""

    def test_default_market_state_reads_zero_tradeable(self):
        # Not a bug in itself — but it is why 0 cannot be read as "weekend".
        svc = _service()
        self.assertEqual(svc.tradeable_market_count(), 0)
        self.assertFalse(svc.has_ever_ticked())

    def test_watchdog_asks_rest_when_nothing_has_ticked(self):
        import main

        svc = _service()
        fake_client = mock.Mock()
        fake_client.get_market_info.return_value = mock.Mock(market_status="TRADEABLE")

        with mock.patch.object(main, "stream_service", svc), \
             mock.patch.object(main, "client", fake_client), \
             mock.patch.object(main, "_watchdog_rest_probed_at", None):
            count = main._tradeable_count_for_watchdog()

        self.assertGreater(
            count, 0,
            "with no ticks ever, the stream's 0 is uninformative and REST must be consulted",
        )
        self.assertTrue(fake_client.get_market_info.called)

    def test_watchdog_trusts_the_stream_once_ticks_exist(self):
        import main

        svc = _service()
        svc.markets["EPIC.A"].last_update = datetime.now()
        svc.markets["EPIC.A"].market_state = "TRADEABLE"
        fake_client = mock.Mock()

        with mock.patch.object(main, "stream_service", svc), \
             mock.patch.object(main, "client", fake_client):
            count = main._tradeable_count_for_watchdog()

        self.assertEqual(count, 1)
        fake_client.get_market_info.assert_not_called()  # free path, no rate-limit cost

    def test_rest_probe_is_cached(self):
        import main

        svc = _service()
        fake_client = mock.Mock()
        fake_client.get_market_info.return_value = mock.Mock(market_status="TRADEABLE")

        with mock.patch.object(main, "stream_service", svc), \
             mock.patch.object(main, "client", fake_client), \
             mock.patch.object(main, "_watchdog_rest_probed_at", None):
            main._tradeable_count_for_watchdog()
            calls_after_first = fake_client.get_market_info.call_count
            main._tradeable_count_for_watchdog()

        self.assertEqual(
            fake_client.get_market_info.call_count, calls_after_first,
            "the probe shares IG's per-key rate limit with the live bot — it must cache",
        )


class TestWatchdogTrips(unittest.TestCase):
    """End to end: the 09-18 state must trip, and a real weekend must not."""

    def setUp(self):
        import main
        self.main = main
        for attr in ("_streaming_disconnect_since", "_streaming_stale_since",
                     "_streaming_recovery_attempted_at", "_watchdog_rest_probed_at",
                     "_subscription_alert_last"):
            self.addCleanup(setattr, main, attr, getattr(main, attr))
            setattr(main, attr, None)

    def _run(self, svc, market_status="TRADEABLE", passes=2):
        """Drive the watchdog as periodic_tasks() does — once per minute.

        The staleness path needs two CONSECUTIVE stale observations before it
        trips: the first only starts the clock (stale_duration == 0). That is
        deliberate anti-flap behaviour and is why the real 09-18 trip read "no
        ticks for 0:04:03" rather than 0:03:00. A single-pass test would be
        testing a state the production loop never stops in.
        """
        fake_client = mock.Mock()
        fake_client.get_market_info.return_value = mock.Mock(market_status=market_status)
        refresh = mock.Mock()
        with mock.patch.object(self.main, "stream_service", svc), \
             mock.patch.object(self.main, "client", fake_client), \
             mock.patch.object(self.main, "refresh_session", refresh), \
             mock.patch.object(self.main, "telegram", None):
            for _ in range(passes):
                self.main._streaming_watchdog()
        return refresh

    def test_connected_but_never_ticked_while_markets_open_trips(self):
        """THE regression. Old code: silent forever. New code: recovers."""
        svc = _service()
        svc.subscribed_at = datetime.now() - timedelta(minutes=30)
        refresh = self._run(svc, market_status="TRADEABLE")
        self.assertTrue(
            refresh.called,
            "connected + zero ticks for 30 min + markets open must trigger recovery",
        )

    def test_staleness_does_not_trip_on_a_single_observation(self):
        """Anti-flap: one stale reading starts the clock, it does not fire."""
        svc = _service()
        svc.subscribed_at = datetime.now() - timedelta(minutes=30)
        refresh = self._run(svc, market_status="TRADEABLE", passes=1)
        self.assertFalse(refresh.called)

    def test_refused_subscription_trips_immediately(self):
        svc = _service()
        svc.subscription_failed = True
        svc.subscribed_at = datetime.now()  # no time for staleness to build
        refresh = self._run(svc)
        self.assertTrue(refresh.called, "a refused subscription is a definite fault")

    def test_genuine_weekend_stays_quiet(self):
        """The property that keeps the watchdog trusted: silence when correct."""
        svc = _service()
        svc.subscribed_at = datetime.now() - timedelta(hours=40)
        refresh = self._run(svc, market_status="CLOSED")
        self.assertFalse(
            refresh.called,
            "no ticks with every market closed is a weekend, not an outage",
        )

    def test_healthy_feed_stays_quiet(self):
        svc = _service()
        for m in svc.markets.values():
            m.last_update = datetime.now()
            m.market_state = "TRADEABLE"
        refresh = self._run(svc)
        self.assertFalse(refresh.called)

    def test_entitlement_refusal_does_not_crash_loop(self):
        """A restart cannot fix an entitlement; it can only burn API budget."""
        svc = _service()
        svc.subscription_failed = True
        self.main._streaming_recovery_attempted_at = (
            datetime.now() - timedelta(minutes=10)  # well past the grace window
        )
        with mock.patch.object(self.main, "os") as fake_os:
            self._run(svc)
        fake_os._exit.assert_not_called()


class TestSubscriptionGroupLadder(unittest.TestCase):
    """IG has revoked two groups on this key; one hardcoded group is a SPOF."""

    def test_ladder_order_is_cheapest_first(self):
        """Quote groups first; CHART is the entitled fallback, not the default."""
        names = [s.name for s in IGStreamService.SUBSCRIPTION_GROUPS]
        self.assertEqual(names, ["MARKET", "L1", "CHART:TICK"])

    def test_parser_resolves_the_epic_for_every_group_in_the_ladder(self):
        """CHART puts the epic in the MIDDLE — prefix-stripping silently fails."""
        from src.streaming import IGStreamListener

        epic = "IX.D.FTSE.DAILY.IP"
        svc = _service(markets=(epic,))
        for spec in IGStreamService.SUBSCRIPTION_GROUPS:
            item = spec.item(epic)
            svc._item_to_epic = {spec.item(epic): epic}
            listener = IGStreamListener(svc, spec=spec)
            self.assertEqual(
                listener._epic_for(item), epic,
                f"cannot resolve the epic from item {item!r} (group {spec.name})",
            )

    def test_epic_resolves_without_the_map_too(self):
        """Fallback path, for an update arriving before the map is populated."""
        from src.streaming import IGStreamListener, GROUP_CHART_TICK, GROUP_MARKET

        svc = _service()
        svc._item_to_epic = {}
        for spec in (GROUP_MARKET, GROUP_CHART_TICK):
            listener = IGStreamListener(svc, spec=spec)
            self.assertEqual(
                listener._epic_for(spec.item("CS.D.EURUSD.TODAY.IP")),
                "CS.D.EURUSD.TODAY.IP",
            )

    def test_listener_latches_the_verdict_instead_of_only_logging_it(self):
        import threading
        from src.streaming import IGStreamListener

        ev = threading.Event()
        listener = IGStreamListener(None, group="MARKET", outcome=ev)
        listener.onSubscriptionError(21, "Invalid group")

        self.assertTrue(ev.is_set(), "the subscriber must be woken by a refusal")
        self.assertFalse(listener.active)
        self.assertEqual(listener.error[0], 21)

    def test_midlife_refusal_latches_on_the_service(self):
        """The 19:31 shape: socket drops, library re-subscribes, IG refuses."""
        from src.streaming import IGStreamListener

        svc = _service()
        svc.subscription_group = "MARKET"  # bound and live
        listener = IGStreamListener(svc, group="MARKET")
        listener.onSubscriptionError(21, "Invalid group")

        self.assertTrue(
            svc.subscription_failed,
            "a refusal on the bound subscription must be a fault immediately",
        )

    def test_ladder_refusal_does_not_false_latch(self):
        """While the ladder is still trying groups, nothing is bound yet."""
        from src.streaming import IGStreamListener

        svc = _service()
        svc.subscription_group = None  # mid-ladder
        listener = IGStreamListener(svc, group="MARKET")
        listener.onSubscriptionError(21, "Invalid group")

        self.assertFalse(
            svc.subscription_failed,
            "the ladder must be free to try the next group without tripping",
        )

    def test_listener_latches_success(self):
        import threading
        from src.streaming import IGStreamListener

        ev = threading.Event()
        listener = IGStreamListener(None, group="L1", outcome=ev)
        listener.onSubscription()

        self.assertTrue(ev.is_set())
        self.assertTrue(listener.active)
        self.assertIsNone(listener.error)


if __name__ == "__main__":
    unittest.main()
