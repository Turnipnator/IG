"""Group specs: the same prices, three different shapes.

IG serves prices through several Lightstreamer groups and has now withdrawn two
of them from this key — L1 around 2026-05-15 and MARKET on 2026-09-18, each time
killing the feed. CHART:TICK was the one still entitled, so the bot must be able
to fall through to it.

The groups are NOT interchangeable by string substitution, which is the trap
this file guards:

  * the item name puts the epic in a DIFFERENT PLACE ("CHART:{epic}:TICK" vs
    "MARKET:{epic}"), so prefix-stripping returns a key matching no market —
    indistinguishable, downstream, from a dead feed;
  * the field is "OFR", not "OFFER" — reading the wrong name yields None, and
    the tick is dropped as malformed;
  * the mode is DISTINCT, not MERGE;
  * there is NO MARKET_STATE, and overwriting the REST-sourced value with
    "UNKNOWN" would silently disable both the watchdog's weekend suppressor and
    the dead-market guard.

The property that matters most: a tick delivered through CHART must produce the
same candle as the identical tick delivered through MARKET. If that ever stops
holding, live quietly stops matching every validated backtest.
"""

import unittest
from datetime import datetime
from unittest import mock

from src.streaming import (
    GROUP_CHART_TICK, GROUP_L1, GROUP_MARKET, GroupSpec,
    IGStreamListener, IGStreamService, MarketStream,
)

EPIC = "CS.D.EURUSD.TODAY.IP"


class _Update:
    """Minimal stand-in for Lightstreamer's ItemUpdate."""

    def __init__(self, item_name, values):
        self._item = item_name
        self._values = values

    def getItemName(self):
        return self._item

    def getValue(self, field):
        return self._values.get(field)


def _service():
    svc = IGStreamService.__new__(IGStreamService)
    svc.markets = {EPIC: MarketStream(epic=EPIC, name="EUR/USD")}
    svc.connected = True
    svc.connection_status = "CONNECTED:WS-STREAMING"
    svc.subscription = None
    svc.subscription_group = None
    svc.subscription_spec = None
    svc.subscription_failed = False
    svc.subscribed_at = datetime.now()
    svc._item_to_epic = {}
    svc.on_price_update = None
    svc.on_candle_complete = None
    return svc


class TestSpecShapes(unittest.TestCase):

    def test_item_templates(self):
        self.assertEqual(GROUP_MARKET.item(EPIC), f"MARKET:{EPIC}")
        self.assertEqual(GROUP_L1.item(EPIC), f"L1:{EPIC}")
        self.assertEqual(GROUP_CHART_TICK.item(EPIC), f"CHART:{EPIC}:TICK")

    def test_chart_is_distinct_mode(self):
        self.assertEqual(GROUP_CHART_TICK.mode, "DISTINCT")
        self.assertEqual(GROUP_MARKET.mode, "MERGE")

    def test_chart_offer_field_is_OFR_not_OFFER(self):
        self.assertEqual(GROUP_CHART_TICK.aliases["offer"], "OFR")
        self.assertEqual(GROUP_MARKET.aliases["offer"], "OFFER")

    def test_every_alias_target_is_actually_subscribed(self):
        """An alias naming a field we never requested always reads None."""
        for spec in (GROUP_MARKET, GROUP_L1, GROUP_CHART_TICK):
            for canonical, actual in spec.aliases.items():
                self.assertIn(
                    actual, spec.fields,
                    f"{spec.name}: alias {canonical}->{actual} is not in the "
                    f"subscribed field list, so it can never return a value",
                )

    def test_chart_does_not_claim_market_state(self):
        self.assertFalse(GROUP_CHART_TICK.serves("market_state"))
        self.assertTrue(GROUP_MARKET.serves("market_state"))

    def test_value_returns_none_for_an_unserved_field(self):
        u = _Update(GROUP_CHART_TICK.item(EPIC), {"MARKET_STATE": "TRADEABLE"})
        self.assertIsNone(GROUP_CHART_TICK.value(u, "market_state"))


class TestParserAcrossGroups(unittest.TestCase):
    """The equivalence property: same tick, same candle, whichever group."""

    def _feed(self, spec, bid, offer, extra=None):
        svc = _service()
        svc._item_to_epic = {spec.item(EPIC): EPIC}
        svc.subscription_spec = spec
        listener = IGStreamListener(svc, spec=spec)
        values = {
            spec.aliases["bid"]: str(bid),
            spec.aliases["offer"]: str(offer),
        }
        values.update(extra or {})
        listener.onItemUpdate(_Update(spec.item(EPIC), values))
        return svc.markets[EPIC]

    def test_same_tick_gives_the_same_mid_through_every_group(self):
        mids = []
        for spec in (GROUP_MARKET, GROUP_L1, GROUP_CHART_TICK):
            m = self._feed(spec, 1.1000, 1.1002)
            mids.append(m.mid_price)
            self.assertEqual(m.bid, 1.1000)
            self.assertEqual(m.offer, 1.1002)
        self.assertEqual(len(set(mids)), 1, f"mid diverges across groups: {mids}")

    def test_same_tick_builds_the_same_candle_through_every_group(self):
        candles = []
        for spec in (GROUP_MARKET, GROUP_CHART_TICK):
            m = self._feed(spec, 1.2000, 1.2004)
            self.assertIsNotNone(m.current_candle, f"{spec.name} built no candle")
            c = m.current_candle
            candles.append((c.open, c.high, c.low, c.close))
        self.assertEqual(
            candles[0], candles[1],
            "a CHART tick must build the same candle as the identical MARKET tick",
        )

    def test_chart_tick_marks_the_market_as_updated(self):
        """Feed liveness is what the watchdog reads; it must advance on CHART."""
        m = self._feed(GROUP_CHART_TICK, 1.1, 1.1002)
        self.assertIsNotNone(m.last_update)

    def test_chart_does_not_clobber_rest_sourced_market_state(self):
        svc = _service()
        svc._item_to_epic = {GROUP_CHART_TICK.item(EPIC): EPIC}
        svc.subscription_spec = GROUP_CHART_TICK
        svc.markets[EPIC].market_state = "TRADEABLE"  # as set by REST
        listener = IGStreamListener(svc, spec=GROUP_CHART_TICK)
        listener.onItemUpdate(_Update(GROUP_CHART_TICK.item(EPIC),
                                      {"BID": "1.1", "OFR": "1.1002"}))
        self.assertEqual(
            svc.markets[EPIC].market_state, "TRADEABLE",
            "CHART has no MARKET_STATE; it must leave the REST value alone",
        )

    def test_quote_group_still_writes_market_state(self):
        m = self._feed(GROUP_MARKET, 1.1, 1.1002, {"MARKET_STATE": "EDITS_ONLY"})
        self.assertEqual(m.market_state, "EDITS_ONLY")

    def test_a_tick_for_an_unknown_item_is_ignored_not_crashed(self):
        svc = _service()
        svc.subscription_spec = GROUP_CHART_TICK
        listener = IGStreamListener(svc, spec=GROUP_CHART_TICK)
        listener.onItemUpdate(_Update("CHART:NOT.A.REAL.EPIC:TICK",
                                      {"BID": "1.0", "OFR": "1.1"}))
        self.assertIsNone(svc.markets[EPIC].last_update)

    def test_bad_ticks_are_still_rejected_on_chart(self):
        """The corruption guards must not be bypassed by the new path."""
        m = self._feed(GROUP_CHART_TICK, -1.0, 1.1002)
        self.assertIsNone(m.last_update, "negative bid must be rejected on CHART too")


class TestRestMarketState(unittest.TestCase):

    def test_needed_only_when_the_group_omits_market_state(self):
        svc = _service()
        svc.subscription_spec = GROUP_MARKET
        self.assertFalse(svc.needs_rest_market_state())
        svc.subscription_spec = GROUP_CHART_TICK
        self.assertTrue(svc.needs_rest_market_state())

    def test_refresh_fills_state_from_rest(self):
        svc = _service()
        svc.subscription_spec = GROUP_CHART_TICK
        svc.subscription_group = "CHART:TICK"
        client = mock.Mock()
        client.get_market_info.return_value = mock.Mock(market_status="TRADEABLE")

        n = svc.refresh_market_states(client)

        self.assertEqual(n, 1)
        self.assertEqual(svc.markets[EPIC].market_state, "TRADEABLE")

    def test_refresh_is_a_noop_on_a_quote_group(self):
        """No pointless REST calls when the stream already carries the field."""
        svc = _service()
        svc.subscription_spec = GROUP_MARKET
        client = mock.Mock()
        self.assertEqual(svc.refresh_market_states(client), 0)
        client.get_market_info.assert_not_called()

    def test_one_failing_epic_does_not_abort_the_refresh(self):
        svc = _service()
        svc.markets["OTHER"] = MarketStream(epic="OTHER", name="Other")
        svc.subscription_spec = GROUP_CHART_TICK
        client = mock.Mock()
        client.get_market_info.side_effect = [
            RuntimeError("boom"), mock.Mock(market_status="TRADEABLE"),
        ]
        self.assertEqual(svc.refresh_market_states(client), 1)


if __name__ == "__main__":
    unittest.main()
