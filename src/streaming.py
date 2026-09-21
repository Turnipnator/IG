"""
IG Markets Streaming Client using Lightstreamer.

Provides real-time price updates without consuming historical data allowance.
Aggregates tick data into candles for indicator calculation.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

import pandas as pd

# Disk persistence for candle data
CANDLE_CACHE_DIR = Path("/app/data") if os.path.exists("/app") else Path("data")
CANDLE_CACHE_FILE = CANDLE_CACHE_DIR / "streamed_candles.json"
# Durable, append-only candle history harvested from the free stream (zero API
# cost). Unlike the 100-candle rolling cache above, this accumulates a permanent
# OHLC history per EPIC so IG-only instruments with no Yahoo equivalent (e.g. AI
# Index) become backtestable on the real contract. Bounded by a daily prune
# (prune_candle_archive) so it can't grow without limit — project HDD rule.
CANDLE_ARCHIVE_DIR = CANDLE_CACHE_DIR / "candle_archive"
# Retention for the durable archive. 365d ≈ 125MB across all markets (~116B/candle)
# — generous (6× Yahoo's 59d 5m depth) yet hard-bounded. Env-tunable (e.g. 90).
CANDLE_ARCHIVE_RETENTION_DAYS = int(os.getenv("CANDLE_ARCHIVE_RETENTION_DAYS", "365"))

try:
    from lightstreamer.client import (
        LightstreamerClient,
        Subscription,
        SubscriptionListener,
        ItemUpdate,
        ClientListener,
    )
    LIGHTSTREAMER_AVAILABLE = True
except ImportError:
    LIGHTSTREAMER_AVAILABLE = False
    LightstreamerClient = None
    Subscription = None
    SubscriptionListener = object
    ItemUpdate = None
    ClientListener = object

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupSpec:
    """One way of asking IG for prices: item shape, mode and field names.

    IG serves the same prices through several Lightstreamer groups and has now
    withdrawn two of them from this key (L1 ~2026-05-15, MARKET 2026-09-18),
    each time killing the feed silently. The groups are NOT interchangeable by
    string substitution — they differ in subscription mode, item-name shape and
    field names — so the ladder carries a full spec per group rather than a bare
    prefix, and the parser reads fields through `aliases` instead of literals.

    `aliases` maps the canonical name the parser uses to that group's actual
    field name. A canonical name absent from the map is simply not served by
    this group (CHART has no MARKET_STATE), and the parser leaves the existing
    value alone rather than overwriting it with "UNKNOWN".
    """
    name: str
    mode: str
    item_template: str           # "{epic}" is substituted
    fields: tuple[str, ...]
    aliases: dict[str, str]

    def item(self, epic: str) -> str:
        return self.item_template.format(epic=epic)

    def value(self, update: Any, canonical: str) -> Optional[Any]:
        """Read a canonical field, or None if this group doesn't serve it."""
        actual = self.aliases.get(canonical)
        if actual is None:
            return None
        return update.getValue(actual)

    def serves(self, canonical: str) -> bool:
        return canonical in self.aliases


# The quote groups. Identical field semantics, so they share a spec shape and
# the parser needs no special-casing between them.
_QUOTE_FIELDS = ("UPDATE_TIME", "BID", "OFFER", "CHANGE", "CHANGE_PCT", "HIGH", "LOW", "MARKET_STATE")
_QUOTE_ALIASES = {
    "bid": "BID", "offer": "OFFER", "high": "HIGH", "low": "LOW",
    "change": "CHANGE", "change_pct": "CHANGE_PCT", "market_state": "MARKET_STATE",
}

GROUP_MARKET = GroupSpec(
    name="MARKET", mode="MERGE", item_template="MARKET:{epic}",
    fields=_QUOTE_FIELDS, aliases=_QUOTE_ALIASES,
)
GROUP_L1 = GroupSpec(
    name="L1", mode="MERGE", item_template="L1:{epic}",
    fields=_QUOTE_FIELDS, aliases=_QUOTE_ALIASES,
)
# Tick-by-tick chart data. Verified entitled on this key 2026-09-21 while both
# quote groups were refused, which is what got the feed back. DISTINCT (not
# MERGE) because every tick is delivered rather than coalesced; "OFR" not
# "OFFER"; and there is no MARKET_STATE, so tradeable state comes from REST.
# DAY_HIGH/DAY_LOW are the session extremes, the same thing HIGH/LOW carried.
GROUP_CHART_TICK = GroupSpec(
    name="CHART:TICK", mode="DISTINCT", item_template="CHART:{epic}:TICK",
    fields=("BID", "OFR", "UTM", "DAY_HIGH", "DAY_LOW", "DAY_NET_CHG_MID", "DAY_PERC_CHG_MID"),
    aliases={
        "bid": "BID", "offer": "OFR", "high": "DAY_HIGH", "low": "DAY_LOW",
        "change": "DAY_NET_CHG_MID", "change_pct": "DAY_PERC_CHG_MID",
        # no market_state — deliberately absent, see _refresh_market_states
    },
)


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    def update(self, price: float) -> None:
        """Update candle with new tick price."""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price


@dataclass
class MarketStream:
    """Holds streaming data for a single market."""
    epic: str
    name: str
    bid: float = 0.0
    offer: float = 0.0
    mid_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    market_state: str = "CLOSED"
    last_update: Optional[datetime] = None
    consecutive_rejections: int = 0

    # Candle data - rolling window
    candles: deque = field(default_factory=lambda: deque(maxlen=100))
    current_candle: Optional[Candle] = None
    candle_interval: int = 5  # minutes

    def to_dataframe(self) -> pd.DataFrame:
        """Convert candles to DataFrame for indicator calculation."""
        if not self.candles:
            return pd.DataFrame()

        data = [{
            "date": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        } for c in self.candles]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)


class IGStreamListener(SubscriptionListener if LIGHTSTREAMER_AVAILABLE else object):
    """Listener for market price updates."""

    def __init__(
        self,
        stream_service: "IGStreamService",
        on_price_update: Optional[Callable] = None,
        on_candle_complete: Optional[Callable] = None,
        group: Optional[str] = None,
        outcome: Optional[threading.Event] = None,
        spec: Optional[GroupSpec] = None,
    ):
        self.stream_service = stream_service
        self.on_price_update = on_price_update
        self.on_candle_complete = on_candle_complete
        # Which item group this listener was subscribed with, and an event the
        # subscriber waits on so a refused subscription is detected synchronously
        # instead of being assumed successful (see subscribe_markets).
        self.spec = spec
        self.group = spec.name if spec is not None else group
        self.outcome = outcome
        self.active = False
        self.error: Optional[tuple[int, str]] = None

    def onItemUpdate(self, update: ItemUpdate) -> None:
        """Handle incoming price update."""
        try:
            item_name = update.getItemName()
            epic = self._epic_for(item_name)
            if epic is None:
                return

            spec = self.spec or GROUP_MARKET
            bid = self._safe_float(spec.value(update, "bid"))
            offer = self._safe_float(spec.value(update, "offer"))

            if bid is None or offer is None:
                return

            # Reject negative or zero prices (corrupted Lightstreamer data)
            if bid <= 0 or offer <= 0:
                logger.warning(f"[{epic}] Rejected bad tick: bid={bid}, offer={offer}")
                return

            mid_price = (bid + offer) / 2

            market = self.stream_service.markets.get(epic)
            if market:
                # Reject ticks >10% from last known price — catches Lightstreamer
                # corruption. Escape hatch: after N consecutive rejections, accept
                # the next tick and re-seed (handles weekend gaps / stale caches
                # where the reference itself is wrong).
                if market.mid_price > 0:
                    pct_change = abs(mid_price - market.mid_price) / market.mid_price
                    if pct_change > 0.10:
                        market.consecutive_rejections += 1
                        if market.consecutive_rejections >= 20:
                            logger.warning(
                                f"[{epic}] Re-seeding mid_price after "
                                f"{market.consecutive_rejections} consecutive rejections: "
                                f"{market.mid_price:.2f} -> {mid_price:.2f} "
                                f"(likely gap / stale cache)"
                            )
                            market.consecutive_rejections = 0
                        else:
                            logger.warning(
                                f"[{epic}] Rejected outlier tick: {mid_price:.2f} "
                                f"vs last {market.mid_price:.2f} ({pct_change:.1%} change)"
                            )
                            return
                    else:
                        market.consecutive_rejections = 0

                market.bid = bid
                market.offer = offer
                market.mid_price = mid_price
                market.high = self._safe_float(spec.value(update, "high")) or market.high
                market.low = self._safe_float(spec.value(update, "low")) or market.low
                market.change = self._safe_float(spec.value(update, "change")) or 0.0
                market.change_pct = self._safe_float(spec.value(update, "change_pct")) or 0.0
                # Only groups that actually serve MARKET_STATE may write it. CHART
                # does not, and clobbering the REST-sourced value with "UNKNOWN"
                # would break the weekend suppressor and the dead-market guard.
                if spec.serves("market_state"):
                    market.market_state = spec.value(update, "market_state") or "UNKNOWN"
                market.last_update = datetime.now()

                # Update candle
                candle_completed = self._update_candle(market, mid_price)

                # Callbacks
                if self.on_price_update:
                    self.on_price_update(epic, market)

                if candle_completed and self.on_candle_complete:
                    self.on_candle_complete(epic, market)

        except Exception as e:
            logger.error(f"Error processing price update: {e}")

    def _epic_for(self, item_name: str) -> Optional[str]:
        """Resolve an item name back to its epic.

        Built from the exact map recorded at subscribe time rather than by
        stripping prefixes: CHART items are "CHART:{epic}:TICK", so the epic sits
        in the MIDDLE and the old `.replace("MARKET:", "")` approach silently
        yields a key that matches no market — which would look exactly like a
        dead feed. The fallback below only covers the two quote groups and exists
        so a stray update arriving before the map is populated still parses.
        """
        svc = self.stream_service
        if svc is not None:
            epic = getattr(svc, "_item_to_epic", {}).get(item_name)
            if epic is not None:
                return epic
        if item_name.startswith("CHART:"):
            parts = item_name.split(":")
            return parts[1] if len(parts) >= 3 else None
        stripped = item_name.replace("L1:", "").replace("MARKET:", "")
        return stripped or None

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _update_candle(self, market: MarketStream, price: float) -> bool:
        """Update current candle and check if a new one should start."""
        now = datetime.now()
        candle_completed = False

        # Determine candle start time (rounded to interval)
        interval_mins = market.candle_interval
        candle_start = now.replace(
            minute=(now.minute // interval_mins) * interval_mins,
            second=0,
            microsecond=0
        )

        if market.current_candle is None:
            # Start first candle
            market.current_candle = Candle(
                timestamp=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
            )
        elif candle_start > market.current_candle.timestamp:
            # New candle period - save current and start new
            market.candles.append(market.current_candle)
            candle_completed = True

            logger.debug(
                f"{market.epic}: Candle completed - "
                f"O:{market.current_candle.open:.2f} "
                f"H:{market.current_candle.high:.2f} "
                f"L:{market.current_candle.low:.2f} "
                f"C:{market.current_candle.close:.2f}"
            )

            market.current_candle = Candle(
                timestamp=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
            )
        else:
            # Update current candle
            market.current_candle.update(price)

        return candle_completed

    def onSubscription(self) -> None:
        logger.info(f"Market subscription active (group: {self.group})")
        self.active = True
        self.error = None
        if self.outcome is not None:
            self.outcome.set()

    def onSubscriptionError(self, code: int, message: str) -> None:
        logger.error(f"Subscription error on group {self.group}: [{code}] {message}")
        self.active = False
        self.error = (code, message)
        if "Invalid account type" in message:
            logger.error(
                "This error typically means your demo account default is set to CFD "
                "instead of Spreadbet. Contact IG helpdesk to change your default "
                "demo account to Spreadbet Demo."
            )
        if code == 21:
            logger.error(
                f"'[21] Invalid group' means this account/key is not entitled to the "
                f"'{self.group}' item group. L1 was revoked ~2026-05-15 and MARKET on "
                f"2026-09-18; the subscriber will try the next group in the ladder."
            )
        # A refusal on the subscription that is CURRENTLY BOUND is the 19:31
        # case: the socket dropped, the library silently re-subscribed, and IG
        # refused. Latch it so the watchdog reacts now instead of inferring it
        # from four minutes of silence. Guarded on subscription_group so a
        # refusal while the ladder is still trying groups — when the service has
        # no bound group yet — cannot false-latch.
        svc = self.stream_service
        if svc is not None and getattr(svc, "subscription_group", None) == self.group:
            logger.error(
                "The live subscription was refused mid-session — treating the feed "
                "as dead immediately rather than waiting for the staleness timer."
            )
            svc.subscription_failed = True
        if self.outcome is not None:
            self.outcome.set()

    def onUnsubscription(self) -> None:
        logger.info("Market subscription ended")


class IGConnectionListener(ClientListener if LIGHTSTREAMER_AVAILABLE else object):
    """Listener for connection status changes."""

    def __init__(self, stream_service: "IGStreamService"):
        self.stream_service = stream_service

    def onStatusChange(self, status: str) -> None:
        logger.info(f"Lightstreamer connection status: {status}")
        self.stream_service.connection_status = status

        if status.startswith("CONNECTED"):
            self.stream_service.connected = True
        elif status.startswith("DISCONNECTED"):
            self.stream_service.connected = False

    def onServerError(self, code: int, message: str) -> None:
        logger.error(f"Lightstreamer server error: [{code}] {message}")

    def onPropertyChange(self, property_name: str) -> None:
        logger.debug(f"Lightstreamer property changed: {property_name}")


class IGStreamService:
    """
    IG Markets Streaming Service.

    Connects to IG's Lightstreamer server for real-time price data.
    Does not consume historical data allowance.
    """

    # Lightstreamer endpoints
    DEMO_ENDPOINT = "https://demo-apd.marketdatasystems.com"
    LIVE_ENDPOINT = "https://apd.marketdatasystems.com"

    # Item groups to try, in order. IG has revoked two of these on this key
    # (L1 ~2026-05-15, MARKET 2026-09-18) and each revocation killed the feed
    # silently, so a single hardcoded group is a known single point of failure.
    #
    # Order is cheapest-first: the quote groups coalesce updates (MERGE) and
    # carry MARKET_STATE, so they are preferred whenever IG will serve them, and
    # the bot drops back to them automatically if entitlement is restored.
    # CHART:TICK is the fallback that is actually entitled as of 2026-09-21 —
    # it is DISTINCT, names its fields differently and has no MARKET_STATE,
    # which is why each group carries a full spec rather than a bare prefix.
    SUBSCRIPTION_GROUPS = (GROUP_MARKET, GROUP_L1, GROUP_CHART_TICK)
    # How long to wait for IG's async verdict on a subscribe before giving up.
    SUBSCRIBE_CONFIRM_TIMEOUT = 8.0

    def __init__(
        self,
        cst: str,
        security_token: str,
        account_id: str,
        is_demo: bool = True,
        on_price_update: Optional[Callable] = None,
        on_candle_complete: Optional[Callable] = None,
    ):
        if not LIGHTSTREAMER_AVAILABLE:
            raise ImportError(
                "lightstreamer-client-lib not installed. "
                "Run: pip install lightstreamer-client-lib"
            )

        self.cst = cst
        self.security_token = security_token
        self.account_id = account_id
        self.endpoint = self.DEMO_ENDPOINT if is_demo else self.LIVE_ENDPOINT

        self.on_price_update = on_price_update
        self.on_candle_complete = on_candle_complete

        self.client: Optional[LightstreamerClient] = None
        self.subscription: Optional[Subscription] = None
        self.markets: dict[str, MarketStream] = {}

        self.connected = False
        self.connection_status = "DISCONNECTED"
        self._lock = threading.Lock()

        # Subscription health. A refused subscription leaves the SOCKET connected
        # (CONNECTED:WS-STREAMING) and only the data flow dead, so `connected` alone
        # says nothing about whether the feed works — on 2026-09-18 that gap hid a
        # 62-hour outage. These three are what the watchdog actually reads.
        self.subscription_group: Optional[str] = None   # group that bound, if any
        self.subscription_spec: Optional[GroupSpec] = None
        self.subscription_failed = False                # every group refused
        self.subscribed_at: Optional[datetime] = None   # when we last tried
        self._item_to_epic: dict[str, str] = {}

        # Append-only candle-archive cursor: last-archived timestamp per epic, so
        # archive_candles_to_disk() appends only NEW closed candles before the
        # rolling deque (maxlen=100) drops them. Seeded once from disk on first run.
        self._archive_cursor: dict[str, datetime] = {}
        self._archive_cursor_loaded = False

    def connect(self) -> bool:
        """Connect to Lightstreamer server."""
        try:
            # Create credentials string
            password = f"CST-{self.cst}|XST-{self.security_token}"

            # Debug: log connection details (mask sensitive parts)
            logger.info(f"Streaming connect - Account: {self.account_id}")
            logger.info(f"Streaming connect - Endpoint: {self.endpoint}")
            logger.info(f"Streaming connect - CST length: {len(self.cst) if self.cst else 0}")
            logger.info(f"Streaming connect - XST length: {len(self.security_token) if self.security_token else 0}")

            # Create client
            self.client = LightstreamerClient(self.endpoint, "DEFAULT")
            self.client.connectionDetails.setUser(self.account_id)
            self.client.connectionDetails.setPassword(password)

            # Add connection listener
            self.client.addListener(IGConnectionListener(self))

            # Connect
            self.client.connect()

            # Wait for connection (with timeout)
            timeout = 10
            start = datetime.now()
            while not self.connected and (datetime.now() - start).seconds < timeout:
                import time
                time.sleep(0.1)

            if self.connected:
                logger.info(f"Connected to Lightstreamer at {self.endpoint}")
                return True
            else:
                logger.error("Lightstreamer connection timeout")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Lightstreamer: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Lightstreamer server."""
        if self.client:
            try:
                if self.subscription:
                    self.client.unsubscribe(self.subscription)
                self.client.disconnect()
                logger.info("Disconnected from Lightstreamer")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

        self.connected = False
        self.client = None
        self.subscription = None
        # Not a failure — an intentional teardown. Leaving the latch set here
        # would trip the watchdog on every planned reconnect.
        self.subscription_failed = False
        self.subscription_group = None
        self.subscribed_at = None

    def subscribe_markets(self, epics: list[str], names: list[str] = None, candle_intervals: list[int] = None) -> bool:
        """
        Subscribe to market price updates.

        Args:
            epics: List of market EPICs to subscribe to
            names: Optional list of market names (for logging)
            candle_intervals: Optional list of candle intervals in minutes per market

        Returns:
            True if subscription successful
        """
        if not self.client or not self.connected:
            logger.error("Not connected to Lightstreamer")
            return False

        try:
            # Initialize market streams
            names = names or epics
            candle_intervals = candle_intervals or [5] * len(epics)
            for epic, name, interval in zip(epics, names, candle_intervals):
                self.markets[epic] = MarketStream(epic=epic, name=name, candle_interval=interval)

            self.subscribed_at = datetime.now()
            self.subscription_group = None
            self.subscription_failed = False

            for spec in self.SUBSCRIPTION_GROUPS:
                if self._try_subscribe(spec, epics):
                    self.subscription_group = spec.name
                    self.subscription_spec = spec
                    logger.info(
                        f"Subscribed to {len(epics)} markets via '{spec.name}': "
                        f"{', '.join(names)}"
                    )
                    if not spec.serves("market_state"):
                        logger.info(
                            f"'{spec.name}' does not serve MARKET_STATE — tradeable "
                            f"state will be sourced from REST (see refresh_market_states)"
                        )
                    return True
                logger.warning(f"Group '{spec.name}' refused — falling back to the next one")

            self.subscription_failed = True
            logger.critical(
                f"Every subscription group "
                f"{[s.name for s in self.SUBSCRIPTION_GROUPS]} was refused. The feed is "
                f"DEAD while the socket stays connected. This is an account entitlement "
                f"problem, not a network one — check the API key at labs.ig.com."
            )
            return False

        except Exception as e:
            logger.error(f"Failed to subscribe to markets: {e}")
            self.subscription_failed = True
            return False

    def _try_subscribe(self, spec: GroupSpec, epics: list[str]) -> bool:
        """Subscribe with one item group and WAIT for IG's verdict.

        The old code called client.subscribe() and immediately logged success.
        That is a lie: the result arrives asynchronously on the listener, so a
        refused subscription was logged as "Subscribed to 13 markets" and the
        [21] error a few lines later went unread by everything downstream.
        """
        outcome = threading.Event()
        items = [spec.item(epic) for epic in epics]
        # Exact item->epic map for the parser. CHART puts the epic in the middle
        # of the item name, so prefix-stripping cannot recover it.
        self._item_to_epic = {spec.item(epic): epic for epic in epics}
        logger.info(f"Subscribing to {len(items)} items via '{spec.name}': {items[:2]}...")

        subscription = Subscription(
            mode=spec.mode,
            items=items,
            fields=list(spec.fields),
        )
        listener = IGStreamListener(
            self,
            on_price_update=self.on_price_update,
            on_candle_complete=self.on_candle_complete,
            spec=spec,
            outcome=outcome,
        )
        subscription.addListener(listener)
        self.client.subscribe(subscription)

        if not outcome.wait(self.SUBSCRIBE_CONFIRM_TIMEOUT):
            # Neither callback fired. Treat silence as failure — an unconfirmed
            # subscription is exactly the state that hid the 09-18 outage.
            logger.error(
                f"Group '{spec.name}': no subscription verdict within "
                f"{self.SUBSCRIBE_CONFIRM_TIMEOUT}s — treating as refused"
            )
            self._safe_unsubscribe(subscription)
            return False

        if listener.active:
            self.subscription = subscription
            return True

        self._safe_unsubscribe(subscription)
        return False

    def _safe_unsubscribe(self, subscription) -> None:
        """Drop a subscription that did not bind, so the next group starts clean."""
        try:
            self.client.unsubscribe(subscription)
        except Exception as e:
            logger.debug(f"Unsubscribe of a non-bound subscription failed (ignoring): {e}")

    def initialize_candles(self, epic: str, historical_df: pd.DataFrame) -> None:
        """
        Initialize candle history from historical data.

        Call this once at startup with data from REST API to prime
        the indicators, then streaming takes over.
        """
        if epic not in self.markets:
            logger.warning(f"Market {epic} not subscribed")
            return

        market = self.markets[epic]
        skipped = 0

        for _, row in historical_df.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]

            # Skip corrupted candles (negative prices or absurd range)
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                skipped += 1
                continue
            if h > 0 and l > 0 and (h - l) / l > 0.5:
                # Range > 50% of low price is almost certainly corrupted
                skipped += 1
                continue

            candle = Candle(
                timestamp=row["date"],
                open=o, high=h, low=l, close=c,
                volume=row.get("volume", 0),
            )
            market.candles.append(candle)

        # Seed mid_price from last candle so tick outlier filter works immediately
        if market.candles:
            market.mid_price = market.candles[-1].close

        if skipped:
            logger.warning(f"Filtered {skipped} corrupted candles for {epic}")
        logger.info(f"Initialized {len(market.candles)} candles for {epic}")

    def get_market_data(self, epic: str) -> Optional[MarketStream]:
        """Get current market data for an epic."""
        return self.markets.get(epic)

    def get_dataframe(self, epic: str) -> Optional[pd.DataFrame]:
        """Get candle data as DataFrame for indicator calculation."""
        market = self.markets.get(epic)
        if market:
            return market.to_dataframe()
        return None

    def is_market_open(self, epic: str) -> bool:
        """Check if market is currently tradeable."""
        market = self.markets.get(epic)
        if market:
            return market.market_state == "TRADEABLE"
        return False

    def save_candles_to_disk(self) -> None:
        """Persist all candle data to disk for surviving restarts."""
        try:
            CANDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            cache_data = {}
            for epic, market in self.markets.items():
                if market.candles:
                    cache_data[epic] = {
                        "name": market.name,
                        "candle_interval": market.candle_interval,
                        "saved_at": datetime.now().isoformat(),
                        "candles": [
                            {
                                "timestamp": c.timestamp.isoformat() if isinstance(c.timestamp, datetime) else str(c.timestamp),
                                "open": c.open,
                                "high": c.high,
                                "low": c.low,
                                "close": c.close,
                                "volume": c.volume,
                            }
                            for c in market.candles
                        ]
                    }

            if cache_data:
                with open(CANDLE_CACHE_FILE, "w") as f:
                    json.dump(cache_data, f)
                total = sum(len(v["candles"]) for v in cache_data.values())
                logger.info(f"Saved {total} candles across {len(cache_data)} markets to disk")

        except Exception as e:
            logger.warning(f"Could not save candles to disk: {e}")

    def _load_archive_cursor(self) -> None:
        """One-time: seed the per-epic archive cursor from the tail of each
        archive file so a restart doesn't re-append candles already on disk."""
        self._archive_cursor_loaded = True
        if not CANDLE_ARCHIVE_DIR.exists():
            return
        for path in CANDLE_ARCHIVE_DIR.glob("*.jsonl"):
            try:
                last = None
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            last = line
                if last:
                    ts = json.loads(last).get("timestamp")
                    if ts:
                        self._archive_cursor[path.stem] = datetime.fromisoformat(ts)
            except Exception as e:
                logger.warning(f"Could not read archive tail {path.name}: {e}")

    def archive_candles_to_disk(self) -> None:
        """Append newly-closed streamed candles to a durable per-epic JSONL archive.

        Zero API cost (streaming data only). The in-memory buffer is a 100-candle
        deque (~8h at 5m) that silently drops older candles; this harvests them
        into permanent history so every EPIC — especially IG-only ones with no
        Yahoo equivalent (AI Index) — becomes backtestable on the REAL instrument
        with no Yahoo-proxy error and no historical-allowance usage. Append-only,
        deduped by an in-memory timestamp cursor (no full-file reads after startup).
        Growth is ~10MB/month across all markets — trim/retention is a later tweak.
        """
        try:
            if not self._archive_cursor_loaded:
                self._load_archive_cursor()
            CANDLE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            total_new = 0
            for epic, market in self.markets.items():
                if not market.candles:
                    continue
                cursor = self._archive_cursor.get(epic)
                new = [
                    c for c in market.candles
                    if isinstance(c.timestamp, datetime)
                    and (cursor is None or c.timestamp > cursor)
                ]
                if not new:
                    continue
                path = CANDLE_ARCHIVE_DIR / f"{epic}.jsonl"
                with open(path, "a") as f:
                    for c in new:
                        f.write(json.dumps({
                            "timestamp": c.timestamp.isoformat(),
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume,
                        }) + "\n")
                self._archive_cursor[epic] = new[-1].timestamp
                total_new += len(new)
            if total_new:
                logger.info(f"Archived {total_new} new candles to durable history")
        except Exception as e:
            logger.warning(f"Could not archive candles: {e}")

    def prune_candle_archive(
        self, retention_days: int = CANDLE_ARCHIVE_RETENTION_DAYS
    ) -> None:
        """Bound the durable archive: drop candles older than retention_days from
        each per-EPIC JSONL via an atomic rewrite. Keeps the archive size-bounded
        (project HDD rule). Only removes OLD lines — never the recent tail — so the
        append cursor stays valid. Runs daily on the SAME scheduler thread as the
        harvester, so there's no append/rewrite race.
        """
        try:
            if not CANDLE_ARCHIVE_DIR.exists():
                return
            cutoff = datetime.now() - timedelta(days=retention_days)
            total_pruned = 0
            for path in CANDLE_ARCHIVE_DIR.glob("*.jsonl"):
                try:
                    kept, pruned = [], 0
                    with open(path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ts = datetime.fromisoformat(json.loads(line)["timestamp"])
                            except (json.JSONDecodeError, KeyError, ValueError):
                                kept.append(line)  # keep unparseable lines, don't lose data
                                continue
                            if ts >= cutoff:
                                kept.append(line)
                            else:
                                pruned += 1
                    if pruned:
                        tmp = path.with_suffix(".jsonl.tmp")
                        with open(tmp, "w") as f:
                            f.write("\n".join(kept) + ("\n" if kept else ""))
                        tmp.replace(path)  # atomic
                        total_pruned += pruned
                except Exception as e:
                    logger.warning(f"Could not prune archive {path.name}: {e}")
            if total_pruned:
                logger.info(
                    f"Pruned {total_pruned} candles older than {retention_days}d from archive"
                )
        except Exception as e:
            logger.warning(f"Archive prune failed: {e}")

    def load_candles_from_disk(self) -> dict:
        """
        Load candle data from disk (saved by previous session).

        Returns:
            Dict of epic -> DataFrame with candle data, or empty dict
        """
        try:
            if not CANDLE_CACHE_FILE.exists():
                return {}

            with open(CANDLE_CACHE_FILE, "r") as f:
                cache_data = json.load(f)

            result = {}
            for epic, item in cache_data.items():
                # Skip cache if interval changed: loading 5m candles into a
                # 1h-configured market would compute indicators on misaligned
                # bars. Falls through to API fetch.
                cached_interval = item.get("candle_interval")
                current_market = self.markets.get(epic)
                if (
                    cached_interval is not None
                    and current_market is not None
                    and cached_interval != current_market.candle_interval
                ):
                    logger.info(
                        f"  {item.get('name', epic)}: skipping disk cache "
                        f"(interval changed {cached_interval}m → {current_market.candle_interval}m)"
                    )
                    continue

                saved_at = datetime.fromisoformat(item["saved_at"])
                age = datetime.now() - saved_at

                # Use if less than 6 hours old — avoids wasting API budget on restart
                if age < timedelta(hours=6):
                    candles = item["candles"]
                    if candles:
                        df = pd.DataFrame(candles)
                        df["date"] = pd.to_datetime(df["timestamp"])
                        df = df.drop(columns=["timestamp"])
                        result[epic] = df
                        logger.info(f"  {item.get('name', epic)}: Loaded {len(candles)} candles from disk (age: {age.seconds // 60}m)")
                else:
                    logger.debug(f"  {epic}: Disk cache too old ({age.seconds // 60}m)")

            if result:
                logger.info(f"Loaded candle data for {len(result)} markets from disk (0 API calls)")

            return result

        except Exception as e:
            logger.warning(f"Could not load candles from disk: {e}")
            return {}

    def most_recent_tick_age(self) -> Optional[timedelta]:
        """Age of the most recent tick across all subscribed markets.

        Returns None if no market has ever ticked (e.g. streaming hasn't
        started yet). The streaming watchdog uses this to detect a "looks
        connected but no data flowing" failure mode.
        """
        latest: Optional[datetime] = None
        for market in self.markets.values():
            if market.last_update and (latest is None or market.last_update > latest):
                latest = market.last_update
        if latest is None:
            return None
        return datetime.now() - latest

    def tradeable_market_count(self) -> int:
        """Markets currently reporting MARKET_STATE=TRADEABLE in their last tick.

        Used to suppress the staleness watchdog when all markets are closed
        (e.g. forex weekend) — no ticks are expected, so silence is fine.

        CAUTION: this is derived from tick data, so it reads 0 both when every
        market is genuinely closed AND when the feed is dead (MarketStream
        defaults market_state to "CLOSED" and no tick ever corrects it). Callers
        must not treat 0 as proof of a weekend — see has_ever_ticked().
        """
        return sum(1 for m in self.markets.values() if m.market_state == "TRADEABLE")

    def needs_rest_market_state(self) -> bool:
        """True when the bound group does not serve MARKET_STATE (CHART does not)."""
        spec = self.subscription_spec
        return spec is not None and not spec.serves("market_state")

    def refresh_market_states(self, client) -> int:
        """Fill MarketStream.market_state from REST for groups that omit it.

        CHART:TICK carries prices but no MARKET_STATE, and that field is not
        decoration: tradeable_market_count() feeds the watchdog's weekend
        suppressor, and _dead_market_guard uses it to catch an epic that has gone
        OFFLINE while ticks keep flowing (the Crude 2026-08-19 failure, 16 days
        silent). Losing it would quietly disable both.

        Caller controls cadence — get_market_info shares IG's per-key
        non-trading rate limit with everything else the bot does, so this is
        driven from the screener's schedule, not per tick. Returns the number of
        markets updated.
        """
        if not self.needs_rest_market_state() or client is None:
            return 0
        updated = 0
        for epic, market in self.markets.items():
            try:
                info = client.get_market_info(epic)
            except Exception as e:
                logger.debug(f"REST market-state refresh failed for {epic}: {e}")
                continue
            if info is not None and getattr(info, "market_status", None):
                market.market_state = info.market_status
                updated += 1
        if updated:
            logger.info(
                f"Refreshed MARKET_STATE from REST for {updated}/{len(self.markets)} "
                f"markets (bound group '{self.subscription_group}' does not serve it)"
            )
        return updated

    def has_ever_ticked(self) -> bool:
        """Whether any market has delivered a tick since the last subscribe."""
        return any(m.last_update is not None for m in self.markets.values())

    def feed_silence(self) -> Optional[timedelta]:
        """How long the feed has been silent, counting from the last subscribe.

        This exists because most_recent_tick_age() returns None when no market
        has ever ticked, and the watchdog used to read that None as "streaming
        hasn't started yet — benign". After a subscribe, "never ticked" is not
        benign: it is precisely the failure mode that went unnoticed for 62
        hours on 2026-09-18, because resubscribing resets every MarketStream and
        so resets last_update to None.

        Returns None only when no subscribe has been attempted at all.
        """
        tick_age = self.most_recent_tick_age()
        if tick_age is not None:
            return tick_age
        if self.subscribed_at is None:
            return None
        return datetime.now() - self.subscribed_at

    def get_status(self) -> dict:
        """Get streaming service status."""
        silence = self.feed_silence()
        return {
            "connected": self.connected,
            "connection_status": self.connection_status,
            "subscription_group": self.subscription_group,
            "subscription_failed": self.subscription_failed,
            "feed_silence_seconds": silence.total_seconds() if silence else None,
            "subscribed_markets": len(self.markets),
            "markets": {
                epic: {
                    "name": m.name,
                    "bid": m.bid,
                    "offer": m.offer,
                    "state": m.market_state,
                    "candles": len(m.candles),
                    "last_update": m.last_update.isoformat() if m.last_update else None,
                }
                for epic, m in self.markets.items()
            }
        }
