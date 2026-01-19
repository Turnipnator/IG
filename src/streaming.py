"""
IG Markets Streaming Client using Lightstreamer.

Provides real-time price updates without consuming historical data allowance.
Aggregates tick data into candles for indicator calculation.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

import pandas as pd

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
    ):
        self.stream_service = stream_service
        self.on_price_update = on_price_update
        self.on_candle_complete = on_candle_complete

    def onItemUpdate(self, update: ItemUpdate) -> None:
        """Handle incoming price update."""
        try:
            item_name = update.getItemName()
            # Item name format: "L1:IX.D.SPTRD.DAILY.IP" or "MARKET:IX.D.SPTRD.DAILY.IP"
            epic = item_name.replace("L1:", "").replace("MARKET:", "")

            bid = self._safe_float(update.getValue("BID"))
            offer = self._safe_float(update.getValue("OFFER"))

            if bid is None or offer is None:
                return

            mid_price = (bid + offer) / 2

            market = self.stream_service.markets.get(epic)
            if market:
                market.bid = bid
                market.offer = offer
                market.mid_price = mid_price
                market.high = self._safe_float(update.getValue("HIGH")) or market.high
                market.low = self._safe_float(update.getValue("LOW")) or market.low
                market.change = self._safe_float(update.getValue("CHANGE")) or 0.0
                market.change_pct = self._safe_float(update.getValue("CHANGE_PCT")) or 0.0
                market.market_state = update.getValue("MARKET_STATE") or "UNKNOWN"
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
        logger.info("Market subscription active")

    def onSubscriptionError(self, code: int, message: str) -> None:
        logger.error(f"Subscription error: [{code}] {message}")
        if "Invalid account type" in message:
            logger.error(
                "This error typically means your demo account default is set to CFD "
                "instead of Spreadbet. Contact IG helpdesk to change your default "
                "demo account to Spreadbet Demo."
            )

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

    def connect(self) -> bool:
        """Connect to Lightstreamer server."""
        try:
            # Create credentials string
            password = f"CST-{self.cst}|XST-{self.security_token}"

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

    def subscribe_markets(self, epics: list[str], names: list[str] = None) -> bool:
        """
        Subscribe to market price updates.

        Args:
            epics: List of market EPICs to subscribe to
            names: Optional list of market names (for logging)

        Returns:
            True if subscription successful
        """
        if not self.client or not self.connected:
            logger.error("Not connected to Lightstreamer")
            return False

        try:
            # Initialize market streams
            names = names or epics
            for epic, name in zip(epics, names):
                self.markets[epic] = MarketStream(epic=epic, name=name)

            # Create subscription using L1 (Level 1) prefix for price data
            # L1: is the standard format for market prices on IG
            items = [f"L1:{epic}" for epic in epics]

            self.subscription = Subscription(
                mode="MERGE",
                items=items,
                fields=[
                    "UPDATE_TIME",
                    "BID",
                    "OFFER",
                    "CHANGE",
                    "CHANGE_PCT",
                    "HIGH",
                    "LOW",
                    "MARKET_STATE",
                ],
            )

            # Add listener
            listener = IGStreamListener(
                self,
                on_price_update=self.on_price_update,
                on_candle_complete=self.on_candle_complete,
            )
            self.subscription.addListener(listener)

            # Subscribe
            self.client.subscribe(self.subscription)

            logger.info(f"Subscribed to {len(epics)} markets: {', '.join(names)}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to markets: {e}")
            return False

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

        for _, row in historical_df.iterrows():
            candle = Candle(
                timestamp=row["date"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0),
            )
            market.candles.append(candle)

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

    def get_status(self) -> dict:
        """Get streaming service status."""
        return {
            "connected": self.connected,
            "connection_status": self.connection_status,
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
