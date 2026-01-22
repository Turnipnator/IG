"""
IG Markets API Client for spread betting.
Handles authentication, market data, and order execution.
"""

import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests
import pandas as pd

from config import IGConfig

logger = logging.getLogger(__name__)


@dataclass
class CachedPriceData:
    """Cached price data with timestamp."""
    data: pd.DataFrame
    fetched_at: datetime
    epic: str


@dataclass
class Position:
    """Represents an open position."""
    deal_id: str
    epic: str
    direction: str
    size: float
    open_level: float
    stop_level: Optional[float]
    limit_level: Optional[float]
    profit_loss: float
    created_date: str


@dataclass
class MarketInfo:
    """Market information for an instrument."""
    epic: str
    instrument_name: str
    bid: float
    offer: float
    high: float
    low: float
    min_deal_size: float
    min_stop_distance: float
    market_status: str
    expiry: str


class IGClient:
    """Client for interacting with IG Markets REST API."""

    def __init__(self, config: IGConfig, cache_ttl_minutes: int = 55):
        self.config = config
        self.session = requests.Session()
        self.cst: Optional[str] = None
        self.security_token: Optional[str] = None
        self.account_id: Optional[str] = None
        self.accounts: list = []
        self._logged_in = False
        self._price_cache: dict[str, CachedPriceData] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._api_calls_today = 0
        self._last_reset_date = datetime.now().date()

    @property
    def is_logged_in(self) -> bool:
        return self._logged_in and self.cst is not None

    def login(self) -> bool:
        """Authenticate with IG API and obtain session tokens."""
        headers = {
            "Content-Type": "application/json",
            "X-IG-API-KEY": self.config.api_key,
            "Version": "2",
        }
        payload = {
            "identifier": self.config.username,
            "password": self.config.password,
        }

        try:
            response = self.session.post(
                f"{self.config.base_url}/session",
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.security_token = response.headers.get("X-SECURITY-TOKEN")

                # Debug: log token status
                logger.debug(f"CST token: {'OK' if self.cst else 'MISSING'} ({len(self.cst) if self.cst else 0} chars)")
                logger.debug(f"Security token: {'OK' if self.security_token else 'MISSING'} ({len(self.security_token) if self.security_token else 0} chars)")

                data = response.json()
                self.account_id = data.get("currentAccountId")
                self.accounts = data.get("accounts", [])

                self._logged_in = True
                logger.info(f"Successfully logged in to IG ({self.config.acc_type})")
                logger.info(f"Account ID: {self.account_id}")

                # Log available accounts
                for acc in self.accounts:
                    logger.info(f"  Available: {acc.get('accountId')} ({acc.get('accountType')})")

                return True
            else:
                logger.error(f"Login failed: {response.status_code} - {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"Login request failed: {e}")
            return False

    def logout(self) -> bool:
        """End the current session."""
        if not self.is_logged_in:
            return True

        try:
            response = self.session.delete(
                f"{self.config.base_url}/session",
                headers=self._get_headers(),
                timeout=30,
            )
            self._logged_in = False
            self.cst = None
            self.security_token = None
            logger.info("Logged out from IG")
            return response.status_code == 200

        except requests.RequestException as e:
            logger.error(f"Logout failed: {e}")
            return False

    def switch_account(self, account_id: str) -> bool:
        """
        Switch to a different account.

        Required for streaming - must switch to SPREADBET account
        before Lightstreamer will work.
        """
        if not self.is_logged_in:
            logger.error("Not logged in")
            return False

        try:
            response = self.session.put(
                f"{self.config.base_url}/session",
                json={"accountId": account_id},
                headers=self._get_headers(version="1"),
                timeout=30,
            )

            if response.status_code == 200:
                # Update tokens if new ones provided
                new_cst = response.headers.get("CST")
                new_token = response.headers.get("X-SECURITY-TOKEN")

                logger.debug(f"Switch account response - New CST: {'YES' if new_cst else 'NO'}, New XST: {'YES' if new_token else 'NO'}")

                if new_cst:
                    old_len = len(self.cst) if self.cst else 0
                    self.cst = new_cst
                    logger.info(f"CST token updated ({old_len} -> {len(new_cst)} chars)")
                if new_token:
                    old_len = len(self.security_token) if self.security_token else 0
                    self.security_token = new_token
                    logger.info(f"Security token updated ({old_len} -> {len(new_token)} chars)")

                self.account_id = account_id
                logger.info(f"Switched to account: {account_id}")
                return True
            else:
                logger.error(f"Failed to switch account: {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"Account switch failed: {e}")
            return False

    def get_spreadbet_account_id(self) -> Optional[str]:
        """Get the SPREADBET account ID if available."""
        for acc in getattr(self, 'accounts', []):
            if acc.get('accountType') == 'SPREADBET':
                return acc.get('accountId')
        return None

    def _get_headers(self, version: str = "2") -> dict:
        """Get authenticated request headers."""
        return {
            "Content-Type": "application/json",
            "X-IG-API-KEY": self.config.api_key,
            "CST": self.cst or "",
            "X-SECURITY-TOKEN": self.security_token or "",
            "Version": version,
        }

    def get_account_info(self) -> Optional[dict]:
        """Get account balance and details."""
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        try:
            headers = self._get_headers(version="1")
            logger.debug(f"Account info request - CST length: {len(headers.get('CST', ''))}, Token length: {len(headers.get('X-SECURITY-TOKEN', ''))}")

            response = self.session.get(
                f"{self.config.base_url}/accounts",
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get account info: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Account info request failed: {e}")
            return None

    def get_balance(self) -> Optional[float]:
        """Get current account balance."""
        account_info = self.get_account_info()
        if account_info and "accounts" in account_info:
            for account in account_info["accounts"]:
                if account.get("accountId") == self.account_id:
                    return account.get("balance", {}).get("balance", 0.0)
        return None

    def get_market_info(self, epic: str) -> Optional[MarketInfo]:
        """Get market information for an instrument."""
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        try:
            response = self.session.get(
                f"{self.config.base_url}/markets/{epic}",
                headers=self._get_headers(version="3"),
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                snapshot = data.get("snapshot", {})
                instrument = data.get("instrument", {})
                dealing_rules = data.get("dealingRules", {})

                return MarketInfo(
                    epic=epic,
                    instrument_name=instrument.get("name", ""),
                    bid=snapshot.get("bid", 0.0),
                    offer=snapshot.get("offer", 0.0),
                    high=snapshot.get("high", 0.0),
                    low=snapshot.get("low", 0.0),
                    min_deal_size=dealing_rules.get("minDealSize", {}).get("value", 0.1),
                    min_stop_distance=dealing_rules.get("minNormalStopOrLimitDistance", {}).get("value", 0.0),
                    market_status=snapshot.get("marketStatus", "CLOSED"),
                    expiry=instrument.get("expiry", ""),
                )
            else:
                logger.error(f"Failed to get market info for {epic}: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Market info request failed: {e}")
            return None

    def _is_cache_valid(self, epic: str) -> bool:
        """Check if cached data for an epic is still valid."""
        if epic not in self._price_cache:
            return False
        cached = self._price_cache[epic]
        age = datetime.now() - cached.fetched_at
        return age < self._cache_ttl

    def _get_cached_prices(self, epic: str) -> Optional[pd.DataFrame]:
        """Get cached price data if valid."""
        if self._is_cache_valid(epic):
            cached = self._price_cache[epic]
            age_mins = (datetime.now() - cached.fetched_at).total_seconds() / 60
            logger.debug(f"Using cached data for {epic} (age: {age_mins:.1f} mins)")
            return cached.data.copy()
        return None

    def _cache_prices(self, epic: str, df: pd.DataFrame) -> None:
        """Store price data in cache."""
        self._price_cache[epic] = CachedPriceData(
            data=df.copy(),
            fetched_at=datetime.now(),
            epic=epic
        )
        logger.debug(f"Cached price data for {epic}")

    def clear_cache(self, epic: Optional[str] = None) -> None:
        """Clear price cache for a specific epic or all."""
        if epic:
            self._price_cache.pop(epic, None)
            logger.info(f"Cleared cache for {epic}")
        else:
            self._price_cache.clear()
            logger.info("Cleared all price cache")

    def get_cache_status(self) -> dict:
        """Get cache status for monitoring."""
        status = {}
        for epic, cached in self._price_cache.items():
            age = datetime.now() - cached.fetched_at
            status[epic] = {
                "age_minutes": age.total_seconds() / 60,
                "valid": age < self._cache_ttl,
                "rows": len(cached.data)
            }
        return status

    def is_weekend(self) -> bool:
        """Check if it's currently weekend (markets closed)."""
        now = datetime.now()
        # Markets closed from Friday ~10pm to Sunday ~10pm UTC
        # Simplify: Saturday and Sunday before 10pm are definitely closed
        return now.weekday() >= 5  # Saturday=5, Sunday=6

    def get_historical_prices(
        self,
        epic: str,
        resolution: str = "MINUTE_5",
        num_points: int = 50,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data with caching.

        Args:
            epic: Instrument identifier
            resolution: Time resolution (MINUTE_5, HOUR, DAY, etc.)
            num_points: Number of data points to fetch (default 50 to conserve allowance)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data or None if failed

        Note:
            IG API has a 10,000 data points/week limit. This method uses caching
            to minimize API calls. Cache TTL is configurable (default 55 mins).
        """
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        # Check cache first
        if use_cache:
            cached_df = self._get_cached_prices(epic)
            if cached_df is not None:
                return cached_df

        # Skip API call on weekends to save allowance
        if self.is_weekend():
            logger.info(f"Weekend - skipping API call for {epic}, using stale cache if available")
            if epic in self._price_cache:
                return self._price_cache[epic].data.copy()
            return None

        try:
            response = self.session.get(
                f"{self.config.base_url}/prices/{epic}",
                params={
                    "resolution": resolution,
                    "max": num_points,
                    "pageSize": num_points,
                },
                headers=self._get_headers(version="3"),
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])

                if not prices:
                    logger.warning(f"No price data returned for {epic}")
                    return None

                df = pd.DataFrame([
                    {
                        "date": p["snapshotTime"],
                        "open": (p["openPrice"]["bid"] + p["openPrice"]["ask"]) / 2,
                        "high": (p["highPrice"]["bid"] + p["highPrice"]["ask"]) / 2,
                        "low": (p["lowPrice"]["bid"] + p["lowPrice"]["ask"]) / 2,
                        "close": (p["closePrice"]["bid"] + p["closePrice"]["ask"]) / 2,
                        "volume": p.get("lastTradedVolume", 0),
                    }
                    for p in prices
                ])

                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)

                # Cache the result
                self._cache_prices(epic, df)
                self._api_calls_today += 1
                logger.info(f"Fetched {len(df)} price points for {epic} (API calls today: {self._api_calls_today})")

                return df

            else:
                error_msg = response.text
                if "exceeded-account-historical-data-allowance" in error_msg:
                    logger.error(f"Historical data allowance exceeded! Using stale cache for {epic}")
                    if epic in self._price_cache:
                        return self._price_cache[epic].data.copy()
                else:
                    logger.error(f"Failed to get prices for {epic}: {error_msg}")
                return None

        except requests.RequestException as e:
            logger.error(f"Price request failed: {e}")
            return None

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if not self.is_logged_in:
            logger.error("Not logged in")
            return []

        try:
            response = self.session.get(
                f"{self.config.base_url}/positions",
                headers=self._get_headers(version="2"),
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                positions = []

                for pos in data.get("positions", []):
                    position = pos.get("position", {})
                    market = pos.get("market", {})

                    positions.append(Position(
                        deal_id=position.get("dealId", ""),
                        epic=market.get("epic", ""),
                        direction=position.get("direction", ""),
                        size=position.get("size", 0.0),
                        open_level=position.get("openLevel", 0.0),
                        stop_level=position.get("stopLevel"),
                        limit_level=position.get("limitLevel"),
                        profit_loss=position.get("profit", 0.0),
                        created_date=position.get("createdDate", ""),
                    ))

                return positions
            else:
                logger.error(f"Failed to get positions: {response.text}")
                return []

        except requests.RequestException as e:
            logger.error(f"Positions request failed: {e}")
            return []

    def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_distance: Optional[float] = None,
        limit_distance: Optional[float] = None,
        guaranteed_stop: bool = False,
    ) -> Optional[dict]:
        """
        Open a new spread bet position.

        Args:
            epic: Instrument identifier
            direction: "BUY" or "SELL"
            size: Position size (stake per point for spread betting)
            stop_distance: Stop loss distance in points
            limit_distance: Take profit distance in points
            guaranteed_stop: Use guaranteed stop (premium applies)

        Returns:
            Deal reference dict or None if failed
        """
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        payload = {
            "epic": epic,
            "direction": direction,
            "size": str(size),
            "orderType": "MARKET",
            "guaranteedStop": guaranteed_stop,
            "forceOpen": True,
            "currencyCode": "GBP",
            "expiry": "DFB",  # Daily Funded Bet - required for spread betting (not "-")
        }

        if stop_distance:
            payload["stopDistance"] = str(stop_distance)
        if limit_distance:
            payload["limitDistance"] = str(limit_distance)

        try:
            response = self.session.post(
                f"{self.config.base_url}/positions/otc",
                json=payload,
                headers=self._get_headers(version="2"),
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                deal_ref = result.get("dealReference")
                logger.info(f"Position opened: {deal_ref} - {direction} {size} {epic}")

                # Confirm the deal
                return self._confirm_deal(deal_ref)
            else:
                logger.error(f"Failed to open position: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Open position request failed: {e}")
            return None

    def close_position(
        self,
        deal_id: str,
        direction: str,
        size: float,
    ) -> Optional[dict]:
        """
        Close an existing position.

        Args:
            deal_id: The deal ID to close
            direction: Original position direction (will be reversed)
            size: Size to close

        Returns:
            Deal confirmation or None if failed
        """
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        # Close requires opposite direction
        close_direction = "SELL" if direction == "BUY" else "BUY"

        headers = self._get_headers(version="1")
        headers["_method"] = "DELETE"

        payload = {
            "dealId": deal_id,
            "direction": close_direction,
            "size": str(size),
            "orderType": "MARKET",
        }

        try:
            response = self.session.post(
                f"{self.config.base_url}/positions/otc",
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                deal_ref = result.get("dealReference")
                logger.info(f"Position closed: {deal_ref}")
                return self._confirm_deal(deal_ref)
            else:
                logger.error(f"Failed to close position: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Close position request failed: {e}")
            return None

    def _confirm_deal(self, deal_reference: str) -> Optional[dict]:
        """Confirm a deal was executed successfully."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/confirms/{deal_reference}",
                headers=self._get_headers(version="1"),
                timeout=30,
            )

            if response.status_code == 200:
                confirmation = response.json()
                status = confirmation.get("dealStatus")

                if status == "ACCEPTED":
                    logger.info(f"Deal confirmed: {confirmation.get('dealId')}")
                    return confirmation
                else:
                    reason = confirmation.get("reason", "Unknown")
                    logger.error(f"Deal rejected: {reason}")
                    return None
            else:
                logger.error(f"Failed to confirm deal: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Deal confirmation failed: {e}")
            return None

    def search_markets(self, search_term: str) -> list[dict]:
        """Search for markets by name or keyword."""
        if not self.is_logged_in:
            logger.error("Not logged in")
            return []

        try:
            response = self.session.get(
                f"{self.config.base_url}/markets",
                params={"searchTerm": search_term},
                headers=self._get_headers(version="1"),
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("markets", [])
            else:
                logger.error(f"Market search failed: {response.text}")
                return []

        except requests.RequestException as e:
            logger.error(f"Market search request failed: {e}")
            return []
