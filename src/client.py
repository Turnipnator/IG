"""
IG Markets API Client for spread betting.
Handles authentication, market data, and order execution.
"""

import json
import logging
import os
import time
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

from config import IGConfig

logger = logging.getLogger(__name__)

# Disk cache for surviving restarts (saves API allowance)
CACHE_DIR = Path("/app/data") if os.path.exists("/app") else Path("data")
PRICE_CACHE_FILE = CACHE_DIR / "price_cache.json"
DISK_CACHE_TTL_MINUTES = 360  # Use disk cache if < 6 hours old (saves API budget on restarts)

# Positions-API watchdog thresholds. get_positions() runs every 60s, so 3
# consecutive failures = ~3 min of broken position state before we try to
# rebuild the requests.Session pool, and 25 = ~25 min before we hard-exit
# and let Docker restart us. The high tolerance is because IG demo's
# /session and /positions endpoints intermittently time out at 30s for
# several minutes at a stretch; 10 was too aggressive and caused restart
# loops every 20-30 min.
POSITION_FAILURES_BEFORE_RESET = 3
POSITION_FAILURES_BEFORE_EXIT = 25

# Stop-amendment rejection backoff. IG refuses an amendment with
# ATTACHED_ORDER_LEVEL_ERROR when the requested level is too close to current
# price — but the exact rule is NOT confirmed to be the entry-order rule
# (minNormalStopOrLimitDistance): on 2026-08-18 the amendment that finally
# SUCCEEDED sat ~3.5pt above DXY's offer against a stated 10.0pt minimum. See
# _log_stop_amend_diagnostics — measuring that rule is the whole point of the
# extra fields on the refusal line. The caller keeps
# the old level in trailing_stop_levels on failure, so it recomputes the SAME
# level and retries every cycle: 124 identical ERROR lines over two blocks, and
# two earlier episodes on 08-05 and 08-06. The level eventually became legal on
# its own each time, so the retry is right; the cadence and the missing context
# are not. Skip the call while the same deal wants the same rejected level, and
# never let that delay a level that has CHANGED — a tighter stop always goes
# straight through.
STOP_AMEND_FAILURES_BEFORE_BACKOFF = 3
STOP_AMEND_BACKOFF_SECONDS = 60
STOP_AMEND_BACKOFF_MAX_SECONDS = 300
STOP_AMEND_RELOG_MINUTES = 60

# minNormalStopOrLimitDistance is an instrument/account property (tier, region,
# CFD-vs-spreadbet), not a tick-by-tick quantity, so a day-long cache is ample.
# Fetched lazily on the first refusal only — a market whose amendments are being
# accepted never pays for it.
MIN_STOP_DISTANCE_TTL_HOURS = 24


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

    def __init__(self, config: IGConfig, cache_ttl_minutes: int = 1380):  # 23 hours — HTF updates once/day
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
        self.last_error: Optional[str] = None

        # Positions API watchdog. On 2026-05-06 the bot's requests.Session
        # connection pool went zombie — every /positions call timed out for
        # 2h while a fresh Session in the same container hit it in 0.2s.
        # On consecutive failures we recreate the Session; if that doesn't
        # help, exit so Docker restarts the container.
        self._consecutive_position_failures = 0

        # Stop-amendment rejection state, keyed by deal_id ->
        # {level, reason, failures, next_attempt, last_logged}. In-memory only:
        # a restart clearing it costs one duplicate log line, whereas a stale
        # backoff surviving a restart would hold back a real stop tighten.
        self._stop_amend_failures: dict[str, dict] = {}

        # epic -> (min_stop_distance, fetched_at). Populated only when an
        # amendment is refused; see MIN_STOP_DISTANCE_TTL_HOURS.
        self._min_stop_distance_cache: dict[str, tuple] = {}

        # Try to load disk cache from previous session
        self._load_disk_cache()

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

    def _is_auth_error(self, response: requests.Response) -> bool:
        """Check if a response indicates an expired/invalid session."""
        if response.status_code not in (401, 403):
            return False
        body = (response.text or "").lower()
        return (
            "client-token-invalid" in body
            or "security-token-invalid" in body
            or "api-key-invalid" in body
            or "oauth-token-invalid" in body
        )

    def _reauthenticate(self) -> bool:
        """Re-login and switch back to the correct account. Used on token expiry."""
        logger.info("Re-authenticating (session expired)...")
        try:
            self.logout()
        except Exception:
            pass
        self._logged_in = False
        self.cst = None
        self.security_token = None
        if not self.login():
            return False
        # Switch back to spreadbet account so subsequent calls work
        spreadbet_id = self.get_spreadbet_account_id()
        if spreadbet_id and spreadbet_id != self.account_id:
            self.switch_account(spreadbet_id)
        return True

    def _reset_session(self) -> bool:
        """Drop the requests.Session pool entirely and re-login.

        _reauthenticate() only refreshes credentials but reuses the same
        Session object — so its underlying urllib3 connection pool is reused
        too. When that pool goes zombie (sockets stuck in CLOSE_WAIT or
        similar), every request times out. This forces a fresh pool.
        """
        logger.warning("Resetting HTTP session pool and re-authenticating...")
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()
        self._logged_in = False
        self.cst = None
        self.security_token = None
        if not self.login():
            return False
        spreadbet_id = self.get_spreadbet_account_id()
        if spreadbet_id and spreadbet_id != self.account_id:
            self.switch_account(spreadbet_id)
        # Reset the failure counter so the watchdog has a clean slate after a
        # successful reset. Without this, a subsequent transient failure would
        # carry forward the pre-reset count and skip past the reset threshold.
        self._consecutive_position_failures = 0
        return True

    def get_account_info(self) -> Optional[dict]:
        """Get account balance and details."""
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        for attempt in range(2):
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

                if attempt == 0 and self._is_auth_error(response):
                    logger.warning(f"Account info: auth error ({response.status_code}), reauth and retry")
                    if self._reauthenticate():
                        continue
                    return None

                logger.error(f"Failed to get account info: {response.text}")
                return None

            except requests.RequestException as e:
                logger.error(f"Account info request failed: {e}")
                return None

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
                    min_deal_size=self._rule(dealing_rules, "minDealSize", 0.1, epic),
                    min_stop_distance=self._rule(
                        dealing_rules, "minNormalStopOrLimitDistance", 0.0, epic
                    ),
                    market_status=snapshot.get("marketStatus", "CLOSED"),
                    expiry=instrument.get("expiry", ""),
                )
            else:
                logger.error(f"Failed to get market info for {epic}: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Market info request failed: {e}")
            return None

    @staticmethod
    def _rule(dealing_rules: dict, key: str, default: float, epic: str) -> float:
        """Read one IG dealing rule, and SAY SO when it is missing.

        Both callers used to fall back silently, which hides the failure in the
        two places it matters most:

        - min_stop_distance -> 0.0 makes the order-time clamp
          (`if info.min_stop_distance > 0`) skip entirely, so the stop goes to IG
          unchecked and a too-tight one is rejected — the signal is lost with no
          hint as to why. Indistinguishable from IG legitimately reporting 0.
        - min_deal_size -> 0.1 is worse: it is the exact wrong value that once had
          IG reject every Gold order ("IG minimum is 1.0 per point (was 0.1 - all
          trades rejected!)", config.py), and it feeds position sizing.

        The defaults are kept, so behaviour is unchanged — this only makes the
        substitution visible. Refusing to trade on unknown dealing rules would be
        a behaviour change and needs its own decision.
        """
        node = dealing_rules.get(key)
        if isinstance(node, dict) and node.get("value") is not None:
            return node["value"]
        logger.warning(
            f"[{epic}] IG returned no {key} — falling back to {default}. "
            f"The order-time safety clamp cannot be applied on this value."
        )
        return default

    @staticmethod
    def _cache_key(epic: str, resolution: str) -> str:
        """Price cache key: (epic, resolution), NOT the bare epic.

        Until 2026-09-04 the key was the epic alone, so a DAY request served
        within the TTL of a HOUR fetch got the HOUR bars back. At boot the
        candle seed fetched 50 HOUR bars for Crude; the daily HTF refresh 24s
        later asked for DAY, received those hourly bars, and ran the EMA9/21
        rule on intraday closes — HTF BEARISH while the real daily series was
        BULLISH. Any market seeded from REST at startup (a new epic, or a cold
        start with a >6h-stale candle cache) was exposed."""
        return f"{epic}|{resolution}"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if the cached data under this (epic|resolution) key is still valid."""
        if key not in self._price_cache:
            return False
        cached = self._price_cache[key]
        age = datetime.now() - cached.fetched_at
        return age < self._cache_ttl

    def _get_cached_prices(self, epic: str, resolution: str) -> Optional[pd.DataFrame]:
        """Get cached price data for this epic AT THIS RESOLUTION, if valid."""
        key = self._cache_key(epic, resolution)
        if self._is_cache_valid(key):
            cached = self._price_cache[key]
            age_mins = (datetime.now() - cached.fetched_at).total_seconds() / 60
            logger.debug(f"Using cached data for {key} (age: {age_mins:.1f} mins)")
            return cached.data.copy()
        return None

    def _cache_prices(self, epic: str, df: pd.DataFrame, resolution: str) -> None:
        """Store price data in cache (memory + disk) under (epic, resolution)."""
        self._price_cache[self._cache_key(epic, resolution)] = CachedPriceData(
            data=df.copy(),
            fetched_at=datetime.now(),
            epic=epic
        )
        logger.debug(f"Cached price data for {epic} ({resolution})")

        # Persist to disk for surviving restarts
        self._save_disk_cache()

    def clear_cache(self, epic: Optional[str] = None) -> None:
        """Clear price cache for a specific epic (every resolution) or all."""
        if epic:
            for key in [k for k in self._price_cache if k.split("|", 1)[0] == epic]:
                self._price_cache.pop(key, None)
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

    def _load_disk_cache(self) -> None:
        """Load price cache from disk if fresh (saves API calls on restart)."""
        try:
            if not PRICE_CACHE_FILE.exists():
                return

            with open(PRICE_CACHE_FILE, "r") as f:
                cache_data = json.load(f)

            loaded_count = 0
            for key, item in cache_data.items():
                if "|" not in key:
                    # Legacy epic-only key (pre-2026-09-04): its resolution is
                    # unknown, so it cannot safely serve any request. Skip it;
                    # the next fetch rewrites the entry under the new key.
                    continue
                fetched_at = datetime.fromisoformat(item["fetched_at"])
                age = datetime.now() - fetched_at

                # Only use if fresh enough
                if age < timedelta(minutes=DISK_CACHE_TTL_MINUTES):
                    df = pd.DataFrame(item["data"])
                    # Convert date column back to datetime
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    self._price_cache[key] = CachedPriceData(
                        data=df,
                        fetched_at=fetched_at,
                        epic=key.split("|", 1)[0]
                    )
                    loaded_count += 1

            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} markets from disk cache (< {DISK_CACHE_TTL_MINUTES} mins old)")

        except Exception as e:
            logger.debug(f"Could not load disk cache: {e}")

    def _save_disk_cache(self) -> None:
        """Save current price cache to disk for surviving restarts."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            cache_data = {}
            for key, cached in self._price_cache.items():
                # Convert DataFrame to JSON-serializable format
                df_dict = cached.data.copy()
                if "date" in df_dict.columns:
                    df_dict["date"] = df_dict["date"].astype(str)
                cache_data[key] = {
                    "fetched_at": cached.fetched_at.isoformat(),
                    "data": df_dict.to_dict(orient="records")
                }

            with open(PRICE_CACHE_FILE, "w") as f:
                json.dump(cache_data, f)

            logger.debug(f"Saved {len(cache_data)} markets to disk cache")

        except Exception as e:
            logger.warning(f"Could not save disk cache: {e}")

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
        cache_key = self._cache_key(epic, resolution)
        if use_cache:
            cached_df = self._get_cached_prices(epic, resolution)
            if cached_df is not None:
                return cached_df

        # Skip API call on weekends to save allowance
        if self.is_weekend():
            logger.info(f"Weekend - skipping API call for {epic}, using stale cache if available")
            if cache_key in self._price_cache:
                return self._price_cache[cache_key].data.copy()
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

                rows = []
                for p in prices:
                    # Skip candles where bid/ask is null (market closed hours)
                    cp = p["closePrice"]
                    if cp.get("bid") is None or cp.get("ask") is None:
                        continue
                    rows.append({
                        "date": p["snapshotTime"],
                        "open": (p["openPrice"]["bid"] + p["openPrice"]["ask"]) / 2,
                        "high": (p["highPrice"]["bid"] + p["highPrice"]["ask"]) / 2,
                        "low": (p["lowPrice"]["bid"] + p["lowPrice"]["ask"]) / 2,
                        "close": (cp["bid"] + cp["ask"]) / 2,
                        "volume": p.get("lastTradedVolume", 0),
                    })

                if not rows:
                    logger.warning(f"No valid price candles for {epic} (all had null bid/ask)")
                    return None

                df = pd.DataFrame(rows)

                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)

                # Cache the result
                self._cache_prices(epic, df, resolution)
                self._api_calls_today += 1
                logger.info(f"Fetched {len(df)} price points for {epic} (API calls today: {self._api_calls_today})")

                return df

            else:
                error_msg = response.text
                if "exceeded-account-historical-data-allowance" in error_msg:
                    logger.error(f"Historical data allowance exceeded! Using stale cache for {epic}")
                    if cache_key in self._price_cache:
                        return self._price_cache[cache_key].data.copy()
                else:
                    logger.error(f"Failed to get prices for {epic}: {error_msg}")
                return None

        except requests.RequestException as e:
            logger.error(f"Price request failed: {e}")
            return None

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if not self.is_logged_in:
            # Stuck-not-logged-in failure mode: a previous _reset_session()
            # got past the session.close() but its login() retry timed out
            # against IG, leaving _logged_in=False with no further recovery.
            # Feed this path into the same watchdog ladder as request failures
            # so a stuck auth state self-heals (reset attempt then hard-exit
            # → Docker restart). Triggered 2026-05-07 10:23 incident.
            self._consecutive_position_failures += 1
            logger.error(
                f"Not logged in ({self._consecutive_position_failures} consecutive)"
            )
            if self._consecutive_position_failures == POSITION_FAILURES_BEFORE_RESET:
                self._reset_session()
            elif self._consecutive_position_failures >= POSITION_FAILURES_BEFORE_EXIT:
                logger.critical(
                    f"Stuck not-logged-in for {self._consecutive_position_failures} "
                    f"consecutive checks — exiting so Docker restarts the container."
                )
                os._exit(1)
            return []

        for attempt in range(2):
            try:
                response = self.session.get(
                    f"{self.config.base_url}/positions",
                    headers=self._get_headers(version="2"),
                    timeout=30,
                )

                if response.status_code == 200:
                    self._consecutive_position_failures = 0
                    data = response.json()
                    positions = []

                    for pos in data.get("positions", []):
                        position = pos.get("position", {})
                        market = pos.get("market", {})

                        # Calculate P&L from current market prices
                        direction = position.get("direction", "")
                        open_level = position.get("level", 0.0)
                        size = position.get("size", 0.0)
                        bid = market.get("bid", 0.0)
                        offer = market.get("offer", 0.0)

                        if direction == "BUY":
                            pnl = (bid - open_level) * size if bid and open_level else 0.0
                        elif direction == "SELL":
                            pnl = (open_level - offer) * size if offer and open_level else 0.0
                        else:
                            pnl = 0.0

                        positions.append(Position(
                            deal_id=position.get("dealId", ""),
                            epic=market.get("epic", ""),
                            direction=direction,
                            size=size,
                            open_level=open_level,
                            stop_level=position.get("stopLevel"),
                            limit_level=position.get("limitLevel"),
                            profit_loss=round(pnl, 2),
                            created_date=position.get("createdDate", ""),
                        ))

                    return positions

                if attempt == 0 and self._is_auth_error(response):
                    logger.warning(f"Positions: auth error ({response.status_code}), reauth and retry")
                    if self._reauthenticate():
                        continue
                    return []

                logger.error(f"Failed to get positions: {response.text}")
                return []

            except requests.RequestException as e:
                self._consecutive_position_failures += 1
                logger.error(
                    f"Positions request failed "
                    f"({self._consecutive_position_failures} consecutive): {e}"
                )
                # Watchdog: zombie connection pool recovery + last-resort exit.
                if self._consecutive_position_failures == POSITION_FAILURES_BEFORE_RESET:
                    self._reset_session()
                elif self._consecutive_position_failures >= POSITION_FAILURES_BEFORE_EXIT:
                    logger.critical(
                        f"Positions API failed {self._consecutive_position_failures} "
                        f"times in a row — session reset did not recover. "
                        f"Exiting so Docker restarts the container."
                    )
                    os._exit(1)
                return []

        return []

    def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_distance: Optional[float] = None,
        limit_distance: Optional[float] = None,
        guaranteed_stop: bool = False,
        expiry: str = "DFB",
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
            "expiry": expiry,
        }

        if stop_distance:
            payload["stopDistance"] = round(stop_distance, 1)
        if limit_distance:
            payload["limitDistance"] = round(limit_distance, 1)

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

                # Confirm the deal - may be rejected by IG
                self.last_error = None
                confirmation = self._confirm_deal(deal_ref)
                if not confirmation and self.last_error and "Confirmation failed" in self.last_error:
                    # Retries exhausted on deal-not-found: fall back to positions API
                    confirmation = self._fallback_position_lookup(epic, direction)
                if confirmation:
                    logger.info(f"Position opened: {confirmation.get('dealId')} - {direction} {size} {epic}")
                return confirmation
            else:
                try:
                    error_data = response.json()
                    self.last_error = error_data.get("errorCode", response.text)
                except Exception:
                    self.last_error = response.text
                logger.error(f"Failed to open position: {self.last_error}")
                return None

        except requests.RequestException as e:
            self.last_error = str(e)
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

    def _confirm_deal(self, deal_reference: str, max_retries: int = 4) -> Optional[dict]:
        """Confirm a deal was executed successfully, with retries for deal-not-found timing issues."""
        for attempt in range(max_retries):
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
                        if attempt > 0:
                            logger.info(f"Deal confirmed on retry {attempt + 1}: {confirmation.get('dealId')}")
                        else:
                            logger.info(f"Deal confirmed: {confirmation.get('dealId')}")
                        self.last_error = None
                        return confirmation
                    else:
                        reason = confirmation.get("reason", "Unknown")
                        self.last_error = reason
                        logger.error(f"Deal rejected: {reason}")
                        return None
                else:
                    try:
                        error_code = response.json().get("errorCode", "")
                    except Exception:
                        error_code = ""

                    if "deal-not-found" in error_code and attempt < max_retries - 1:
                        wait = attempt + 1
                        logger.warning(
                            f"Deal not found yet (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait}s..."
                        )
                        time.sleep(wait)
                        continue

                    self.last_error = f"Confirmation failed: HTTP {response.status_code}"
                    logger.error(f"Failed to confirm deal: {response.text}")
                    return None

            except requests.RequestException as e:
                self.last_error = str(e)
                logger.error(f"Deal confirmation failed: {e}")
                return None

        return None

    def _fallback_position_lookup(self, epic: str, direction: str) -> Optional[dict]:
        """
        Fallback when confirm endpoint returns deal-not-found after all retries.
        Looks up the newly opened position via the positions API by EPIC.
        """
        time.sleep(2)
        positions = self.get_positions()
        for pos in positions:
            if pos.epic == epic and pos.direction == direction:
                logger.warning(
                    f"Confirm fallback: found position {pos.deal_id} for {epic} via positions API"
                )
                return {
                    "dealId": pos.deal_id,
                    "epic": epic,
                    "direction": direction,
                    "size": pos.size,
                    "level": pos.open_level,
                    "stopLevel": pos.stop_level,
                    "limitLevel": pos.limit_level,
                    "dealStatus": "ACCEPTED",
                }
        logger.error(f"Confirm fallback failed: no {direction} position found for {epic}")
        return None

    def update_position_stop(
        self,
        deal_id: str,
        new_stop_level: float,
        new_limit_level: Optional[float] = None,
        *,
        epic: Optional[str] = None,
        bid: Optional[float] = None,
        offer: Optional[float] = None,
    ) -> bool:
        """
        Update the stop level on an existing position (used for break-even trail).

        Args:
            deal_id: The deal ID to update
            new_stop_level: New stop price level
            new_limit_level: New limit price level (unchanged if None)
            epic: Instrument, for diagnostics on refusal only — never used to
                decide anything. Omitting it costs a less useful log line.
            bid, offer: Live streaming prices at the moment of the request,
                for the same purpose.

        Returns:
            True if successfully updated
        """
        if not self.is_logged_in:
            logger.error("Not logged in")
            return False

        if self._stop_amend_backed_off(deal_id, new_stop_level):
            return False

        payload = {
            "stopLevel": new_stop_level,
        }
        if new_limit_level is not None:
            payload["limitLevel"] = new_limit_level

        try:
            response = self.session.put(
                f"{self.config.base_url}/positions/otc/{deal_id}",
                json=payload,
                headers=self._get_headers(version="2"),
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                deal_ref = result.get("dealReference")
                confirmation = self._confirm_deal(deal_ref)
                if confirmation:
                    self._stop_amend_failures.pop(deal_id, None)
                    logger.info(f"Stop updated for {deal_id}: new stop={new_stop_level}")
                    return True
                self._note_stop_amend_failure(
                    deal_id, new_stop_level, self.last_error or "rejected",
                    epic=epic, bid=bid, offer=offer,
                )
                return False
            # Position already closed (race between stop-hit and BE/trail update).
            # Treat as success so the caller stops retrying.
            if "position.details.null" in response.text:
                self._stop_amend_failures.pop(deal_id, None)
                logger.info(
                    f"Stop update skipped for {deal_id}: position already closed"
                )
                return True
            self._note_stop_amend_failure(
                deal_id, new_stop_level, f"HTTP {response.status_code}: {response.text}",
                epic=epic, bid=bid, offer=offer,
            )
            return False

        except requests.RequestException as e:
            # Transport failure, not a level the broker refuses — don't let it
            # arm the backoff, or one timeout would hold back the next tighten.
            logger.error(f"Update stop request failed: {e}")
            return False

    def _stop_amend_backed_off(self, deal_id: str, new_stop_level: float) -> bool:
        """True if this deal is still backing off from rejections of THIS level.

        A changed level clears the backoff outright: the trail only ever moves a
        stop tighter, so a new level is strictly better protection than the one
        already on the book and must never queue behind a stale rejection."""
        state = self._stop_amend_failures.get(deal_id)
        if state is None:
            return False
        if abs(state["level"] - new_stop_level) > 1e-6:
            del self._stop_amend_failures[deal_id]
            return False
        return datetime.now() < state["next_attempt"]

    def _cached_min_stop_distance(self, epic: str) -> Optional[float]:
        """IG's stated minimum stop distance for an epic, cached for a day."""
        cached = self._min_stop_distance_cache.get(epic)
        if cached and datetime.now() - cached[1] < timedelta(
            hours=MIN_STOP_DISTANCE_TTL_HOURS
        ):
            return cached[0]
        info = self.get_market_info(epic)
        if info is None:
            return None
        self._min_stop_distance_cache[epic] = (info.min_stop_distance, datetime.now())
        return info.min_stop_distance

    def _log_stop_amend_diagnostics(
        self,
        epic: Optional[str],
        new_stop_level: float,
        bid: Optional[float],
        offer: Optional[float],
    ) -> str:
        """Measure — don't assume — what makes IG refuse an amendment.

        The entry path clamps to minNormalStopOrLimitDistance (main.py:1687) and
        it was the obvious explanation here too, but the numbers don't close: on
        2026-08-18 the DXY amendment that SUCCEEDED asked for a level ~3.5pt
        above the offer against a stated 10.0pt minimum, while ones refused an
        hour earlier were ~3.9pt away. So log the distance to BOTH sides of the
        book and the stated minimum, and let a real episode settle which
        quantity IG actually tests. Picking a side here would bake in the very
        assumption this line exists to check. Diagnostics only — nothing below
        feeds a decision, and any failure is swallowed."""
        if not epic:
            return ""
        try:
            parts = []
            if bid is not None and offer is not None:
                parts.append(f"bid={bid} offer={offer}")
                parts.append(f"dist_bid={abs(new_stop_level - bid):.2f}")
                parts.append(f"dist_offer={abs(new_stop_level - offer):.2f}")
            ig_min = self._cached_min_stop_distance(epic)
            if ig_min is not None:
                parts.append(f"ig_min={ig_min:.2f}")
            return f" [{epic} " + " ".join(parts) + "]" if parts else ""
        except Exception as e:  # never let instrumentation break the trail
            logger.debug(f"Stop-amend diagnostics failed for {epic}: {e}")
            return ""

    def _note_stop_amend_failure(
        self,
        deal_id: str,
        new_stop_level: float,
        reason: str,
        epic: Optional[str] = None,
        bid: Optional[float] = None,
        offer: Optional[float] = None,
    ) -> None:
        """Record a refused amendment and log it at a rate that stays readable.

        The bare 'Deal rejected: <reason>' from _confirm_deal names neither the
        deal nor the level, so 124 of them in a row (2026-08-17) said nothing
        about which position was affected or what it was asking for. Log the
        full picture at ERROR on the first failure and on any change of reason,
        then drop to DEBUG and re-surface hourly while the streak continues."""
        now = datetime.now()
        state = self._stop_amend_failures.get(deal_id)
        if state is None or abs(state["level"] - new_stop_level) > 1e-6:
            state = {"level": new_stop_level, "reason": None, "failures": 0,
                     "next_attempt": now, "last_logged": None}
            self._stop_amend_failures[deal_id] = state

        previous_reason = state["reason"]
        state["failures"] += 1
        state["reason"] = reason

        if state["failures"] >= STOP_AMEND_FAILURES_BEFORE_BACKOFF:
            backoff = min(
                STOP_AMEND_BACKOFF_SECONDS
                * 2 ** (state["failures"] - STOP_AMEND_FAILURES_BEFORE_BACKOFF),
                STOP_AMEND_BACKOFF_MAX_SECONDS,
            )
            state["next_attempt"] = now + timedelta(seconds=backoff)
        else:
            backoff = 0

        first = state["last_logged"] is None
        changed = previous_reason is not None and previous_reason != reason
        stale = state["last_logged"] is not None and (
            now - state["last_logged"] >= timedelta(minutes=STOP_AMEND_RELOG_MINUTES)
        )
        message = (
            f"Stop amendment refused for {deal_id}: requested stop={new_stop_level} "
            f"— {reason} (consecutive={state['failures']}"
            + (f", retrying in {backoff}s)" if backoff else ")")
        )
        if first or changed or stale:
            state["last_logged"] = now
            # Diagnostics only on the lines we actually surface, so a long
            # suppressed streak never fetches market info once a minute.
            logger.error(
                message + self._log_stop_amend_diagnostics(
                    epic, new_stop_level, bid, offer
                )
            )
        else:
            logger.debug(message)

    @staticmethod
    def _parse_pnl(pnl_str: str) -> float:
        """Parse IG P&L string like '£-14.90' or '- £14.90' into a float."""
        cleaned = pnl_str.replace("£", "").replace("$", "").replace("€", "").replace(",", "").strip()
        cleaned = cleaned.replace("- ", "-")
        return float(cleaned)

    def get_recent_transactions(self, hours: int = 24) -> list[dict]:
        """Fetch recent IG transaction history. Costs 1 REST call (no data points)."""
        if not self.is_logged_in:
            return []
        try:
            now = datetime.utcnow()
            from_date = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
            to_date = now.strftime("%Y-%m-%dT%H:%M:%S")
            response = self.session.get(
                f"{self.config.base_url}/history/transactions",
                params={"type": "ALL", "from": from_date, "to": to_date, "pageSize": 50},
                headers=self._get_headers(version="2"),
                timeout=30,
            )
            if response.status_code == 200:
                return response.json().get("transactions", [])
            logger.warning(f"Transaction history request failed: {response.status_code}")
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch transactions: {e}")
            return []

    def find_close_transaction(
        self,
        open_level: float,
        direction: str,
        transactions: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """
        Find the IG DEAL transaction that closed a position with the given
        open_level + direction. Returns the raw txn dict (with profitAndLoss,
        closeLevel, etc.) so callers can extract whichever fields they need.

        If `transactions` is provided, search within it (avoids extra API call —
        useful when reconciling many provisional rows from a single fetch).
        """
        if transactions is None:
            transactions = self.get_recent_transactions(hours=24)
        for txn in transactions:
            if txn.get("transactionType") != "DEAL":
                continue
            try:
                txn_open = float(txn.get("openLevel", 0))
                txn_size = float(str(txn.get("size", "0")).replace("+", ""))
                txn_dir = "BUY" if txn_size > 0 else "SELL"
                if abs(txn_open - open_level) < 1.0 and txn_dir == direction:
                    return txn
            except (ValueError, TypeError):
                continue
        return None

    def get_closed_position_pnl(
        self, deal_id: str, open_level: float = 0.0, direction: str = ""
    ) -> Optional[float]:
        """
        Backward-compat: return only the P&L for a recently closed position.
        Prefer find_close_transaction() if you also need closeLevel.
        """
        if not (open_level and direction):
            return None
        txn = self.find_close_transaction(open_level, direction)
        if txn is None:
            logger.warning(f"No transaction match for deal {deal_id}")
            return None
        pnl = self._parse_pnl(txn.get("profitAndLoss", "0"))
        logger.info(f"P&L for {deal_id}: £{pnl:.2f} (matched by level+direction)")
        return pnl

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
