"""
Economic calendar integration.
Fetches high-impact events and reports when a market is near one.
Uses ForexFactory's free calendar feed.

2026-09-22: the feed stopped sending a separate `time` field — `date` is now one ISO
timestamp with its offset ("2026-09-23T21:30:00-04:00") — and the parser dropped
every event, so the block had never fired ("0 high-impact events" on every refresh).
Times are now timezone-aware UTC. Whether a caller BLOCKS on would_block() is the
caller's decision; main.py is log-only by default (CALENDAR_ENFORCE).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

# ForexFactory free JSON calendar feed
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map currencies to affected EPICs
CURRENCY_EPIC_MAP = {
    "USD": [
        "IX.D.SPTRD.DAILY.IP",   # S&P 500
        "IX.D.NASDAQ.CASH.IP",   # NASDAQ
        "CS.D.EURUSD.TODAY.IP",  # EUR/USD (USD side)
        "CS.D.GBPUSD.TODAY.IP",  # GBP/USD (USD side)
        "CS.D.USCGC.TODAY.IP",   # Gold (USD-denominated)
        "CC.D.CL.USS.IP",       # Crude Oil (USD-denominated)
        "CC.D.DX.USS.IP",       # Dollar Index
    ],
    "EUR": [
        "CS.D.EURUSD.TODAY.IP",  # EUR/USD (EUR side)
    ],
    "GBP": [
        "CS.D.GBPUSD.TODAY.IP",  # GBP/USD (GBP side)
    ],
    "JPY": [],
    "AUD": [],
    "CAD": [
        "CC.D.CL.USS.IP",       # Crude Oil (Canada is major producer)
    ],
    "CHF": [],
    "NZD": [],
    "CNY": [],
}


@dataclass
class EconomicEvent:
    """A single economic calendar event."""
    title: str
    country: str
    currency: str
    impact: str  # "High", "Medium", "Low"
    event_time: Optional[datetime]


class EconomicCalendar:
    """
    Fetches and caches economic calendar events.
    Provides a check to block trading around high-impact events.
    """

    def __init__(self, buffer_minutes: int = 30):
        """
        Args:
            buffer_minutes: Minutes before/after event to block trading.
        """
        self.buffer_minutes = buffer_minutes
        self.events: list[EconomicEvent] = []
        self.last_fetch: Optional[datetime] = None
        self.fetch_interval = timedelta(hours=6)  # Refresh every 6 hours
        # A failing feed (HTTP 429 is common) must not be re-fetched on every call.
        self.last_attempt: Optional[datetime] = None
        self.retry_interval = timedelta(minutes=30)

    def refresh(self) -> bool:
        """Fetch calendar data from ForexFactory feed. On failure the previous
        events are kept."""
        self.last_attempt = datetime.now()
        try:
            response = requests.get(CALENDAR_URL, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Calendar fetch failed: HTTP {response.status_code}")
                return False

            data = response.json()
            events: list[EconomicEvent] = []
            high_rows = 0

            for event in data:
                impact = event.get("impact", "")
                if impact != "High":
                    continue  # Only track high-impact events
                high_rows += 1

                # Parse event time
                date_str = event.get("date", "")
                time_str = event.get("time", "")

                event_time = self._parse_event_time(date_str, time_str)
                if not event_time:
                    continue

                events.append(EconomicEvent(
                    title=event.get("title", "Unknown"),
                    country=event.get("country", ""),
                    currency=event.get("country", ""),  # FF uses country code as currency
                    impact=impact,
                    event_time=event_time,
                ))

            self.events = events
            self.last_fetch = datetime.now()
            if high_rows and not events:
                # The failure this module had for eight months: a format change that
                # parses to nothing and reports itself as a quiet week.
                logger.warning(
                    f"Economic calendar: parsed 0 of {high_rows} high-impact rows — "
                    f"the feed format has probably changed")
            else:
                logger.info(f"Economic calendar: {len(events)} high-impact events this week")
            return True

        except requests.RequestException as e:
            logger.warning(f"Calendar fetch error: {e}")
            return False
        except (ValueError, KeyError) as e:
            logger.warning(f"Calendar parse error: {e}")
            return False

    def _parse_event_time(self, date_str: str, time_str: str = "") -> Optional[datetime]:
        """Event time as timezone-aware UTC, or None for all-day/tentative/unparseable.

        Current feed: `date` is ISO with an offset and there is no `time`. The old
        split format (`date` "2026-01-23" + `time` "8:30am", US Eastern) is still
        accepted, converted with the real ET offset rather than a fixed +5h."""
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc)
        except ValueError:
            dt = None

        if not time_str or time_str in ("All Day", "Tentative"):
            return None
        combined = f"{date_str[:10]} {time_str.strip().lower()}"
        for fmt in ("%Y-%m-%d %I:%M%p", "%Y-%m-%d %H:%M"):
            try:
                et = datetime.strptime(combined, fmt).replace(tzinfo=ZoneInfo("America/New_York"))
                return et.astimezone(timezone.utc)
            except ValueError:
                continue
        return None

    def would_block(self, epic: str, allow_refresh: bool = True) -> Optional[str]:
        """Reason string if a mapped high-impact event is within the buffer of now,
        else None. `allow_refresh=False` never touches the network — for the order
        path, where a 15s fetch must not sit in front of an order."""
        if allow_refresh and self._stale() and self._may_retry():
            self.refresh()
        if not self.events:
            return None

        now = datetime.now(timezone.utc)
        buffer = timedelta(minutes=self.buffer_minutes)
        for event in self.events:
            if not event.event_time or abs(now - event.event_time) > buffer:
                continue
            if epic in CURRENCY_EPIC_MAP.get(event.currency, []):
                mins_to_event = (event.event_time - now).total_seconds() / 60
                direction = "in" if mins_to_event > 0 else "ago"
                return (f"High-impact event: {event.title} ({event.currency}) "
                        f"{abs(mins_to_event):.0f} mins {direction}")
        return None

    def is_safe_to_trade(self, epic: str) -> tuple[bool, str]:
        """(is_safe, reason_if_blocked) — kept for callers of the old API."""
        reason = self.would_block(epic)
        return (reason is None), (reason or "")

    def _stale(self) -> bool:
        return self.last_fetch is None or datetime.now() - self.last_fetch > self.fetch_interval

    def _may_retry(self) -> bool:
        return self.last_attempt is None or datetime.now() - self.last_attempt > self.retry_interval

    def get_upcoming_events(self, hours: int = 24) -> list[EconomicEvent]:
        """Get high-impact events in the next N hours."""
        if self.last_fetch is None:
            self.refresh()

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)

        return [
            e for e in self.events
            if e.event_time and now <= e.event_time <= cutoff
        ]
