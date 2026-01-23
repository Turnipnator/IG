"""
Economic calendar integration.
Fetches high-impact events and blocks trading during volatile periods.
Uses ForexFactory's free calendar feed.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

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
        "CS.D.USCGC.TODAY.IP",   # Gold (USD-denominated)
        "EN.D.CL.Month1.IP",    # Crude Oil (USD-denominated)
        "CO.D.DX.Month1.IP",    # Dollar Index
    ],
    "EUR": [
        "CS.D.EURUSD.TODAY.IP",  # EUR/USD (EUR side)
    ],
    "GBP": [],  # No GBP pairs currently traded
    "JPY": [],
    "AUD": [],
    "CAD": [
        "EN.D.CL.Month1.IP",    # Crude Oil (Canada is major producer)
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

    def refresh(self) -> bool:
        """Fetch calendar data from ForexFactory feed."""
        try:
            response = requests.get(CALENDAR_URL, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Calendar fetch failed: HTTP {response.status_code}")
                return False

            data = response.json()
            self.events = []

            for event in data:
                impact = event.get("impact", "")
                if impact != "High":
                    continue  # Only track high-impact events

                # Parse event time
                date_str = event.get("date", "")
                time_str = event.get("time", "")

                event_time = self._parse_event_time(date_str, time_str)
                if not event_time:
                    continue

                self.events.append(EconomicEvent(
                    title=event.get("title", "Unknown"),
                    country=event.get("country", ""),
                    currency=event.get("country", ""),  # FF uses country code as currency
                    impact=impact,
                    event_time=event_time,
                ))

            self.last_fetch = datetime.now()
            logger.info(f"Economic calendar: {len(self.events)} high-impact events this week")
            return True

        except requests.RequestException as e:
            logger.warning(f"Calendar fetch error: {e}")
            return False
        except (ValueError, KeyError) as e:
            logger.warning(f"Calendar parse error: {e}")
            return False

    def _parse_event_time(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Parse ForexFactory date/time format."""
        if not date_str:
            return None

        # Time might be empty for "All Day" or "Tentative" events
        if not time_str or time_str in ("All Day", "Tentative", ""):
            return None

        try:
            # FF format: "2026-01-23" and "8:30am" (ET timezone)
            # Convert to 24h format
            combined = f"{date_str} {time_str}"

            # Handle various time formats from FF
            for fmt in ("%Y-%m-%d %I:%M%p", "%Y-%m-%d %I:%Mam", "%Y-%m-%d %I:%Mpm",
                        "%Y-%m-%d %H:%M"):
                try:
                    # FF times are in US Eastern
                    et_time = datetime.strptime(combined, fmt)
                    # Convert ET to UTC (ET is UTC-5, adjust for DST later if needed)
                    utc_time = et_time + timedelta(hours=5)
                    return utc_time
                except ValueError:
                    continue

            return None
        except Exception:
            return None

    def is_safe_to_trade(self, epic: str) -> tuple[bool, str]:
        """
        Check if it's safe to trade a given market.

        Returns:
            Tuple of (is_safe, reason_if_blocked)
        """
        # Refresh if stale
        if self.last_fetch is None or datetime.now() - self.last_fetch > self.fetch_interval:
            self.refresh()

        if not self.events:
            return True, ""

        now = datetime.utcnow()
        buffer = timedelta(minutes=self.buffer_minutes)

        for event in self.events:
            if not event.event_time:
                continue

            # Check if we're within the buffer window
            if abs((now - event.event_time).total_seconds()) > buffer.total_seconds():
                continue

            # Check if this event affects this epic
            affected_epics = CURRENCY_EPIC_MAP.get(event.currency, [])
            if epic in affected_epics:
                mins_to_event = (event.event_time - now).total_seconds() / 60
                direction = "in" if mins_to_event > 0 else "ago"
                mins_abs = abs(mins_to_event)
                return False, (
                    f"High-impact event: {event.title} ({event.currency}) "
                    f"{mins_abs:.0f} mins {direction}"
                )

        return True, ""

    def get_upcoming_events(self, hours: int = 24) -> list[EconomicEvent]:
        """Get high-impact events in the next N hours."""
        if self.last_fetch is None:
            self.refresh()

        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours)

        return [
            e for e in self.events
            if e.event_time and now <= e.event_time <= cutoff
        ]
