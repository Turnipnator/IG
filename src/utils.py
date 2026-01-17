"""
Utility functions and logging setup.
"""

import logging
import sys
from datetime import datetime, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure logging with console and optional rotating file handler.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        max_bytes: Max size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def is_market_open(market_type: str = "forex") -> bool:
    """
    Check if markets are likely open based on time.

    Note: This is a simplified check. Real implementation should
    use IG API to check actual market status.

    Args:
        market_type: Type of market (forex, indices, commodities)

    Returns:
        True if market is likely open
    """
    now = datetime.utcnow()
    weekday = now.weekday()

    # Forex: Sunday 22:00 UTC to Friday 22:00 UTC
    if market_type == "forex":
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            return now.hour >= 22
        if weekday == 4:  # Friday
            return now.hour < 22
        return True

    # Indices: Roughly follow their exchange hours
    if market_type == "indices":
        if weekday in (5, 6):  # Weekend
            return False
        # US indices: ~14:30 - 21:00 UTC
        # UK indices: ~08:00 - 16:30 UTC
        # This is simplified - check actual hours
        return time(7, 0) <= now.time() <= time(21, 30)

    # Commodities: Vary by product
    if market_type == "commodities":
        if weekday in (5, 6):
            return False
        return True

    return True


def format_currency(value: float, currency: str = "GBP") -> str:
    """Format a value as currency."""
    symbols = {"GBP": "£", "USD": "$", "EUR": "€"}
    symbol = symbols.get(currency, currency)

    if value >= 0:
        return f"{symbol}{value:,.2f}"
    return f"-{symbol}{abs(value):,.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.{decimal_places}f}%"


def format_points(value: float) -> str:
    """Format a value as points."""
    if value >= 0:
        return f"+{value:.2f} pts"
    return f"{value:.2f} pts"


def calculate_pip_value(
    epic: str,
    size: float,
    account_currency: str = "GBP",
) -> float:
    """
    Calculate the value of one pip/point for a position.

    For spread betting, this is simply the stake size.
    For CFDs, it depends on the contract specification.

    Args:
        epic: Instrument EPIC
        size: Position size
        account_currency: Account currency

    Returns:
        Value per pip in account currency
    """
    # For spread betting, pip value = stake size
    return size


def get_market_type(epic: str) -> str:
    """
    Determine market type from EPIC.

    Args:
        epic: IG EPIC identifier

    Returns:
        Market type (forex, indices, commodities)
    """
    if epic.startswith("CS.D."):
        return "forex"
    if epic.startswith("IX.D."):
        return "indices"
    if epic.startswith("CC.D."):
        return "commodities"
    return "unknown"


class RateLimiter:
    """
    Simple rate limiter for API calls.

    IG has rate limits:
    - Historical prices: 30 requests per minute
    - Trading: 60 requests per minute
    """

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request: Optional[datetime] = None

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        import time as time_module

        if self.last_request:
            elapsed = (datetime.now() - self.last_request).total_seconds()
            if elapsed < self.interval:
                time_module.sleep(self.interval - elapsed)

        self.last_request = datetime.now()
