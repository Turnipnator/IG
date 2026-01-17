"""
Utility functions and logging setup.
"""

import logging
import os
import sys
from datetime import datetime, time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB per file
    backup_count: int = 5,  # Keep 5 backup files (25MB total max)
    use_timed_rotation: bool = False,
    rotation_when: str = "midnight",
    rotation_interval: int = 1,
) -> logging.Logger:
    """
    Configure logging with console and rotating file handler.

    Circular logging ensures HDD doesn't fill up on VPS:
    - RotatingFileHandler: Rotates when file reaches max_bytes
    - TimedRotatingFileHandler: Rotates at intervals (daily by default)

    Default config: 5 files × 5MB = 25MB maximum disk usage

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        max_bytes: Max size of each log file before rotation (default 5MB)
        backup_count: Number of backup files to keep (default 5)
        use_timed_rotation: Use time-based rotation instead of size-based
        rotation_when: When to rotate ('midnight', 'H', 'D', 'W0'-'W6')
        rotation_interval: Interval for rotation

    Returns:
        Configured root logger
    """
    # Allow override from environment
    log_level = os.getenv("LOG_LEVEL", log_level)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Detailed format for file logging
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Simpler format for console
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler - always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # File handler (rotating) - if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if use_timed_rotation:
            # Time-based rotation (e.g., daily at midnight)
            file_handler = TimedRotatingFileHandler(
                log_file,
                when=rotation_when,
                interval=rotation_interval,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            # Size-based rotation (default - better for containers)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )

        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        logger.addHandler(file_handler)

        logger.info(f"Logging to {log_file} (max {max_bytes/1024/1024:.1f}MB × {backup_count} files)")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Message")

    Args:
        name: Logger name (typically __name__)

    Returns:
        Named logger instance
    """
    return logging.getLogger(name)


def log_trade_signal(logger: logging.Logger, signal: dict) -> None:
    """Log a trade signal with consistent formatting."""
    logger.info(
        f"SIGNAL | {signal.get('market', 'N/A')} | "
        f"{signal.get('direction', 'N/A')} | "
        f"Confidence: {signal.get('confidence', 0):.0%} | "
        f"Reason: {signal.get('reason', 'N/A')}"
    )


def log_trade_execution(
    logger: logging.Logger,
    action: str,
    market: str,
    direction: str,
    size: float,
    price: float,
    deal_id: str = "",
) -> None:
    """Log a trade execution with consistent formatting."""
    logger.info(
        f"TRADE | {action} | {market} | "
        f"{direction} | Size: {size} | "
        f"Price: {price:.2f} | "
        f"Deal: {deal_id}"
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: str,
    **extra,
) -> None:
    """Log an error with additional context."""
    extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
    logger.error(
        f"ERROR | {context} | {type(error).__name__}: {error} | {extra_str}",
        exc_info=True,
    )


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
        self.request_count = 0
        self.logger = get_logger(__name__)

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        import time as time_module

        if self.last_request:
            elapsed = (datetime.now() - self.last_request).total_seconds()
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time_module.sleep(wait_time)

        self.last_request = datetime.now()
        self.request_count += 1

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "total_requests": self.request_count,
            "last_request": self.last_request.isoformat() if self.last_request else None,
        }


class HealthCheck:
    """
    Simple health check for monitoring.

    Tracks:
    - Last successful API call
    - Last successful trade
    - Error count
    - Uptime
    """

    def __init__(self):
        self.start_time = datetime.now()
        self.last_api_success: Optional[datetime] = None
        self.last_trade: Optional[datetime] = None
        self.error_count = 0
        self.trade_count = 0
        self.logger = get_logger(__name__)

    def record_api_success(self) -> None:
        """Record successful API call."""
        self.last_api_success = datetime.now()

    def record_trade(self) -> None:
        """Record a trade execution."""
        self.last_trade = datetime.now()
        self.trade_count += 1

    def record_error(self) -> None:
        """Record an error."""
        self.error_count += 1

    def get_status(self) -> dict:
        """Get health status."""
        now = datetime.now()
        uptime = now - self.start_time

        # Consider unhealthy if no API success in 15 minutes
        api_healthy = (
            self.last_api_success is not None
            and (now - self.last_api_success).total_seconds() < 900
        )

        return {
            "healthy": api_healthy,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_human": str(uptime).split(".")[0],
            "last_api_success": self.last_api_success.isoformat() if self.last_api_success else None,
            "last_trade": self.last_trade.isoformat() if self.last_trade else None,
            "trade_count": self.trade_count,
            "error_count": self.error_count,
        }

    def log_status(self) -> None:
        """Log current health status."""
        status = self.get_status()
        self.logger.info(
            f"HEALTH | Healthy: {status['healthy']} | "
            f"Uptime: {status['uptime_human']} | "
            f"Trades: {status['trade_count']} | "
            f"Errors: {status['error_count']}"
        )
