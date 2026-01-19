"""
Configuration management for IG Trading Bot.
Loads settings from environment variables.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class IGConfig:
    """IG Markets API configuration."""
    api_key: str
    username: str
    password: str
    acc_type: str

    @property
    def base_url(self) -> str:
        if self.acc_type.upper() == "DEMO":
            return "https://demo-api.ig.com/gateway/deal"
        return "https://api.ig.com/gateway/deal"


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str
    chat_id: str
    enabled: bool = True


@dataclass
class TradingConfig:
    """Trading parameters."""
    risk_per_trade: float  # Fraction of account to risk per trade
    max_positions: int     # Maximum concurrent positions
    trading_enabled: bool  # Kill switch
    check_interval: int    # Minutes between market checks
    price_data_points: int # Number of historical data points to fetch
    cache_ttl_minutes: int # How long to cache price data


@dataclass
class MarketConfig:
    """Market instrument configuration."""
    epic: str
    name: str
    sector: str
    min_stop_distance: float
    default_size: float


# Load configurations from environment
def load_ig_config() -> IGConfig:
    return IGConfig(
        api_key=os.getenv("IG_API_KEY", ""),
        username=os.getenv("IG_USERNAME", ""),
        password=os.getenv("IG_PASSWORD", ""),
        acc_type=os.getenv("IG_ACC_TYPE", "DEMO"),
    )


def load_telegram_config() -> TelegramConfig:
    return TelegramConfig(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        enabled=os.getenv("TELEGRAM_ENABLED", "true").lower() == "true",
    )


def load_trading_config() -> TradingConfig:
    return TradingConfig(
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
        max_positions=int(os.getenv("MAX_POSITIONS", "5")),
        trading_enabled=os.getenv("TRADING_ENABLED", "true").lower() == "true",
        check_interval=int(os.getenv("CHECK_INTERVAL", "60")),  # 60 mins to conserve API allowance
        price_data_points=int(os.getenv("PRICE_DATA_POINTS", "50")),  # 50 points (saves 50% vs 100)
        cache_ttl_minutes=int(os.getenv("CACHE_TTL_MINUTES", "55")),  # Cache for 55 mins
    )


# Market configurations - IG EPICs for spread betting
# Note: These are spread betting EPICs. CFD EPICs may differ slightly.
# Min stop distances from IG API (verified 2024)
MARKETS = [
    MarketConfig(
        epic="IX.D.SPTRD.DAILY.IP",
        name="S&P 500",
        sector="Indices",
        min_stop_distance=1.0,
        default_size=1.0,
    ),
    MarketConfig(
        epic="IX.D.NASDAQ.CASH.IP",
        name="NASDAQ 100",
        sector="Indices",
        min_stop_distance=4.0,
        default_size=0.2,
    ),
    MarketConfig(
        epic="CC.D.CL.UNC.IP",
        name="Crude Oil",
        sector="Commodities",
        min_stop_distance=12.0,
        default_size=0.1,
    ),
    MarketConfig(
        epic="CC.D.DX.UMP.IP",
        name="Dollar Index (DXY)",
        sector="Forex",
        min_stop_distance=20.0,
        default_size=1.0,
    ),
    MarketConfig(
        epic="CS.D.EURUSD.TODAY.IP",
        name="EUR/USD",
        sector="Forex",
        min_stop_distance=2.0,
        default_size=0.5,
    ),
    MarketConfig(
        epic="CS.D.USCGC.TODAY.IP",
        name="Gold",
        sector="Commodities",
        min_stop_distance=1.0,
        default_size=0.1,
    ),
]


# Strategy parameters
STRATEGY_PARAMS = {
    "ema_fast": 9,
    "ema_medium": 21,
    "ema_slow": 50,
    "rsi_period": 7,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
}
