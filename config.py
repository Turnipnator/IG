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
class StrategyConfig:
    """Strategy parameters for a specific market type."""
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50
    rsi_period: int = 7
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_buy_max: int = 60
    rsi_sell_min: int = 40
    adx_threshold: int = 25
    stop_atr_mult: float = 1.5
    reward_risk: float = 2.0
    min_confidence: float = 0.5
    use_macd_exit: bool = True
    require_htf: bool = False


@dataclass
class MarketConfig:
    """Market instrument configuration."""
    epic: str
    name: str
    sector: str
    min_stop_distance: float
    default_size: float
    expiry: str = "DFB"  # DFB for daily funded bets, or specific like "MAR-26"
    candle_interval: int = 5  # Candle duration in minutes (5 for indices/commodities, 15 for forex)
    min_confidence: float = 0.5  # Minimum confidence to enter (higher = more selective)
    strategy: str = "default"  # Strategy profile to use: "default" or "indices"


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


# =============================================================================
# STRATEGY PROFILES
# =============================================================================
# Based on comprehensive backtesting (30 days, £10k account)
#
# "Big Winners" Strategy (Gold, EUR/USD, Dollar Index, Crude Oil):
#   - Low win rate (27%) but huge winners that dwarf losses
#   - Average win £241, average loss £53 = 4.5:1 ratio
#   - Monthly return: +24.6% (£2,462)
#   - Key: High R:R (4.0), no MACD exit, low confidence threshold
#
# "Momentum" Strategy (S&P 500, NASDAQ 100):
#   - High win rate (59%) with many smaller wins
#   - Faster EMAs to catch momentum moves
#   - Monthly return: +5.1% (£505)
#   - Key: Fast EMAs (5/12/26), MACD exit ON, require HTF alignment

STRATEGY_PROFILES = {
    # Default strategy for most markets: "Big Winners"
    # Optimized for Gold, EUR/USD, Dollar Index, Crude Oil
    "default": StrategyConfig(
        ema_fast=9,
        ema_medium=21,
        ema_slow=50,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=60,
        rsi_sell_min=40,
        adx_threshold=25,
        stop_atr_mult=1.5,
        reward_risk=4.0,       # High R:R - let winners run big
        min_confidence=0.4,    # Lower threshold - more entries
        use_macd_exit=False,   # Don't cut winners short
        require_htf=False,
    ),

    # Indices strategy: "Momentum"
    # Optimized for S&P 500 and NASDAQ 100
    "indices": StrategyConfig(
        ema_fast=5,            # Faster EMAs for momentum
        ema_medium=12,
        ema_slow=26,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=65,        # Wider RSI band
        rsi_sell_min=35,       # Wider RSI band
        adx_threshold=20,      # Lower ADX - catch more moves
        stop_atr_mult=1.5,
        reward_risk=2.0,       # Standard R:R - take profits
        min_confidence=0.4,
        use_macd_exit=True,    # MACD exit helps on indices
        require_htf=True,      # Only trade with the trend
    ),
}


# =============================================================================
# MARKET CONFIGURATIONS
# =============================================================================
# IG EPICs for spread betting. CFD EPICs may differ.
# Min stop distances from IG API (verified 2024)

MARKETS = [
    # --- INDICES (Momentum Strategy) ---
    MarketConfig(
        epic="IX.D.SPTRD.DAILY.IP",
        name="S&P 500",
        sector="Indices",
        min_stop_distance=1.0,
        default_size=1.0,
        min_confidence=0.4,
        strategy="indices",    # Use Momentum strategy
    ),
    MarketConfig(
        epic="IX.D.NASDAQ.CASH.IP",
        name="NASDAQ 100",
        sector="Indices",
        min_stop_distance=4.0,
        default_size=0.2,
        min_confidence=0.4,
        strategy="indices",    # Use Momentum strategy
    ),

    # --- COMMODITIES (Big Winners Strategy) ---
    MarketConfig(
        epic="EN.D.CL.Month1.IP",
        name="Crude Oil",
        sector="Commodities",
        min_stop_distance=12.0,
        default_size=0.1,
        expiry="MAR-26",
        candle_interval=15,
        min_confidence=0.4,    # Lowered from 0.7 - Big Winners uses 0.4
        strategy="default",
    ),
    MarketConfig(
        epic="CS.D.USCGC.TODAY.IP",
        name="Gold",
        sector="Commodities",
        min_stop_distance=1.0,
        default_size=0.1,
        min_confidence=0.4,
        strategy="default",    # Gold is the star performer!
    ),

    # --- FOREX (Big Winners Strategy) ---
    MarketConfig(
        epic="CO.D.DX.Month1.IP",
        name="Dollar Index (DXY)",
        sector="Forex",
        min_stop_distance=20.0,
        default_size=1.0,
        expiry="MAR-26",
        candle_interval=15,
        min_confidence=0.4,
        strategy="default",
    ),
    MarketConfig(
        epic="CS.D.EURUSD.TODAY.IP",
        name="EUR/USD",
        sector="Forex",
        min_stop_distance=2.0,
        default_size=0.5,
        candle_interval=15,
        min_confidence=0.4,
        strategy="default",
    ),
]


def get_strategy_for_market(market: MarketConfig) -> StrategyConfig:
    """Get the strategy configuration for a market."""
    return STRATEGY_PROFILES.get(market.strategy, STRATEGY_PROFILES["default"])


# Legacy STRATEGY_PARAMS for backward compatibility
# New code should use get_strategy_for_market() instead
STRATEGY_PARAMS = {
    "ema_fast": 9,
    "ema_medium": 21,
    "ema_slow": 50,
    "rsi_period": 7,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_buy_max": 60,
    "rsi_sell_min": 40,
    "adx_threshold": 25,
}
