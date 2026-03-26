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
    pullback_pct: float = 0.3  # Max % distance from fast EMA to enter (0.3 = 0.3%)
    breakeven_trigger_pct: float = 0.7  # Move stop to break-even when profit >= X% of stop distance (0.7 = 70%)
    atr_trail_mult: float = 1.5  # ATR multiplier for trailing stop distance (after break-even)


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
    trading_start: int = 4   # UTC hour to start trading (inclusive)
    trading_end: int = 20    # UTC hour to stop trading (exclusive)


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
    # =================================================================
    # DEFAULT — baseline for markets without a specific profile
    # Used by: Soybeans, NY Cocoa, NY Cotton, Dollar Index, T-Notes
    # =================================================================
    "default": StrategyConfig(
        ema_fast=9,
        ema_medium=21,
        ema_slow=50,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=1.8,
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=False,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.7,  # Raised from 0.5 — 50% triggered on noise
        atr_trail_mult=1.5,
    ),

    # =================================================================
    # GOLD — the star. Very fast EMAs catch trends early, wide stops
    # let them run, RSI 85/15 stops cutting winners short.
    # Backtest: £+286 (60d) vs £+45 with default. PF=1.88.
    # =================================================================
    "gold": StrategyConfig(
        ema_fast=3,            # Very fast — catch Gold trends early
        ema_medium=8,
        ema_slow=21,
        rsi_period=7,
        rsi_overbought=85,    # Wide — Gold trends push RSI high, don't exit early
        rsi_oversold=15,
        rsi_buy_max=60,       # Wider entry range for fast EMAs
        rsi_sell_min=40,
        adx_threshold=30,
        stop_atr_mult=2.5,    # Wide stops — Gold's ATR is large, needs room
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=False,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.7,  # Let trades breathe before locking BE
        atr_trail_mult=1.5,
    ),

    # =================================================================
    # FOREX — tighter stops work best on currency pairs.
    # EUR/USD: £+19 (60d) with 1.0x stops vs £+1 with 1.8x. PF=1.31.
    # USD/JPY: £+91 (60d) with 1.0x stops + RSI 80/20 + BE 90%. PF=1.79.
    # =================================================================
    "forex": StrategyConfig(
        ema_fast=9,
        ema_medium=21,
        ema_slow=50,
        rsi_period=7,
        rsi_overbought=70,    # Forex trends are shorter — keep tight RSI exit
        rsi_oversold=30,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=1.0,    # Tight stops — smaller losses, more frequent
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=False,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.7,  # Let trades develop before BE
        atr_trail_mult=1.5,
    ),

    # =================================================================
    # USD/JPY specific — best performer. Tight stops + wide RSI + high BE.
    # Backtest: £+91 (60d), PF=1.79, 45% WR.
    # =================================================================
    "usdjpy": StrategyConfig(
        ema_fast=9,
        ema_medium=21,
        ema_slow=50,
        rsi_period=7,
        rsi_overbought=80,    # Wider than other forex — JPY trends run further
        rsi_oversold=20,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=1.0,    # Tight stops
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=False,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.9,  # Almost at target before locking BE
        atr_trail_mult=1.5,
    ),

    # =================================================================
    # INDICES — MACD exit ON, per-index stop tuning.
    # NASDAQ: £+41 (60d) with 2.5x stops. PF=2.33.
    # FTSE: £+30 (60d) with 1.0x stops. PF=2.09.
    # =================================================================
    "indices": StrategyConfig(
        ema_fast=5,
        ema_medium=12,
        ema_slow=26,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=1.5,    # Default for indices — overridden per market below
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=True,
        require_htf=True,
        pullback_pct=0.2,
        breakeven_trigger_pct=0.7,  # Raised from 0.5 — indices tight stops triggered BE on noise
        atr_trail_mult=1.5,
    ),

    # NASDAQ needs wider stops to avoid premature stop-outs
    "indices_wide": StrategyConfig(
        ema_fast=5,
        ema_medium=12,
        ema_slow=26,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=2.5,    # Wide stops — NASDAQ PF=2.33 at 2.5x vs 2.04 at 1.5x
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=True,
        require_htf=True,
        pullback_pct=0.2,
        breakeven_trigger_pct=0.7,  # Raised from 0.5 — indices tight stops triggered BE on noise
        atr_trail_mult=1.5,
    ),

    # FTSE needs tight stops — PF=2.09 at 1.0x vs 1.24 at 1.5x
    "indices_tight": StrategyConfig(
        ema_fast=5,
        ema_medium=12,
        ema_slow=26,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=1.0,    # Tight stops — FTSE trends are cleaner
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=True,
        require_htf=True,
        pullback_pct=0.2,
        breakeven_trigger_pct=0.7,  # Raised from 0.5 — indices tight stops triggered BE on noise
        atr_trail_mult=1.5,
    ),

    # =================================================================
    # CRUDE OIL — wide RSI to stop killing every trade at break-even.
    # All 70 trades exited at BE with RSI 70/30. RSI 80/20 helps marginally.
    # =================================================================
    "crude": StrategyConfig(
        ema_fast=9,
        ema_medium=21,
        ema_slow=50,
        rsi_period=7,
        rsi_overbought=80,    # Wider — stop killing trades at BE
        rsi_oversold=20,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=1.0,    # Tight stops
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=False,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.7,
        atr_trail_mult=1.5,
    ),

    # Silver — disabled but kept for re-enabling
    "silver": StrategyConfig(
        ema_fast=9,
        ema_medium=21,
        ema_slow=50,
        rsi_period=7,
        rsi_overbought=80,
        rsi_oversold=20,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=30,
        stop_atr_mult=1.2,
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=False,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.7,
        atr_trail_mult=1.5,
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
        min_stop_distance=2.0,  # Raised from 1.0 — cap was 20pts, ATR*1.5 can exceed that
        default_size=1.0,
        min_confidence=0.55,   # Raised from 0.4 for quality entries
        strategy="indices",
    ),
    MarketConfig(
        epic="IX.D.NASDAQ.CASH.IP",
        name="NASDAQ 100",
        sector="Indices",
        min_stop_distance=4.0,
        default_size=0.2,
        min_confidence=0.55,
        strategy="indices_wide",  # NASDAQ: wider 2.5x stops, PF=2.33 (60d backtest)
    ),
    MarketConfig(
        epic="IX.D.RUSSELL.DAILY.IP",
        name="US Russell 2000",
        sector="Indices",
        min_stop_distance=1.0,
        default_size=1.0,
        min_confidence=0.55,
        strategy="indices",    # Use Momentum strategy (same as S&P/NASDAQ)
    ),

    MarketConfig(
        epic="IX.D.DAX.DAILY.IP",
        name="Germany 40",
        sector="Indices",
        min_stop_distance=2.0,
        default_size=0.5,
        min_confidence=0.55,
        strategy="indices",
    ),
    MarketConfig(
        epic="IX.D.DOW.DAILY.IP",
        name="Wall Street",
        sector="Indices",
        min_stop_distance=4.0,
        default_size=0.1,
        min_confidence=0.55,
        strategy="indices",
    ),
    MarketConfig(
        epic="IX.D.FTSE.DAILY.IP",
        name="FTSE 100",
        sector="Indices",
        min_stop_distance=1.0,
        default_size=1.0,
        min_confidence=0.55,
        strategy="indices_tight",  # FTSE: tight 1.0x stops, PF=2.09 (60d backtest)
    ),
    MarketConfig(
        epic="IX.D.AIIDX.DAILY.IP",
        name="AI Index",
        sector="Indices",
        min_stop_distance=1.0,
        default_size=1.0,
        min_confidence=0.55,
        strategy="indices",
    ),
    # --- COMMODITIES (Big Winners Strategy) ---
    MarketConfig(
        epic="EN.D.CL.Month1.IP",
        name="Crude Oil",
        sector="Commodities",
        min_stop_distance=12.0,
        default_size=0.1,
        expiry="JUN-26",
        candle_interval=15,
        min_confidence=0.55,
        strategy="crude",      # Custom: RSI 80/20, tight stops. Stops BE-exit problem.
        trading_start=23,      # Nearly 24h market — avoid IG reset window (21-23 UTC)
        trading_end=21,
    ),
    MarketConfig(
        epic="CS.D.USCGC.TODAY.IP",
        name="Gold",
        sector="Commodities",
        min_stop_distance=2.0,  # Raised from 1.0 — cap was 20pts (20x), ATR*2.5 = 25-35, every trade capped
        default_size=1.0,      # IG minimum is 1.0 per point (was 0.1 - all trades rejected!)
        min_confidence=0.55,   # Raised from 0.4 for quality entries
        strategy="gold",       # Custom: fast EMAs 3/8/21, RSI 85/15, 2.5x stops. £+286 (60d)
        trading_start=23,
        trading_end=21,
    ),

    # Spot Silver — disabled during choppy Iran-conflict conditions (3/3 losses)
    # MarketConfig(
    #     epic="CS.D.USCSI.TODAY.IP",
    #     name="Spot Silver",
    #     sector="Commodities",
    #     min_stop_distance=4.0,
    #     default_size=0.5,
    #     min_confidence=0.55,
    #     strategy="silver",
    #     trading_start=23,
    #     trading_end=21,
    # ),
    MarketConfig(
        epic="CS.D.COPPER.TODAY.IP",
        name="Copper",
        sector="Commodities",
        min_stop_distance=12.0,    # Spread is 22.0 — need comfortable margin
        default_size=1.0,          # IG minimum is 1.0 (0.1/0.2 rejected with MINIMUM_ORDER_SIZE_ERROR)
        min_confidence=0.55,
        strategy="default",
        trading_start=23,
        trading_end=21,
    ),

    # --- SOFT COMMODITIES ---
    # Soybeans — disabled. No backtest data (no Yahoo ticker), ATR/spread 1.7x (marginal),
    # hourly candles too slow, generic strategy. Lost £12 on first trade.
    # MarketConfig(
    #     epic="CO.D.S.Month1.IP",
    #     name="Soybeans",
    #     sector="Commodities",
    #     min_stop_distance=4.0,
    #     default_size=0.04,
    #     expiry="MAY-26",
    #     candle_interval=60,
    #     min_confidence=0.55,
    #     strategy="default",
    # ),
    MarketConfig(
        epic="CO.D.CC.Month2.IP",
        name="NY Cocoa",
        sector="Commodities",
        min_stop_distance=10.0,
        default_size=0.04,
        expiry="MAY-26",
        candle_interval=15,
        min_confidence=0.55,
        strategy="default",
    ),
    MarketConfig(
        epic="CO.D.CT.Month1.IP",
        name="NY Cotton",
        sector="Commodities",
        min_stop_distance=40.0,
        default_size=0.04,
        expiry="DEC-26",
        candle_interval=15,
        min_confidence=0.55,
        strategy="default",
    ),

    # --- FOREX (Big Winners Strategy) ---
    MarketConfig(
        epic="CO.D.DX.Month1.IP",
        name="Dollar Index (DXY)",
        sector="Forex",
        min_stop_distance=20.0,
        default_size=1.0,
        expiry="SEP-26",
        candle_interval=15,
        min_confidence=0.55,   # Raised from 0.4 for quality entries
        strategy="default",
        trading_start=23,      # Forex trades 24/5 — avoid IG reset window
        trading_end=21,
    ),
    MarketConfig(
        epic="CS.D.EURUSD.TODAY.IP",
        name="EUR/USD",
        sector="Forex",
        min_stop_distance=2.0,
        default_size=0.5,
        candle_interval=15,
        min_confidence=0.55,
        strategy="forex",      # Tight 1.0x stops, PF=1.31 (60d)
        trading_start=23,
        trading_end=21,
    ),
    MarketConfig(
        epic="CS.D.GBPUSD.TODAY.IP",
        name="GBP/USD",
        sector="Forex",
        min_stop_distance=2.0,
        default_size=0.5,
        candle_interval=15,
        min_confidence=0.55,
        strategy="forex",      # Tight 1.0x stops
        trading_start=23,
        trading_end=21,
    ),
    MarketConfig(
        epic="CS.D.USDJPY.TODAY.IP",
        name="USD/JPY",
        sector="Forex",
        min_stop_distance=2.0,     # Spread is 1.0 — tight spread, good for trading
        default_size=0.5,
        candle_interval=15,
        min_confidence=0.55,
        strategy="usdjpy",     # Custom: RSI 80/20, tight stops, BE 90%. £+91 (60d), PF=1.79
        trading_start=23,
        trading_end=21,
    ),

    # --- RATES / BONDS (Big Winners Strategy) ---
    MarketConfig(
        epic="IR.D.02YEAR100.Month2.IP",
        name="US 2-Year T-Note",
        sector="Rates",
        min_stop_distance=6.0,     # Spread is 2.0 — need 3x spread minimum
        default_size=1.0,
        expiry="JUN-26",
        candle_interval=15,
        min_confidence=0.55,
        strategy="default",
    ),
    MarketConfig(
        epic="IR.D.10YEAR100.Month2.IP",
        name="US 10-Year T-Note",
        sector="Rates",
        min_stop_distance=10.0,    # Spread is 4.0 — ATTACHED_ORDER_LEVEL_ERROR at 4.0
        default_size=1.0,
        expiry="JUN-26",
        candle_interval=15,
        min_confidence=0.55,
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
