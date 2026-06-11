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
    # Absolute £ ceiling on the loss a single trade may carry at its stop. When an
    # instrument's minimum dealing size (default_size) forces the £ risk above this
    # amount, the trade is skipped rather than silently over-risked. Replaced the
    # old `max_risk_multiple × per-trade budget` (2026-06-08): a balance-relative
    # multiple DRIFTS — as the account draws down or is topped up, the same trade
    # flips between blocked and admitted, and the multiple had to be re-tuned each
    # time (1.3→1.8 chased exactly this). An absolute £ is stable: a trade that
    # risks £50 is too risky at any balance. Set it >= the per-trade budget (only
    # min-size-forced trades can exceed the budget, so only they trip it). £45
    # admits the un-truncated index stops (FTSE £24 / AI £36 / NASDAQ £38 / Wall
    # St £15) while blocking genuinely large min-size risk (Germany-type £90+,
    # corrupt stops). Market SELECTION (no-edge markets) is the screener/disable's
    # job, not this ceiling's — it is a safety cap, not an edge filter.
    max_risk_gbp: float = 45.0


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
    breakeven_lock_pct: float = 0.0  # When BE triggers, lock in this fraction of stop distance as profit
                                     # (0.0 = stop at exact entry; 0.25 = lock ~25% of stop as guaranteed profit).
                                     # Must be < breakeven_trigger_pct so the locked stop stays behind price.
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
    candle_interval: int = 5  # Candle duration in minutes (5 for indices/commodities, 15 for forex, 60 for natgas)
    htf_resolution: str = "HOUR"  # IG resolution for HTF trend (HOUR, HOUR_4, DAY etc). Use DAY for 1h-candle markets.
    min_confidence: float = 0.5  # Minimum confidence to enter (higher = more selective)
    strategy: str = "default"  # Strategy profile to use: "default" or "indices"
    trading_start: int = 4   # UTC hour to start trading (inclusive)
    trading_end: int = 20    # UTC hour to stop trading (exclusive)
    # Leg-size (exhaustion) filter — blocks entries chasing a move that already
    # ran > leg_filter_threshold × ATR in the trade direction over the last
    # leg_filter_lookback candles. Ported from the Oanda_Gold EMA-Trend bot.
    # leg_filter_lookback=0 disables it. leg_filter_enforce=False makes it
    # OBSERVATIONAL (log + journal only, trade still proceeds) so we can measure
    # the realised P&L of would-be-blocked entries before enforcing.
    # NB threshold is in legATR units calibrated to the candle interval — on 5m
    # a 12-candle (1h) leg sits ~3.5-6.5× ATR, so ~5.0 blocks the most extended.
    leg_filter_lookback: int = 0
    leg_filter_threshold: float = 5.0
    leg_filter_enforce: bool = False
    # ADX-ceiling (exhaustion) filter — skips entries whose ADX exceeds
    # adx_ceiling, on the thesis that an extreme ADX marks a climax about to
    # mean-revert rather than a trend to ride. adx_ceiling=0 disables it.
    # adx_ceiling_enforce=False makes it OBSERVATIONAL (log + journal only,
    # trade still proceeds) — same gather-data-before-enforcing pattern as the
    # leg filter above. Backtested via scripts/backtest_adx_ceiling.py (Yahoo).
    adx_ceiling: float = 0.0
    adx_ceiling_enforce: bool = False
    # Side the ceiling applies to. "" (default) = both directions; "SELL" caps
    # only short entries, "BUY" only longs. Exhaustion is empirically ONE-SIDED
    # per market (the direction-split backtest 2026-06-09): index/most-commodity
    # climaxes are capitulation LOWS (cap SELLs), but some — e.g. Cocoa — blow
    # off at the TOP (cap BUYs). ADX itself is non-directional, so without this a
    # symmetric cap would also block the healthy-continuation side (it removed 5
    # winning Copper longs in backtest). Set per market to the climax side.
    adx_ceiling_direction: str = ""
    # Correlation-cluster filter — markets sharing a non-empty correlation_group
    # are treated as the same underlying bet. When a 2nd market in the group
    # tries to open the SAME direction within CLUSTER_FILTER_WINDOW_MIN of the
    # first (main.py), it is a doubled position that historically whipsaws:
    # journal mining (2026-06-11) showed same-window same-direction equity-index
    # clusters at -£8.23 avg / PF 0.13 vs solo index entries +£2.62 / PF 1.81.
    # OBSERVATIONAL by default (CLUSTER_FILTER_ENFORCE=False in main.py) — logs +
    # journals the would-block, trade proceeds; query rejected_signals LIKE
    # 'Cluster-filter%'. "" disables. Only the 6 backtested equity indices are
    # grouped; AI Index is a candidate to add once data confirms.
    correlation_group: str = ""
    # Hard direction restriction. "" (default) = trade both sides; "BUY" = long
    # only (skip SELL signals), "SELL" = short only. For markets whose edge is
    # empirically one-sided — unlike adx_ceiling_direction (which only scopes the
    # exhaustion ceiling), this blocks the disallowed side outright. Set on US
    # 10-Year T-Note ("BUY") 2026-06-11: cull review showed long PF 1.26 vs short
    # PF 0.63 (700d Yahoo), so the short side is dead weight.
    allowed_direction: str = ""


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
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
        max_positions=int(os.getenv("MAX_POSITIONS", "5")),
        trading_enabled=os.getenv("TRADING_ENABLED", "true").lower() == "true",
        check_interval=int(os.getenv("CHECK_INTERVAL", "60")),  # 60 mins to conserve API allowance
        price_data_points=int(os.getenv("PRICE_DATA_POINTS", "50")),  # 50 points (saves 50% vs 100)
        cache_ttl_minutes=int(os.getenv("CACHE_TTL_MINUTES", "55")),  # Cache for 55 mins
        max_risk_gbp=float(os.getenv("MAX_RISK_GBP", "45")),
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
    # NATURAL GAS — 1h candles with daily HTF. Default profile + wider
    # stops + R:R 3 was the clear winner across 730d / 73 trades.
    # Backtest: +8.80% (PF 1.30, WR 54.8%, n=73 over 2 years).
    # Volatile market — wider stops give trades room to develop.
    # =================================================================
    "natgas": StrategyConfig(
        ema_fast=9,
        ema_medium=21,
        ema_slow=50,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=60,
        rsi_sell_min=40,
        adx_threshold=25,
        stop_atr_mult=2.5,    # Wider than default 1.8 — natgas needs room (PF 1.30 vs 1.26)
        reward_risk=3.0,      # Higher R:R — let winners run on big swings
        min_confidence=0.55,
        use_macd_exit=True,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.7,
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
        adx_threshold=35,     # Was 30 — filters 17 low-quality ranging entries
        stop_atr_mult=1.5,    # Was 2.5 — tighter stops, smaller losses per trade
        reward_risk=3.0,      # Was 2.0 — bigger winners compensate for tighter stops
        min_confidence=0.55,
        use_macd_exit=False,
        require_htf=True,
        pullback_pct=0.3,
        breakeven_trigger_pct=0.5,  # Lowered 0.7→0.5 (2026-05-31): Yahoo backtest
                                    # shows +0.33%/55d (5m) and +17.19%/365d PF 4.99
                                    # (1h) vs live 0.7 at -1.01% / +15.96% PF 3.78
        breakeven_lock_pct=0.25,    # 2026-06-10: lock ~25% of stop (~2-3pts) as profit
                                    # instead of dead-entry. Gold ATR ~10 snapped BE-at-entry
                                    # back to £0 on noise (e.g. 01:10 SELL +£12→£0). 0.25 < 0.5
                                    # trigger so the locked stop stays behind price at arm time.
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
        adx_threshold=35,     # Raised from 30 — filters 6 noise trades, WR 18%→57%, PF 0.67→3.33
        stop_atr_mult=1.5,    # Widened from 1.0 — gives trades room to develop
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

    # Germany 40 / US Russell 2000 — ADX 30 lets through too much chop;
    # ADX 40 over-filters. 60d backtest: DAX -0.06% → +0.30% (PF 0.96 → 1.87),
    # Russell -0.22% → -0.19% with halved trade count.
    "indices_adx35": StrategyConfig(
        ema_fast=5,
        ema_medium=12,
        ema_slow=26,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=35,
        stop_atr_mult=1.5,
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=True,
        require_htf=True,
        pullback_pct=0.2,
        breakeven_trigger_pct=0.7,
        atr_trail_mult=1.5,
    ),

    # S&P 500 needs high ADX selectivity — too choppy at ADX 30
    "indices_selective": StrategyConfig(
        ema_fast=5,
        ema_medium=12,
        ema_slow=26,
        rsi_period=7,
        rsi_overbought=70,
        rsi_oversold=30,
        rsi_buy_max=55,
        rsi_sell_min=45,
        adx_threshold=40,     # Only trade strong trends — ADX 30 was -£11, ADX 40 is +£10
        stop_atr_mult=1.5,
        reward_risk=2.0,
        min_confidence=0.55,
        use_macd_exit=True,
        require_htf=True,
        pullback_pct=0.2,
        breakeven_trigger_pct=0.7,
        atr_trail_mult=1.5,
    ),

    # NASDAQ needs tighter stops to cut losers fast
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
        stop_atr_mult=2.0,    # Widened 1.0→2.0 (2026-06-04 stop-width sweep): tight 1.0x whipsawed sound entries out before MACD could manage them. NASDAQ WR 64→77%, PF 1.58→2.11 (59d, regime stop-override neutralized). Paper-trial, review after a few trades.
        reward_risk=2.0,      # Raised 1.5→2.0 to match the wider stop (2026-06-04 sweep)
        min_confidence=0.55,
        use_macd_exit=True,
        require_htf=True,
        pullback_pct=0.2,
        breakeven_trigger_pct=0.7,  # Raised from 0.5 — indices tight stops triggered BE on noise
        atr_trail_mult=1.5,
    ),

    # FTSE: stop 2.0x / R:R 2.0 since 2026-06-04 (profile name now a misnomer —
    # no longer "tight"). 1.0x was full-stopping sound entries on 5m noise.
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
        stop_atr_mult=2.0,    # Widened 1.0→2.0 (2026-06-04 stop-width sweep): 1.0x full-stopped good entries on 5m noise before MACD could manage them. FTSE WR 52→61%, PF 1.70→2.77, P&L ~2x (59d, 28-31t, monotonic). Supersedes the old 30d "tight" result — that sweep's stop param was masked by the regime override.
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
        use_macd_exit=True,   # Switched from ADX exit — ADX exit was killing every trade at BE
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
        strategy="indices_selective",  # ADX 40 — S&P too choppy at 30, only trade strong trends
        correlation_group="equity_index",  # cluster filter (2026-06-11)
    ),
    MarketConfig(
        epic="IX.D.NASDAQ.CASH.IP",
        name="NASDAQ 100",
        sector="Indices",
        min_stop_distance=4.0,
        default_size=0.2,
        min_confidence=0.55,
        strategy="indices_wide",  # NASDAQ: stop 2.0x / R:R 2.0 (2026-06-04 sweep, paper-trial)
        # Leg-size filter OBSERVATIONAL (log-only, trade proceeds). 59d Yahoo
        # sweep (scripts/backtest_leg_filter.py, 2026-06-04): lb12/blkTop35%
        # (legATR>5.0) lifted PF 1.79→2.28, P&L +0.71%→+0.88%. NASDAQ was the
        # one market with a consistent edge across configs. Gathering live data
        # before enforcing — set leg_filter_enforce=True once confirmed.
        leg_filter_lookback=12,
        leg_filter_threshold=5.0,
        leg_filter_enforce=False,
        # ADX-ceiling OBSERVATIONAL (log-only, trade proceeds). 59d Yahoo sweep
        # (scripts/backtest_adx_ceiling.py, 2026-06-09): capping at ADX 55
        # removed only 2/15 trades (the ADX 55-60 tail) yet lifted PF 2.59→4.65,
        # P&L +2.03→+2.44%. Triggered by a live S&P SELL @ ADX 57.2 losing on a
        # bounce. Gathering live data before enforcing — set
        # adx_ceiling_enforce=True once would-blocks confirm as net losers.
        adx_ceiling=55.0,
        adx_ceiling_enforce=False,
        adx_ceiling_direction="SELL",  # short-side: longs never reach 55 (dir-split 2026-06-09)
        correlation_group="equity_index",  # cluster filter (2026-06-11)
    ),
    # Disabled 2026-05-01 — strategy doesn't fit. Tested 5m/15m/30m/1h timeframes,
    # ADX 30/35/40, slower EMAs, long-only, wide stops — no variant produced a
    # clean edge. Best 1h/ADX 40 result was 4 trades over 730d (statistically
    # meaningless). Live: 13 trades, 38% WR, -£11.50 over 57d. Revisit if we
    # ever build a non-trend strategy (mean reversion / breakout).
    # MarketConfig(
    #     epic="IX.D.RUSSELL.DAILY.IP",
    #     name="US Russell 2000",
    #     sector="Indices",
    #     min_stop_distance=1.0,
    #     default_size=1.0,
    #     min_confidence=0.55,
    #     strategy="indices_adx35",
    # ),

    # DISABLED 2026-06-08: no real edge. The config's documented "ADX30 PF 3.47"
    # did NOT reproduce — that figure came from the backtest's regime stop-override
    # (which live never applies). Re-run forcing the live profile stop (1.5x), the
    # ^GDAXI 365d 1h result is PF 1.05 / +0.37% — break-even — degrading to PF 0.60
    # over the recent 120d, and -£21.50 live across 8 trades. Per the per-EPIC
    # profitability principle (prefer fewer profitable markets), benched. Profile
    # kept; uncomment to revive if a future regime re-establishes an edge.
    # MarketConfig(
    #     epic="IX.D.DAX.DAILY.IP",
    #     name="Germany 40",
    #     sector="Indices",
    #     min_stop_distance=2.0,
    #     default_size=0.5,
    #     candle_interval=60,        # 1h candles (was 5m default)
    #     htf_resolution="DAY",      # Daily HTF since 1h is the entry timeframe
    #     min_confidence=0.55,
    #     strategy="indices",
    #     trading_start=8,       # Xetra cash open 08:00 UTC
    #     trading_end=17,        # Include 16:30 close auction (peak liquidity)
    # ),
    MarketConfig(
        epic="IX.D.DOW.DAILY.IP",
        name="Wall Street",
        sector="Indices",
        min_stop_distance=4.0,
        default_size=0.1,
        min_confidence=0.55,
        strategy="indices",
        correlation_group="equity_index",  # cluster filter (2026-06-11)
    ),
    MarketConfig(
        epic="IX.D.FTSE.DAILY.IP",
        name="FTSE 100",
        sector="Indices",
        min_stop_distance=1.0,
        default_size=1.0,
        min_confidence=0.55,
        strategy="indices_tight",  # FTSE: stop 2.0x / R:R 2.0 (2026-06-04 sweep)
        correlation_group="equity_index",  # cluster filter (2026-06-11)
        trading_start=8,       # LSE cash open 08:00 UTC
        trading_end=17,        # Include 16:30 close auction (peak liquidity)
    ),
    MarketConfig(
        epic="IX.D.AIIDX.DAILY.IP",
        name="AI Index",
        sector="Indices",
        min_stop_distance=1.0,
        default_size=1.0,
        min_confidence=0.55,
        strategy="indices",
        trading_start=4,
        trading_end=22,        # Extended from 20 (2026-05-07): 5m backtest showed
                               # +£35/mo theoretical edge but live had only 1 trade
                               # in 3 weeks. Hypothesis: end-of-US-session moves
                               # were being cut off. Observe over next week.
    ),

    # --- ASIAN INDICES (added 2026-05-07) ---
    # Both filling the previously-dead 22:00-09:00 UTC window where the bot's
    # only active markets were forex with usually-low overnight ADX.
    # Backtest 30d (5m, indices strategy):
    #   Japan 225:      3 trades, 66.7% WR, PF 6.24, +£93.94
    #   Hong Kong HS50: 6 trades, 50.0% WR, PF 2.54, +£30.37
    # Sample sizes are small (cash sessions only generate ~5–10 candle days),
    # so default_size kept conservative until live data accumulates.
    MarketConfig(
        epic="IX.D.NIKKEI.DAILY.IP",
        name="Japan 225",
        sector="Indices",
        min_stop_distance=20.0,
        default_size=0.5,
        min_confidence=0.55,
        strategy="indices",
        correlation_group="equity_index",  # cluster filter (2026-06-11)
        trading_start=0,       # Tokyo cash session opens 00:00 UTC (09:00 JST)
        trading_end=8,         # closes ~06:00 UTC; pad to 8 for late-print candles
    ),
    MarketConfig(
        epic="IX.D.HANGSENG.DAILY.IP",
        name="Hong Kong HS50",
        sector="Indices",
        min_stop_distance=20.0,
        default_size=0.5,
        min_confidence=0.55,
        strategy="indices",
        correlation_group="equity_index",  # cluster filter (2026-06-11)
        trading_start=1,       # HK cash opens 01:30 UTC (09:30 HKT)
        trading_end=9,         # closes 08:00 UTC; pad to 9
    ),

    # iShares US Home Construction ETF — disabled. EPIC exists on demo
    # (SI.D.ITBUS.DAILY.IP) but Lightstreamer rejects the L1 subscription
    # with "[-1] Incorrect instrument setup", killing the entire streaming
    # session for all markets (same failure mode as CC.D.* CFD-only EPICs).
    # Re-enable only if IG confirms streaming support for this ETF.
    # MarketConfig(
    #     epic="SI.D.ITBUS.DAILY.IP",
    #     name="US Home Construction",
    #     sector="Indices",
    #     min_stop_distance=2.0,
    #     default_size=0.24,
    #     min_confidence=0.55,
    #     strategy="indices",
    # ),
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
        # ADX-ceiling OBSERVATIONAL (log-only, trade proceeds). All-EPIC sweep
        # (scripts/backtest_adx_ceiling_all.py, 2026-06-09, CL=F 15m/59d 49t):
        # capping ADX>50 → PF 1.82→2.31, P&L +8.10→+9.42% (−9t). NB Yahoo
        # continuous futures ≠ IG contract exactly, so gather live would-blocks
        # before enforcing. Set adx_ceiling_enforce=True once confirmed.
        adx_ceiling=50.0,
        adx_ceiling_enforce=False,
        adx_ceiling_direction="SELL",  # short-side: SELL helps +1.62, BUY hurts (dir-split 2026-06-09)
    ),
    # Disabled 2026-05-21 — edge has decayed. Added 2026-05-01, never traded live
    # (rarely fires + screened out). The 720d backtest's +5.06% was all earned
    # mid-2024→mid-2025; the trailing 365d is -5.29% (every ADX negative) and the
    # last ~6 months ≈ -7.4%. No config rescues it: tighter stops -3.84%,
    # ADX20/30 -5.29%/-10.51%, long-only -4.86%, short-only -0.43% (best, still
    # not profitable). Looks like a NatGas regime shift trend-following can't fit.
    # natgas profile kept defined for a future revival. See scripts/backtest_natgas_review.py.
    # MarketConfig(
    #     epic="EN.D.NG.Month1.IP",
    #     name="Natural Gas",
    #     sector="Commodities",
    #     min_stop_distance=60.0,    # IG min normal stop = 60pts
    #     default_size=0.5,          # 0.5/pt × 60pt stop = 30 USD risk per trade (~£24)
    #     expiry="JUL-26",           # Front month — last dealing 2026-06-25, roll required
    #     candle_interval=60,        # 1h candles — sweet spot for NG (5m too noisy)
    #     htf_resolution="DAY",      # Daily HTF since 1h is the entry timeframe
    #     min_confidence=0.55,
    #     strategy="natgas",
    #     trading_start=23,          # Same as Crude — nearly 24h, avoid IG reset window
    #     trading_end=21,
    # ),
    MarketConfig(
        epic="CS.D.USCGC.TODAY.IP",
        name="Gold",
        sector="Commodities",
        min_stop_distance=2.0,  # Raised from 1.0 — cap was 20pts (20x), ATR*2.5 = 25-35, every trade capped
        default_size=1.0,      # IG minimum is 1.0 per point (was 0.1 - all trades rejected!)
        min_confidence=0.60,   # Lowered 0.70→0.60 (2026-05-31): 0.70 produced zero trades
                               # in 2.5 weeks; need samples to evaluate the current profile
        strategy="gold",       # Custom: fast EMAs 3/8/21, RSI 85/15, 1.5x stops, R:R 3.0
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
        # 1h candles after 5m showed modest edge only (60d PF 1.28, +1.08%).
        # 365d 1h backtest: PF 2.16, +7.10%, 41 trades, WR 51%. Same pattern
        # as Germany 40 — interval is the dominant factor on 1h.
        epic="CS.D.COPPER.TODAY.IP",
        name="Copper",
        sector="Commodities",
        min_stop_distance=12.0,    # Spread is 22.0 — need comfortable margin
        default_size=1.0,          # IG minimum is 1.0 (0.1/0.2 rejected with MINIMUM_ORDER_SIZE_ERROR)
        candle_interval=60,        # 1h candles (was 5m default)
        htf_resolution="DAY",      # Daily HTF since 1h is the entry timeframe
        min_confidence=0.55,
        strategy="default",
        trading_start=13,      # CME Copper: US session only. Asia hours are illiquid
        trading_end=20,        # and the spread guard was rejecting overnight signals anyway
        # ADX-ceiling OBSERVATIONAL (log-only, trade proceeds). All-EPIC sweep
        # (scripts/backtest_adx_ceiling_all.py, 2026-06-09, HG=F 1h/700d 92t):
        # high-ADX tail is a net loser — capping ~55-60 → PF 1.22→1.41,
        # P&L +5.47→+9.33%. ΔP&L noisy across ceilings (exact value approximate).
        # Yahoo continuous futures ≠ IG contract; gather live before enforcing.
        adx_ceiling=55.0,
        adx_ceiling_enforce=False,
        adx_ceiling_direction="SELL",  # short-side STRONG: SELL +2.39, BUY HURTS -2.02 (dir-split 2026-06-09)
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
        # 1h candles: 365d backtest PF 7.03, +22.43%, 17 trades, 65% WR vs
        # 15m PF 0.20, -2.55%, 6 trades. Cocoa moves in slow ~hour-long
        # waves — 5m/15m chops it into noise.
        epic="CO.D.CC.Month2.IP",
        name="NY Cocoa",
        sector="Commodities",
        min_stop_distance=10.0,
        default_size=0.04,
        expiry="MAY-26",
        candle_interval=60,        # 1h candles (was 15m)
        htf_resolution="DAY",      # Daily HTF since 1h is the entry timeframe
        min_confidence=0.55,
        strategy="default",
        trading_start=13,      # ICE US softs session 12:45-17:30 UTC
        trading_end=18,
        # ADX-ceiling OBSERVATIONAL (log-only, trade proceeds). All-EPIC sweep
        # (scripts/backtest_adx_ceiling_all.py, 2026-06-09, CC=F 1h/700d 47t):
        # capping ADX>55 → PF 1.24→1.49, P&L +8.84→+14.46% (−6t). Yahoo
        # continuous futures ≠ IG contract; gather live before enforcing.
        adx_ceiling=55.0,
        adx_ceiling_enforce=False,
        adx_ceiling_direction="BUY",  # LONG-side (the inverse!): BUY +5.61, shorts never reach 55 (dir-split 2026-06-09)
    ),
    # Disabled 2026-06-11 (cull review) — INERT, not a loser: bare `default`
    # profile generates ~1 trade in 60d (15m Yahoo) and 0 live trades ever. No
    # edge to validate, just an idle market slot. Profile kept; re-enable only
    # with a dedicated profile that actually trades.
    # MarketConfig(
    #     epic="CO.D.CT.Month1.IP",
    #     name="NY Cotton",
    #     sector="Commodities",
    #     min_stop_distance=40.0,
    #     default_size=0.04,
    #     expiry="DEC-26",
    #     candle_interval=15,
    #     min_confidence=0.55,
    #     strategy="default",
    #     trading_start=13,      # ICE US softs session 12:45-17:30 UTC
    #     trading_end=18,
    # ),

    # --- FOREX (Big Winners Strategy) ---

    # Dollar Index (DXY) — disabled, negative P&L across all 36 param combos on 15m backtest
    # MarketConfig(
    #     epic="CO.D.DX.Month1.IP",
    #     name="Dollar Index (DXY)",
    #     sector="Forex",
    #     min_stop_distance=20.0,
    #     default_size=1.0,
    #     expiry="SEP-26",
    #     candle_interval=15,
    #     min_confidence=0.55,
    #     strategy="default",
    #     trading_start=23,
    #     trading_end=21,
    # ),

    MarketConfig(
        # 1h candles: 365d backtest PF 1.25, +0.46%, 48 trades, 48% WR vs
        # 5m PF 0.73, -0.28%, 50 trades, 36% WR. Prior 15m→5m tuning was
        # for an earlier regime; 1h has the most stable edge over 12 months.
        epic="CS.D.EURUSD.TODAY.IP",
        name="EUR/USD",
        sector="Forex",
        min_stop_distance=3.0,  # Raised from 2.0 — 5m ATR is only 2-3pts, IG rejects stops at minimum
        default_size=0.5,
        candle_interval=60,        # 1h candles (was 5m)
        htf_resolution="DAY",      # Daily HTF since 1h is the entry timeframe
        min_confidence=0.65,   # Raised from 0.55 — 365d backtest: PF 1.37→1.99, +0.64%→+1.13%, WR 49%→54% (37 vs 47 trades)
        strategy="forex",      # Tight 1.0x stops
        trading_start=23,
        trading_end=21,
    ),
    MarketConfig(
        # 1h candles: 365d backtest +1.52%, PF 1.94, 56% WR vs 5m +0.50% PF 2.01.
        # 5m looked OK on paper but bled live (-£31.72): every loss a 3-28min
        # stop-out on a ~4.5pip stop, where IG spread is 30-45% of the stop.
        # 1h's wider ATR stops dilute spread cost — same fix EUR/USD already got.
        epic="CS.D.GBPUSD.TODAY.IP",
        name="GBP/USD",
        sector="Forex",
        min_stop_distance=4.0,  # Raised from 3.0 — IG rejects at 3.0 when pre-London spread widens
        default_size=0.5,
        candle_interval=60,    # Switched from 5m — 5m bled live on spread vs tiny stops; 1h has the durable edge
        htf_resolution="DAY",  # Daily HTF since 1h is the entry timeframe
        min_confidence=0.55,
        strategy="forex",      # Tight 1.0x stops
        trading_start=7,       # London open — avoid illiquid pre-London spread widening
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
    # US 2-Year T-Note disabled 2026-06-11 (cull review) — NO EDGE: 44t/59d
    # Yahoo dead flat (PF 0.93, -0.02%), neither side works (long 0.72 / short
    # 1.07). 1 live trade ever (-£11.80). Profile kept.
    # MarketConfig(
    #     epic="IR.D.02YEAR100.Month2.IP",
    #     name="US 2-Year T-Note",
    #     sector="Rates",
    #     min_stop_distance=6.0,     # Spread is 2.0 — need 3x spread minimum
    #     default_size=1.0,
    #     expiry="JUN-26",
    #     candle_interval=15,
    #     min_confidence=0.55,
    #     strategy="default",
    # ),
    MarketConfig(
        # 1h candles: 365d backtest PF 1.44, +0.55%, 39 trades, 51% WR vs
        # 15m PF 0.60, -0.38%, 20 trades, 40% WR. Treasury futures move
        # on macro headlines — slow timeframe matches the data-release cadence.
        epic="IR.D.10YEAR100.Month2.IP",
        name="US 10-Year T-Note",
        sector="Rates",
        min_stop_distance=10.0,    # Spread is 4.0 — ATTACHED_ORDER_LEVEL_ERROR at 4.0
        default_size=1.0,
        expiry="JUN-26",
        candle_interval=60,        # 1h candles (was 15m)
        htf_resolution="DAY",      # Daily HTF since 1h is the entry timeframe
        min_confidence=0.55,
        strategy="default",
        allowed_direction="BUY",   # 2026-06-11 cull review: long-only. 700d Yahoo
                                   # long PF 1.26 vs short PF 0.63; both-sides was a
                                   # net loser only because shorts dragged it down.
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
