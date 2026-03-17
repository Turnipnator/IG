"""
Dynamic Market Screener for IG Spread Betting.

Discovers markets from the IG API, scores them on tradeability,
and dynamically activates/deactivates markets based on conditions.

Runs daily at startup and on schedule. Zero API cost for scoring
(uses streaming data). Discovery costs ~1 API call per search term.
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from config import MarketConfig, StrategyConfig, get_strategy_for_market

logger = logging.getLogger(__name__)

SCREENER_CACHE = Path("/app/data") if Path("/app").exists() else Path("data")
UNIVERSE_FILE = SCREENER_CACHE / "market_universe.json"
SCORES_FILE = SCREENER_CACHE / "market_scores.json"

# Search terms to discover spread bet markets on IG
# Each returns multiple results — we filter for tradeable spread bets
DISCOVERY_SEARCHES = [
    # Indices
    "FTSE", "S&P 500", "NASDAQ", "DAX", "Dow Jones", "Russell",
    "Nikkei", "Hang Seng", "ASX", "CAC 40", "IBEX",
    # Major Forex
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
    "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF",
    # Commodities
    "Gold", "Silver", "Crude Oil", "Copper", "Platinum",
    "Natural Gas", "Palladium",
    # Soft Commodities
    "Cocoa", "Coffee", "Cotton", "Sugar", "Soybeans", "Wheat", "Corn",
    # Rates
    "T-Note", "T-Bond", "Gilt", "Bund",
    # Crypto
    "Bitcoin", "Ethereum",
]

# EPICs to exclude (known CFD-only or problematic)
EPIC_BLACKLIST = {
    "CC.D.",   # CFD-only prefix — breaks streaming
    "IN.D.VIX",  # VIX unavailable on demo
}

# Minimum requirements for a market to be in the universe
MIN_SPREAD_RATIO = 0.01  # Spread must be < 1% of price


@dataclass
class MarketScore:
    """Scoring result for a market."""
    epic: str
    name: str
    sector: str
    score: float  # 0-100
    adx: float
    atr: float
    atr_spread_ratio: float  # ATR / spread — higher = better
    trend_clarity: float  # EMA separation as %
    volatility_regime: str  # "LOW", "NORMAL", "HIGH"
    htf_trend: str
    is_active: bool
    reason: str
    scored_at: str = ""


class MarketScreener:
    """
    Discovers and scores markets for dynamic trading.

    Two phases:
    1. Discovery (weekly): Search IG API for spread bet markets, cache to disk
    2. Scoring (daily): Score markets using streaming data, activate top N
    """

    def __init__(self, client=None, max_active: int = 15):
        self.client = client
        self.max_active = max_active
        self.universe: list[dict] = []  # All discovered markets
        self.scores: list[MarketScore] = []
        self.active_epics: set[str] = set()

    def discover_markets(self, force: bool = False) -> list[dict]:
        """
        Discover spread bet markets from IG API.
        Caches results to disk — only re-discovers weekly or when forced.
        """
        # Check cache
        if not force and UNIVERSE_FILE.exists():
            age = datetime.now().timestamp() - UNIVERSE_FILE.stat().st_mtime
            if age < 7 * 86400:  # 7 days
                self.universe = self._load_universe()
                logger.info(f"Loaded {len(self.universe)} markets from universe cache")
                return self.universe

        if not self.client:
            logger.warning("No client — cannot discover markets")
            return self._load_universe()

        logger.info("Discovering spread bet markets from IG API...")
        seen_epics = set()
        markets = []

        for term in DISCOVERY_SEARCHES:
            try:
                results = self.client.search_markets(term)
                for m in results:
                    epic = m.get("epic", "")
                    if not epic or epic in seen_epics:
                        continue

                    # Skip blacklisted prefixes
                    if any(epic.startswith(prefix) for prefix in EPIC_BLACKLIST):
                        continue

                    # Only include spread bet instruments
                    # Spread bet EPICs typically: IX.D.*, CS.D.*, EN.D.*, CO.D.*, IR.D.*
                    inst_type = m.get("instrumentType", "")
                    if inst_type in ("OPT_COMMODITIES", "OPT_CURRENCIES", "SHARES"):
                        continue

                    seen_epics.add(epic)
                    markets.append({
                        "epic": epic,
                        "name": m.get("instrumentName", ""),
                        "type": inst_type,
                        "expiry": m.get("expiry", "DFB"),
                    })
            except Exception as e:
                logger.warning(f"Search failed for '{term}': {e}")

        logger.info(f"Discovered {len(markets)} spread bet markets")
        self.universe = markets
        self._save_universe()
        return markets

    def get_market_details(self, epic: str) -> Optional[dict]:
        """Get full market details from IG API (costs 1 API call)."""
        if not self.client:
            return None
        info = self.client.get_market_info(epic)
        if info:
            return {
                "epic": info.epic,
                "name": info.instrument_name,
                "bid": info.bid,
                "offer": info.offer,
                "spread": info.offer - info.bid if info.bid and info.offer else 0,
                "min_deal_size": info.min_deal_size,
                "min_stop_distance": info.min_stop_distance,
                "market_status": info.market_status,
                "expiry": info.expiry,
            }
        return None

    def score_market(
        self,
        epic: str,
        name: str,
        df: pd.DataFrame,
        spread: float = 0,
        htf_trend: str = "NEUTRAL",
    ) -> Optional[MarketScore]:
        """
        Score a market based on its streaming data.

        Scoring criteria (0-100):
        - ADX trend strength (0-25): Higher ADX = stronger trend
        - ATR/spread ratio (0-25): Can we profit after spread?
        - Trend clarity (0-20): EMA separation = clear direction
        - Volatility regime (0-15): Normal/high vol preferred
        - HTF alignment (0-15): Clear HTF trend = bonus
        """
        if df is None or len(df) < 50:
            return MarketScore(
                epic=epic, name=name, sector="", score=0,
                adx=0, atr=0, atr_spread_ratio=0, trend_clarity=0,
                volatility_regime="UNKNOWN", htf_trend=htf_trend,
                is_active=False, reason="Insufficient data",
            )

        try:
            from src.indicators import add_all_indicators, calculate_ema

            # Add indicators
            params = {"ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7}
            df_ind = add_all_indicators(df.copy(), params)

            latest = df_ind.iloc[-1]
            adx = float(latest.get("adx", 0))
            atr = float(latest.get("atr", 0))
            rsi = float(latest.get("rsi", 50))
            ema_fast = float(latest.get("ema_fast", 0))
            ema_slow = float(latest.get("ema_slow", 0))
            close = float(latest.get("close", 0))

            if pd.isna(adx) or pd.isna(atr) or close <= 0:
                return MarketScore(
                    epic=epic, name=name, sector="", score=0,
                    adx=0, atr=0, atr_spread_ratio=0, trend_clarity=0,
                    volatility_regime="UNKNOWN", htf_trend=htf_trend,
                    is_active=False, reason="Invalid indicators",
                )

            # --- Scoring ---

            # 1. ADX trend strength (0-25)
            if adx >= 40:
                adx_score = 25
            elif adx >= 30:
                adx_score = 20
            elif adx >= 25:
                adx_score = 12
            elif adx >= 20:
                adx_score = 5
            else:
                adx_score = 0

            # 2. ATR/spread ratio (0-25)
            if spread > 0:
                atr_spread = atr / spread
            else:
                atr_spread = 10  # No spread info = assume good
            if atr_spread >= 5:
                spread_score = 25
            elif atr_spread >= 3:
                spread_score = 20
            elif atr_spread >= 2:
                spread_score = 12
            elif atr_spread >= 1.5:
                spread_score = 5
            else:
                spread_score = 0  # ATR < 1.5x spread = untradeable

            # 3. Trend clarity — EMA separation (0-20)
            if ema_slow != 0:
                ema_sep = abs(ema_fast - ema_slow) / abs(ema_slow) * 100
            else:
                ema_sep = 0
            if ema_sep >= 1.0:
                trend_score = 20
            elif ema_sep >= 0.5:
                trend_score = 15
            elif ema_sep >= 0.2:
                trend_score = 8
            else:
                trend_score = 0

            # 4. Volatility regime (0-15)
            # Compare current ATR to 20-period ATR average
            if "atr" in df_ind.columns:
                atr_series = df_ind["atr"].dropna()
                if len(atr_series) >= 20:
                    atr_avg = atr_series.iloc[-20:].mean()
                    vol_ratio = atr / atr_avg if atr_avg > 0 else 1.0
                else:
                    vol_ratio = 1.0
            else:
                vol_ratio = 1.0

            if vol_ratio >= 1.5:
                vol_score = 15
                vol_regime = "HIGH"
            elif vol_ratio >= 0.8:
                vol_score = 10
                vol_regime = "NORMAL"
            else:
                vol_score = 2
                vol_regime = "LOW"

            # 5. HTF alignment (0-15)
            if htf_trend in ("BULLISH", "BEARISH"):
                htf_score = 15
            else:
                htf_score = 5

            total_score = adx_score + spread_score + trend_score + vol_score + htf_score

            return MarketScore(
                epic=epic,
                name=name,
                sector="",
                score=total_score,
                adx=round(adx, 1),
                atr=round(atr, 4),
                atr_spread_ratio=round(atr_spread, 1),
                trend_clarity=round(ema_sep, 2),
                volatility_regime=vol_regime,
                htf_trend=htf_trend,
                is_active=False,
                reason="",
                scored_at=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error scoring {name}: {e}")
            return MarketScore(
                epic=epic, name=name, sector="", score=0,
                adx=0, atr=0, atr_spread_ratio=0, trend_clarity=0,
                volatility_regime="ERROR", htf_trend=htf_trend,
                is_active=False, reason=f"Error: {e}",
            )

    def run_screen(
        self,
        stream_service,
        htf_trends: dict = None,
        spreads: dict = None,
    ) -> list[MarketScore]:
        """
        Score all streaming markets and activate the top N.

        Args:
            stream_service: IGStreamService with live market data
            htf_trends: Dict of epic -> HTF trend string
            spreads: Dict of epic -> spread in points
        """
        htf_trends = htf_trends or {}
        spreads = spreads or {}

        scores = []
        for epic, market_stream in stream_service.markets.items():
            df = market_stream.to_dataframe()
            spread = spreads.get(epic, 0)

            # Calculate spread from bid/offer if not provided
            if spread == 0 and market_stream.bid > 0 and market_stream.offer > 0:
                spread = market_stream.offer - market_stream.bid

            htf = htf_trends.get(epic, "NEUTRAL")
            score = self.score_market(epic, market_stream.name, df, spread, htf)
            if score:
                scores.append(score)

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)

        # Activate top N
        self.active_epics = set()
        for i, s in enumerate(scores):
            if i < self.max_active and s.score >= 40:  # Minimum score threshold
                s.is_active = True
                s.reason = f"Rank #{i + 1}"
                self.active_epics.add(s.epic)
            else:
                s.is_active = False
                if s.score < 40:
                    s.reason = f"Score too low ({s.score})"
                else:
                    s.reason = f"Below top {self.max_active}"

        self.scores = scores
        self._save_scores()

        active_count = sum(1 for s in scores if s.is_active)
        logger.info(
            f"Screener: {len(scores)} markets scored, "
            f"{active_count} active (threshold: 40, max: {self.max_active})"
        )

        return scores

    def is_active(self, epic: str) -> bool:
        """Check if a market is currently active for trading."""
        # If no screening has been done, allow all (backwards compatible)
        if not self.scores:
            return True
        return epic in self.active_epics

    def build_market_config(
        self,
        epic: str,
        name: str,
        details: dict,
        sector: str = "Auto",
        strategy: str = "default",
    ) -> MarketConfig:
        """
        Build a MarketConfig from discovered market details.
        Used to add new markets dynamically.
        """
        spread = details.get("spread", 0)
        min_stop = details.get("min_stop_distance", 0)
        min_deal = details.get("min_deal_size", 0.1)
        expiry = details.get("expiry", "DFB")

        # Auto-detect sector from EPIC prefix
        if sector == "Auto":
            if epic.startswith("IX.D."):
                sector = "Indices"
                strategy = "indices"
            elif epic.startswith("CS.D.") and any(c in name.upper() for c in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]):
                sector = "Forex"
            elif epic.startswith(("EN.D.", "CO.D.", "CS.D.USCGC", "CS.D.COPPER", "CS.D.USCSI")):
                sector = "Commodities"
            elif epic.startswith("IR.D."):
                sector = "Rates"
            else:
                sector = "Other"

        # Auto-detect candle interval
        candle_interval = 5 if sector == "Indices" else 15

        # Auto-detect trading hours
        if sector in ("Forex", "Commodities"):
            trading_start, trading_end = 23, 21
        else:
            trading_start, trading_end = 4, 20

        # Ensure min_stop is reasonable
        if min_stop <= 0:
            # Estimate from spread: min stop = 3x spread
            min_stop = max(spread * 3, 1.0)

        # Ensure min_deal is reasonable
        if min_deal <= 0:
            min_deal = 0.1

        return MarketConfig(
            epic=epic,
            name=name,
            sector=sector,
            min_stop_distance=min_stop,
            default_size=min_deal,
            expiry=expiry if expiry else "DFB",
            candle_interval=candle_interval,
            min_confidence=0.55,
            strategy=strategy,
            trading_start=trading_start,
            trading_end=trading_end,
        )

    def get_scores_text(self) -> str:
        """Format scores for Telegram display."""
        if not self.scores:
            return "No scores available. Run /screener first."

        lines = ["<b>Market Screener</b>\n"]
        for s in self.scores:
            icon = "🟢" if s.is_active else "⚪"
            lines.append(
                f"{icon} <b>{s.name}</b>: {s.score}/100\n"
                f"   ADX={s.adx} | ATR/Spread={s.atr_spread_ratio}x | "
                f"Trend={s.trend_clarity}% | Vol={s.volatility_regime} | HTF={s.htf_trend}"
            )
        lines.append(f"\n<i>Active: {sum(1 for s in self.scores if s.is_active)}/{len(self.scores)}</i>")
        return "\n".join(lines)

    def _save_universe(self):
        """Save discovered markets to disk."""
        try:
            SCREENER_CACHE.mkdir(parents=True, exist_ok=True)
            with open(UNIVERSE_FILE, "w") as f:
                json.dump(self.universe, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save universe: {e}")

    def _load_universe(self) -> list[dict]:
        """Load discovered markets from disk."""
        try:
            if UNIVERSE_FILE.exists():
                with open(UNIVERSE_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load universe: {e}")
        return []

    def _save_scores(self):
        """Save scores to disk."""
        try:
            SCREENER_CACHE.mkdir(parents=True, exist_ok=True)
            data = [asdict(s) for s in self.scores]
            with open(SCORES_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save scores: {e}")

    def load_scores(self) -> list[MarketScore]:
        """Load scores from disk."""
        try:
            if SCORES_FILE.exists():
                with open(SCORES_FILE, "r") as f:
                    data = json.load(f)
                self.scores = [MarketScore(**d) for d in data]
                self.active_epics = {s.epic for s in self.scores if s.is_active}
                return self.scores
        except Exception as e:
            logger.warning(f"Could not load scores: {e}")
        return []
