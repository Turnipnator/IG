"""
Backtesting module for IG Trading Bot strategy.

Uses Yahoo Finance data to avoid IG API limits.
Supports parameter testing and comparison.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from src.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_adx,
    calculate_macd,
    calculate_atr,
)
from src.regime import (
    MarketRegime,
    classify_regime,
    get_regime_params,
    TrendState,
)

logger = logging.getLogger(__name__)


# Map IG markets to Yahoo Finance tickers
TICKER_MAP = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "Dollar Index": "DX-Y.NYB",
}

# Minimum stop distances per market in PRICE UNITS (not IG points)
# These ensure reasonable stop distances that align with each market's volatility
# Calculated as approximately 0.5% of typical price for each instrument
MIN_STOP_DISTANCE_MAP = {
    "S&P 500": 30.0,      # ~0.5% of 6000 = 30 pts
    "NASDAQ 100": 100.0,  # ~0.5% of 20000 = 100 pts
    "Gold": 25.0,         # ~0.5% of 5000 = 25 pts (~$25)
    "Crude Oil": 0.35,    # ~0.5% of 70 = 0.35 (~35 cents)
    "EUR/USD": 0.005,     # ~0.5% of 1.08 = 0.005 (~50 pips)
    "Dollar Index": 0.50, # ~0.5% of 108 = 0.54
}

# Minimum confidence thresholds per market (matching config.py)
MIN_CONFIDENCE_MAP = {
    "S&P 500": 0.5,
    "NASDAQ 100": 0.5,
    "Gold": 0.5,
    "Crude Oil": 0.7,  # Higher threshold due to choppiness
    "EUR/USD": 0.5,
    "Dollar Index": 0.5,
}

# Market-specific reward:risk ratios
# Lower ratio for markets with smaller typical moves
REWARD_RISK_MAP = {
    "S&P 500": 2.0,
    "NASDAQ 100": 2.0,
    "Gold": 2.0,
    "Crude Oil": 1.5,  # Reduced - typical moves don't reach 2:1 targets
    "EUR/USD": 2.0,
    "Dollar Index": 2.0,
}

# Markets where MACD exit should be disabled (rely on stop/TP only)
# Currently empty - MACD exits help overall performance
DISABLE_MACD_EXIT: set[str] = set()

# Default strategy parameters (matching config.py)
DEFAULT_PARAMS = {
    "ema_fast": 9,
    "ema_medium": 21,
    "ema_slow": 50,
    "rsi_period": 7,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_buy_max": 60,
    "rsi_sell_min": 40,
    "adx_threshold": 25,
    "atr_period": 14,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
}


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    market: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: Optional[float]
    stop_price: float
    limit_price: float
    size: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    htf_trend: str = "NEUTRAL"
    confidence: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    market: str
    period: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: list[Trade] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"Backtest Results: {self.market}\n"
            f"Period: {self.period}\n"
            f"{'='*50}\n"
            f"Total Trades:    {self.total_trades}\n"
            f"Winning Trades:  {self.winning_trades}\n"
            f"Losing Trades:   {self.losing_trades}\n"
            f"Win Rate:        {self.win_rate:.1%}\n"
            f"Total P&L:       {self.total_pnl:+.2f}%\n"
            f"Avg Win:         {self.avg_win:+.2f}%\n"
            f"Avg Loss:        {self.avg_loss:.2f}%\n"
            f"Profit Factor:   {self.profit_factor:.2f}\n"
            f"Max Drawdown:    {self.max_drawdown:.1%}\n"
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}\n"
            f"{'='*50}\n"
        )


class Backtester:
    """
    Backtests the trading strategy using historical data from Yahoo Finance.

    Usage:
        bt = Backtester()
        result = bt.run("S&P 500", days=30)
        print(result)

        # Compare with HTF requirement
        result_htf = bt.run("S&P 500", days=30, require_htf_alignment=True)
    """

    def __init__(self, params: dict = None):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        self.params = params or DEFAULT_PARAMS.copy()
        self.trades: list[Trade] = []

    def fetch_data(
        self,
        market: str,
        days: int = 30,
        interval: str = "5m",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            market: Market name (e.g., "S&P 500")
            days: Number of days of history
            interval: Candle interval ("1m", "5m", "15m", "1h", "1d")

        Returns:
            DataFrame with OHLCV data
        """
        ticker = TICKER_MAP.get(market)
        if not ticker:
            logger.error(f"Unknown market: {market}")
            return None

        try:
            # Yahoo Finance limits: 5m data only available for last 60 days
            if interval in ["1m", "2m", "5m", "15m", "30m"]:
                max_days = 60
                days = min(days, max_days)

            end = datetime.now()
            start = end - timedelta(days=days)

            logger.info(f"Fetching {market} ({ticker}) data: {days} days, {interval} interval")

            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
            )

            if df.empty:
                logger.warning(f"No data returned for {market}")
                return None

            # Handle multi-level columns from yfinance (newer versions)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten multi-index columns (take first level - Price name)
                df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]

            # Reset index to get datetime as a column
            df = df.reset_index()

            # Standardize the datetime column name
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "date"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "date"})

            # Convert date to datetime if needed
            if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df["date"] = pd.to_datetime(df["date"])

            logger.info(f"Fetched {len(df)} candles for {market}")
            return df

        except Exception as e:
            logger.exception(f"Error fetching data for {market}: {e}")
            return None

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame."""
        df = df.copy()

        # EMAs
        df["ema_fast"] = calculate_ema(df["close"], self.params["ema_fast"])
        df["ema_medium"] = calculate_ema(df["close"], self.params["ema_medium"])
        df["ema_slow"] = calculate_ema(df["close"], self.params["ema_slow"])

        # RSI
        df["rsi"] = calculate_rsi(df["close"], self.params["rsi_period"])

        # ADX
        df["adx"] = calculate_adx(
            df["high"], df["low"], df["close"], period=14
        )

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])

        # ATR
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], self.params["atr_period"])

        return df

    def calculate_htf_trend(
        self,
        market: str,
        current_time: datetime,
        htf_data: pd.DataFrame,
    ) -> str:
        """
        Determine the higher timeframe trend at a specific point in time.

        Args:
            market: Market name
            current_time: The timestamp to check
            htf_data: Hourly DataFrame with EMAs

        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if htf_data is None or htf_data.empty:
            return "NEUTRAL"

        # Get the most recent hourly candle before current_time
        mask = htf_data["date"] <= current_time
        if not mask.any():
            return "NEUTRAL"

        latest = htf_data[mask].iloc[-1]

        ema_9 = latest.get("ema_9")
        ema_21 = latest.get("ema_21")
        close = latest.get("close")

        if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(close):
            return "NEUTRAL"

        if ema_9 > ema_21 and close > ema_21:
            return "BULLISH"
        elif ema_9 < ema_21 and close < ema_21:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def check_entry_signal(
        self,
        row: pd.Series,
        htf_trend: str,
        require_htf_alignment: bool = False,
    ) -> tuple[Optional[str], float, str]:
        """
        Check if current candle generates an entry signal.

        Returns:
            Tuple of (direction, confidence, reason) or (None, 0, "")
        """
        ema_fast = row["ema_fast"]
        ema_medium = row["ema_medium"]
        ema_slow = row["ema_slow"]
        rsi = row["rsi"]
        adx = row["adx"]
        close = row["close"]
        macd_hist = row["macd_hist"]

        # Skip if indicators not ready
        if pd.isna(ema_slow) or pd.isna(adx) or pd.isna(rsi):
            return None, 0, ""

        # ADX filter
        if adx < self.params["adx_threshold"]:
            return None, 0, f"ADX too low ({adx:.1f})"

        # Check for bullish setup
        bullish_ema = ema_fast > ema_medium > ema_slow
        price_above_ema = close > ema_slow
        rsi_buy_valid = self.params["rsi_oversold"] < rsi < self.params["rsi_buy_max"]

        # Check for bearish setup
        bearish_ema = ema_fast < ema_medium < ema_slow
        price_below_ema = close < ema_slow
        rsi_sell_valid = self.params["rsi_sell_min"] < rsi < self.params["rsi_overbought"]

        # MACD pre-check
        macd_already_bearish = macd_hist < 0 if not pd.isna(macd_hist) else False
        macd_already_bullish = macd_hist > 0 if not pd.isna(macd_hist) else False

        # BUY signal
        if bullish_ema and price_above_ema and rsi_buy_valid:
            if macd_already_bearish:
                return None, 0, "MACD already bearish"

            # HTF filter
            if require_htf_alignment and htf_trend != "BULLISH":
                return None, 0, f"HTF not aligned ({htf_trend})"
            elif htf_trend == "BEARISH":
                return None, 0, f"HTF opposing ({htf_trend})"

            confidence = self._calculate_confidence("BUY", rsi, adx, htf_trend)
            return "BUY", confidence, f"Bullish EMA, RSI={rsi:.1f}, ADX={adx:.1f}, HTF={htf_trend}"

        # SELL signal
        if bearish_ema and price_below_ema and rsi_sell_valid:
            if macd_already_bullish:
                return None, 0, "MACD already bullish"

            # HTF filter
            if require_htf_alignment and htf_trend != "BEARISH":
                return None, 0, f"HTF not aligned ({htf_trend})"
            elif htf_trend == "BULLISH":
                return None, 0, f"HTF opposing ({htf_trend})"

            confidence = self._calculate_confidence("SELL", rsi, adx, htf_trend)
            return "SELL", confidence, f"Bearish EMA, RSI={rsi:.1f}, ADX={adx:.1f}, HTF={htf_trend}"

        return None, 0, ""

    def _calculate_confidence(
        self,
        direction: str,
        rsi: float,
        adx: float,
        htf_trend: str,
    ) -> float:
        """Calculate confidence score (0-1)."""
        confidence = 0.0

        # ADX strength (0-0.3)
        confidence += min((adx - 25) / 50, 0.3)

        # RSI position (0-0.3)
        if direction == "BUY":
            rsi_factor = (60 - rsi) / 30  # Better if RSI is lower
        else:
            rsi_factor = (rsi - 40) / 30  # Better if RSI is higher
        confidence += max(0, min(rsi_factor * 0.3, 0.3))

        # HTF alignment (0-0.4)
        if (direction == "BUY" and htf_trend == "BULLISH") or \
           (direction == "SELL" and htf_trend == "BEARISH"):
            confidence += 0.4
        elif htf_trend == "NEUTRAL":
            confidence += 0.2

        return min(confidence, 1.0)

    def run(
        self,
        market: str,
        days: int = 30,
        interval: str = "5m",
        require_htf_alignment: bool = False,
        min_confidence: Optional[float] = None,
        account_size: float = 10000,
        risk_per_trade: float = 0.01,
    ) -> BacktestResult:
        """
        Run backtest for a market.

        Args:
            market: Market name
            days: Days of history to test
            interval: Candle interval for entries
            require_htf_alignment: If True, require HTF trend to match direction
            min_confidence: Minimum confidence to take trade (uses market default if None)
            account_size: Starting account size
            risk_per_trade: Risk per trade as fraction of account

        Returns:
            BacktestResult with performance metrics
        """
        self.trades = []

        # Use market-specific min_confidence if not explicitly provided
        if min_confidence is None:
            min_confidence = MIN_CONFIDENCE_MAP.get(market, 0.5)

        # Fetch main timeframe data
        df = self.fetch_data(market, days, interval)
        if df is None or df.empty:
            return self._empty_result(market, days)

        # Add indicators
        df = self.add_indicators(df)

        # Fetch hourly data for HTF trend
        htf_df = self.fetch_data(market, days, "1h")
        if htf_df is not None and not htf_df.empty:
            htf_df["ema_9"] = calculate_ema(htf_df["close"], 9)
            htf_df["ema_21"] = calculate_ema(htf_df["close"], 21)

        # Track state
        position: Optional[Trade] = None
        equity = account_size
        peak_equity = account_size
        max_drawdown = 0.0
        daily_returns = []
        last_close_time = None
        cooldown_until = None

        # Iterate through candles
        for i in range(self.params["ema_slow"], len(df)):
            row = df.iloc[i]
            current_time = row["date"]
            close = row["close"]
            atr = row["atr"]

            # Skip if in cooldown
            if cooldown_until and current_time < cooldown_until:
                continue

            # If in position, check for exit
            if position:
                exit_reason = None
                exit_price = None

                # Check stop loss
                if position.direction == "BUY" and close <= position.stop_price:
                    exit_reason = "Stop loss"
                    exit_price = position.stop_price
                elif position.direction == "SELL" and close >= position.stop_price:
                    exit_reason = "Stop loss"
                    exit_price = position.stop_price

                # Check take profit
                if position.direction == "BUY" and close >= position.limit_price:
                    exit_reason = "Take profit"
                    exit_price = position.limit_price
                elif position.direction == "SELL" and close <= position.limit_price:
                    exit_reason = "Take profit"
                    exit_price = position.limit_price

                # Check MACD exit (unless disabled for this market)
                if market not in DISABLE_MACD_EXIT:
                    macd_hist = row["macd_hist"]
                    if position.direction == "BUY" and macd_hist < 0:
                        # Check 3 consecutive negative
                        if i >= 3:
                            last_3 = [df.iloc[i-j]["macd_hist"] for j in range(3)]
                            if all(h < 0 for h in last_3 if not pd.isna(h)):
                                exit_reason = "MACD bearish"
                                exit_price = close
                    elif position.direction == "SELL" and macd_hist > 0:
                        if i >= 3:
                            last_3 = [df.iloc[i-j]["macd_hist"] for j in range(3)]
                            if all(h > 0 for h in last_3 if not pd.isna(h)):
                                exit_reason = "MACD bullish"
                                exit_price = close

                # Close position if exit triggered
                if exit_reason:
                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason

                    # Calculate P&L
                    if position.direction == "BUY":
                        position.pnl_percent = (exit_price - position.entry_price) / position.entry_price * 100
                    else:
                        position.pnl_percent = (position.entry_price - exit_price) / position.entry_price * 100

                    position.pnl = position.pnl_percent * position.size
                    equity += position.pnl * account_size / 100

                    self.trades.append(position)

                    # Set cooldown if loss
                    if position.pnl_percent < 0:
                        # 1 hour cooldown (12 x 5min candles)
                        cooldown_until = current_time + timedelta(hours=1)

                    last_close_time = current_time
                    position = None

                    # Track drawdown
                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity
                    max_drawdown = max(max_drawdown, drawdown)

                continue  # Don't look for new entries while in position

            # Check for entry signal
            htf_trend = self.calculate_htf_trend(market, current_time, htf_df)
            direction, confidence, reason = self.check_entry_signal(
                row, htf_trend, require_htf_alignment
            )

            # Classify market regime (need at least 20 candles for ATR median)
            regime = None
            regime_params = None
            if i >= 20:
                try:
                    regime_df = df.iloc[:i+1].copy()
                    regime = classify_regime(regime_df)
                    regime_params = get_regime_params(regime)
                except (ValueError, KeyError):
                    pass  # Not enough data or missing columns

            # Apply regime filters
            if regime_params:
                # Block trades in untradeable regimes (RANGING_HIGH)
                if not regime.is_tradeable:
                    continue

                # Block trend-following in ranging regimes
                if direction and not regime_params.allow_trend_follow:
                    continue

                # Adjust min_confidence based on regime
                effective_min_confidence = max(min_confidence, regime_params.min_confidence)
            else:
                effective_min_confidence = min_confidence

            if direction and confidence >= effective_min_confidence:
                # Calculate stop and limit
                if pd.isna(atr):
                    continue

                # Get ATR multiplier (regime-adjusted if available)
                stop_atr_mult = self.params["stop_atr_multiplier"]
                if regime_params:
                    stop_atr_mult = regime_params.stop_atr_multiplier

                # Calculate ATR-based stop, but respect minimum stop distance from config
                atr_stop = atr * stop_atr_mult
                min_stop = MIN_STOP_DISTANCE_MAP.get(market, 0.0)
                stop_distance = max(atr_stop, min_stop)

                # Use market-specific reward:risk ratio
                rr_ratio = REWARD_RISK_MAP.get(market, self.params["reward_risk_ratio"])
                limit_distance = stop_distance * rr_ratio

                if direction == "BUY":
                    stop_price = close - stop_distance
                    limit_price = close + limit_distance
                else:
                    stop_price = close + stop_distance
                    limit_price = close - limit_distance

                # Calculate position size (risk-based, regime-adjusted)
                risk_amount = account_size * risk_per_trade
                size = risk_amount / stop_distance if stop_distance > 0 else 0

                # Apply regime size multiplier
                if regime_params:
                    size = size * regime_params.size_multiplier

                position = Trade(
                    entry_time=current_time,
                    exit_time=None,
                    market=market,
                    direction=direction,
                    entry_price=close,
                    exit_price=None,
                    stop_price=stop_price,
                    limit_price=limit_price,
                    size=size,
                    htf_trend=htf_trend,
                    confidence=confidence,
                )

        # Close any open position at end
        if position:
            position.exit_time = df.iloc[-1]["date"]
            position.exit_price = df.iloc[-1]["close"]
            position.exit_reason = "End of test"
            if position.direction == "BUY":
                position.pnl_percent = (position.exit_price - position.entry_price) / position.entry_price * 100
            else:
                position.pnl_percent = (position.entry_price - position.exit_price) / position.entry_price * 100
            position.pnl = position.pnl_percent * position.size
            self.trades.append(position)

        return self._calculate_results(market, days, require_htf_alignment, max_drawdown)

    def _empty_result(self, market: str, days: int) -> BacktestResult:
        """Return empty result when no data available."""
        return BacktestResult(
            market=market,
            period=f"{days} days",
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            trades=[],
            params=self.params.copy(),
        )

    def _calculate_results(
        self,
        market: str,
        days: int,
        require_htf: bool,
        max_drawdown: float,
    ) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return self._empty_result(market, days)

        winning = [t for t in self.trades if t.pnl_percent > 0]
        losing = [t for t in self.trades if t.pnl_percent < 0]

        total_trades = len(self.trades)
        winning_trades = len(winning)
        losing_trades = len(losing)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(t.pnl_percent for t in self.trades)
        avg_win = np.mean([t.pnl_percent for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_percent for t in losing]) if losing else 0

        gross_profit = sum(t.pnl_percent for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl_percent for t in losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified)
        returns = [t.pnl_percent for t in self.trades]
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            market=market,
            period=f"{days} days",
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            trades=self.trades.copy(),
            params={**self.params, "require_htf_alignment": require_htf},
        )

    def compare_htf_impact(
        self,
        market: str,
        days: int = 30,
    ) -> tuple[BacktestResult, BacktestResult]:
        """
        Run backtest with and without HTF requirement and compare.

        Returns:
            Tuple of (result_without_htf, result_with_htf)
        """
        print(f"\n{'='*60}")
        print(f"Comparing HTF Impact for {market}")
        print(f"{'='*60}")

        result_no_htf = self.run(market, days, require_htf_alignment=False)
        result_htf = self.run(market, days, require_htf_alignment=True)

        print("\n--- WITHOUT HTF Requirement ---")
        print(result_no_htf)

        print("\n--- WITH HTF Requirement ---")
        print(result_htf)

        # Summary comparison
        print("\n--- COMPARISON ---")
        print(f"{'Metric':<20} {'No HTF':>12} {'With HTF':>12} {'Diff':>12}")
        print("-" * 56)
        print(f"{'Trades':<20} {result_no_htf.total_trades:>12} {result_htf.total_trades:>12} {result_htf.total_trades - result_no_htf.total_trades:>+12}")
        print(f"{'Win Rate':<20} {result_no_htf.win_rate:>11.1%} {result_htf.win_rate:>11.1%} {(result_htf.win_rate - result_no_htf.win_rate)*100:>+11.1f}%")
        print(f"{'Total P&L':<20} {result_no_htf.total_pnl:>+11.2f}% {result_htf.total_pnl:>+11.2f}% {result_htf.total_pnl - result_no_htf.total_pnl:>+11.2f}%")
        print(f"{'Profit Factor':<20} {result_no_htf.profit_factor:>12.2f} {result_htf.profit_factor:>12.2f} {result_htf.profit_factor - result_no_htf.profit_factor:>+12.2f}")
        print(f"{'Max Drawdown':<20} {result_no_htf.max_drawdown:>11.1%} {result_htf.max_drawdown:>11.1%} {(result_htf.max_drawdown - result_no_htf.max_drawdown)*100:>+11.1f}%")

        return result_no_htf, result_htf


def run_full_backtest():
    """Run backtest across all markets and compare HTF impact."""
    bt = Backtester()

    markets = ["S&P 500", "NASDAQ 100", "Gold", "Crude Oil"]

    all_results = []

    for market in markets:
        try:
            no_htf, with_htf = bt.compare_htf_impact(market, days=30)
            all_results.append({
                "market": market,
                "no_htf": no_htf,
                "with_htf": with_htf,
            })
        except Exception as e:
            print(f"Error backtesting {market}: {e}")

    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    total_no_htf = sum(r["no_htf"].total_pnl for r in all_results)
    total_htf = sum(r["with_htf"].total_pnl for r in all_results)

    trades_no_htf = sum(r["no_htf"].total_trades for r in all_results)
    trades_htf = sum(r["with_htf"].total_trades for r in all_results)

    wins_no_htf = sum(r["no_htf"].winning_trades for r in all_results)
    wins_htf = sum(r["with_htf"].winning_trades for r in all_results)

    print(f"\nWithout HTF Requirement:")
    print(f"  Total Trades: {trades_no_htf}")
    print(f"  Win Rate: {wins_no_htf/trades_no_htf:.1%}" if trades_no_htf > 0 else "  Win Rate: N/A")
    print(f"  Total P&L: {total_no_htf:+.2f}%")

    print(f"\nWith HTF Requirement:")
    print(f"  Total Trades: {trades_htf}")
    print(f"  Win Rate: {wins_htf/trades_htf:.1%}" if trades_htf > 0 else "  Win Rate: N/A")
    print(f"  Total P&L: {total_htf:+.2f}%")

    print(f"\nDifference:")
    print(f"  Trades: {trades_htf - trades_no_htf:+d}")
    print(f"  P&L: {total_htf - total_no_htf:+.2f}%")

    recommendation = "WITH" if total_htf > total_no_htf else "WITHOUT"
    print(f"\n>>> RECOMMENDATION: Trade {recommendation} HTF requirement")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_backtest()
