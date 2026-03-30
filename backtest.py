#!/usr/bin/env python3
"""
Backtester for IG Trading Bot.

Reuses the live strategy logic (TradingStrategy.analyze, should_close_position)
to simulate trades on historical data. Supports parameter sweeps to find
optimal R:R, stop multiplier, and ADX threshold combinations.

Usage:
    python backtest.py                              # All markets, 30 days
    python backtest.py --market "Gold"              # Single market
    python backtest.py --days 60                    # 60 days of data
    python backtest.py --sweep                      # Parameter sweep
    python backtest.py --market "S&P 500" --sweep   # Sweep on one market
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MARKETS, MarketConfig, StrategyConfig, get_strategy_for_market, STRATEGY_PROFILES
from src.indicators import add_all_indicators, calculate_ema, calculate_atr
from src.strategy import TradingStrategy, Signal, should_close_position

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/backtest_cache")

# Yahoo Finance ticker mapping for markets we can backtest
TICKER_MAP = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "US Russell 2000": "^RUT",
    "Germany 40": "^GDAXI",
    "Wall Street": "^DJI",
    "FTSE 100": "^FTSE",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "Copper": "HG=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "Dollar Index (DXY)": "DX-Y.NYB",
}

# IG points multiplier: Yahoo prices are in real units, IG uses points
# e.g. EUR/USD: Yahoo=1.0850, IG=10850.0 (multiply by 10000)
# Gold: Yahoo=3000, IG=3000 (same)
# S&P: Yahoo=5800, IG=5800 (same for indices)
IG_POINTS_MULT = {
    "EUR/USD": 10000,
    "GBP/USD": 10000,
    "USD/JPY": 100,
    "Dollar Index (DXY)": 100,
    "Copper": 100,  # Yahoo cents/lb -> IG uses pence
}


@dataclass
class Trade:
    """A simulated trade."""
    market: str
    direction: str  # BUY or SELL
    entry_price: float
    entry_time: datetime
    stop_distance: float
    limit_distance: float
    size: float
    confidence: float
    # Exit fields
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0
    # State
    stop_price: float = 0.0
    limit_price: float = 0.0
    breakeven_applied: bool = False


@dataclass
class BacktestResult:
    """Results for one backtest run."""
    market: str
    strategy: str
    params: dict
    trades: list
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    be_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0


def fetch_data(market_name: str, days: int = 30, interval: str = "5m", no_cache: bool = False) -> Optional[pd.DataFrame]:
    """Fetch historical data from Yahoo Finance with disk caching."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return None

    ticker = TICKER_MAP.get(market_name)
    if not ticker:
        return None

    # Check cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = market_name.replace("/", "_").replace(" ", "_")
    cache_file = CACHE_DIR / f"{safe_name}_{interval}_{days}d.json"

    if not no_cache and cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < 86400:  # 24 hours
            df = pd.read_json(cache_file)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                logger.info(f"  {market_name}: Loaded {len(df)} candles from cache")
                return df

    # Fetch from Yahoo
    # Yahoo limits 5m data to 60 days
    actual_days = min(days, 59)
    start = datetime.now() - timedelta(days=actual_days)

    logger.info(f"  {market_name}: Fetching {actual_days} days of {interval} data from Yahoo...")
    data = yf.download(ticker, start=start, interval=interval, progress=False)

    if data is None or data.empty:
        logger.warning(f"  {market_name}: No data returned")
        return None

    # Handle multi-level columns from newer yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = pd.DataFrame({
        "date": data.index,
        "open": data["Open"].values,
        "high": data["High"].values,
        "low": data["Low"].values,
        "close": data["Close"].values,
        "volume": data["Volume"].values if "Volume" in data.columns else 0,
    }).reset_index(drop=True)

    # Remove timezone info for consistency
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Cache to disk
    df.to_json(cache_file, date_format="iso")
    logger.info(f"  {market_name}: Got {len(df)} candles")

    return df


def fetch_htf_data(market_name: str, days: int = 60, no_cache: bool = False) -> Optional[pd.DataFrame]:
    """Fetch hourly data for HTF trend calculation."""
    return fetch_data(market_name, days=min(days * 2, 59), interval="1h", no_cache=no_cache)


def calculate_htf_trend(htf_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate HTF trend at each timestamp using EMA 9/21."""
    if htf_df is None or htf_df.empty:
        return pd.DataFrame()

    htf_df = htf_df.copy()
    htf_df["ema9"] = calculate_ema(htf_df["close"], 9)
    htf_df["ema21"] = calculate_ema(htf_df["close"], 21)

    def get_trend(row):
        if pd.isna(row["ema9"]) or pd.isna(row["ema21"]):
            return "NEUTRAL"
        if row["ema9"] > row["ema21"]:
            return "BULLISH"
        elif row["ema9"] < row["ema21"]:
            return "BEARISH"
        return "NEUTRAL"

    htf_df["htf_trend"] = htf_df.apply(get_trend, axis=1)
    return htf_df[["date", "htf_trend"]]


def lookup_htf_trend(htf_trends: pd.DataFrame, timestamp: datetime) -> str:
    """Find the HTF trend at a given timestamp."""
    if htf_trends is None or htf_trends.empty:
        return "NEUTRAL"
    # Find the most recent HTF candle before this timestamp
    mask = htf_trends["date"] <= timestamp
    if not mask.any():
        return "NEUTRAL"
    return htf_trends.loc[mask, "htf_trend"].iloc[-1]


def convert_to_ig_points(market_name: str, price_distance: float, price_level: float) -> float:
    """Convert a price distance from Yahoo units to IG points."""
    mult = IG_POINTS_MULT.get(market_name, 1)
    return price_distance * mult


def run_backtest(
    market_config: MarketConfig,
    strategy_override: Optional[StrategyConfig] = None,
    days: int = 30,
    no_cache: bool = False,
) -> Optional[BacktestResult]:
    """Run backtest for a single market."""
    market_name = market_config.name
    strategy = strategy_override or get_strategy_for_market(market_config)

    # Fetch data using the market's actual candle interval (5m or 15m)
    candle_interval = f"{market_config.candle_interval}m"
    df = fetch_data(market_name, days=days, interval=candle_interval, no_cache=no_cache)
    if df is None or len(df) < strategy.ema_slow + 20:
        return None

    # Fetch HTF data
    htf_df = fetch_htf_data(market_name, days=days, no_cache=no_cache)
    htf_trends = calculate_htf_trend(htf_df) if htf_df is not None else pd.DataFrame()

    # Points multiplier for this market
    pts_mult = IG_POINTS_MULT.get(market_name, 1)

    # Build indicator params
    indicator_params = {
        "ema_fast": strategy.ema_fast,
        "ema_medium": strategy.ema_medium,
        "ema_slow": strategy.ema_slow,
        "rsi_period": strategy.rsi_period,
        "rsi_overbought": strategy.rsi_overbought,
        "rsi_oversold": strategy.rsi_oversold,
        "rsi_buy_max": strategy.rsi_buy_max,
        "rsi_sell_min": strategy.rsi_sell_min,
        "adx_threshold": strategy.adx_threshold,
    }

    # Add all indicators to the full dataset once
    df_with_indicators = add_all_indicators(df.copy(), indicator_params)

    # Strategy instance
    strat = TradingStrategy()

    # Simulation state
    trades = []
    active_trade: Optional[Trade] = None
    cooldown_until = None  # Candle index
    min_candles = max(strategy.ema_slow + 10, 50)  # Need enough history for indicators

    for i in range(min_candles, len(df_with_indicators)):
        row = df_with_indicators.iloc[i]
        timestamp = row["date"]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row["atr"] if "atr" in row and not pd.isna(row.get("atr", float("nan"))) else 0

        htf_trend = lookup_htf_trend(htf_trends, timestamp)

        # --- Check exit for active trade ---
        if active_trade is not None:
            trade = active_trade
            closed = False

            # 1. Stop/limit check using high/low
            if trade.direction == "BUY":
                if low <= trade.stop_price:
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "Stop hit"
                    closed = True
                elif high >= trade.limit_price:
                    trade.exit_price = trade.limit_price
                    trade.exit_reason = "Limit hit"
                    closed = True
            else:  # SELL
                if high >= trade.stop_price:
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "Stop hit"
                    closed = True
                elif low <= trade.limit_price:
                    trade.exit_price = trade.limit_price
                    trade.exit_reason = "Limit hit"
                    closed = True

            if not closed:
                # 2. Break-even check
                if not trade.breakeven_applied:
                    if trade.direction == "BUY":
                        profit_pts = close - trade.entry_price
                    else:
                        profit_pts = trade.entry_price - close

                    stop_dist = trade.stop_distance / pts_mult  # Convert back to price units
                    trigger = stop_dist * strategy.breakeven_trigger_pct
                    if profit_pts >= trigger:
                        trade.stop_price = trade.entry_price
                        trade.breakeven_applied = True

                # 3. ATR trailing stop (after break-even)
                if trade.breakeven_applied and atr > 0:
                    trail_dist = atr * strategy.atr_trail_mult
                    if trade.direction == "BUY":
                        new_trail = close - trail_dist
                        if new_trail > trade.stop_price:
                            trade.stop_price = new_trail
                    else:
                        new_trail = close + trail_dist
                        if new_trail < trade.stop_price:
                            trade.stop_price = new_trail

                # 4. Strategy exit checks (RSI extreme, MACD, dynamic)
                rsi_val = row.get("rsi", 50)
                adx_now = row.get("adx", 50)
                if not pd.isna(rsi_val):
                    if trade.direction == "BUY" and rsi_val > strategy.rsi_overbought:
                        trade.exit_price = close
                        trade.exit_reason = f"RSI overbought ({rsi_val:.1f})"
                        closed = True
                    elif trade.direction == "SELL" and rsi_val < strategy.rsi_oversold:
                        trade.exit_price = close
                        trade.exit_reason = f"RSI oversold ({rsi_val:.1f})"
                        closed = True

                if not closed and strategy.use_macd_exit and i >= 3:
                    macds = [df_with_indicators.iloc[i - j].get("macd_hist", 0) for j in range(3)]
                    if trade.direction == "BUY" and all(h < 0 for h in macds if not pd.isna(h)):
                        trade.exit_price = close
                        trade.exit_reason = "MACD histogram negative for 3 candles"
                        closed = True
                    elif trade.direction == "SELL" and all(h > 0 for h in macds if not pd.isna(h)):
                        trade.exit_price = close
                        trade.exit_reason = "MACD histogram positive for 3 candles"
                        closed = True

                if not closed and not strategy.use_macd_exit and not pd.isna(adx_now):
                    adx_exit = strategy.adx_threshold - 3
                    if adx_now < adx_exit:
                        trade.exit_price = close
                        trade.exit_reason = f"Market turned ranging (ADX {adx_now:.1f} < {adx_exit})"
                        closed = True
                    elif trade.direction == "BUY" and htf_trend == "BEARISH":
                        trade.exit_price = close
                        trade.exit_reason = "HTF trend reversed to BEARISH"
                        closed = True
                    elif trade.direction == "SELL" and htf_trend == "BULLISH":
                        trade.exit_price = close
                        trade.exit_reason = "HTF trend reversed to BULLISH"
                        closed = True

            if closed:
                trade.exit_time = timestamp
                if trade.direction == "BUY":
                    pnl_pts = (trade.exit_price - trade.entry_price) * pts_mult
                else:
                    pnl_pts = (trade.entry_price - trade.exit_price) * pts_mult
                trade.pnl = round(pnl_pts * trade.size, 2)
                trades.append(trade)
                active_trade = None
                # Cooldown: 12 candles after loss, 3 after any close
                cooldown_candles = 12 if trade.pnl < 0 else 3
                cooldown_until = i + cooldown_candles
                continue

        # --- Check entry ---
        if active_trade is not None:
            continue  # Already in a trade

        if cooldown_until and i < cooldown_until:
            continue  # In cooldown

        # Fast entry check using pre-computed indicators (avoids re-computing on every candle)
        current_price = close
        ema_fast = row.get("ema_fast", 0)
        ema_medium = row.get("ema_medium", 0)
        ema_slow_val = row.get("ema_slow", 0)
        rsi = row.get("rsi", 50)
        adx_val = row.get("adx", 0)

        if pd.isna(ema_fast) or pd.isna(adx_val) or pd.isna(atr):
            continue

        # ADX filter
        if adx_val < strategy.adx_threshold:
            continue

        # ADX direction filter
        if i > 0:
            prev_adx = df_with_indicators.iloc[i - 1].get("adx", adx_val)
            if not pd.isna(prev_adx) and adx_val < prev_adx - 0.5:
                continue

        # Determine signal direction
        bullish = ema_fast > ema_medium > ema_slow_val and close > ema_slow_val
        bearish = ema_fast < ema_medium < ema_slow_val and close < ema_slow_val

        signal_dir = None
        if bullish and strategy.rsi_oversold < rsi < strategy.rsi_buy_max:
            if htf_trend != "BEARISH":
                signal_dir = "BUY"
        elif bearish and strategy.rsi_sell_min < rsi < strategy.rsi_overbought:
            if htf_trend != "BULLISH":
                signal_dir = "SELL"

        if signal_dir is None:
            continue

        # Confidence check (simplified)
        if market_config.min_confidence > 0.5:
            # Quick confidence estimate
            ema_sep = abs(ema_fast - ema_slow_val) / abs(ema_slow_val) if ema_slow_val != 0 else 0
            conf = min(0.25, ema_sep * 10) + min(0.15, (adx_val - 25) / 100) + 0.4
            if htf_trend == signal_dir.replace("BUY", "BULLISH").replace("SELL", "BEARISH"):
                conf += 0.15
            if conf < market_config.min_confidence:
                continue

        # Calculate stop/limit from ATR
        stop_dist_price = max(atr * strategy.stop_atr_mult, market_config.min_stop_distance / pts_mult)
        # Cap stop
        max_stop_price = (market_config.min_stop_distance * 20) / pts_mult
        stop_dist_price = min(stop_dist_price, max_stop_price)
        limit_dist_price = stop_dist_price * strategy.reward_risk

        stop_dist_ig = stop_dist_price * pts_mult  # For recording

        if stop_dist_price <= 0:
            continue

        # Calculate stop and limit prices
        if signal_dir == "BUY":
            stop_price = current_price - stop_dist_price
            limit_price = current_price + limit_dist_price
        else:
            stop_price = current_price + stop_dist_price
            limit_price = current_price - limit_dist_price

        active_trade = Trade(
            market=market_name,
            direction=signal_dir,
            entry_price=current_price,
            entry_time=timestamp,
            stop_distance=stop_dist_ig,
            limit_distance=stop_dist_ig * strategy.reward_risk,
            size=market_config.default_size,
            confidence=0.7,
            stop_price=stop_price,
            limit_price=limit_price,
        )

    # Close any remaining open trade at last close
    if active_trade is not None:
        trade = active_trade
        trade.exit_price = df_with_indicators.iloc[-1]["close"]
        trade.exit_time = df_with_indicators.iloc[-1]["date"]
        trade.exit_reason = "End of data"
        if trade.direction == "BUY":
            pnl_pts = (trade.exit_price - trade.entry_price) * pts_mult
        else:
            pnl_pts = (trade.entry_price - trade.exit_price) * pts_mult
        trade.pnl = round(pnl_pts * trade.size, 2)
        trades.append(trade)

    # Calculate results
    result = BacktestResult(
        market=market_name,
        strategy=market_config.strategy,
        params={
            "reward_risk": strategy.reward_risk,
            "stop_atr_mult": strategy.stop_atr_mult,
            "adx_threshold": strategy.adx_threshold,
            "breakeven_trigger_pct": strategy.breakeven_trigger_pct,
        },
        trades=trades,
    )
    result.params["candle_interval"] = market_config.candle_interval

    wins = [t for t in trades if t.pnl > 0.50]
    losses = [t for t in trades if t.pnl < -0.50]
    bes = [t for t in trades if -0.50 <= t.pnl <= 0.50]

    result.win_count = len(wins)
    result.loss_count = len(losses)
    result.be_count = len(bes)
    result.total_pnl = sum(t.pnl for t in trades)
    result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

    # Profit factor
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    running_pnl = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running_pnl += t.pnl
        peak = max(peak, running_pnl)
        dd = peak - running_pnl
        max_dd = max(max_dd, dd)
    result.max_drawdown = max_dd

    return result


def print_result(result: BacktestResult):
    """Print formatted backtest results."""
    total = result.win_count + result.loss_count + result.be_count
    wr = (result.win_count / (result.win_count + result.loss_count) * 100) if (result.win_count + result.loss_count) > 0 else 0

    candle_str = f" | {result.params['candle_interval']}m candles" if 'candle_interval' in result.params else ""
    print(f"\n{'=' * 60}")
    print(f"  {result.market} ({result.strategy} strategy)")
    print(f"{'=' * 60}")
    print(f"  R:R = {result.params['reward_risk']:.1f}:1 | "
          f"Stop ATR = {result.params['stop_atr_mult']}x | "
          f"ADX = {result.params['adx_threshold']} | "
          f"BE = {result.params['breakeven_trigger_pct']:.0%}"
          f"{candle_str}")
    print(f"  Trades: {total} ({result.win_count}W / {result.be_count}BE / {result.loss_count}L)")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total P&L: £{result.total_pnl:>+.2f}")
    print(f"  Avg Win: £{result.avg_win:>+.2f}  |  Avg Loss: £{result.avg_loss:>+.2f}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Max Drawdown: £{result.max_drawdown:.2f}")

    # Exit breakdown
    exit_counts = {}
    for t in result.trades:
        key = t.exit_reason.split("(")[0].strip()  # Group similar reasons
        exit_counts[key] = exit_counts.get(key, 0) + 1
    if exit_counts:
        print(f"  Exits: {', '.join(f'{k}={v}' for k, v in sorted(exit_counts.items(), key=lambda x: -x[1]))}")


def run_sweep(
    market_config: MarketConfig,
    days: int = 30,
    no_cache: bool = False,
) -> list[BacktestResult]:
    """Run parameter sweep for a market."""
    base_strategy = get_strategy_for_market(market_config)

    # Parameter combinations to test
    rr_values = [1.5, 2.0, 3.0, 4.0]
    stop_values = [1.0, 1.5, 2.0]
    adx_values = [30, 35, 40]

    results = []
    combos = [(rr, stop, adx) for rr in rr_values for stop in stop_values for adx in adx_values]
    print(f"\n  Testing {len(combos)} parameter combinations for {market_config.name}...")

    for rr, stop_mult, adx_thresh in combos:
        # Create modified strategy and inject into STRATEGY_PROFILES temporarily
        modified = replace(
            base_strategy,
            reward_risk=rr,
            stop_atr_mult=stop_mult,
            adx_threshold=adx_thresh,
        )
        sweep_key = "_sweep_temp"
        STRATEGY_PROFILES[sweep_key] = modified
        sweep_market = replace(market_config, strategy=sweep_key)

        result = run_backtest(sweep_market, strategy_override=modified, days=days, no_cache=no_cache)
        if result:
            result.market = market_config.name  # Restore original name
            result.params = {
                "reward_risk": rr,
                "stop_atr_mult": stop_mult,
                "adx_threshold": adx_thresh,
                "breakeven_trigger_pct": base_strategy.breakeven_trigger_pct,
            }
            results.append(result)

    # Cleanup
    STRATEGY_PROFILES.pop("_sweep_temp", None)
    return results


def print_sweep_results(results: list[BacktestResult], market_name: str):
    """Print sweep results sorted by profit factor."""
    if not results:
        print(f"\n  No results for {market_name}")
        return

    # Sort by total P&L
    results.sort(key=lambda r: r.total_pnl, reverse=True)

    print(f"\n{'=' * 90}")
    print(f"  PARAMETER SWEEP: {market_name}")
    print(f"{'=' * 90}")
    print(f"  {'R:R':>5} {'Stop':>5} {'ADX':>5} | {'Trades':>6} {'WR%':>6} | {'P&L':>10} {'AvgW':>8} {'AvgL':>8} {'PF':>6} {'MaxDD':>8}")
    print(f"  {'-' * 80}")

    for r in results[:20]:  # Top 20
        total = r.win_count + r.loss_count + r.be_count
        wr = (r.win_count / (r.win_count + r.loss_count) * 100) if (r.win_count + r.loss_count) > 0 else 0
        marker = " <-- CURRENT" if (
            r.params["reward_risk"] == get_strategy_for_market(next(m for m in MARKETS if m.name == market_name)).reward_risk
            and r.params["stop_atr_mult"] == get_strategy_for_market(next(m for m in MARKETS if m.name == market_name)).stop_atr_mult
            and r.params["adx_threshold"] == get_strategy_for_market(next(m for m in MARKETS if m.name == market_name)).adx_threshold
        ) else ""
        print(
            f"  {r.params['reward_risk']:>5.1f} {r.params['stop_atr_mult']:>5.1f} {r.params['adx_threshold']:>5} | "
            f"{total:>6} {wr:>5.1f}% | "
            f"£{r.total_pnl:>+9.2f} £{r.avg_win:>+7.2f} £{r.avg_loss:>+7.2f} {r.profit_factor:>5.2f} £{r.max_drawdown:>7.2f}"
            f"{marker}"
        )


def main():
    parser = argparse.ArgumentParser(description="IG Trading Bot Backtester")
    parser.add_argument("--market", type=str, help="Run for a specific market (e.g. 'Gold')")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data (max 59 for 5m/15m Yahoo)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Filter to available markets (those with Yahoo tickers)
    available = [m for m in MARKETS if m.name in TICKER_MAP]

    if args.market:
        available = [m for m in available if m.name == args.market]
        if not available:
            names = [m.name for m in MARKETS if m.name in TICKER_MAP]
            print(f"Market '{args.market}' not found. Available: {', '.join(names)}")
            return

    print(f"\n{'#' * 60}")
    print(f"  IG TRADING BOT BACKTESTER")
    print(f"  Period: {args.days} days | Markets: {len(available)}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#' * 60}")

    if args.sweep:
        # Parameter sweep mode
        for market in available:
            results = run_sweep(market, days=args.days, no_cache=args.no_cache)
            print_sweep_results(results, market.name)
    else:
        # Normal backtest with current parameters
        all_results = []
        print(f"\nFetching data...")

        for market in available:
            result = run_backtest(market, days=args.days, no_cache=args.no_cache)
            if result:
                all_results.append(result)
                print_result(result)

        # Summary
        if all_results:
            total_trades = sum(r.win_count + r.loss_count + r.be_count for r in all_results)
            total_wins = sum(r.win_count for r in all_results)
            total_losses = sum(r.loss_count for r in all_results)
            total_bes = sum(r.be_count for r in all_results)
            total_pnl = sum(r.total_pnl for r in all_results)
            wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

            print(f"\n{'=' * 60}")
            print(f"  PORTFOLIO SUMMARY")
            print(f"{'=' * 60}")
            print(f"  Markets tested: {len(all_results)}")
            print(f"  Total trades: {total_trades} ({total_wins}W / {total_bes}BE / {total_losses}L)")
            print(f"  Overall Win Rate: {wr:.1f}%")
            print(f"  Total P&L: £{total_pnl:>+.2f}")
            print(f"\n  Note: Spread costs not simulated. Real P&L will be lower.")


if __name__ == "__main__":
    main()
