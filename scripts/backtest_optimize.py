#!/usr/bin/env python3
"""
Comprehensive strategy optimization backtest.

Tests multiple parameter combinations to find optimal settings for maximum profit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from itertools import product
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

from src.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_adx,
    calculate_macd,
    calculate_atr,
)

logging.basicConfig(level=logging.WARNING)


TICKER_MAP = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "Dollar Index": "DX-Y.NYB",
}

# Minimum stop distances in price units (for realistic stops)
MIN_STOP_MAP = {
    "S&P 500": 30.0,
    "NASDAQ 100": 100.0,
    "Gold": 25.0,
    "Crude Oil": 0.35,
    "EUR/USD": 0.005,
    "Dollar Index": 0.50,
}

# Point values for P&L calculation (£ per point move)
POINT_VALUE_MAP = {
    "S&P 500": 1.0,      # £1 per point
    "NASDAQ 100": 1.0,
    "Gold": 1.0,
    "Crude Oil": 1.0,
    "EUR/USD": 10000.0,  # Forex is per pip (0.0001)
    "Dollar Index": 1.0,
}


@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    market: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    stop_price: float
    limit_price: float
    size: float
    pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    params: dict
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    sharpe: float
    trades: list = field(default_factory=list)

    def score(self) -> float:
        """Calculate overall score for ranking strategies."""
        if self.total_trades < 10:
            return -999  # Not enough trades

        # Weighted score: prioritize P&L but penalize drawdown and reward consistency
        pnl_score = self.total_pnl
        dd_penalty = self.max_drawdown * 2  # Penalize drawdown
        consistency = self.win_rate * 0.5   # Bonus for high win rate

        return pnl_score - dd_penalty + consistency


def fetch_data(market: str, days: int = 60, interval: str = "5m") -> Optional[pd.DataFrame]:
    """Fetch historical data."""
    ticker = TICKER_MAP.get(market)
    if not ticker:
        return None

    try:
        if interval in ["1m", "2m", "5m", "15m", "30m"]:
            days = min(days, 60)

        end = datetime.now()
        start = end - timedelta(days=days)

        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        df = df.reset_index()
        for col in df.columns:
            if col.lower() in ("datetime", "date", "index"):
                df = df.rename(columns={col: "date"})
                break

        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        return df
    except Exception as e:
        return None


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add technical indicators with given parameters."""
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], params["ema_fast"])
    df["ema_medium"] = calculate_ema(df["close"], params["ema_medium"])
    df["ema_slow"] = calculate_ema(df["close"], params["ema_slow"])
    df["rsi"] = calculate_rsi(df["close"], params["rsi_period"])
    df["adx"] = calculate_adx(df["high"], df["low"], df["close"], period=14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], params["atr_period"])
    return df


def run_backtest(
    market: str,
    df: pd.DataFrame,
    htf_df: pd.DataFrame,
    params: dict,
    account_size: float = 10000,
    risk_per_trade: float = 0.01,
) -> BacktestResult:
    """Run backtest with given parameters."""

    df = add_indicators(df, params)

    if htf_df is not None and not htf_df.empty:
        htf_df = htf_df.copy()
        htf_df["ema_9"] = calculate_ema(htf_df["close"], 9)
        htf_df["ema_21"] = calculate_ema(htf_df["close"], 21)

    trades = []
    position = None
    equity = account_size
    peak_equity = account_size
    max_drawdown = 0.0

    min_bars = max(params["ema_slow"], 20) + 5

    for i in range(min_bars, len(df)):
        row = df.iloc[i]
        current_time = row["date"]
        close = row["close"]
        atr = row["atr"]

        # Exit logic
        if position:
            exit_reason = None
            exit_price = None

            if position.direction == "BUY":
                if close <= position.stop_price:
                    exit_reason, exit_price = "Stop", position.stop_price
                elif close >= position.limit_price:
                    exit_reason, exit_price = "TP", position.limit_price
            else:
                if close >= position.stop_price:
                    exit_reason, exit_price = "Stop", position.stop_price
                elif close <= position.limit_price:
                    exit_reason, exit_price = "TP", position.limit_price

            # MACD exit (if enabled)
            if params.get("use_macd_exit", True) and not exit_reason and i >= 3:
                macd_hist = row["macd_hist"]
                last_3 = [df.iloc[i-j]["macd_hist"] for j in range(3)]
                if position.direction == "BUY" and all(h < 0 for h in last_3 if not pd.isna(h)):
                    exit_reason, exit_price = "MACD", close
                elif position.direction == "SELL" and all(h > 0 for h in last_3 if not pd.isna(h)):
                    exit_reason, exit_price = "MACD", close

            if exit_reason:
                position.exit_time = current_time
                position.exit_price = exit_price
                position.exit_reason = exit_reason

                # Calculate P&L in account currency
                point_value = POINT_VALUE_MAP.get(market, 1.0)
                if position.direction == "BUY":
                    points = exit_price - position.entry_price
                else:
                    points = position.entry_price - exit_price

                position.pnl = points * position.size * point_value
                equity += position.pnl

                trades.append(position)
                position = None

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                max_drawdown = max(max_drawdown, dd)

            continue

        # Entry logic
        ema_fast = row["ema_fast"]
        ema_medium = row["ema_medium"]
        ema_slow = row["ema_slow"]
        rsi = row["rsi"]
        adx = row["adx"]
        macd_hist = row["macd_hist"]

        if pd.isna(ema_slow) or pd.isna(adx) or pd.isna(rsi) or pd.isna(atr):
            continue

        # ADX filter
        if adx < params["adx_threshold"]:
            continue

        # HTF trend
        htf_trend = "NEUTRAL"
        if htf_df is not None and not htf_df.empty:
            mask = htf_df["date"] <= current_time
            if mask.any():
                latest_htf = htf_df[mask].iloc[-1]
                ema_9 = latest_htf.get("ema_9")
                ema_21 = latest_htf.get("ema_21")
                htf_close = latest_htf.get("close")
                if not pd.isna(ema_9) and not pd.isna(ema_21):
                    if ema_9 > ema_21 and htf_close > ema_21:
                        htf_trend = "BULLISH"
                    elif ema_9 < ema_21 and htf_close < ema_21:
                        htf_trend = "BEARISH"

        direction = None
        confidence = 0.0

        # Bullish signal
        if (ema_fast > ema_medium > ema_slow and
            close > ema_slow and
            params["rsi_oversold"] < rsi < params["rsi_buy_max"]):

            if params.get("check_macd_entry", True) and not pd.isna(macd_hist) and macd_hist < 0:
                pass  # Skip if MACD bearish
            elif params.get("require_htf", False) and htf_trend != "BULLISH":
                pass  # Skip if HTF not aligned
            elif htf_trend == "BEARISH":
                pass  # Never trade against HTF
            else:
                confidence = min((adx - 25) / 50, 0.3)
                confidence += max(0, min((60 - rsi) / 100, 0.3))
                if htf_trend == "BULLISH":
                    confidence += 0.4
                elif htf_trend == "NEUTRAL":
                    confidence += 0.2
                direction = "BUY"

        # Bearish signal
        elif (ema_fast < ema_medium < ema_slow and
              close < ema_slow and
              params["rsi_sell_min"] < rsi < params["rsi_overbought"]):

            if params.get("check_macd_entry", True) and not pd.isna(macd_hist) and macd_hist > 0:
                pass
            elif params.get("require_htf", False) and htf_trend != "BEARISH":
                pass
            elif htf_trend == "BULLISH":
                pass
            else:
                confidence = min((adx - 25) / 50, 0.3)
                confidence += max(0, min((rsi - 40) / 100, 0.3))
                if htf_trend == "BEARISH":
                    confidence += 0.4
                elif htf_trend == "NEUTRAL":
                    confidence += 0.2
                direction = "SELL"

        if not direction:
            continue

        # Confidence filter
        if confidence < params["min_confidence"]:
            continue

        # Calculate stops
        stop_mult = params["stop_atr_mult"]
        rr_ratio = params["reward_risk"]

        stop_distance = max(atr * stop_mult, MIN_STOP_MAP.get(market, 0))
        limit_distance = stop_distance * rr_ratio

        if direction == "BUY":
            stop_price = close - stop_distance
            limit_price = close + limit_distance
        else:
            stop_price = close + stop_distance
            limit_price = close - limit_distance

        # Position sizing based on risk
        risk_amount = equity * risk_per_trade
        point_value = POINT_VALUE_MAP.get(market, 1.0)
        size = risk_amount / (stop_distance * point_value) if stop_distance > 0 else 0

        # Apply max position size limit
        max_size = params.get("max_size", 10)
        size = min(size, max_size)

        if size <= 0:
            continue

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
        )

    # Close open position at end
    if position:
        position.exit_time = df.iloc[-1]["date"]
        position.exit_price = df.iloc[-1]["close"]
        position.exit_reason = "EOT"
        point_value = POINT_VALUE_MAP.get(market, 1.0)
        if position.direction == "BUY":
            points = position.exit_price - position.entry_price
        else:
            points = position.entry_price - position.exit_price
        position.pnl = points * position.size * point_value
        trades.append(position)

    # Calculate results
    if not trades:
        return BacktestResult(
            params=params, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl=0, max_drawdown=0, profit_factor=0,
            avg_win=0, avg_loss=0, sharpe=0, trades=[]
        )

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(winning) / len(trades) if trades else 0

    avg_win = np.mean([t.pnl for t in winning]) if winning else 0
    avg_loss = np.mean([t.pnl for t in losing]) if losing else 0

    gross_profit = sum(t.pnl for t in winning) if winning else 0
    gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    returns = [t.pnl for t in trades]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

    return BacktestResult(
        params=params,
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        sharpe=sharpe,
        trades=trades,
    )


def optimize_strategy(markets: list[str], days: int = 30, account_size: float = 10000):
    """Run optimization across parameter space."""

    print("=" * 80)
    print("STRATEGY OPTIMIZATION")
    print(f"Account: £{account_size:,.0f} | Period: {days} days | Markets: {', '.join(markets)}")
    print("=" * 80)

    # Fetch data once for all tests
    print("\nFetching market data...")
    market_data = {}
    htf_data = {}

    for market in markets:
        print(f"  {market}...", end=" ")
        df = fetch_data(market, days, "5m")
        htf = fetch_data(market, days, "1h")
        if df is not None and not df.empty:
            market_data[market] = df
            htf_data[market] = htf
            print(f"{len(df)} candles")
        else:
            print("FAILED")

    if not market_data:
        print("No data available!")
        return

    # Parameter space to test
    param_grid = {
        "ema_fast": [5, 9, 12],
        "ema_medium": [15, 21, 26],
        "ema_slow": [40, 50, 60],
        "rsi_period": [7, 14],
        "rsi_oversold": [30],
        "rsi_overbought": [70],
        "rsi_buy_max": [55, 60, 65],
        "rsi_sell_min": [35, 40, 45],
        "adx_threshold": [20, 25, 30],
        "atr_period": [14],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "reward_risk": [1.5, 2.0, 2.5, 3.0],
        "min_confidence": [0.4, 0.5, 0.6],
        "use_macd_exit": [True, False],
        "check_macd_entry": [True, False],
        "require_htf": [False],  # Test with and without
        "max_size": [5],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))

    print(f"\nTesting {len(combinations)} parameter combinations...")

    results = []
    best_pnl = -float('inf')

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Skip invalid EMA combinations
        if params["ema_fast"] >= params["ema_medium"] or params["ema_medium"] >= params["ema_slow"]:
            continue

        # Run backtest across all markets
        combined_result = BacktestResult(
            params=params, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl=0, max_drawdown=0, profit_factor=0,
            avg_win=0, avg_loss=0, sharpe=0, trades=[]
        )

        all_trades = []
        total_max_dd = 0

        for market in market_data:
            result = run_backtest(
                market,
                market_data[market],
                htf_data.get(market),
                params,
                account_size,
                risk_per_trade=0.01
            )

            combined_result.total_trades += result.total_trades
            combined_result.winning_trades += result.winning_trades
            combined_result.losing_trades += result.losing_trades
            combined_result.total_pnl += result.total_pnl
            total_max_dd = max(total_max_dd, result.max_drawdown)
            all_trades.extend(result.trades)

        combined_result.max_drawdown = total_max_dd
        combined_result.trades = all_trades

        if combined_result.total_trades > 0:
            combined_result.win_rate = combined_result.winning_trades / combined_result.total_trades

            winning = [t for t in all_trades if t.pnl > 0]
            losing = [t for t in all_trades if t.pnl <= 0]
            combined_result.avg_win = np.mean([t.pnl for t in winning]) if winning else 0
            combined_result.avg_loss = np.mean([t.pnl for t in losing]) if losing else 0

            gross_profit = sum(t.pnl for t in winning) if winning else 0
            gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
            combined_result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        results.append(combined_result)

        # Progress update
        if combined_result.total_pnl > best_pnl and combined_result.total_trades >= 10:
            best_pnl = combined_result.total_pnl
            print(f"\r  [{idx+1}/{len(combinations)}] New best: £{best_pnl:+,.0f} ({combined_result.total_trades} trades)", end="")

    print("\n")

    # Sort by P&L
    results.sort(key=lambda x: x.total_pnl, reverse=True)

    # Filter to results with enough trades
    valid_results = [r for r in results if r.total_trades >= 10]

    if not valid_results:
        print("No valid results with sufficient trades!")
        return

    # Show top 10 configurations
    print("=" * 80)
    print("TOP 10 STRATEGIES BY P&L")
    print("=" * 80)

    for i, result in enumerate(valid_results[:10]):
        print(f"\n#{i+1}: £{result.total_pnl:+,.0f} P&L | {result.total_trades} trades | "
              f"{result.win_rate:.0%} win | {result.max_drawdown:.1%} DD | PF {result.profit_factor:.1f}")
        print(f"    EMA: {result.params['ema_fast']}/{result.params['ema_medium']}/{result.params['ema_slow']} | "
              f"RSI: {result.params['rsi_period']}p ({result.params['rsi_buy_max']}/{result.params['rsi_sell_min']}) | "
              f"ADX: {result.params['adx_threshold']}")
        print(f"    Stop: {result.params['stop_atr_mult']}x ATR | R:R {result.params['reward_risk']} | "
              f"Conf: {result.params['min_confidence']} | MACD exit: {result.params['use_macd_exit']}")

    # Best result details
    best = valid_results[0]
    print("\n" + "=" * 80)
    print("BEST STRATEGY DETAILS")
    print("=" * 80)

    print(f"\nTotal P&L: £{best.total_pnl:+,.2f}")
    print(f"Return on £{account_size:,}: {best.total_pnl/account_size*100:+.1f}%")
    print(f"Total Trades: {best.total_trades}")
    print(f"Winning: {best.winning_trades} ({best.win_rate:.0%})")
    print(f"Losing: {best.losing_trades}")
    print(f"Avg Win: £{best.avg_win:+,.2f}")
    print(f"Avg Loss: £{best.avg_loss:,.2f}")
    print(f"Profit Factor: {best.profit_factor:.2f}")
    print(f"Max Drawdown: {best.max_drawdown:.1%}")

    print("\nOptimal Parameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    # Trade breakdown by market
    print("\nP&L by Market:")
    market_pnl = {}
    for trade in best.trades:
        if trade.market not in market_pnl:
            market_pnl[trade.market] = {"pnl": 0, "trades": 0, "wins": 0}
        market_pnl[trade.market]["pnl"] += trade.pnl
        market_pnl[trade.market]["trades"] += 1
        if trade.pnl > 0:
            market_pnl[trade.market]["wins"] += 1

    for market, stats in sorted(market_pnl.items(), key=lambda x: -x[1]["pnl"]):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"  {market}: £{stats['pnl']:+,.0f} ({stats['trades']} trades, {wr:.0f}% win)")

    # Show recent trades
    print("\nRecent Trades (last 20):")
    print("-" * 100)
    for trade in best.trades[-20:]:
        time_str = trade.entry_time.strftime("%m/%d %H:%M") if hasattr(trade.entry_time, 'strftime') else str(trade.entry_time)[:16]
        result = "WIN" if trade.pnl > 0 else "LOSS"
        print(f"  {time_str} {trade.market:<15} {trade.direction:4} £{trade.pnl:+8,.2f} [{trade.exit_reason:5}] {result}")

    # Compare to current strategy
    print("\n" + "=" * 80)
    print("COMPARISON: CURRENT vs OPTIMAL")
    print("=" * 80)

    current_params = {
        "ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
        "rsi_period": 7, "rsi_oversold": 30, "rsi_overbought": 70,
        "rsi_buy_max": 60, "rsi_sell_min": 40, "adx_threshold": 25,
        "atr_period": 14, "stop_atr_mult": 1.5, "reward_risk": 2.0,
        "min_confidence": 0.5, "use_macd_exit": True, "check_macd_entry": True,
        "require_htf": False, "max_size": 5,
    }

    current_result = BacktestResult(
        params=current_params, total_trades=0, winning_trades=0, losing_trades=0,
        win_rate=0, total_pnl=0, max_drawdown=0, profit_factor=0,
        avg_win=0, avg_loss=0, sharpe=0, trades=[]
    )

    for market in market_data:
        result = run_backtest(
            market, market_data[market], htf_data.get(market),
            current_params, account_size, risk_per_trade=0.01
        )
        current_result.total_trades += result.total_trades
        current_result.winning_trades += result.winning_trades
        current_result.total_pnl += result.total_pnl
        current_result.max_drawdown = max(current_result.max_drawdown, result.max_drawdown)

    if current_result.total_trades > 0:
        current_result.win_rate = current_result.winning_trades / current_result.total_trades

    print(f"\n{'Metric':<20} {'Current':>15} {'Optimal':>15} {'Improvement':>15}")
    print("-" * 65)
    print(f"{'P&L':<20} £{current_result.total_pnl:>+14,.0f} £{best.total_pnl:>+14,.0f} £{best.total_pnl - current_result.total_pnl:>+14,.0f}")
    print(f"{'Trades':<20} {current_result.total_trades:>15} {best.total_trades:>15} {best.total_trades - current_result.total_trades:>+15}")
    print(f"{'Win Rate':<20} {current_result.win_rate:>14.0%} {best.win_rate:>14.0%} {(best.win_rate - current_result.win_rate)*100:>+14.0f}%")
    print(f"{'Max Drawdown':<20} {current_result.max_drawdown:>14.1%} {best.max_drawdown:>14.1%}")

    improvement = best.total_pnl - current_result.total_pnl
    print(f"\nPotential improvement: £{improvement:+,.0f}/month ({improvement/account_size*100:+.1f}%)")

    return best


if __name__ == "__main__":
    # Run optimization
    markets = ["EUR/USD", "Dollar Index", "S&P 500", "NASDAQ 100", "Gold", "Crude Oil"]
    best = optimize_strategy(markets, days=30, account_size=10000)
