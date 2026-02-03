"""
Technical indicators for trading strategy.
Pure Python/NumPy implementations.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        series: Price series (typically close)
        period: Number of periods

    Returns:
        SMA series
    """
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        series: Price series (typically close)
        period: Number of periods

    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        series: Price series (typically close)
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        series: Price series (typically close)
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        series: Price series (typically close)
        period: SMA period
        std_dev: Number of standard deviations

    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle = calculate_sma(series, period)
    std = series.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).

    ADX measures trend strength regardless of direction.
    Values above 25 indicate a strong trend, below 20 indicates ranging.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ADX period (default 14)

    Returns:
        ADX series (0-100)
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Smoothed averages (Wilder's smoothing)
    atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr

    # DX and ADX (guard against division by zero when +DI and -DI both zero)
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period
        d_period: %D period

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return k, d


def add_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data
        params: Strategy parameters dict

    Returns:
        DataFrame with indicators added
    """
    df = df.copy()

    # EMAs
    df["ema_fast"] = calculate_ema(df["close"], params.get("ema_fast", 9))
    df["ema_medium"] = calculate_ema(df["close"], params.get("ema_medium", 21))
    df["ema_slow"] = calculate_ema(df["close"], params.get("ema_slow", 50))

    # RSI
    df["rsi"] = calculate_rsi(df["close"], params.get("rsi_period", 7))

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])

    # Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(
        df["close"]
    )

    # ATR for volatility-based stops
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"])

    # ADX (trend strength)
    df["adx"] = calculate_adx(df["high"], df["low"], df["close"])

    # Stochastic
    df["stoch_k"], df["stoch_d"] = calculate_stochastic(
        df["high"], df["low"], df["close"]
    )

    return df
