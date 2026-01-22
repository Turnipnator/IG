> **Common Patterns**: See `~/trading-bot-skill.md` for deployment, Docker, Telegram, and strategy patterns shared across all trading bots.

---

# Automated Trading Platform Guide for IG Markets

## Overview

This document outlines a trading strategy for an automated trading platform using IG Markets as the broker. The strategy focuses on CFDs/Spread Betting across Indices, Forex, and Commodities sectors. This guide provides high-level strategy, implementation insights, and code skeletons to build the platform.

The platform will use real-time data integration, analysis layers, and automated execution via IG's REST API. Remember, this is for educational purposes—trading involves risk, especially with leveraged products like CFDs. Backtest thoroughly and understand the risks.

## Key Components of the Strategy

### Portfolio Focus

- **Markets**: Indices (e.g., UK100, US500, Germany40), Forex (e.g., GBP/USD, EUR/USD, USD/JPY), Commodities (e.g., Gold, Silver, Oil).
- **Allocation**: Aim for 30-40% in each sector to balance volatility. Rebalance based on market conditions.
- **Position Sizing**: Risk no more than 1-2% of account per trade (CFDs are leveraged).
- **Trade Type**: Both long and short positions available with CFDs.

### Data Integration

Use APIs for real-time feeds:

- **Market Data**: IG's Streaming API for real-time prices, or REST API for historical data.
- **News/Sentiment**: Alpha Vantage or NewsAPI; parse with NLP for sentiment scores.
- **Economic Calendar**: Monitor high-impact events (NFP, interest rate decisions, etc.).

### Analysis Layers

- **Technical Analysis**: Use indicators like Moving Averages (SMA/EMA), RSI, MACD to identify buy/sell signals.
- **Fundamental Analysis**: For Forex, monitor interest rate differentials; for indices, track earnings seasons.
- **Sentiment Analysis**: Analyze news and market sentiment for directional bias.
- **Risk Management**: Implement stop-loss (use guaranteed stops for volatile markets), take-profit, and leverage limits.

### Decision Engine

**Hybrid**: Rules-based for basics + ML for advanced predictions.

**Example Logic**:

- **Buy**: If 50-period EMA > 200-period EMA (bullish trend) AND RSI < 70 AND sentiment positive.
- **Sell**: If RSI > 70 (overbought) OR negative news event OR drawdown > threshold.

### Execution

- **Broker**: IG Markets via their REST API.
- **Automation**: Run on a schedule (e.g., every 5 minutes) using cron jobs or a framework like Apache Airflow.

## Implementation Steps

### Tech Stack

- **Language**: Python 3.12+
- **Libraries**:
  - **Data**: pandas, numpy
  - **Technicals**: Custom indicators (pure Python/NumPy) or TA-Lib
  - **APIs**: requests (for IG REST API), trading_ig (community library)
  - **ML**: scikit-learn or PyTorch for sentiment/models
  - **Logging**: logging module which is circular to not fill HDD
- **Environment**: Deploy on VPS (Contabo) as well as local Macbook for initial development. Use Github for versioning and use Docker for portability.

### IG API Authentication

IG uses a two-step authentication:
1. Login to get CST and X-SECURITY-TOKEN headers
2. Use these headers for all subsequent requests

### Code Skeleton

Below is a basic Python script structure. Expand as needed.

```python
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime

# IG API Configuration
IG_API_KEY = "your_api_key"
IG_USERNAME = "your_username"
IG_PASSWORD = "your_password"
IG_ACC_TYPE = "DEMO"  # or "LIVE"

# API URLs
IG_BASE_URL = "https://demo-api.ig.com/gateway/deal" if IG_ACC_TYPE == "DEMO" else "https://api.ig.com/gateway/deal"

class IGClient:
    def __init__(self):
        self.session = requests.Session()
        self.cst = None
        self.security_token = None

    def login(self):
        """Authenticate with IG API."""
        headers = {
            "Content-Type": "application/json",
            "X-IG-API-KEY": IG_API_KEY,
            "Version": "2"
        }
        payload = {
            "identifier": IG_USERNAME,
            "password": IG_PASSWORD
        }

        response = self.session.post(
            f"{IG_BASE_URL}/session",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            self.cst = response.headers.get("CST")
            self.security_token = response.headers.get("X-SECURITY-TOKEN")
            logging.info("Successfully logged in to IG")
            return True
        else:
            logging.error(f"Login failed: {response.text}")
            return False

    def get_headers(self):
        """Get authenticated headers."""
        return {
            "Content-Type": "application/json",
            "X-IG-API-KEY": IG_API_KEY,
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.security_token,
            "Version": "2"
        }

    def get_market_info(self, epic):
        """Get market information for an instrument."""
        response = self.session.get(
            f"{IG_BASE_URL}/markets/{epic}",
            headers=self.get_headers()
        )
        return response.json() if response.status_code == 200 else None

    def get_historical_prices(self, epic, resolution="MINUTE_5", num_points=100):
        """
        Fetch historical price data.

        Resolutions: SECOND, MINUTE, MINUTE_2, MINUTE_3, MINUTE_5,
                     MINUTE_10, MINUTE_15, MINUTE_30, HOUR, HOUR_2,
                     HOUR_3, HOUR_4, DAY, WEEK, MONTH
        """
        response = self.session.get(
            f"{IG_BASE_URL}/prices/{epic}?resolution={resolution}&max={num_points}&pageSize={num_points}",
            headers=self.get_headers()
        )

        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", [])

            df = pd.DataFrame([{
                "date": p["snapshotTime"],
                "open": (p["openPrice"]["bid"] + p["openPrice"]["ask"]) / 2,
                "high": (p["highPrice"]["bid"] + p["highPrice"]["ask"]) / 2,
                "low": (p["lowPrice"]["bid"] + p["lowPrice"]["ask"]) / 2,
                "close": (p["closePrice"]["bid"] + p["closePrice"]["ask"]) / 2,
                "volume": p.get("lastTradedVolume", 0)
            } for p in prices])

            return df
        return None

    def open_position(self, epic, direction, size, stop_distance=None, limit_distance=None):
        """
        Open a new position.

        Args:
            epic: Instrument identifier (e.g., "IX.D.FTSE.DAILY.IP")
            direction: "BUY" or "SELL"
            size: Position size
            stop_distance: Stop loss distance in points
            limit_distance: Take profit distance in points
        """
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "orderType": "MARKET",
            "guaranteedStop": False,
            "forceOpen": True
        }

        if stop_distance:
            payload["stopDistance"] = stop_distance
        if limit_distance:
            payload["limitDistance"] = limit_distance

        response = self.session.post(
            f"{IG_BASE_URL}/positions/otc",
            json=payload,
            headers=self.get_headers()
        )

        return response.json() if response.status_code == 200 else None

    def close_position(self, deal_id, direction, size):
        """Close an existing position."""
        # IG requires opposite direction to close
        close_direction = "SELL" if direction == "BUY" else "BUY"

        headers = self.get_headers()
        headers["_method"] = "DELETE"

        payload = {
            "dealId": deal_id,
            "direction": close_direction,
            "size": size,
            "orderType": "MARKET"
        }

        response = self.session.post(
            f"{IG_BASE_URL}/positions/otc",
            json=payload,
            headers=headers
        )

        return response.json() if response.status_code == 200 else None

    def get_positions(self):
        """Get all open positions."""
        response = self.session.get(
            f"{IG_BASE_URL}/positions",
            headers=self.get_headers()
        )
        return response.json() if response.status_code == 200 else None


# Define Markets (IG EPICs)
markets = {
    'Indices': [
        'IX.D.FTSE.DAILY.IP',    # UK 100
        'IX.D.SPTRD.DAILY.IP',   # US 500
        'IX.D.DAX.DAILY.IP'      # Germany 40
    ],
    'Forex': [
        'CS.D.GBPUSD.TODAY.IP',  # GBP/USD
        'CS.D.EURUSD.TODAY.IP',  # EUR/USD
        'CS.D.USDJPY.TODAY.IP'   # USD/JPY
    ],
    'Commodities': [
        'CS.D.USCGC.TODAY.IP',   # Gold
        'CS.D.USCSI.TODAY.IP',   # Silver
        'CC.D.CL.UNC.IP'         # Crude Oil
    ]
}

# Technical Analysis (same as IBKR bot)
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def analyze_technicals(df):
    df['ema_9'] = calculate_ema(df['close'], 9)
    df['ema_21'] = calculate_ema(df['close'], 21)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['rsi'] = calculate_rsi(df['close'], 7)
    return df

# Decision Logic
def make_decision(df):
    latest = df.iloc[-1]

    ema_9 = latest['ema_9']
    ema_21 = latest['ema_21']
    ema_50 = latest['ema_50']
    rsi = latest['rsi']
    close = latest['close']

    # Bullish: EMAs aligned upward, RSI not overbought
    if ema_9 > ema_21 > ema_50 and close > ema_50 and rsi < 70:
        return 'BUY'
    # Bearish: EMAs aligned downward, RSI not oversold
    elif ema_9 < ema_21 < ema_50 and close < ema_50 and rsi > 30:
        return 'SELL'
    return 'HOLD'

# Main Loop
def run_trader():
    client = IGClient()
    if not client.login():
        return

    for sector, epics in markets.items():
        for epic in epics:
            df = client.get_historical_prices(epic, resolution="MINUTE_5", num_points=100)
            if df is None or df.empty:
                continue

            df = analyze_technicals(df)
            decision = make_decision(df)

            logging.info(f"{epic}: {decision}")

            if decision == 'BUY':
                # Open long position with stop and limit
                client.open_position(
                    epic=epic,
                    direction="BUY",
                    size=1,  # Minimum size, adjust based on risk
                    stop_distance=50,  # Points
                    limit_distance=30   # Points
                )
            elif decision == 'SELL':
                # Open short position
                client.open_position(
                    epic=epic,
                    direction="SELL",
                    size=1,
                    stop_distance=50,
                    limit_distance=30
                )

# Run periodically
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_trader()
```

## Risk and Backtesting

- **Backtest**: Use historical data from IG to simulate trades. Calculate Sharpe Ratio, max drawdown.
- **Leverage Risk**: CFDs are leveraged products. A small move can result in significant gains or losses.
- **Edge Cases**: Handle API failures, market halts, weekends, rollover periods.
- **Margin**: Monitor margin requirements and avoid margin calls.

## IG-Specific Considerations

### EPICs
IG uses "EPICs" as instrument identifiers. Find them via:
- IG's market search API
- The web platform (inspect network requests)

### Market Hours
Different markets have different trading hours:
- Indices: Usually follow exchange hours
- Forex: 24/5 (Sunday evening to Friday evening)
- Commodities: Vary by product

### Spread Betting vs CFDs
- **Spread Betting**: Tax-free profits in UK, stake per point
- **CFDs**: Taxable, trade in contracts

### Guaranteed Stops
IG offers guaranteed stops (for a premium) which protect against slippage in volatile markets.

## Next Steps

1. Apply for IG API access (demo account first)
2. Test in IG demo account for at least a few weeks
3. Integrate the technical analysis from IBKR bot (can reuse indicators.py)
4. Add detailed Telegram notifications
5. Monitor: Use same Telegram bot setup as IBKR for notifications

---
