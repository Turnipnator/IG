# IG Spread Betting Bot

Automated spread betting platform for IG Markets. Uses real-time Lightstreamer streaming for price data, technical indicators for signal generation, and built-in risk management with break-even and ATR trailing stops.

## Features

- **6 Markets**: S&P 500, NASDAQ 100, Gold, EUR/USD, Dollar Index (DXY), Crude Oil
- **Real-Time Streaming**: Lightstreamer for live prices (free, doesn't count against API limits)
- **Two Strategy Profiles**:
  - *Momentum* (Indices): Fast EMAs, MACD exit, 2:1 R:R
  - *Big Winners* (Forex/Commodities): High R:R (4:1), no MACD exit, let winners run
- **Smart Filters**: ADX trend strength, pullback-to-EMA, HTF alignment, market regime
- **Trailing Stops**: Break-even stop at 50% of target, then ATR trail ratchets profit
- **Risk Management**: ATR-based position sizing, max position limits, daily loss limit, loss cooldowns
- **Telegram Bot**: Real-time alerts, remote control (/status, /positions, /emergency)
- **API-Friendly**: Disk caching, weekend detection, rate limiting (stays well under IG's 10k/week limit)
- **Docker Support**: One-command deployment with persistent data volumes

## Prerequisites

- Python 3.12+
- IG Markets account (Demo or Live)
- IG API key from [IG Labs](https://labs.ig.com/)
- Telegram bot (optional, for notifications)

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Turnipnator/IG.git
cd IG
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# IG Markets API Configuration
IG_API_KEY=your_api_key_here
IG_USERNAME=your_username_here
IG_PASSWORD=your_password_here
IG_ACC_TYPE=DEMO  # or LIVE

# Telegram Bot Configuration (set TELEGRAM_ENABLED=false to run without Telegram)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ENABLED=true

# Trading Configuration
RISK_PER_TRADE=0.01     # 1% of account per trade
MAX_POSITIONS=5          # Maximum concurrent positions
TRADING_ENABLED=true     # Set to false to disable trade execution in main.py

# API Rate Limiting (IG has 10,000 data points/week limit)
CHECK_INTERVAL=60        # Minutes between market checks (polling mode fallback)
PRICE_DATA_POINTS=50     # Historical candles to fetch (50 saves allowance vs 100)
CACHE_TTL_MINUTES=55     # How long to cache price data before re-fetching
```

> **No Telegram?** Set `TELEGRAM_ENABLED=false` and leave the token/chat_id as placeholders. The bot runs fine without it.

### 3. Get Your IG API Key

1. Go to [IG Labs](https://labs.ig.com/)
2. Log in with your IG account (use Demo account first!)
3. Create a new API key
4. Copy the key to your `.env` file

### 4. Set Up Telegram Notifications (Optional)

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the bot token to `TELEGRAM_BOT_TOKEN`
4. Start a chat with your new bot and send any message
5. Run this to get your chat ID:
   ```bash
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
   ```
6. Copy your chat ID to `TELEGRAM_CHAT_ID`

### 5. Verify Setup

```bash
python verify_epics.py  # Check IG API connection and market EPICs
```

### 6. Test Run (Safe - No Trades)

```bash
python test_run.py
```

This analyses all 6 markets, shows signals with confidence scores, and sends Telegram alerts if configured. **It never executes trades** regardless of the `TRADING_ENABLED` setting - it's purely read-only.

### 7. Start the Production Bot

```bash
python main.py
```

This starts the full bot with Lightstreamer streaming, real-time analysis on candle completion, and automated trade execution. Use `TRADING_ENABLED=false` in `.env` to observe signals without executing trades.

## Docker Deployment

Build and run:

```bash
docker compose up -d --build
```

View logs:

```bash
docker compose logs -f
```

Stop:

```bash
docker compose down
```

Data persists across rebuilds via Docker volumes (`./logs` and `./data`).

## Strategy

### Two Profiles

| Parameter | Momentum (Indices) | Big Winners (Forex/Commodities) |
|-----------|-------------------|-------------------------------|
| Markets | S&P 500, NASDAQ 100 | Gold, EUR/USD, DXY, Crude Oil |
| EMAs | 5 / 12 / 26 | 9 / 21 / 50 |
| R:R | 2:1 | 4:1 |
| ADX Threshold | 23 | 25 |
| MACD Exit | Yes | No |
| HTF Required | Yes | Yes |

### Entry Conditions

All must be true:
- EMA alignment (fast > medium > slow for BUY, reversed for SELL)
- Price above/below slow EMA
- RSI in valid range (not overbought/oversold)
- ADX above threshold (trend is strong enough)
- ADX not declining (trend not weakening)
- Price within pullback distance of fast EMA (not chasing)
- Higher timeframe trend aligned
- Market regime allows direction (S&P 500 sets regime for all markets)

### Exit Conditions

- **Indices**: MACD histogram opposite for 3 consecutive candles
- **Forex/Commodities**: ADX drops below threshold-3 (market turned ranging) or HTF reversal
- **All**: RSI extreme (>70 overbought, <30 oversold), stop loss, or take profit

### Trailing Stop System

1. **Entry**: Stop set at ATR x 1.5 (forex/commodities: ATR x 1.8)
2. **Break-Even**: When profit reaches 50% of stop distance, stop moves to entry price
3. **ATR Trail**: After break-even, stop continuously ratchets behind price at ATR x 1.5 distance. Never moves backwards. Minimum 20% ATR move between updates.

## Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| Risk per trade | 1% | Maximum account % to risk per trade |
| Max positions | 5 | Maximum concurrent open positions |
| Stop loss | ATR-based | Dynamic stops based on market volatility |
| Daily loss limit | 5% | Trading pauses if daily loss exceeds this |
| Loss cooldown | 60 min | Cooldown after a losing trade on same market |
| Entry cooldown | 30 min | Cooldown after any close before re-entering |

## Markets

| Market | EPIC | Strategy | Candle |
|--------|------|----------|--------|
| S&P 500 | IX.D.SPTRD.DAILY.IP | Momentum | 5 min |
| NASDAQ 100 | IX.D.NASDAQ.CASH.IP | Momentum | 5 min |
| Gold | CS.D.USCGC.TODAY.IP | Big Winners | 5 min |
| EUR/USD | CS.D.EURUSD.TODAY.IP | Big Winners | 15 min |
| Dollar Index | CO.D.DX.Month1.IP | Big Winners | 15 min |
| Crude Oil | EN.D.CL.Month1.IP | Big Winners | 15 min |

> **Note**: EPICs are for spread betting accounts. CFD accounts use different EPICs (CC.D.* prefix) which don't support Lightstreamer streaming on spread bet accounts.

## Project Structure

```
IG/
├── .env                    # Your credentials (git ignored)
├── .env.example            # Template for credentials
├── config.py               # Configuration, strategy profiles & market definitions
├── main.py                 # Production bot (streaming + auto-trading)
├── test_run.py             # Test script (analysis only, no trades)
├── verify_epics.py         # Verify market EPICs via IG API
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build
├── docker-compose.yml      # Container orchestration
└── src/
    ├── client.py           # IG REST API client (auth, orders, positions)
    ├── streaming.py        # Lightstreamer real-time price streaming
    ├── indicators.py       # Technical indicators (EMA, RSI, MACD, ADX, ATR)
    ├── strategy.py         # Trading strategy & signal generation
    ├── risk_manager.py     # Position sizing & risk controls
    ├── regime.py           # Market regime classification
    ├── calendar.py         # Economic calendar integration
    ├── telegram_bot.py     # Telegram bot (async) + notifier (sync)
    └── utils.py            # Logging & helpers
```

## Troubleshooting

### Login Failed

- **Invalid identifier**: Use your IG username, not email
- **Invalid details**: Check password is correct
- **Account migrated**: Regenerate API key at IG Labs
- **API key invalid**: Ensure key matches account type (Demo/Live)

### No Signals Generated

- Markets may be ranging (ADX below threshold) - this is normal
- RSI may be in overbought/oversold territory
- Higher timeframe trend may not be aligned
- Check logs for specific hold reasons (logged every candle)

### API Allowance Exceeded

- IG allows 10,000 historical data points per rolling 7-day window
- The bot uses disk caching and streaming to minimize API usage (~630 pts/week normal operation)
- If exceeded, the bot continues with streaming data and cached candles
- Allowance recovers automatically over the 7-day rolling window

### Telegram Not Working

- Verify bot token is correct
- Ensure you've started a chat with the bot and sent at least one message
- Check `TELEGRAM_ENABLED=true` in `.env`
- Set `TELEGRAM_ENABLED=false` to run without Telegram entirely

## Safety Features

1. **Kill Switch**: Set `TRADING_ENABLED=false` to stop all trade execution
2. **Demo Mode**: Always test with `IG_ACC_TYPE=DEMO` first
3. **Max Positions**: Limits total exposure
4. **Daily Loss Limit**: Pauses trading if losses exceed 5%
5. **Loss Cooldown**: 60-minute cooldown after a losing trade
6. **Market Regime**: Only trades in the direction of S&P 500 trend
7. **Session Refresh**: Auto-refreshes IG session every 6 hours
8. **Graceful Shutdown**: Saves candle data to disk on SIGTERM/SIGINT

## Disclaimer

This bot is for educational purposes. Trading CFDs and spread bets carries significant risk. You can lose more than your initial deposit. Past performance is not indicative of future results. Only trade with money you can afford to lose.

## License

MIT
