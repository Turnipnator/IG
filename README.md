# IG Spread Betting Bot

Automated spread betting platform for IG Markets. Analyses markets using technical indicators and executes trades automatically with built-in risk management.

## Features

- **6 Markets**: S&P 500, NASDAQ 100, Crude Oil, Dollar Index (DXY), EUR/USD, Gold
- **Technical Strategy**: EMA crossover + RSI with confidence scoring
- **Risk Management**: Position sizing based on account percentage, max position limits
- **Telegram Alerts**: Real-time notifications for signals, trades, and daily summaries
- **Docker Support**: Easy deployment to VPS or local machine

## Prerequisites

- Python 3.12+
- IG Markets account (Demo or Live)
- IG API key from [IG Labs](https://labs.ig.com/)
- Telegram bot (optional, for notifications)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Turnipnator/IG.git
cd IG
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

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

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ENABLED=true

# Trading Configuration
RISK_PER_TRADE=0.01    # 1% of account per trade
MAX_POSITIONS=5         # Maximum concurrent positions
TRADING_ENABLED=false   # Set to true when ready to trade
CHECK_INTERVAL=5        # Minutes between market checks
```

### 4. Get Your IG API Key

1. Go to [IG Labs](https://labs.ig.com/)
2. Log in with your IG account (use Demo account first!)
3. Create a new API key
4. Copy the key to your `.env` file

### 5. Set Up Telegram Notifications (Optional)

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the bot token to `TELEGRAM_BOT_TOKEN`
4. Start a chat with your new bot and send any message
5. Run this to get your chat ID:
   ```bash
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
   ```
6. Copy your chat ID to `TELEGRAM_CHAT_ID`

### 6. Verify Setup

```bash
python verify_epics.py  # Check market connections
python test_run.py      # Run a test cycle (no trades)
```

### 7. Start Trading

Once you're confident everything works:

```bash
# Enable trading in .env
TRADING_ENABLED=true

# Run the bot
python main.py
```

## Docker Deployment

Build and run with Docker:

```bash
docker-compose up -d
```

View logs:

```bash
docker-compose logs -f
```

Stop:

```bash
docker-compose down
```

## Strategy

### Entry Signals

**BUY (Long)**:
- Fast EMA (9) > Medium EMA (21) > Slow EMA (50)
- Price above Slow EMA
- RSI below 70 (not overbought)

**SELL (Short)**:
- Fast EMA (9) < Medium EMA (21) < Slow EMA (50)
- Price below Slow EMA
- RSI above 30 (not oversold)

### Exit Signals

- RSI reaches overbought (>70) or oversold (<30)
- MACD histogram crosses zero against position
- Stop loss or take profit hit

### Confidence Scoring

Each signal gets a confidence score (0-100%) based on:
- EMA separation (trend strength)
- RSI distance from threshold
- MACD confirmation

Trades only execute when confidence exceeds 50%.

## Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| Risk per trade | 1% | Maximum account % to risk per trade |
| Max positions | 5 | Maximum concurrent open positions |
| Stop loss | ATR-based | Dynamic stops based on volatility |
| Take profit | 1.5x stop | Reward:risk ratio of 1.5:1 |
| Daily loss limit | 5% | Trading pauses if daily loss exceeds this |

### Position Sizing

Position size is calculated to risk exactly 1% of account:

```
Size = (Account Balance × Risk %) / Stop Distance
```

## Markets

| Market | EPIC | Min Stop | Notes |
|--------|------|----------|-------|
| S&P 500 | IX.D.SPTRD.DAILY.IP | 1.0 | US equity index |
| NASDAQ 100 | IX.D.NASDAQ.CASH.IP | 4.0 | US tech index |
| Crude Oil | CC.D.CL.UNC.IP | 12.0 | WTI crude |
| Dollar Index | CC.D.DX.UMP.IP | 20.0 | DXY basket |
| EUR/USD | CS.D.EURUSD.TODAY.IP | 2.0 | Major forex pair |
| Gold | CS.D.USCGC.TODAY.IP | 1.0 | Spot gold |

### Market Hours (UK Time)

- **Forex (EUR/USD)**: Sunday 10pm - Friday 10pm
- **Gold**: Sunday 11pm - Friday 10pm
- **US Indices**: Monday-Friday, ~2:30pm - 9pm (cash hours vary)
- **Crude Oil**: Sunday 11pm - Friday 10pm

## Project Structure

```
IG/
├── .env                    # Your credentials (git ignored)
├── .env.example            # Template for credentials
├── config.py               # Configuration & market definitions
├── main.py                 # Main entry point
├── test_run.py             # Test without trading
├── verify_epics.py         # Verify market EPICs
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build
├── docker-compose.yml      # Container orchestration
└── src/
    ├── client.py           # IG API client
    ├── indicators.py       # Technical indicators (EMA, RSI, MACD, etc.)
    ├── strategy.py         # Trading strategy & signals
    ├── risk_manager.py     # Position sizing & risk controls
    ├── telegram_bot.py     # Notification system
    └── utils.py            # Logging & helpers
```

## Configuration

### Adding New Markets

Edit `config.py` and add to the `MARKETS` list:

```python
MarketConfig(
    epic="XX.D.XXXXX.XXXXX.IP",  # Find via verify_epics.py
    name="Market Name",
    sector="Indices",  # or "Forex", "Commodities"
    min_stop_distance=5.0,
    default_size=1.0,
),
```

### Adjusting Strategy Parameters

Edit `STRATEGY_PARAMS` in `config.py`:

```python
STRATEGY_PARAMS = {
    "ema_fast": 9,       # Fast EMA period
    "ema_medium": 21,    # Medium EMA period
    "ema_slow": 50,      # Slow EMA period
    "rsi_period": 7,     # RSI calculation period
    "rsi_overbought": 70,
    "rsi_oversold": 30,
}
```

## Troubleshooting

### Login Failed

- **Invalid identifier**: Use your IG username, not email
- **Invalid details**: Check password is correct
- **Account migrated**: Regenerate API key at IG Labs
- **API key invalid**: Ensure key matches account type (Demo/Live)

### Market Not Tradeable

- **EDITS_ONLY**: Market is closed (weekend/out of hours)
- **OFFLINE**: Market temporarily unavailable
- Check market hours above

### No Signals Generated

- Markets may be ranging (no clear trend)
- RSI may be in overbought/oversold territory
- Check logs for specific reasons

### Telegram Not Working

- Verify bot token is correct
- Ensure you've started a chat with the bot
- Check `TELEGRAM_ENABLED=true` in `.env`

## Safety Features

1. **Kill Switch**: Set `TRADING_ENABLED=false` to stop all trading
2. **Demo Mode**: Always test with `IG_ACC_TYPE=DEMO` first
3. **Max Positions**: Limits total exposure
4. **Daily Loss Limit**: Pauses trading if losses exceed 5%
5. **Session Refresh**: Auto-refreshes IG session every 6 hours

## Disclaimer

This bot is for educational purposes. Trading CFDs and spread bets carries significant risk. You can lose more than your initial deposit. Past performance is not indicative of future results. Only trade with money you can afford to lose.

## License

MIT
