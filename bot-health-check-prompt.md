# Bot Health Check - Claude Code Prompt Template

## Quick Usage

Copy and paste the command below into Claude Code, replacing `[BOT_NAME]` with your specific bot (e.g., `hyperliquid-bot`, `betfair-bot`, etc.):

---

## The Prompt (Copy This)

```
Please perform a comprehensive health check on [BOT_NAME]. Work through each section systematically and provide a summary at the end.

## 1. PROCESS STATUS
- Is the bot process running? Check with `ps aux | grep [bot]` or `docker ps` if containerised
- How long has it been running (uptime)?
- Any recent restarts or crashes?

## 2. LOG ANALYSIS
- Check the last 100 lines of logs for errors, warnings, or anomalies
- Identify any recurring error patterns
- Check log rotation is working (logs not growing unbounded)
- Look for any connection issues (API, websocket, database)

## 3. SIGNAL GENERATION
- Is the bot actively producing trading signals?
- What was the last signal generated and when?
- Are signals being logged/recorded properly?
- Any gaps in signal generation that suggest issues?

## 4. PERFORMANCE METRICS
- Check current trades/positions if applicable
- Review recent P&L or results if logged
- Compare signal frequency to expected baseline
- Any latency issues visible in logs?

## 5. SYSTEM RESOURCES
Run these checks and report findings:
- `free -h` (RAM usage)
- `df -h` (disk space, especially where logs/data stored)
- `top -bn1 | head -20` (CPU usage, load average)
- Check if bot is consuming excessive resources

## 6. CONFIGURATION REVIEW
- Are API keys/credentials still valid (check for auth errors in logs)?
- Any config drift or missing environment variables?
- Is the bot using the intended strategy settings?

## 7. CODE EFFICIENCY REVIEW
- Any obvious inefficiencies in the main loop?
- Memory leaks (growing RAM usage over time)?
- Unnecessary API calls that could be optimised?
- Error handling gaps that could cause silent failures?

## 8. STRATEGY EDGE ASSESSMENT
- Based on logged results, is the strategy performing as expected?
- Any market condition changes that might affect the edge?
- Suggest any parameter tweaks that might improve performance
- Flag if backtesting/paper trading might be needed before changes

## 9. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 10. SUMMARY DASHBOARD
Present a quick status summary:
| Check | Status | Notes |
|-------|--------|-------|
| Process Running | ✅/❌ | |
| Logs Healthy | ✅/⚠️/❌ | |
| Signals Active | ✅/❌ | |
| Resources OK | ✅/⚠️/❌ | |
| Strategy Edge | ✅/⚠️/❌ | |
```

---

## Shorter Version (Quick Check)

If you want a faster, less comprehensive check:

```
Quick health check on [BOT_NAME]:
1. Is process running and for how long?
2. Last 50 log lines - any errors or warnings?
3. Last signal generated and when?
4. RAM/CPU/disk status (free -h, df -h, top snapshot)
5. Any immediate concerns?
Give me a traffic light summary: 🟢 All good / 🟡 Minor issues / 🔴 Needs attention
```

---

## Bot-Specific Variations

### For Hyperliquid/Crypto Bots
Add to the prompt:
```
Also check:
- WebSocket connection status to exchange
- Current open positions and unrealised P&L
- Funding rate considerations if holding perps
- API rate limit usage (are we near limits?)
```

### For Betfair Bot
Add to the prompt:
```
Also check:
- Betfair API session validity
- Current market subscriptions active
- Betting P&L from logs
- Any market suspension handling issues
```

### For OANDA/Forex Bots
Add to the prompt:
```
Also check:
- Spread conditions in recent trades
- Slippage metrics if logged
- Account margin usage
- Any requote or rejection issues
```

---

## Setting Up as a Shell Alias (Optional)

Add this to your `~/.bashrc` or `~/.zshrc` to create a quick command:

```bash
# Bot health check function
# Usage: botcheck hyperliquid-bot
botcheck() {
    if [ -z "$1" ]; then
        echo "Usage: botcheck [bot-name]"
        echo "Example: botcheck hyperliquid-bot"
        return 1
    fi
    
    # This echoes the prompt - you'd paste this into Claude Code
    cat << EOF
Please perform a comprehensive health check on $1. Work through each section:

1. PROCESS STATUS - Is it running? Uptime? Recent crashes?
2. LOG ANALYSIS - Last 100 lines, errors, warnings, log rotation
3. SIGNAL GENERATION - Active signals? Last signal when?
4. PERFORMANCE - Recent P&L, signal frequency, latency
5. SYSTEM RESOURCES - RAM (free -h), disk (df -h), CPU (top)
6. CONFIG - API keys valid? Env vars set? Strategy settings correct?
7. CODE EFFICIENCY - Any obvious optimisations needed?
8. STRATEGY EDGE - Is it performing as expected? Tweaks needed?
9. RECOMMENDATIONS - P1/P2/P3 prioritised list
10. SUMMARY TABLE - Traffic light status for each area

EOF
}
```

---

## Pro Tips

1. **Run weekly**: Set a reminder to run this on each bot weekly
2. **Keep a log**: Save the outputs somewhere so you can track trends
3. **Threshold alerts**: Ask Claude Code to set up actual monitoring scripts that alert you automatically
4. **Before bed check**: Use the "shorter version" for quick daily checks

---

## Next Level: Automated Monitoring Script

If you want, ask Claude Code to build an actual monitoring script that:
- Runs these checks automatically on a schedule
- Outputs to a daily report file
- Sends alerts (email/Telegram) if critical issues found

Just ask: "Build me an automated monitoring script based on these health checks that runs every 6 hours and alerts me via [Telegram/email] if there are P1 issues"
