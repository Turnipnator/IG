---
name: healthcheck
description: Run a comprehensive health check on the IG trading bot
---

# IG Trading Bot Health Check

Run a comprehensive health check on the ig-trading-bot. Work through each section systematically and provide a summary dashboard at the end.

## VPS Details
- Server: 149.102.144.190
- SSH Key: ~/.ssh/id_ed25519_vps
- Container: ig-trading-bot

## 1. PROCESS STATUS
- Is the bot process running? Check with `docker ps`
- How long has it been running (uptime)?
- Any recent restarts or crashes?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' | grep ig"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of logs for errors, warnings, or anomalies
- Identify any recurring error patterns
- Look for any connection issues (API, websocket)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs ig-trading-bot --tail 100 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs ig-trading-bot 2>&1 | grep -iE 'error|warn|fail|exception' | tail -20"
```

## 3. SIGNAL GENERATION
- Is the bot actively producing trading signals?
- What was the last signal generated and when?
- Check data files for recent activity

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "ls -la /root/IG/data/ 2>/dev/null || ls -la /opt/IG/data/ 2>/dev/null"
```

## 4. PERFORMANCE METRICS
- Check current trades/positions
- Review recent P&L if logged
- Check IG API connectivity

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker exec ig-trading-bot cat /app/data/trading_stats.json 2>/dev/null || echo 'No stats file'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker exec ig-trading-bot cat /app/data/active_positions.json 2>/dev/null || echo 'No positions file'"
```

## 5. SYSTEM RESOURCES
- RAM usage, disk space, CPU usage

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h && echo '---' && df -h / && echo '---' && top -bn1 | head -12"
```

## 6. CONFIGURATION REVIEW
- Check key environment variables are set correctly

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E 'ENABLE_|MODE|STRATEGY' /root/IG/.env 2>/dev/null | head -15"
```

## 7. IG-SPECIFIC CHECKS
- Spread conditions in recent trades
- Slippage metrics if logged
- Account margin usage
- Any requote or rejection issues

## 8. STRATEGY EDGE ASSESSMENT
- Calculate win rate from stats
- Is the strategy performing as expected?
- Any parameter tweaks recommended?

## 9. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 10. SUMMARY DASHBOARD
Present a quick status summary table:

| Check | Status | Notes |
|-------|--------|-------|
| Process Running | ?/? | |
| Logs Healthy | ?/?/? | |
| Signals Active | ?/? | |
| Resources OK | ?/?/? | |
| Strategy Edge | ?/?/? | |

Traffic light summary: ? All good / ? Minor issues / ? Needs attention
