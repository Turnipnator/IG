#!/bin/bash
# rebuild-watcher.sh
# Watches for a rebuild trigger from the Telegram /rebuild command.
# Runs on the HOST (not inside Docker). Install as a systemd service:
# see rebuild-watcher.service.
#
# PARITY GUARANTEE
# ----------------
# This script syncs the deployment to upstream `main` with a *hard reset*,
# not a `git pull`. A plain pull silently fails on any local drift (an edited
# config, a stray commit, an uncommitted change) and leaves the bot running
# STALE code while still reporting success. Hard reset wipes local drift so
# every bot that runs /rebuild ends up byte-identical to origin/main.
#
# If the sync fails (network, git error), the rebuild is ABORTED — the bot
# keeps running its current code rather than rebuilding from a half-synced
# tree. Failures are pushed to Telegram (best-effort) so they are never silent.
#
# NOTE: data/ and .env are gitignored, so the hard reset never touches runtime
# state, the trade journal, the candle cache, or your credentials.

BOT_DIR="/root/ig-bot"
TRIGGER_FILE="$BOT_DIR/data/rebuild_trigger"
UPSTREAM_REMOTE="origin"
UPSTREAM_BRANCH="main"

# Best-effort Telegram notification from the host. Sources the bot's own
# .env so we reuse its token/chat id. Never fails the script.
notify() {
    local text="$1"
    [ -f "$BOT_DIR/.env" ] || return 0
    local token chat
    token=$(grep -E '^TELEGRAM_BOT_TOKEN=' "$BOT_DIR/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'[:space:]')
    chat=$(grep -E '^TELEGRAM_CHAT_ID=' "$BOT_DIR/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'[:space:]')
    [ -n "$token" ] && [ -n "$chat" ] || return 0
    curl -s -m 10 -o /dev/null \
        "https://api.telegram.org/bot${token}/sendMessage" \
        --data-urlencode "chat_id=${chat}" \
        --data-urlencode "text=${text}" \
        --data-urlencode "parse_mode=Markdown" || true
}

echo "[rebuild-watcher] Watching for rebuild triggers (hard-reset parity mode)..."

while true; do
    if [ -f "$TRIGGER_FILE" ]; then
        echo "[rebuild-watcher] Rebuild triggered at $(cat "$TRIGGER_FILE")"
        rm -f "$TRIGGER_FILE"

        cd "$BOT_DIR" || { echo "[rebuild-watcher] BOT_DIR missing"; sleep 5; continue; }

        # 1. Fetch upstream. Abort the rebuild on failure — never build stale.
        echo "[rebuild-watcher] Fetching $UPSTREAM_REMOTE/$UPSTREAM_BRANCH..."
        if ! git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" 2>&1; then
            echo "[rebuild-watcher] git fetch FAILED — aborting, staying on current code"
            notify "❌ *Rebuild aborted*: git fetch failed. Bot left on current code (no stale build)."
            continue
        fi

        # 2. Hard reset to upstream — discards ALL local drift for guaranteed parity.
        echo "[rebuild-watcher] Hard-resetting to $UPSTREAM_REMOTE/$UPSTREAM_BRANCH..."
        if ! git reset --hard "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" 2>&1; then
            echo "[rebuild-watcher] git reset FAILED — aborting, staying on current code"
            notify "❌ *Rebuild aborted*: git reset failed. Bot left on current code (no stale build)."
            continue
        fi
        git clean -fd 2>&1   # remove untracked files so the tree matches upstream exactly

        HEAD_DESC=$(git log -1 --pretty=format:'%h %s' 2>/dev/null)
        echo "[rebuild-watcher] Now at: $HEAD_DESC"

        # 3. Rebuild + restart. Clean image to avoid Docker caching old code.
        echo "[rebuild-watcher] Rebuilding container..."
        docker compose down 2>&1
        docker compose build --no-cache 2>&1
        docker compose up -d 2>&1

        echo "[rebuild-watcher] Rebuild complete at $(date)"
        notify "✅ *Rebuild complete* — synced to \`${HEAD_DESC}\`. Container restarted."
    fi
    sleep 5
done
