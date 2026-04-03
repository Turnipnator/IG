#!/bin/bash
# rebuild-watcher.sh
# Watches for rebuild trigger from Telegram /rebuild command.
# Runs on the HOST (not inside Docker).
# Install as systemd service: see rebuild-watcher.service

BOT_DIR="/root/ig-bot"
TRIGGER_FILE="$BOT_DIR/data/rebuild_trigger"

echo "[rebuild-watcher] Watching for rebuild triggers..."

while true; do
    if [ -f "$TRIGGER_FILE" ]; then
        echo "[rebuild-watcher] Rebuild triggered at $(cat "$TRIGGER_FILE")"
        rm -f "$TRIGGER_FILE"

        cd "$BOT_DIR" || exit 1

        echo "[rebuild-watcher] Pulling latest code..."
        git pull origin main 2>&1

        echo "[rebuild-watcher] Rebuilding container..."
        docker compose down 2>&1
        docker compose build --no-cache 2>&1
        docker compose up -d 2>&1

        echo "[rebuild-watcher] Rebuild complete at $(date)"
    fi
    sleep 5
done
