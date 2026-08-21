#!/bin/bash
# watchdog.sh
# Out-of-band liveness watchdog for the IG bot. Runs on the HOST from cron,
# NOT inside Docker — the whole point is that it keeps working when the bot
# does not.
#
# WHY
# ---
# Every other alert this system produces comes from the bot itself, which means
# a dead or degraded bot tells you nothing. Polling-mode degradation ran 2.5
# days undetected in 2026-07 for exactly that reason. This closes the loop from
# outside the process.
#
# It also watches rebuild-watcher, because if THAT dies, `touch
# data/rebuild_trigger` does nothing at all — silently, forever, with no
# symptom until you need a deploy.
#
# ALERT-ONLY. It never restarts, stops, or otherwise touches the bot: every
# probe is read-only. `restart: unless-stopped` already handles process
# crashes, and a watchdog that restarted a bot holding an open position could
# re-adopt it repeatedly. Diagnosis is a machine's job; remediation here is not.
#
# NOISE
# -----
# The primary failure mode of a watchdog is crying wolf, not missing an event —
# an alert channel people mute is worse than no alert channel. So alerts are
# EDGE-TRIGGERED off a state file: one message when a problem appears, one
# re-nag per day while it persists, one message when it clears.
#
# THRESHOLDS
# ----------
# Measured, not guessed. 11 days of logs/ig_bot.log showed ZERO gaps over 20
# minutes, weekends included (Sat/Sun are quieter — ~600-850 lines/day vs
# ~5500 — but never silent). 30 min is ~50% headroom over the worst observed.
#
# Reads logs/ig_bot.log, NOT `docker logs`: compose down/up REMOVES the
# container, so docker's log buffer resets on every deploy and would report a
# freshly-deployed bot as silent.

set -uo pipefail

# Defaults to wherever this script lives — it ships inside the repo it watches,
# so its own directory is the answer, and no host hardcodes a path into a
# tracked file that `git reset --hard` would wipe on the next deploy.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="${WATCHDOG_BOT_DIR:-$SCRIPT_DIR}"
CONTAINER="${WATCHDOG_CONTAINER:-ig-trading-bot}"
WATCHER_UNIT="${WATCHDOG_WATCHER_UNIT:-rebuild-watcher}"
LOG_FILE="${WATCHDOG_LOG_FILE:-$BOT_DIR/logs/ig_bot.log}"
STATE_FILE="${WATCHDOG_STATE_FILE:-$BOT_DIR/data/.watchdog_state}"
HEARTBEAT_FILE="${WATCHDOG_HEARTBEAT_FILE:-$BOT_DIR/data/.watchdog_heartbeat}"
STALE_MIN="${WATCHDOG_STALE_MIN:-30}"
GRACE_MIN="${WATCHDOG_GRACE_MIN:-5}"
TRIGGER_STALE_MIN="${WATCHDOG_TRIGGER_STALE_MIN:-10}"
RENAG_SEC="${WATCHDOG_RENAG_SEC:-86400}"

NOW=$(date +%s)

# ---------------------------------------------------------------- utilities

# GNU stat (VPS, CI) and BSD stat (macOS dev) disagree on flags; tests run on
# both, so probe rather than assume.
mtime_of() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null
}

# Docker reports StartedAt as RFC3339 with nanoseconds. `date -d` is GNU-only;
# macOS needs `date -j -f`. Probing rather than assuming keeps the deploy-grace
# window working identically on the VPS, in CI, and on the dev machine —
# without this the grace silently never engaged on macOS, so the one test that
# proves deploys stay quiet could not run where it is most often read.
to_epoch() {
    local s="$1" trimmed
    date -d "$s" +%s 2>/dev/null && return
    trimmed="${s%%.*}"; trimmed="${trimmed%Z}"
    date -j -u -f "%Y-%m-%dT%H:%M:%S" "$trimmed" +%s 2>/dev/null
}

notify() {
    local text="$1"
    # Tests set WATCHDOG_NOTIFY_FILE to capture messages instead of sending.
    if [ -n "${WATCHDOG_NOTIFY_FILE:-}" ]; then
        printf '%s\n' "$text" >> "$WATCHDOG_NOTIFY_FILE"
        return 0
    fi
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

inspect() { docker inspect -f "$1" "$CONTAINER" 2>/dev/null; }

# ------------------------------------------------------------------- probes
# Each problem gets a stable KEY (used for edge-triggering) and a message.

KEYS=()
MSGS=()
problem() { KEYS+=("$1"); MSGS+=("$2"); }

running=$(inspect '{{.State.Running}}')
started=$(inspect '{{.State.StartedAt}}')
health=$(inspect '{{.State.Health.Status}}')
# RestartCount lives at the TOP LEVEL of docker inspect, not under .State —
# `{{.State.RestartCount}}` is a template parsing error, not an empty string,
# so this probe was silently dead until a live run exposed it. The test stub
# had encoded the same wrong assumption, which is why the suite was green.
restarts=$(inspect '{{.RestartCount}}')

# A container that started moments ago is mid-deploy or mid-restart: its health
# is legitimately "starting" and its log legitimately thin. Suppress only the
# checks that a fresh start makes meaningless — NEVER the crash-loop or
# not-running checks, which is precisely when a young container matters most.
in_grace=0
if [ -n "$started" ]; then
    started_epoch=$(to_epoch "$started"); started_epoch="${started_epoch:-0}"
    if [ "$started_epoch" -gt 0 ] && [ $(( (NOW - started_epoch) / 60 )) -lt "$GRACE_MIN" ]; then
        in_grace=1
    fi
fi

if [ -z "$running" ]; then
    problem "container_missing" "🚨 *Bot watchdog*: container \`$CONTAINER\` does not exist. The bot is NOT running."
elif [ "$running" != "true" ]; then
    problem "container_stopped" "🚨 *Bot watchdog*: container \`$CONTAINER\` is STOPPED. The bot is not trading."
else
    if [ "$in_grace" -eq 0 ] && [ -n "$health" ] && [ "$health" != "healthy" ] && [ "$health" != "<no value>" ]; then
        problem "container_unhealthy" "⚠️ *Bot watchdog*: container health is *$health* (not healthy)."
    fi

    # Crash loop. RestartCount resets to 0 on a compose down/up, so a climbing
    # value means Docker is restarting the process in place — the exact
    # signature of a container that boots, dies, and is restarted by
    # `restart: unless-stopped` while looking superficially "up".
    prev_restarts=$(grep -E '^_restarts=' "$STATE_FILE" 2>/dev/null | cut -d= -f2)
    if [ -n "$restarts" ] && [ -n "${prev_restarts:-}" ] && [ "$restarts" -gt "${prev_restarts:-0}" ]; then
        problem "crash_loop" "🚨 *Bot watchdog*: container restart count rose ${prev_restarts} → ${restarts}. Likely crash-looping."
    fi

    if [ "$in_grace" -eq 0 ]; then
        log_m=$(mtime_of "$LOG_FILE")
        if [ -z "$log_m" ]; then
            problem "log_missing" "⚠️ *Bot watchdog*: log file \`$LOG_FILE\` is missing."
        else
            age_min=$(( (NOW - log_m) / 60 ))
            if [ "$age_min" -ge "$STALE_MIN" ]; then
                problem "log_stale" "⚠️ *Bot watchdog*: no log output for *${age_min} min* (threshold ${STALE_MIN}). 11 days of history never exceeded 20 min, weekends included."
            fi
        fi

        # Polling-mode degradation: the bot falls back from streaming to REST
        # polling and keeps running, quieter and worse, with no alarm of its own.
        if [ -f "$LOG_FILE" ] && tail -400 "$LOG_FILE" 2>/dev/null | grep -q "Polling cycle started"; then
            problem "polling_mode" "⚠️ *Bot watchdog*: *POLLING MODE* detected — streaming has degraded to REST polling. Check \`[STREAM]\` lines."
        fi
    fi
fi

# The watchdog on the deploy path. If rebuild-watcher is dead, writing the
# trigger file does nothing and reports nothing.
if ! systemctl is-active --quiet "$WATCHER_UNIT" 2>/dev/null; then
    problem "watcher_down" "🚨 *Bot watchdog*: \`$WATCHER_UNIT\` is NOT active — deploys will silently do nothing."
fi

# A trigger file still sitting there means the watcher never consumed it.
trigger="$BOT_DIR/data/rebuild_trigger"
if [ -f "$trigger" ]; then
    t_m=$(mtime_of "$trigger")
    if [ -n "$t_m" ] && [ $(( (NOW - t_m) / 60 )) -ge "$TRIGGER_STALE_MIN" ]; then
        problem "trigger_wedged" "⚠️ *Bot watchdog*: a rebuild trigger has sat unconsumed for ≥${TRIGGER_STALE_MIN} min — the watcher is wedged."
    fi
fi

# --------------------------------------------------------- edge-triggering
#
# Indexed arrays and linear lookups rather than `declare -A`. Associative
# arrays need bash 4+, and macOS still ships bash 3.2 — the script would then
# work on the VPS and in CI but break on the dev machine, which for a watchdog
# is the wrong way round: you would stop trusting your local run.

PREV_K=(); PREV_V=()
if [ -f "$STATE_FILE" ]; then
    while IFS='=' read -r k v; do
        case "$k" in ""|_*) continue ;; esac
        PREV_K+=("$k"); PREV_V+=("$v")
    done < "$STATE_FILE"
fi

prev_seen_at() {                 # echo the epoch this key was last alerted, or ""
    local want="$1" i
    for i in "${!PREV_K[@]}"; do
        if [ "${PREV_K[$i]}" = "$want" ]; then printf '%s' "${PREV_V[$i]}"; return; fi
    done
}

is_current() {                   # 0 if the key is a problem right now
    local want="$1" k
    for k in ${KEYS[@]+"${KEYS[@]}"}; do
        [ "$k" = "$want" ] && return 0
    done
    return 1
}

new_state=""
for i in ${KEYS[@]+"${!KEYS[@]}"}; do
    key="${KEYS[$i]}"
    last=$(prev_seen_at "$key")
    if [ -z "$last" ]; then
        notify "${MSGS[$i]}"                            # newly broken
        new_state="${new_state}${key}=${NOW}"$'\n'
    elif [ $(( NOW - last )) -ge "$RENAG_SEC" ]; then
        notify "${MSGS[$i]} (still unresolved)"         # daily re-nag
        new_state="${new_state}${key}=${NOW}"$'\n'
    else
        new_state="${new_state}${key}=${last}"$'\n'    # already alerted, stay quiet
    fi
done

for i in ${PREV_K[@]+"${!PREV_K[@]}"}; do
    if ! is_current "${PREV_K[$i]}"; then
        notify "✅ *Bot watchdog*: recovered — \`${PREV_K[$i]}\` is clear."
    fi
done

mkdir -p "$(dirname "$STATE_FILE")"
{ printf '%s' "$new_state"; [ -n "$restarts" ] && printf '_restarts=%s\n' "$restarts"; } > "$STATE_FILE"

# Heartbeat: proof this script RAN and reached the end.
#
# Without it, a healthy run is completely invisible — it writes nothing to its
# cron log, and `>>` with no output does not even touch that file's mtime. So a
# watchdog whose cron entry had been removed would look EXACTLY like a watchdog
# reporting all-clear: silent, with an old log. The thing that watches
# everything else could not show that it was alive.
#
# The state file above happens to be rewritten every run too, so its mtime is
# an accidental heartbeat — but only accidentally. Anyone optimising that write
# to "only if changed" would silently remove the signal, with nothing to say it
# was load-bearing. This file exists to be that signal on purpose, and carries a
# readable timestamp and verdict rather than requiring a stat.
#
# Written LAST so it attests completion, not merely invocation: a run that dies
# in a probe leaves the old timestamp, which is the honest answer.
mkdir -p "$(dirname "$HEARTBEAT_FILE")"
if [ "${#KEYS[@]}" -eq 0 ]; then
    printf '%s ok\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$HEARTBEAT_FILE"
else
    printf '%s %d problem(s): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${#KEYS[@]}" "$(printf '%s,' "${KEYS[@]}" | sed 's/,$//')" > "$HEARTBEAT_FILE"
fi

[ "${#KEYS[@]}" -eq 0 ] && exit 0 || exit 1
