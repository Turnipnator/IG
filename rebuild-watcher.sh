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
#
# CI GATE
# -------
# Before syncing, the target commit must have a PASSING `verify` job in
# .github/workflows/ci.yml. Until this gate existed, any push to main became
# the live trading bot unconditionally: a broken import produced a
# crash-looping container under `restart: unless-stopped` AND a Telegram
# message saying "Rebuild complete".
#
# The gate FAILS CLOSED. If GitHub is unreachable, rate-limited, or CI is
# still running past CI_WAIT_SECONDS, the rebuild is refused and the bot stays
# on its current, known-good code. To deploy anyway (CI outage, emergency
# rollback to a pre-CI commit), use the force trigger:
#
#     touch /root/ig-bot/data/rebuild_trigger_force
#
# BUILD ORDER
# -----------
# `docker compose build` runs BEFORE `down`. The previous order stopped the
# bot first, so any build failure — a yanked wheel, an unresolvable pin — left
# it stopped indefinitely. Building first means a failed build costs nothing:
# the old container is still up on the old image.

# PORTABILITY
# -----------
# Nothing here is hardcoded to one host or one fork, because this script is a
# TRACKED file: `git reset --hard` below would wipe any local edit to it on
# every single deploy. Anyone editing constants in this file to suit their
# machine would silently lose that edit the first time they deployed. So the
# two host-specific values are DERIVED, and everything else takes an env
# override (settable via systemd `Environment=` without touching the repo).

# The bot directory is simply wherever this script lives — it ships inside the
# repo it deploys, so its own location is the answer.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="${IG_BOT_DIR:-$SCRIPT_DIR}"

TRIGGER_FILE="$BOT_DIR/data/rebuild_trigger"
FORCE_TRIGGER_FILE="$BOT_DIR/data/rebuild_trigger_force"
UPSTREAM_REMOTE="${IG_UPSTREAM_REMOTE:-origin}"
UPSTREAM_BRANCH="${IG_UPSTREAM_BRANCH:-main}"

# CI gate configuration.
# The repo is read from the git remote rather than hardcoded, so a fork gates
# against ITS OWN CI. Hardcoding one owner would make a fork query the upstream
# repo for a SHA that only exists in the fork — a 404 on every deploy, i.e. a
# permanently blocked deploy path, with a confusing "was it pushed?" message.
#
# Public repos need no token for check-runs (60 req/hr per IP, unauthenticated).
# CI_JOB_NAME must match the `name:` of the job in .github/workflows/ci.yml
# EXACTLY — rename one without the other and the gate fails closed on every
# deploy. tests/test_packaging_hygiene.py asserts the two agree.
derive_github_repo() {
    local url
    url=$(git -C "$BOT_DIR" remote get-url "$UPSTREAM_REMOTE" 2>/dev/null) || return 1
    case "$url" in
        git@github.com:*)          url="${url#git@github.com:}" ;;
        ssh://git@github.com/*)    url="${url#ssh://git@github.com/}" ;;
        https://github.com/*)      url="${url#https://github.com/}" ;;
        http://github.com/*)       url="${url#http://github.com/}" ;;
        *) return 1 ;;   # not GitHub — the check-runs API cannot answer for it
    esac
    url="${url%.git}"
    url="${url%/}"
    # Must look like owner/repo and nothing more.
    case "$url" in
        */*/*|*/) return 1 ;;
        */*) printf '%s' "$url" ;;
        *) return 1 ;;
    esac
}

GITHUB_REPO="${IG_GITHUB_REPO:-$(derive_github_repo)}"
CI_JOB_NAME="${IG_CI_JOB_NAME:-verify}"
CI_WAIT_SECONDS="${IG_CI_WAIT_SECONDS:-600}"   # wait for an in-progress run
CI_POLL_SECONDS="${IG_CI_POLL_SECONDS:-30}"    # ~20 API calls worst case, inside 60/hr

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

# ci_gate <sha> -> 0 if the `verify` job passed for that commit, 1 otherwise.
# Sets CI_GATE_REASON to a human-readable explanation on failure.
#
# Reads /check-runs, NOT /commits/{sha}/status: GitHub Actions reports as
# check runs, and the legacy combined-status endpoint returns
# `state=pending, statuses=[]` forever on an Actions-only repo — a gate built
# on it either blocks every deploy or never blocks anything.
#
# It also selects ONE named job rather than requiring every check run to be
# green. Dependabot check runs and any future `continue-on-error` job show up
# in the same list, so "all green" would false-reject perfectly good commits.
ci_gate() {
    local sha="$1"
    local deadline=$(( $(date +%s) + CI_WAIT_SECONDS ))
    local url="https://api.github.com/repos/${GITHUB_REPO}/commits/${sha}/check-runs?per_page=100"

    while :; do
        local response http body status conclusion
        response=$(curl -s -m 20 -w $'\n%{http_code}' \
            -H 'Accept: application/vnd.github+json' \
            -H 'User-Agent: ig-bot-rebuild-watcher' \
            "$url")
        http=$(printf '%s' "$response" | tail -n1)
        body=$(printf '%s' "$response" | sed '$d')

        if [ "$http" = "403" ] || [ "$http" = "429" ]; then
            CI_GATE_REASON="GitHub API rate-limited (HTTP ${http})"
            return 1
        fi
        if [ "$http" = "404" ] || [ "$http" = "422" ]; then
            # Distinct from a network failure and worth its own wording: this
            # is what you get for a commit GitHub has never seen. Almost
            # always means the commit was never pushed.
            CI_GATE_REASON="commit \`${sha:0:7}\` not found on GitHub (HTTP ${http}) — was it pushed?"
            return 1
        fi
        if [ "$http" != "200" ]; then
            CI_GATE_REASON="GitHub API unreachable (HTTP ${http:-none})"
            return 1
        fi

        # Most recently started run with this name. `last` on an empty array
        # yields null, hence the `// {}` before indexing.
        read -r status conclusion <<<"$(printf '%s' "$body" | jq -r --arg n "$CI_JOB_NAME" '
            (([.check_runs[] | select(.name == $n)] | sort_by(.started_at) | last) // {})
            | "\(.status // "absent") \(.conclusion // "none")"' 2>/dev/null)"

        case "$status" in
            completed)
                if [ "$conclusion" = "success" ]; then
                    return 0
                fi
                CI_GATE_REASON="CI job \`${CI_JOB_NAME}\` concluded *${conclusion}*"
                return 1
                ;;
            queued|in_progress|waiting|pending|requested)
                : # still running — fall through to the wait below
                ;;
            absent)
                : # workflow not started yet, or this commit predates ci.yml
                ;;
            *)
                CI_GATE_REASON="unexpected CI status '${status}'"
                return 1
                ;;
        esac

        if [ "$(date +%s)" -ge "$deadline" ]; then
            if [ "$status" = "absent" ]; then
                CI_GATE_REASON="no \`${CI_JOB_NAME}\` job found for this commit after ${CI_WAIT_SECONDS}s (does it predate ci.yml?)"
            else
                CI_GATE_REASON="CI still *${status}* after ${CI_WAIT_SECONDS}s"
            fi
            return 1
        fi
        echo "[rebuild-watcher] CI ${status} for ${sha:0:7} — waiting ${CI_POLL_SECONDS}s..."
        sleep "$CI_POLL_SECONDS"
    done
}

echo "[rebuild-watcher] Watching for rebuild triggers (hard-reset parity mode, CI-gated)..."

while true; do
    if [ -f "$TRIGGER_FILE" ] || [ -f "$FORCE_TRIGGER_FILE" ]; then
        FORCED=0
        [ -f "$FORCE_TRIGGER_FILE" ] && FORCED=1

        if [ "$FORCED" -eq 1 ]; then
            echo "[rebuild-watcher] FORCE rebuild triggered at $(cat "$FORCE_TRIGGER_FILE" 2>/dev/null)"
        else
            echo "[rebuild-watcher] Rebuild triggered at $(cat "$TRIGGER_FILE" 2>/dev/null)"
        fi
        rm -f "$TRIGGER_FILE" "$FORCE_TRIGGER_FILE"

        cd "$BOT_DIR" || { echo "[rebuild-watcher] BOT_DIR missing"; sleep 5; continue; }

        # 1. Fetch upstream. Abort the rebuild on failure — never build stale.
        echo "[rebuild-watcher] Fetching $UPSTREAM_REMOTE/$UPSTREAM_BRANCH..."
        if ! git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" 2>&1; then
            echo "[rebuild-watcher] git fetch FAILED — aborting, staying on current code"
            notify "❌ *Rebuild aborted*: git fetch failed. Bot left on current code (no stale build)."
            continue
        fi

        # 2. Resolve the exact commit we are about to deploy, and gate on THAT
        #    sha rather than on the branch ref — otherwise a push landing
        #    between the gate and the reset would deploy ungated code.
        TARGET_SHA=$(git rev-parse "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" 2>/dev/null)
        if [ -z "$TARGET_SHA" ]; then
            echo "[rebuild-watcher] could not resolve $UPSTREAM_REMOTE/$UPSTREAM_BRANCH — aborting"
            notify "❌ *Rebuild aborted*: could not resolve \`$UPSTREAM_REMOTE/$UPSTREAM_BRANCH\`. Bot left on current code."
            continue
        fi

        # 3. CI gate.
        if [ "$FORCED" -eq 0 ] && [ -z "$GITHUB_REPO" ]; then
            # Fail closed rather than deploy ungated. Happens when the remote is
            # not GitHub, or is missing entirely.
            echo "[rebuild-watcher] cannot determine GitHub repo from remote '$UPSTREAM_REMOTE' — aborting"
            notify "⛔ *Rebuild BLOCKED*: cannot determine the GitHub repo from remote \`$UPSTREAM_REMOTE\`.
Set \`IG_GITHUB_REPO=owner/repo\`, or force with \`touch data/rebuild_trigger_force\`."
            continue
        fi
        if [ "$FORCED" -eq 1 ]; then
            echo "[rebuild-watcher] CI gate BYPASSED (force trigger) for ${TARGET_SHA:0:7}"
            notify "⚠️ *Force rebuild* — CI gate BYPASSED for \`${TARGET_SHA:0:7}\`. Deploying unverified code."
        else
            CI_GATE_REASON=""
            echo "[rebuild-watcher] Checking CI for ${TARGET_SHA:0:7}..."
            if ! ci_gate "$TARGET_SHA"; then
                echo "[rebuild-watcher] CI GATE BLOCKED: $CI_GATE_REASON"
                notify "⛔ *Rebuild BLOCKED* for \`${TARGET_SHA:0:7}\`: ${CI_GATE_REASON}.
Bot left running on current code. Fix CI, or force with \`touch data/rebuild_trigger_force\`."
                continue
            fi
            echo "[rebuild-watcher] CI passed for ${TARGET_SHA:0:7}"
        fi

        # 4. Hard reset to the gated commit — discards ALL local drift for
        #    guaranteed parity.
        echo "[rebuild-watcher] Hard-resetting to ${TARGET_SHA:0:7}..."
        if ! git reset --hard "$TARGET_SHA" 2>&1; then
            echo "[rebuild-watcher] git reset FAILED — aborting, staying on current code"
            notify "❌ *Rebuild aborted*: git reset failed. Bot left on current code (no stale build)."
            continue
        fi
        git clean -fd 2>&1   # remove untracked files so the tree matches upstream exactly

        HEAD_DESC=$(git log -1 --pretty=format:'%h %s' 2>/dev/null)
        echo "[rebuild-watcher] Now at: $HEAD_DESC"

        # 5. Build FIRST, while the old container is still serving. A failed
        #    build here costs nothing — the bot is still up on the old image.
        echo "[rebuild-watcher] Building new image..."
        if ! docker compose build --no-cache 2>&1; then
            echo "[rebuild-watcher] docker build FAILED — bot left RUNNING on the previous image"
            notify "❌ *Rebuild failed*: docker build errored for \`${HEAD_DESC}\`.
Bot is *still running* on the previous image — no downtime. Source tree is synced; fix and re-trigger."
            continue
        fi

        # 6. Only now swap containers.
        echo "[rebuild-watcher] Restarting container..."
        docker compose down 2>&1
        if ! docker compose up -d 2>&1; then
            echo "[rebuild-watcher] docker compose up FAILED — BOT IS DOWN"
            notify "🚨 *CRITICAL*: image built but \`compose up\` failed for \`${HEAD_DESC}\`. *THE BOT IS DOWN* — manual intervention needed."
            continue
        fi

        echo "[rebuild-watcher] Rebuild complete at $(date)"
        notify "✅ *Rebuild complete* — synced to \`${HEAD_DESC}\` (CI verified). Container restarted."
    fi
    sleep 5
done
