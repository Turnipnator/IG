"""Container healthcheck: is the BOT working, not just is the internet up.

The previous check was `requests.get('https://demo-api.ig.com')`. That tests
the container's route to IG and nothing else — it passes while the bot is
wedged, while the feed is dead, and it would pass if main.py had never started.
It reported `healthy` for every one of the 62 hours the market feed was down
from 2026-09-18, which is how the outage survived a whole weekend unnoticed.

Two signals, both read from local files so this costs no API calls and no
rate-limit budget (it runs every 5 minutes, forever):

  1. the process is writing logs at all;
  2. the candle cache is advancing while markets are open.

Bias: a false UNHEALTHY is expensive (watchdog.sh alerts, and a human is woken)
and this is only a backstop — the in-process streaming watchdog is the primary
detector and reacts in ~4 minutes rather than 30. So anything ambiguous, any
unparsable file, any unexpected exception reports HEALTHY and lets the real
watchdog make the call. The one thing this must never do is report healthy
when the feed is provably frozen during a trading session.

Exit 0 = healthy, exit 1 = unhealthy.
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("/app/data")
LOGS = Path("/app/logs")
CANDLE_CACHE = DATA / "streamed_candles.json"
LOG_FILE = LOGS / "ig_bot.log"

# The bot logs something (screener, candle save, HTF x-check) at least every
# 15 minutes; 45 gives three missed cycles of slack.
LOG_SILENCE_LIMIT = timedelta(minutes=45)
# The in-process watchdog trips at ~4 minutes. 30 means this only ever fires
# for an outage that watchdog has already failed to fix.
CANDLE_STALE_LIMIT = timedelta(minutes=30)


def markets_should_be_open(now_utc: datetime) -> bool:
    """True outside the forex weekend, when at least one market is normally live.

    Deliberately coarse. The forex week runs Sunday ~22:00 UTC to Friday ~22:00
    UTC, and across 13 markets something is trading essentially all of that
    window. Wide margins on both edges: this decides whether to WAKE SOMEONE,
    so it must not fire on a quiet Friday evening or a Sunday night warm-up.
    """
    weekday = now_utc.weekday()  # Mon=0 .. Sun=6
    if weekday == 5:                                  # Saturday
        return False
    if weekday == 4 and now_utc.hour >= 21:           # Friday evening onward
        return False
    if weekday == 6 and now_utc.hour < 23:            # most of Sunday
        return False
    return True


def newest_candle_time() -> datetime | None:
    """Newest timestamp across every market in the restart cache, or None."""
    raw = json.loads(CANDLE_CACHE.read_text())
    newest = None
    for value in raw.values():
        candles = value["candles"] if isinstance(value, dict) else value
        if not candles:
            continue
        ts = datetime.fromisoformat(candles[-1]["timestamp"])
        if newest is None or ts > newest:
            newest = ts
    return newest


def check() -> tuple[bool, str]:
    now_utc = datetime.now(timezone.utc)

    if LOG_FILE.exists():
        log_age = timedelta(seconds=time.time() - LOG_FILE.stat().st_mtime)
        if log_age > LOG_SILENCE_LIMIT:
            return False, f"no log output for {log_age} (bot wedged or dead)"

    if not markets_should_be_open(now_utc):
        return True, "markets closed — feed silence expected"

    if not CANDLE_CACHE.exists():
        # Normal for the first minutes after a cold start.
        return True, "no candle cache yet"

    newest = newest_candle_time()
    if newest is None:
        return True, "candle cache present but empty"

    # Candle timestamps are container-local (Europe/London), so compare local.
    age = datetime.now() - newest
    if age > CANDLE_STALE_LIMIT:
        return False, (
            f"newest candle is {age} old while markets are open — "
            f"feed frozen (see the streaming watchdog)"
        )
    return True, f"feed live, newest candle {age} old"


def main() -> int:
    try:
        ok, detail = check()
    except Exception as e:
        # Never fail the container over a bug in the healthcheck itself.
        print(f"healthcheck inconclusive ({e}) — reporting healthy")
        return 0
    print(detail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
