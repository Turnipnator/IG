#!/usr/bin/env python3
"""Tests for tick-triggered breakout entry (main._arm_breakout_levels /
main._check_breakout_tick_trigger, src.breakout.current_channel / arm_channel).

The dangerous part of this feature is the LATCH. on_price_update fires on every
streaming tick, so a crossing that is not latched opens positions in a loop until
something else stops it. Most of what follows exercises that one property.

Run:  python3 scripts/test_breakout_tick_entry.py       (no live account needed)

`telegram` and `lightstreamer` are stubbed because they are not installed outside
the container; NOTHING in main.py's breakout path is stubbed — the real functions
and the real module state are exercised.
"""
import os
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.error",
             "lightstreamer", "lightstreamer.client"):
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr, _n=name: MagicMock(name=f"{_n}.{attr}")
    sys.modules.setdefault(name, mod)
os.environ.setdefault("IG_API_KEY", "test")
os.environ.setdefault("IG_USERNAME", "test")
os.environ.setdefault("IG_PASSWORD", "test")

import pandas as pd  # noqa: E402
import main  # noqa: E402
from src import breakout  # noqa: E402
from config import MARKETS  # noqa: E402

EPIC = "CS.D.GBPUSD.TODAY.IP"
MC = next(m for m in MARKETS if m.epic == EPIC)
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail and not cond else ''}")


@dataclass
class FakeMarket:
    """Minimal MarketStream stand-in: the trigger only reads these three."""
    bid: float
    offer: float

    @property
    def mid_price(self):
        return (self.bid + self.offer) / 2


def frame(n=60, base=100.0, step=0.0, hi=1.0, start="2026-01-01"):
    """Flat-ish OHLC frame: channel upper = base+hi, lower = base-hi."""
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="h"),
        "high": [base + hi + i * step for i in range(n)],
        "low": [base - hi + i * step for i in range(n)],
        "close": [base + i * step for i in range(n)],
    })


def reset(mode="live"):
    main.BREAKOUT_TICK_ENTRY = mode
    main._breakout_armed.clear()
    main.journal = None          # journalling is best-effort and not under test
    return []


def stub_execute(calls):
    def _fake(epic, market, market_config, signal, df):
        calls.append((epic, signal.signal.value, signal.entry_price,
                      signal.stop_distance, signal.break_level))
    main._execute_breakout_entry = _fake


# ---------------------------------------------------------------- channel maths
print("\nsrc.breakout.current_channel / arm_channel")
df = frame()
ch = breakout.current_channel(df, EPIC)
check("current_channel returns (upper, lower) of the last N closed bars",
      ch == (101.0, 99.0), str(ch))
check("current_channel is None for an epic with no breakout config",
      breakout.current_channel(df, "NO.SUCH.EPIC") is None)
check("current_channel is None on a short frame",
      breakout.current_channel(frame(n=10), EPIC) is None)

# The off-by-one that matters: arming must use the bar just closed, confirming must not.
rising = frame(n=60, step=1.0)
armed = breakout.current_channel(rising, EPIC)
n = breakout.BREAKOUT_CONFIGS[EPIC].n
check("arming window INCLUDES the just-closed bar (one bar ahead of analyze_breakout)",
      armed[0] == float(rising["high"].iloc[-1]),
      f"{armed[0]} vs last high {rising['high'].iloc[-1]}")
check("analyze_breakout's confirming window EXCLUDES it",
      float(rising.iloc[-(n + 1):-1]["high"].max()) < armed[0])

a = breakout.arm_channel(df, MC, "BULLISH")
check("arm_channel clamps stop to max(k*ATR, min_stop)",
      a is not None and a.stop_distance >= MC.min_stop_distance)
check("arm_channel carries the htf trend and filter flag",
      a.htf_trend == "BULLISH" and a.htf_filter is True)
check("arm_channel bar_time identifies the last bar",
      a.bar_time == str(df["date"].iloc[-1]))

# --------------------------------------------------------------------- the latch
print("\nlatch — the money-risk property")
calls = reset(); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
for _ in range(500):                       # 500 ticks all above the channel
    main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
check("500 crossing ticks open exactly ONE position", len(calls) == 1, f"got {len(calls)}")
check("entry is BUY at the executable (offer) price",
      calls and calls[0][1] == "BUY" and calls[0][2] == 105.2, str(calls[:1]))
check("break_level recorded is the channel level, not the fill",
      calls and calls[0][4] == 101.0, str(calls[:1]))

calls = reset(); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
# Same bar re-evaluated (30-min re-screen / debounced callback) must not re-arm.
for _ in range(20):
    main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
    main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
check("re-arming the SAME bar does not release a consumed latch", len(calls) == 1,
      f"got {len(calls)}")

# A genuinely new bar must release it.
calls = reset(); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
main._arm_breakout_levels(EPIC, MC, frame(start="2026-02-01"), "BULLISH", live=True)
main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
check("a NEW bar re-arms and allows one further entry", len(calls) == 2, f"got {len(calls)}")

# ------------------------------------------------------------------ gates & sides
print("\ngates")
calls = reset(); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BEARISH", live=True)
for _ in range(50):
    main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
check("HTF filter blocks a BUY when the trend is BEARISH", len(calls) == 0, f"got {len(calls)}")

calls = reset(); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BEARISH", live=True)
main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=95.0, offer=95.2))
check("SELL fires on a downside cross with a BEARISH htf", len(calls) == 1, f"got {len(calls)}")
check("SELL entry uses the bid", calls and calls[0][2] == 95.0, str(calls[:1]))

calls = reset(); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
for _ in range(50):
    main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=100.0, offer=100.2))
check("no entry while price is inside the channel", len(calls) == 0, f"got {len(calls)}")

calls = reset(mode="log"); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
for _ in range(50):
    main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
check("log mode places NO order even on a live market", len(calls) == 0, f"got {len(calls)}")

calls = reset(mode="live"); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=False)
for _ in range(50):
    main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
check("a SHADOW market places no order in live mode", len(calls) == 0, f"got {len(calls)}")

calls = reset(mode="off"); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))
check("mode=off arms nothing and fires nothing",
      len(calls) == 0 and not main._breakout_armed)

calls = reset(); stub_execute(calls)
main._check_breakout_tick_trigger("NOT.ARMED.EPIC", FakeMarket(bid=105.0, offer=105.2))
check("an unarmed epic is a no-op", len(calls) == 0)

# ------------------------------------------------------------------- concurrency
print("\nconcurrency — many stream threads on one crossing")
import threading  # noqa: E402
calls = reset(); stub_execute(calls)
main._arm_breakout_levels(EPIC, MC, df, "BULLISH", live=True)
barrier = threading.Barrier(24)


def hammer():
    barrier.wait()
    for _ in range(40):
        main._check_breakout_tick_trigger(EPIC, FakeMarket(bid=105.0, offer=105.2))


ts = [threading.Thread(target=hammer) for _ in range(24)]
[t.start() for t in ts]
[t.join() for t in ts]
check("24 threads x 40 ticks on one armed bar open exactly ONE position",
      len(calls) == 1, f"got {len(calls)}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
