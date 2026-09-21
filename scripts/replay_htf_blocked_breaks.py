#!/usr/bin/env python3
"""Did the breakouts the HTF gate refused actually win?

    IG_ARCHIVE_DIR=/tmp/ig_archive_replay scripts/replay_htf_blocked_breaks.py

The question people ask after watching a blocked break run away is "how much is
that filter costing me?". Answering it from the log is a trap: the blocked-break
rows in `rejected_signals` are throttled to one per epic per hour, so counting
them measures the throttle, and five remembered winners measure memory.

Method. Run the breakout engine with NO HTF gate so every break is taken, then
tag each resulting trade with what the LIVE gate (htf_resolution, DAY on every
market as of 2026-09-21) would have said at that bar. That partitions ONE trade
set into TAKEN and BLOCKED, so both halves share identical sequencing — unlike
comparing two separate runs, where blocking a trade frees the engine to take a
different one later and the sets stop being comparable.

What this does and does not answer:
  * DOES answer — of the breaks the gate refused, what did they return?
  * does NOT answer — what the bot's P&L would be with the gate off. That is the
    HTF={NONE,HOUR,DAY} ladder (scripts/backtest_breakout_htf_ladder_ignative.py),
    whose sequencing differs. Both numbers are needed; they are not the same number.

Warm-up is excluded. DAY HTF needs 21 closed daily bars, and before that the
series sits at its NEUTRAL default, which blocks everything. Counting those as
"the gate refused this" would conflate missing data with a decision — and it is
exactly what makes the ladder's DAY column look thin.

Engine and HTF join are imported from the ladder's base module, whose join is
look-ahead safe (a daily bar is only used once `date + span <= refresh_instant`)
and mirrors the live 21:30 UTC scheduler, holding its value between refreshes.
The readiness flag recomputed here is asserted equal to that module's own trend
output, so this script cannot silently drift from the engine it borrows.
"""
import os, sys, importlib, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
ARCH = os.environ.get("IG_ARCHIVE_DIR", "data/candle_archive")
os.environ.setdefault("IG_ARCHIVE_DIR", ARCH)

bt = importlib.import_module("backtest_index_breakout_ignative")
from config import MARKETS
from src.breakout import BREAKOUT_CONFIGS
from src.indicators import calculate_atr, calculate_ema

# Measured trading-hours spreads, copied from backtest_breakout_htf_ladder_ignative
# rather than imported: that module runs its whole ladder at import time (no
# __main__ guard), so importing it would re-run a multi-minute backtest and print
# its output into ours. Keep in sync if the spread study is redone.
# [[project-spread-measurement-2026-09]]: measured inside each market's own hours.
SPREAD = {
    'IX.D.SPTRD.DAILY.IP': 0.61, 'IX.D.NASDAQ.CASH.IP': 2.28, 'IX.D.NIKKEI.DAILY.IP': 9.63,
    'IX.D.HANGSENG.DAILY.IP': 7.11, 'IX.D.DOW.DAILY.IP': 3.2, 'IX.D.FTSE.DAILY.IP': 1.3,
    'IX.D.AIIDX.DAILY.IP': 7.9, 'IX.D.RUSSELL.DAILY.IP': 0.3, 'CS.D.USCGC.TODAY.IP': 0.4,
    'CS.D.EURUSD.TODAY.IP': 0.6, 'CS.D.GBPUSD.TODAY.IP': 0.9, 'CC.D.CL.USS.IP': 3.0,
    'CC.D.DX.USS.IP': 8.1,
}


def htf_with_ready(df1h, resolution, trade_index):
    """Same trend series as the base module, plus a per-bar 'was it warmed up' flag.

    Deliberately a copy of the base logic rather than a wrapper, because the base
    returns only the trend. The assert below is what keeps the copy honest.
    """
    if resolution == "DAY":
        h = df1h.set_index("date").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    else:
        h = df1h.set_index("date")[["open", "high", "low", "close"]].copy()
    h["ema9"] = calculate_ema(h["close"], 9)
    h["ema21"] = calculate_ema(h["close"], 21)
    span = pd.Timedelta("1D") if resolution == "DAY" else pd.Timedelta("1h")

    trend_out = pd.Series("NEUTRAL", index=trade_index)
    ready_out = pd.Series(False, index=trade_index)
    start, end = trade_index.min(), trade_index.max()
    refreshes = pd.date_range(start.normalize() + pd.Timedelta("22:30:00"),
                              end.normalize() + pd.Timedelta("22:30:00"), freq="1D")
    trend, ready = "NEUTRAL", False
    prev = start
    for r in refreshes:
        seg = (trade_index >= prev) & (trade_index < r)
        trend_out[seg] = trend
        ready_out[seg] = ready
        closed = h[(h.index + span) <= r].tail(30)       # closed bars only
        if len(closed) >= 21 and not closed[["ema9", "ema21", "close"]].iloc[-1].isna().any():
            ready = True
            last = closed.iloc[-1]
            if last["ema9"] > last["ema21"] and last["close"] > last["ema21"]:
                trend = "BULLISH"
            elif last["ema9"] < last["ema21"] and last["close"] < last["ema21"]:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
        prev = r
    trend_out[trade_index >= prev] = trend
    ready_out[trade_index >= prev] = ready
    return trend_out, ready_out


def run_ungated(epic, mk, resolution="DAY", fill="close"):
    """Take EVERY break; tag each trade with the live gate's verdict at that bar."""
    cfg = BREAKOUT_CONFIGS[epic]
    df = bt.load_1h(epic)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
    idx = pd.Index(df["date"])
    trend, ready = htf_with_ready(df, resolution, idx)
    # Guard: our trend must match the engine's own, or this script is measuring
    # a different filter from the one the ladder and the live bot use.
    assert (trend.values == bt.htf_trend_series(df, resolution, idx).values).all(), \
        f"{epic}: HTF series diverged from the base module"
    df["htf"], df["htf_ready"] = trend.values, ready.values

    N, M, k = cfg.n, cfg.m, cfg.stop_atr_mult
    h_lo, h_hi = (mk.trading_start + 1) % 24, (mk.trading_end + 1) % 24
    out = []
    i, n = N + 1, len(df)
    while i < n:
        r = df.iloc[i]; atr = r["atr"]
        if not np.isfinite(atr) or atr <= 0:
            i += 1; continue
        prior = df.iloc[i - N:i]
        upper, lower = prior["high"].max(), prior["low"].min()
        d = None
        if r["high"] >= upper:   d, level = "BUY", upper
        elif r["low"] <= lower:  d, level = "SELL", lower
        if d is None:
            i += 1; continue
        hh = r["date"].hour
        if (hh < h_lo or hh >= h_hi) if h_lo < h_hi else (h_hi <= hh < h_lo):
            i += 1; continue

        if fill == "level":
            entry = max(r["open"], level) if d == "BUY" else min(r["open"], level)
            cost = 0.286 * atr
        else:
            entry = r["close"]; cost = SPREAD.get(epic, 0.0)
        sd = max(atr * k, mk.min_stop_distance)
        stop = entry - sd if d == "BUY" else entry + sd

        j, xp, why = i + 1, None, None
        while j < n:
            b = df.iloc[j]
            if d == "BUY" and b["low"] <= stop:   xp, why = min(b["open"], stop), "stop"; break
            if d == "SELL" and b["high"] >= stop: xp, why = max(b["open"], stop), "stop"; break
            if j - M >= 0:
                pm = df.iloc[j - M:j]
                if d == "BUY":
                    lvl = pm["low"].min()
                    if b["low"] <= lvl: xp, why = min(b["open"], lvl), "trail"; break
                else:
                    lvl = pm["high"].max()
                    if b["high"] >= lvl: xp, why = max(b["open"], lvl), "trail"; break
            j += 1
        if xp is None:
            xp, why, j = df.iloc[-1]["close"], "open-at-end", n - 1

        g = (xp - entry) if d == "BUY" else (entry - xp)
        want = "BULLISH" if d == "BUY" else "BEARISH"
        htf = r["htf"]
        out.append(dict(
            epic=epic, market=mk.name, date=r["date"], d=d, r=(g - cost) / sd, why=why,
            htf=htf, ready=bool(r["htf_ready"]),
            blocked=(htf != want),
            block_kind=("warmup" if not r["htf_ready"] else
                        "none" if htf == want else
                        "NEUTRAL" if htf == "NEUTRAL" else "OPPOSING"),
        ))
        i = j + 1
    return pd.DataFrame(out)


def stats(t):
    if len(t) == 0:
        return "n=0"
    n, s, m = len(t), t.r.sum(), t.r.mean()
    sd = t.r.std(ddof=1) if n > 1 else float("nan")
    z = m / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    w = int((t.r > 0).sum())
    win = t.r[t.r > 0].sum(); loss = -t.r[t.r <= 0].sum()
    pf = win / loss if loss > 0 else float("inf")
    return (f"n={n:3d} {w:2d}W/{n-w:2d}L  ΣR={s:+7.2f}  R/t={m:+6.3f}  "
            f"PF={pf:4.2f}  z={z:+5.2f}")


if __name__ == "__main__":
    fill = sys.argv[1] if len(sys.argv) > 1 else "close"
    mks = [m for m in MARKETS if m.epic in BREAKOUT_CONFIGS
           and os.path.exists(f"{ARCH}/{m.epic}.jsonl")
           and os.path.getsize(f"{ARCH}/{m.epic}.jsonl") > 200_000]

    print(f"FILL = {fill}   (live mechanism = close + trading-hours spread)")
    print(f"Gate under test: htf_resolution=DAY, the live setting on every market\n")
    print(f"{'market':18s} | {'BLOCKED by the gate':52s} | {'TAKEN (gate allowed)':52s}")
    print("-" * 128)
    allt = []
    for mk in mks:
        t = run_ungated(mk.epic, mk, "DAY", fill)
        allt.append(t)
        warm = t[t.block_kind == "warmup"]
        t = t[t.block_kind != "warmup"]              # warm-up is not a decision
        b, k_ = t[t.blocked], t[~t.blocked]
        print(f"{mk.name:18s} | {stats(b):52s} | {stats(k_):52s}")
    A = pd.concat(allt)
    warm_n = int((A.block_kind == "warmup").sum())
    A = A[A.block_kind != "warmup"]
    print("-" * 128)
    print(f"{'** POOLED **':18s} | {stats(A[A.blocked]):52s} | {stats(A[~A.blocked]):52s}")
    print(f"\n  (excluded {warm_n} pre-warm-up breaks — DAY HTF needs 21 closed daily bars)")

    print("\n=== blocked population split by WHY it was blocked ===")
    for kind in ("NEUTRAL", "OPPOSING"):
        print(f"  {kind:9s} {stats(A[A.block_kind == kind])}")
    print("\n=== blocked population by direction ===")
    for d in ("BUY", "SELL"):
        print(f"  {d:9s} {stats(A[(A.blocked) & (A.d == d)])}")
    print("\n=== live breakout markets only (Gold + GBP/USD) ===")
    live = A[A.epic.isin(["CS.D.USCGC.TODAY.IP", "CS.D.GBPUSD.TODAY.IP"])]
    print(f"  blocked  {stats(live[live.blocked])}")
    print(f"  taken    {stats(live[~live.blocked])}")

    # Exit-reason mix. A subset that is all 'stop' at ~-1R each is a degenerate
    # distribution: its z is meaningless (near-zero variance), and it usually
    # means the breaks failed immediately rather than that the sample is strong.
    print("\n=== exit reason mix (sanity: watch for all-stop subsets) ===")
    for label, sub in (("blocked", A[A.blocked]), ("taken", A[~A.blocked])):
        mix = sub.why.value_counts().to_dict()
        print(f"  {label:8s} {mix}")
    print("\n  S&P taken subset (flagged: z=-52 implies near-zero variance):")
    spx = A[(A.epic == "IX.D.SPTRD.DAILY.IP") & (~A.blocked)]
    print(f"    {stats(spx)}   exits={spx.why.value_counts().to_dict()}")
    print(f"    R values: {[round(x, 3) for x in spx.r.tolist()]}")
