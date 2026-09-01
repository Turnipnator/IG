#!/usr/bin/env python3
"""
Index breakout on IG-NATIVE archive data (v3 items 2/3; closes the "indices UNTESTED"
note in src/breakout.py).

src/breakout.py says of the eight index breakout configs:
    "Indices — UNTESTED as breakout. Yahoo cash proxies can't test them (no overnight
     bars vs IG's ~24h sessions) — the shadow observer on the IG archive IS the test."
and the cost-vs-edge study STRUCK all seven index rows rather than refuting them: Yahoo
sees only 43-89% of the bars inside the bot's own trading window, and 42.6% of IG-native
breakout entries fire in hours Yahoo cannot see (and those entries are WORSE). This runs
the same strategy on the IG archive, which is the instrument and session actually traded.

Faithful to the live path (src/breakout.py + main._breakout_frame_1h):
  * 1h frame resampled from the archive; the forming bar is never used.
  * Entry: break of the prior N=55 CLOSED bars (df[-(n+1):-1]), HTF-filtered.
  * Stop: max(2.0 x ATR(14), market.min_stop_distance). NO take-profit.
  * Exit: Donchian-M trail, M = N//2 = 27, measured on the prior M closed bars.
  * HTF: each market's OWN htf_resolution (7 indices HOUR, Russell DAY) — quoting a
    DAY-HTF number for an HOUR-HTF market is the documented error that overstated Gold
    as 1.28-1.51 when its live config was 1.09.
  * HTF is refreshed ONCE PER DAY at 21:30 UTC from 30 closed bars and held constant
    for the next 24h, exactly as the live scheduler does. Recomputing it every bar
    would be a look-ahead the bot never enjoys.
  * Fills follow 117a4e8: entry gap-adjusted to the adverse side of (open, level);
    stop and trail likewise; stop takes precedence within a bar.
  * Cost charged as a fraction of ATR per trade (default 0.286xATR = 0.143R at a
    2xATR stop), the convention the entry-fill-gap study established.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETS
from src.breakout import BREAKOUT_CONFIGS
from src.indicators import calculate_atr, calculate_ema

ARCHIVE = Path(os.environ.get("IG_ARCHIVE_DIR", "data/candle_archive"))


def load_1h(epic):
    rows = [json.loads(l) for l in (ARCHIVE / f"{epic}.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")
    g = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return g.reset_index()


def htf_trend_series(df1h, resolution, trade_index):
    """Daily-refreshed HTF trend, mirroring the 21:30 UTC scheduler (=22:30 BST).

    Returns a Series aligned to trade_index. Between refreshes the value is HELD.
    """
    if resolution == "DAY":
        h = df1h.set_index("date").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    else:                                   # HOUR — the 1h frame itself
        h = df1h.set_index("date")[["open", "high", "low", "close"]].copy()
    h["ema9"] = calculate_ema(h["close"], 9)
    h["ema21"] = calculate_ema(h["close"], 21)
    span = pd.Timedelta("1D") if resolution == "DAY" else pd.Timedelta("1h")

    out = pd.Series("NEUTRAL", index=trade_index)
    # refresh instants: 22:30 BST each day (21:30 UTC)
    start, end = trade_index.min(), trade_index.max()
    refreshes = pd.date_range(start.normalize() + pd.Timedelta("22:30:00"),
                              end.normalize() + pd.Timedelta("22:30:00"), freq="1D")
    trend = "NEUTRAL"
    prev = start
    for r in refreshes:
        seg = (trade_index >= prev) & (trade_index < r)
        out[seg] = trend
        closed = h[(h.index + span) <= r].tail(30)      # 30 COMPLETED bars, no look-ahead
        if len(closed) >= 21 and not closed[["ema9", "ema21", "close"]].iloc[-1].isna().any():
            last = closed.iloc[-1]
            if last["ema9"] > last["ema21"] and last["close"] > last["ema21"]:
                trend = "BULLISH"
            elif last["ema9"] < last["ema21"] and last["close"] < last["ema21"]:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
        prev = r
    out[trade_index >= prev] = trend
    return out


def run(epic, mk, cost_atr_frac, lo=None, hi=None):
    cfg = BREAKOUT_CONFIGS[epic]
    df = load_1h(epic)
    if lo is not None:
        df = df.iloc[lo:hi].reset_index(drop=True)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
    htf = htf_trend_series(df, mk.htf_resolution, pd.Index(df["date"]))
    df["htf"] = htf.values

    N, M, k = cfg.n, cfg.m, cfg.stop_atr_mult
    # Live refuses breakout entries outside the market's hours (main.py:933). Archive
    # stamps are Europe/London and this window is entirely BST, so live UTC hours are
    # +1 frame-local. Entries only -- an open position is managed round the clock.
    h_lo, h_hi = (mk.trading_start + 1) % 24, (mk.trading_end + 1) % 24
    trades = []
    i = N + 1
    n = len(df)
    while i < n:
        r = df.iloc[i]
        atr = r["atr"]
        if not np.isfinite(atr) or atr <= 0:
            i += 1; continue
        prior = df.iloc[i - N:i]                      # the N bars BEFORE bar i
        upper, lower = prior["high"].max(), prior["low"].min()
        direction = None
        if r["high"] >= upper and r["htf"] == "BULLISH":
            direction, level = "BUY", upper
        elif r["low"] <= lower and r["htf"] == "BEARISH":
            direction, level = "SELL", lower
        if direction is None:
            i += 1; continue
        hh = r["date"].hour
        if (hh < h_lo or hh >= h_hi) if h_lo < h_hi else (h_hi <= hh < h_lo):
            i += 1; continue

        # entry fill: adverse side of (open, level) — a bar that gapped through the
        # channel fills at the open, not the level (117a4e8 convention).
        o = r["open"]
        entry = max(o, level) if direction == "BUY" else min(o, level)
        stop_dist = max(atr * k, mk.min_stop_distance)
        stop = entry - stop_dist if direction == "BUY" else entry + stop_dist

        j = i + 1
        exit_price, why = None, None
        while j < n:
            b = df.iloc[j]
            # stop first — a bar carries no intrabar ordering, so assume the adverse one
            if direction == "BUY" and b["low"] <= stop:
                exit_price, why = min(b["open"], stop), "stop"; break
            if direction == "SELL" and b["high"] >= stop:
                exit_price, why = max(b["open"], stop), "stop"; break
            # Donchian-M trail on the prior M CLOSED bars (exit_channel semantics)
            if j - M >= 0:
                pm = df.iloc[j - M:j]
                if direction == "BUY":
                    lvl = pm["low"].min()
                    if b["low"] <= lvl:
                        exit_price, why = min(b["open"], lvl), "trail"; break
                else:
                    lvl = pm["high"].max()
                    if b["high"] >= lvl:
                        exit_price, why = max(b["open"], lvl), "trail"; break
            j += 1
        if exit_price is None:
            exit_price, why, j = df.iloc[-1]["close"], "eod", n - 1

        gross = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
        rmult = (gross - cost_atr_frac * atr) / stop_dist
        trades.append(dict(direction=direction, r=rmult, why=why, bars=j - i,
                           date=r["date"]))
        i = j + 1
    return trades


def summarize(name, trades, note=""):
    if not trades:
        print("  %-20s no trades %s" % (name, note)); return None
    rs = [t["r"] for t in trades]
    w = [x for x in rs if x > 0]; l = [x for x in rs if x <= 0]
    pf = (sum(w) / abs(sum(l))) if l and sum(l) != 0 else float("inf")
    print("  %-20s n=%3d %2dW/%2dL  totR=%+7.2f  R/t=%+6.3f  PF=%5.2f  maxW=%+5.2f %s"
          % (name, len(rs), len(w), len(l), sum(rs), np.mean(rs), pf,
             max(rs) if rs else 0, note))
    return dict(n=len(rs), totR=sum(rs), rt=float(np.mean(rs)), pf=pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.286,
                    help="cost as a fraction of ATR per trade (0.286 = 0.143R at a 2xATR stop)")
    a = ap.parse_args()
    idx = [m for m in MARKETS if m.epic in BREAKOUT_CONFIGS and m.sector == "Indices"]

    for cost in (0.0, a.cost, a.cost * 2):
        lab = "frictionless" if cost == 0 else ("%.3fxATR (=%.3fR)" % (cost, cost / 2))
        print("\n%s\n== COST %s ==\n%s" % ("=" * 92, lab, "=" * 92))
        pooled = []
        for mk in idx:
            f = ARCHIVE / f"{mk.epic}.jsonl"
            if not f.exists() or f.stat().st_size < 1000:
                print("  %-20s (no archive)" % mk.name); continue
            t = run(mk.epic, mk, cost)
            summarize(mk.name, t, "htf=%s" % mk.htf_resolution)
            pooled += t
        summarize("** POOLED **", pooled)

    # split-half stability at the live cost
    print("\n%s\n== SPLIT-HALF at %.3fxATR (in-sample stability check) ==\n%s"
          % ("=" * 92, a.cost, "=" * 92))
    for mk in idx:
        f = ARCHIVE / f"{mk.epic}.jsonl"
        if not f.exists() or f.stat().st_size < 1000: continue
        d = load_1h(mk.epic); half = len(d) // 2
        t1 = run(mk.epic, mk, a.cost, 0, half)
        t2 = run(mk.epic, mk, a.cost, half, len(d))
        r1 = sum(x["r"] for x in t1); r2 = sum(x["r"] for x in t2)
        agree = "same sign" if (r1 > 0) == (r2 > 0) else "*** FLIPS ***"
        print("  %-20s H1 n=%2d %+7.2fR | H2 n=%2d %+7.2fR   %s"
              % (mk.name, len(t1), r1, len(t2), r2, agree))


if __name__ == "__main__":
    main()
