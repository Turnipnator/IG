#!/usr/bin/env python3
"""
Per-market breakout ENTRY FILL GAP.

The gap is the difference between the fill a backtest assumes and the fill the bot
actually gets:

  idealised (backtest)  : filled AT the channel level the moment price touches it
  actual   (live close) : the break is confirmed on a CLOSED 1h bar, then a MARKET
                          order is sent -- so you pay the rest of the breaking bar's
                          move, plus half the spread

    gap = (bar_close - level) + spread/2      [BUY]
        = (level - bar_close) + spread/2      [SELL]

Expressed in R against the live stop (max(2xATR, min_stop_distance)), so it is directly
comparable to the flat **0.286xATR = 0.143R** convention the corpus charges in breakout
sweeps -- a convention the cost-vs-edge study already flagged as wrong per-market by
+/-0.207 PF, larger than the effects it was used to decompose. This measures it per
market instead of assuming it.

Two independent measurements:
  A. ARCHIVE REPLAY  -- every bar where the live entry rule would fire (N=55 channel on
     prior CLOSED bars, HTF-filtered on the market's OWN htf_resolution, HTF refreshed
     daily at 21:30 UTC and held). Large n.
  B. LOGGED LIVE SIGNALS -- the Breakout-tick[log] rows record the true `level` for
     breaks the bot actually detected, plus the tick's own wall-clock timestamp;
     pairing the level with the archive close of the bar CONTAINING that tick gives a
     ground-truth gap on real signals.
     NB use the row timestamp, NOT the `bar=` field: `bar=` is ArmedChannel.bar_time,
     the bar the channel was ARMED on, which by construction sits INSIDE the channel
     (arm_channel snapshots for entry on the NEXT bar). Pairing against it makes the
     close look systematically favourable and inverts the result.

NB the tick-entry experiment measured something DIFFERENT and should not be confused
with this: it compared a TICK fill against the close fill (paired saving -0.020R at
n=39, REFUTED). This measures the close fill against the idealised channel fill.
"""
import argparse, json, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETS
from src.breakout import BREAKOUT_CONFIGS
from src.indicators import calculate_atr, calculate_ema

ARCHIVE = Path(os.environ.get("IG_ARCHIVE_DIR", "data/candle_archive"))
# trading-hours medians measured 2026-08-31 (see research_notes.md 2026-08-31(b)).
# None = not measured; the spread half is then reported as n/a rather than guessed.
SPREAD = {
    "S&P 500": 0.61, "NASDAQ 100": 2.28, "Japan 225": 9.63, "Hong Kong HS50": 7.11,
    "Wall Street": 4.19, "FTSE 100": 1.17, "AI Index": 0.95, "US Russell 2000": 0.33,
    "GBP/USD": 1.21,
}


def load_1h(epic):
    rows = [json.loads(l) for l in (ARCHIVE / f"{epic}.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")
    g = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return g.reset_index()


def htf_series(df1h, resolution, idx):
    if resolution == "DAY":
        h = df1h.set_index("date").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        span = pd.Timedelta("1D")
    else:
        h = df1h.set_index("date")[["open", "high", "low", "close"]].copy()
        span = pd.Timedelta("1h")
    h["e9"] = calculate_ema(h["close"], 9); h["e21"] = calculate_ema(h["close"], 21)
    out = pd.Series("NEUTRAL", index=idx); trend = "NEUTRAL"; prev = idx.min()
    for r in pd.date_range(idx.min().normalize() + pd.Timedelta("22:30:00"),
                           idx.max().normalize() + pd.Timedelta("22:30:00"), freq="1D"):
        out[(idx >= prev) & (idx < r)] = trend
        c = h[(h.index + span) <= r].tail(30)
        if len(c) >= 21 and not c[["e9", "e21", "close"]].iloc[-1].isna().any():
            l = c.iloc[-1]
            trend = ("BULLISH" if l["e9"] > l["e21"] and l["close"] > l["e21"]
                     else "BEARISH" if l["e9"] < l["e21"] and l["close"] < l["e21"] else "NEUTRAL")
        prev = r
    out[idx >= prev] = trend
    return out


def replay(mk):
    cfg = BREAKOUT_CONFIGS[mk.epic]
    df = load_1h(mk.epic)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
    df["htf"] = htf_series(df, mk.htf_resolution, pd.Index(df["date"])).values
    out = []
    for i in range(cfg.n + 1, len(df)):
        r = df.iloc[i]; atr = r["atr"]
        if not np.isfinite(atr) or atr <= 0:
            continue
        prior = df.iloc[i - cfg.n:i]
        up, lo = prior["high"].max(), prior["low"].min()
        if r["high"] >= up and r["htf"] == "BULLISH":
            d, lvl = "BUY", up
        elif r["low"] <= lo and r["htf"] == "BEARISH":
            d, lvl = "SELL", lo
        else:
            continue
        stop = max(atr * cfg.stop_atr_mult, mk.min_stop_distance)
        over = (r["close"] - lvl) if d == "BUY" else (lvl - r["close"])
        out.append(dict(over=over, atr=atr, stop=stop, over_r=over / stop))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tickrows", default="")
    a = ap.parse_args()
    mkt = {m.epic: m for m in MARKETS if m.epic in BREAKOUT_CONFIGS}

    print("=" * 104)
    print("A. ARCHIVE REPLAY — entry fill gap per market (positive = ADVERSE, i.e. worse than the backtest)")
    print("=" * 104)
    print("%-20s %4s %9s %9s %9s %9s %9s %9s" % (
        "MARKET", "n", "medOver", "med R", "mean R", "+halfSpr", "TOTAL R", "vs 0.143"))
    rows = []
    for epic, mk in mkt.items():
        f = ARCHIVE / f"{epic}.jsonl"
        if not f.exists() or f.stat().st_size < 1000:
            continue
        s = replay(mk)
        if len(s) < 5:
            print("%-20s %4d   (too few breaks to measure)" % (mk.name, len(s))); continue
        med_over = float(np.median([x["over"] for x in s]))
        med_r = float(np.median([x["over_r"] for x in s]))
        mean_r = float(np.mean([x["over_r"] for x in s]))
        sp = SPREAD.get(mk.name)
        half_r = (sp / 2) / float(np.median([x["stop"] for x in s])) if sp else None
        tot = med_r + half_r if half_r is not None else None
        print("%-20s %4d %9.2f %+9.3f %+9.3f %9s %9s %9s" % (
            mk.name, len(s), med_over, med_r, mean_r,
            ("%.3f" % half_r) if half_r is not None else "n/a",
            ("%+.3f" % tot) if tot is not None else "n/a",
            ("%+.3f" % (tot - 0.143)) if tot is not None else "n/a"))
        rows.append((mk.name, tot, med_r, len(s)))
    ok = [r for r in rows if r[1] is not None]
    if ok:
        print("\n  pooled median TOTAL gap = %+.3fR   (flat convention charges 0.143R)"
              % float(np.median([r[1] for r in ok])))
        print("  range across markets    = %+.3fR .. %+.3fR"
              % (min(r[1] for r in ok), max(r[1] for r in ok)))

    if a.tickrows and Path(a.tickrows).exists():
        print("\n" + "=" * 104)
        print("B. GROUND TRUTH — logged live breaks (Breakout-tick rows) paired with the archive close of that bar")
        print("=" * 104)
        tr = json.load(open(a.tickrows))
        cache = {}
        agg = {}
        for r in tr:
            m = re.search(r"level=([\d.]+) exec=([\d.]+).*?stop=([\d.]+) bar=", r["reject_reason"])
            if not m or r["epic"] not in mkt:
                continue
            lvl, ex, stop = float(m.group(1)), float(m.group(2)), float(m.group(3))
            # the BREAKING bar is the one containing the tick, not the arming bar
            bar = pd.Timestamp(r["timestamp"]).floor("1h")
            if r["epic"] not in cache:
                f = ARCHIVE / f"{r['epic']}.jsonl"
                if not f.exists() or f.stat().st_size < 1000:
                    cache[r["epic"]] = None
                else:
                    d = load_1h(r["epic"]); cache[r["epic"]] = d.set_index("date")
            d = cache[r["epic"]]
            if d is None or stop <= 0:
                continue
            try:
                close = float(d.loc[bar, "close"])
            except (KeyError, TypeError):
                continue
            over = (close - lvl) if r["direction"] == "BUY" else (lvl - close)
            tick = (ex - lvl) if r["direction"] == "BUY" else (lvl - ex)
            agg.setdefault(r["market_name"], []).append((over / stop, tick / stop))
        print("%-20s %4s %12s %12s %12s" % ("MARKET", "n", "closeGap R", "tickGap R", "tick saves"))
        for k, v in sorted(agg.items(), key=lambda x: -np.median([y[0] for y in x[1]])):
            c = float(np.median([y[0] for y in v])); t = float(np.median([y[1] for y in v]))
            print("%-20s %4d %+12.3f %+12.3f %+12.3f" % (k, len(v), c, t, c - t))
        allv = [y for v in agg.values() for y in v]
        if allv:
            print("%-20s %4d %+12.3f %+12.3f %+12.3f" % (
                "** POOLED **", len(allv),
                float(np.median([y[0] for y in allv])), float(np.median([y[1] for y in allv])),
                float(np.median([y[0] - y[1] for y in allv]))))


if __name__ == "__main__":
    main()
