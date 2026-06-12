#!/usr/bin/env python3
"""Screener veto-vs-outcome counterfactual (2026-06-12).

Question: when the screener vetoes a signal that PASSED every per-EPIC gate
("would have traded but inactive"), would that trade have WON or LOST? If vetoed
signals are net-negative, the screener earns its keep; if net-positive, it's
costing us edge.

Method: parse the `Screener veto:` events from the VPS persistent logs (saved to
/tmp/veto_events.txt), then for each DISTINCT setup simulate the trade the bot
WOULD have opened — entry at the just-closed candle, ATR stop + R:R target from
the market's live profile — forward-walking Yahoo candles to the first stop/limit
hit (horizon-capped, marked-to-market on timeout). Reported in R-multiples.

Caveats (same as every backtest here): Yahoo cash/continuous-future != IG DFB, so
this is a LEAD not proof; index profiles use MACD/ranging exits live which a pure
stop/limit sim ignores (it tends to overstate hold time -> treat index timeouts
loosely). Repeated vetoes of one setup are collapsed (the bot opens once, holds).
Zero IG API cost (Yahoo only).
"""
import os
import re
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

from src.indicators import calculate_atr

# market -> (yahoo ticker, candle_interval_min, stop_atr_mult, reward_risk)
# US indices + Nikkei use the FUTURES contract (trades ~23h) not the cash index:
# IG's DFB index is priced off the future out of hours, so the future is both the
# better proxy AND it lets us adjudicate the many vetoes that fire outside the
# cash session. FTSE/HK have no liquid Yahoo future -> cash only (hours-limited).
MK = {
    "S&P 500":        ("ES=F",  5, 1.5, 2.0),
    "NASDAQ 100":     ("NQ=F",  5, 2.0, 2.0),
    "Wall Street":    ("YM=F",  5, 1.5, 2.0),
    "FTSE 100":       ("^FTSE", 5, 2.0, 2.0),
    "Japan 225":      ("NKD=F", 5, 1.5, 2.0),
    "Hong Kong HS50": ("^HSI",  5, 1.5, 2.0),
    "Gold":           ("GC=F",  5, 1.5, 3.0),
    "NASDAQ":         ("NQ=F",  5, 2.0, 2.0),
    "USD/JPY":        ("JPY=X", 15, 1.5, 2.0),
    "GBP/USD":        ("GBPUSD=X", 60, 1.0, 2.0),
    "EUR/USD":        ("EURUSD=X", 60, 1.0, 2.0),
    "NY Cotton":      ("CT=F",  60, 1.8, 2.0),
    # AI Index -> no Yahoo equivalent, skipped
}
INTERVAL_STR = {5: "5m", 15: "15m", 60: "60m"}
HORIZON = {5: 78, 15: 40, 60: 30}   # forward candles before mark-to-market

VETO_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Screener veto: "
    r"(?P<mkt>.+?) (?P<dir>BUY|SELL) @ (?P<conf>\d+)%.*?\(score (?P<score>\d+)"
)


def parse_events(path):
    out = []
    for line in open(path):
        m = VETO_RE.search(line)
        if not m:
            continue
        out.append({
            "ts": pd.Timestamp(m["ts"], tz="UTC"),
            "mkt": m["mkt"].strip(),
            "dir": m["dir"],
            "conf": int(m["conf"]),
            "score": int(m["score"]),
        })
    return out


def collapse(events):
    """Collapse repeated vetoes of the same market+direction within ~6 candles
    into one setup (the bot opens once and holds, it doesn't re-enter each bar)."""
    events = sorted(events, key=lambda e: (e["mkt"], e["dir"], e["ts"]))
    trades, last = [], {}
    for e in events:
        ci = MK.get(e["mkt"], (None, 5))[1]
        gap = timedelta(minutes=ci * 6)
        key = (e["mkt"], e["dir"])
        if key in last and (e["ts"] - last[key]) <= gap:
            last[key] = e["ts"]
            continue
        last[key] = e["ts"]
        trades.append(e)
    return trades


def fetch(ticker, interval, days):
    df = yf.download(ticker, period=f"{days}d", interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                      for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    dcol = next((c for c in ["datetime", "date", "index"] if c in df.columns), None)
    df = df.rename(columns={dcol: "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)   # -> UTC, removes tz ambiguity
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
    return df.dropna(subset=["atr"]).reset_index(drop=True)


def simulate(df, e):
    ticker, ci, stop_mult, rr = MK[e["mkt"]]
    # entry = the candle that had just closed when the veto fired (ts - interval)
    target_close = e["ts"] - timedelta(minutes=ci)
    tol = timedelta(minutes=ci)
    df["d"] = (df["ts"] - target_close).abs()
    i = int(df["d"].idxmin())
    if df.loc[i, "d"] > tol:
        return "nocandle"  # no Yahoo candle near veto (market closed / gap)
    entry = float(df.loc[i, "close"])
    atr = float(df.loc[i, "atr"])
    min_stop = float(df["close"].median()) * 0.005
    stop_dist = max(stop_mult * atr, min_stop)
    if stop_dist <= 0:
        return None
    tgt_dist = rr * stop_dist
    horizon = HORIZON[ci]
    fwd = df.iloc[i + 1:i + 1 + horizon]
    for _, r in fwd.iterrows():
        hi, lo = float(r["high"]), float(r["low"])
        if e["dir"] == "BUY":
            if lo <= entry - stop_dist:     # stop first (conservative on same-bar)
                return -1.0, "stop"
            if hi >= entry + tgt_dist:
                return rr, "target"
        else:
            if hi >= entry + stop_dist:
                return -1.0, "stop"
            if lo <= entry - tgt_dist:
                return rr, "target"
    if len(fwd) == 0:
        return None
    last = float(fwd.iloc[-1]["close"])
    mtm = (last - entry) if e["dir"] == "BUY" else (entry - last)
    return mtm / stop_dist, "timeout"   # marked-to-market in R


def main():
    events = parse_events("/tmp/veto_events.txt")
    print(f"Parsed {len(events)} raw veto events")
    skipped_ai = [e for e in events if e["mkt"] not in MK]
    if skipped_ai:
        from collections import Counter
        print("  (no Yahoo ticker, skipped):",
              dict(Counter(e["mkt"] for e in skipped_ai)))
    trades = [t for t in collapse(events) if t["mkt"] in MK]
    print(f"Collapsed to {len(trades)} distinct setups\n")

    frames = {}
    rows, by_mkt = [], {}
    drop = {"nofetch": 0, "nocandle": 0}
    drop_mkt = {}
    for t in trades:
        ticker, ci, _, _ = MK[t["mkt"]]
        days = 59 if ci < 60 else 360
        key = (ticker, ci)
        if key not in frames:
            frames[key] = fetch(ticker, INTERVAL_STR[ci], days)
        df = frames[key]
        if df is None:
            drop["nofetch"] += 1
            continue
        res = simulate(df.copy(), t)
        if res is None or res == "nocandle":
            drop["nocandle"] += 1
            drop_mkt[t["mkt"]] = drop_mkt.get(t["mkt"], 0) + 1
            continue
        r, how = res
        rows.append((t, r, how))
        by_mkt.setdefault(t["mkt"], []).append((t, r, how))

    print(f"Measurable: {len(rows)}/{len(trades)} setups  "
          f"(dropped: {drop['nocandle']} no-candle [market closed in Yahoo], "
          f"{drop['nofetch']} no-fetch)")
    if drop_mkt:
        print("  unmeasurable by market (fired outside Yahoo session):",
              {k: v for k, v in sorted(drop_mkt.items(), key=lambda x: -x[1])})
    print()

    print(f"{'Market':16s} {'Dir':4s} {'setups':>6s} {'win':>4s} {'loss':>5s} "
          f"{'t/o':>4s} {'netR':>7s}")
    print("-" * 52)
    grand = 0.0
    for mkt in sorted(by_mkt, key=lambda m: sum(r for _, r, _ in by_mkt[m])):
        for d in ("BUY", "SELL"):
            sub = [(t, r, h) for (t, r, h) in by_mkt[mkt] if t["dir"] == d]
            if not sub:
                continue
            w = sum(1 for _, r, h in sub if h == "target")
            l = sum(1 for _, r, h in sub if h == "stop")
            to = sum(1 for _, r, h in sub if h == "timeout")
            net = sum(r for _, r, _ in sub)
            grand += net
            print(f"{mkt:16s} {d:4s} {len(sub):>6d} {w:>4d} {l:>5d} {to:>4d} "
                  f"{net:>+7.2f}")
    n = len(rows)
    wins = sum(1 for _, r, h in rows if h == "target")
    losses = sum(1 for _, r, h in rows if h == "stop")
    tos = sum(1 for _, r, h in rows if h == "timeout")
    print("-" * 52)
    print(f"{'TOTAL':16s} {'':4s} {n:>6d} {wins:>4d} {losses:>5d} {tos:>4d} "
          f"{grand:>+7.2f}")
    if n:
        wr = wins / (wins + losses) * 100 if (wins + losses) else 0
        print(f"\nVetoed-signal hit-rate (stop/target only): {wr:.0f}% over "
              f"{wins + losses} resolved")
        print(f"Net counterfactual: {grand:+.2f}R across {n} setups "
              f"({grand / n:+.2f}R/setup)")
        print("  Net NEGATIVE -> screener saved money (vetoed losers).")
        print("  Net POSITIVE -> screener cost edge (vetoed winners).")
        print("  1R ~= the per-trade max loss (live ~GBP12-20).")
    # score-band cut
    print("\nBy screener-score band:")
    for lo, hi in [(0, 35), (35, 45), (45, 101)]:
        sub = [(t, r, h) for (t, r, h) in rows if lo <= t["score"] < hi]
        if sub:
            net = sum(r for _, r, _ in sub)
            print(f"  score {lo:>2d}-{hi - 1:>3d}: {len(sub):>3d} setups  net {net:>+6.2f}R")


if __name__ == "__main__":
    main()
