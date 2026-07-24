#!/usr/bin/env python3
"""Oil (and Gold, for context) Donchian-breakout viability test — 2026-07-24.

Motivation: user wants a discretionary /mode <market> breakout toggle (tips from
contacts that a market is about to trend). Before wiring oil live, test whether
N-bar HTF-filtered Donchian — the exact live forex implementation (src/breakout.py:
break of prior-N channel, 2xATR stop, Donchian N/2 trail, no TP, daily-EMA9/21 HTF
gate) — has any standalone edge on 730d of 1h Yahoo data at IG cost.

Cost: IG US Crude spread ~2.8pts on a ~6900pt contract ≈ 4bps/round-trip (charge
6bps to be safe); Gold ~0.3pts on ~4000 ≈ 1bp (charge 2bps). Yahoo CL=F/GC=F are
futures, near-1:1 proxies for IG's Month contracts (unlike cash-index proxies).
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, period=14):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def run(df, n, cost_frac, htf_filter=True):
    m = max(2, n // 2)
    df = df.copy()
    df["atr"] = atr(df)
    daily = df.set_index("date").resample("1D").agg(
        {"close": "last"}).dropna()
    e9, e21 = ema(daily.close, 9), ema(daily.close, 21)
    trend = pd.Series("NEUTRAL", index=daily.index)
    trend[(e9 > e21) & (daily.close > e21)] = "BULLISH"
    trend[(e9 < e21) & (daily.close < e21)] = "BEARISH"

    def htf_at(ts):
        idx = trend.index[trend.index <= ts - pd.Timedelta(days=1)]
        return trend.loc[idx[-1]] if len(idx) else "NEUTRAL"

    trades = []
    i = n + 15
    N = len(df)
    while i < N - 2:
        row = df.iloc[i]
        prior = df.iloc[i - n:i]
        upper, lower = prior.high.max(), prior.low.min()
        sig = None
        if row.high >= upper:
            sig = "BUY"
        elif row.low <= lower:
            sig = "SELL"
        if sig is None or pd.isna(row.atr):
            i += 1
            continue
        if htf_filter:
            ht = htf_at(row.date)
            if (sig == "BUY" and ht != "BULLISH") or (sig == "SELL" and ht != "BEARISH"):
                i += 1
                continue
        entry = df.iloc[i + 1].open
        sd = row.atr * 2.0
        stop = entry - sd if sig == "BUY" else entry + sd
        exit_px = None
        j = i + 1
        for j in range(i + 1, N):
            bar = df.iloc[j]
            ch = df.iloc[max(0, j - m):j]
            if sig == "BUY":
                trail = max(stop, ch.low.min()) if len(ch) >= m else stop
                if bar.low <= trail:
                    exit_px = trail
                    break
                stop = trail
            else:
                trail = min(stop, ch.high.max()) if len(ch) >= m else stop
                if bar.high >= trail:
                    exit_px = trail
                    break
                stop = trail
        if exit_px is None:
            break
        pts = (exit_px - entry) if sig == "BUY" else (entry - exit_px)
        ret = pts / entry - cost_frac
        trades.append(dict(ts=df.iloc[i + 1].date, ret=ret, r=pts / sd))
        i = j + 1
    return trades


def report(tag, trades):
    if not trades:
        print(f"    {tag:22s} n=0")
        return
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) else float("inf")
    mark = "+" if sum(rets) > 0.002 else ("x" if sum(rets) < -0.002 else ".")
    print(f"    {tag:22s} {mark} n={len(rets):3d} WR={len(wins)/len(rets)*100:3.0f}% "
          f"P&L={sum(rets)*100:+7.2f}% PF={pf:5.2f}")


for ticker, name, cost in [("CL=F", "US Crude", 0.0006), ("GC=F", "Gold", 0.0002)]:
    raw = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        print(f"{name}: NO DATA")
        continue
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    df = raw.reset_index().rename(columns={"Datetime": "date", "index": "date", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date", "open", "high", "low", "close"]].dropna().reset_index(drop=True)
    print(f"\n{'='*74}\n{name} ({ticker}) — {len(df)} 1h candles, cost {cost*1e4:.0f}bps/rt\n{'='*74}")
    for n in (40, 55, 70):
        print(f"  N={n} (trail M={max(2, n//2)}), HTF-filtered:")
        trades = run(df, n, cost)
        report("FULL 730d", trades)
        if trades:
            qs = np.array_split(trades, 4)
            for qi, q in enumerate(qs, 1):
                report(f"Q{qi}", list(q))
    print(f"  N=55 WITHOUT HTF filter (context):")
    report("FULL 730d", run(df, 55, cost, htf_filter=False))
