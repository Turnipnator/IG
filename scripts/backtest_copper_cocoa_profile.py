#!/usr/bin/env python3
"""Build dedicated Copper / Cocoa profiles (2026-06-11). Cull review established
the direction edge (Copper long PF 1.38; Cocoa short PF 2.26, long a net loser).
Both run bare `default` (EMA 9/21/50, ADX 30, stop 1.8x, R:R 2.0). This sweeps
ADX floor × stop × R:R on the RELEVANT direction to find a tuned, robust profile
— pick a config that's good across its neighbours, not the single peak (anti-
overfit). Yahoo HG=F / CC=F 1h/700d.

LEAN loop: indicators are computed ONCE per market (EMA/RSI/ATR periods are
fixed), then each (adx, stop, rr) combo is swept over the same frame using the
core EMA+RSI+ADX entry gate and stop/TP/MACD-3-candle exits — no per-candle
regime classification or HTF (the O(n^2) cost that made the first version hang).
Immediate entry, direction-gated. Yahoo continuous != IG contract → directional.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from src.backtest import Backtester

logging.basicConfig(level=logging.ERROR)

BASE = {"ema_fast": 9, "ema_medium": 21, "ema_slow": 50, "rsi_period": 7,
        "rsi_overbought": 70, "rsi_oversold": 30, "rsi_buy_max": 60,
        "rsi_sell_min": 40, "atr_period": 14, "adx_threshold": 30,
        "stop_atr_multiplier": 1.8, "reward_risk_ratio": 2.0}

TARGETS = [
    ("Copper",   "HG=F", "BUY",  (30, 1.8, 2.0)),
    ("NY Cocoa", "CC=F", "SELL", (30, 1.8, 2.0)),
]
ADX, STOP, RR = [25, 30, 35], [1.5, 1.8, 2.2], [2.0, 2.5, 3.0]
MIN_N = 15


def daily_htf(df):
    """Per-1h-candle daily HTF trend (BULLISH/BEARISH/NEUTRAL) from the PRIOR
    completed day's EMA9/EMA21 — O(n), mirrors Backtester.calculate_htf_trend."""
    d = df.copy()
    d.index = pd.to_datetime(d["date"])
    daily = d["close"].resample("1D").last().dropna()
    e9 = daily.ewm(span=9, adjust=False).mean()
    e21 = daily.ewm(span=21, adjust=False).mean()
    trend = pd.Series("NEUTRAL", index=daily.index)
    trend[(e9 > e21) & (daily > e21)] = "BULLISH"
    trend[(e9 < e21) & (daily < e21)] = "BEARISH"
    trend = trend.shift(1).fillna("NEUTRAL")          # use the prior completed day
    return trend.reindex(d.index, method="ffill").fillna("NEUTRAL").to_numpy()


def signal(row, adx_floor):
    if pd.isna(row["ema_slow"]) or pd.isna(row["adx"]) or pd.isna(row["rsi"]):
        return None
    if row["adx"] < adx_floor:
        return None
    ef, em, es, c, rsi = row["ema_fast"], row["ema_medium"], row["ema_slow"], row["close"], row["rsi"]
    if ef > em > es and c > es and BASE["rsi_oversold"] < rsi < BASE["rsi_buy_max"]:
        return "BUY"
    if ef < em < es and c < es and BASE["rsi_sell_min"] < rsi < BASE["rsi_overbought"]:
        return "SELL"
    return None


def simulate(df, side, adx_floor, stop_mult, rr, min_stop, htf_arr=None):
    trades = []
    pos = None
    cooldown_until = -1
    n = len(df)
    for i in range(50, n):
        row = df.iloc[i]
        close, atr = row["close"], row["atr"]
        if pos:
            d, reason, px = pos["dir"], None, None
            if d == "BUY" and close <= pos["stop"]: reason, px = "Stop", pos["stop"]
            elif d == "SELL" and close >= pos["stop"]: reason, px = "Stop", pos["stop"]
            if d == "BUY" and close >= pos["limit"]: reason, px = "TP", pos["limit"]
            elif d == "SELL" and close <= pos["limit"]: reason, px = "TP", pos["limit"]
            if i >= 3 and not reason:
                last3 = [df.iloc[i - j]["macd_hist"] for j in range(3)]
                if d == "BUY" and all((not pd.isna(h)) and h < 0 for h in last3): reason, px = "MACD", close
                elif d == "SELL" and all((not pd.isna(h)) and h > 0 for h in last3): reason, px = "MACD", close
            if reason:
                pnlp = ((px - pos["entry"]) if d == "BUY" else (pos["entry"] - px)) / pos["entry"] * 100
                trades.append(pnlp)
                if pnlp < 0: cooldown_until = i + 1
                pos = None
            continue
        if i <= cooldown_until or pd.isna(atr):
            continue
        sig = signal(row, adx_floor)
        if sig == side and htf_arr is not None:
            need = "BULLISH" if side == "BUY" else "BEARISH"
            if htf_arr[i] != need:
                sig = None
        if sig == side:
            sd = max(atr * stop_mult, min_stop)
            ld = sd * rr
            if side == "BUY":
                pos = {"dir": "BUY", "entry": close, "stop": close - sd, "limit": close + ld}
            else:
                pos = {"dir": "SELL", "entry": close, "stop": close + sd, "limit": close - ld}
    return trades


def stat(trades):
    n = len(trades)
    if not n: return (0, 0, 0, 0)
    w = sum(1 for t in trades if t > 0)
    gp = sum(t for t in trades if t > 0); gl = -sum(t for t in trades if t < 0)
    pf = gp / gl if gl > 0 else 999
    return (n, w / n * 100, sum(trades), pf)


def main():
    for name, ticker, side, base in TARGETS:
        bt = Backtester(params=BASE.copy())
        raw = bt.fetch_data(name, 700, "1h")
        if raw is None or len(raw) < 100:
            # fetch uses TICKER_MAP; register the ticker
            from src.backtest import TICKER_MAP
            TICKER_MAP[name] = ticker
            raw = bt.fetch_data(name, 700, "1h")
        if raw is None or len(raw) < 100:
            print(f"\n{name}: NO/THIN data ({0 if raw is None else len(raw)})", flush=True); continue
        df = bt.add_indicators(raw)
        min_stop = float(df["close"].median()) * 0.005
        htf = daily_htf(df)
        print(f"\n{'='*72}\n{name} ({ticker}) — {side}-side, 1h/{len(df)}c, min_stop {min_stop:.2f}\n{'='*72}", flush=True)
        for label, harr in (("no-HTF", None), ("HTF-gated", htf)):
            n, wr, pnl, pf = stat(simulate(df, side, *base, min_stop, harr))
            print(f"  [{label}] BASELINE adx{base[0]}/stop{base[1]}/rr{base[2]}: {n}t WR {wr:.0f}% P&L {pnl:+.2f}% PF {pf:.2f}", flush=True)
            rows = []
            for adx in ADX:
                for stop in STOP:
                    for rr in RR:
                        n, wr, pnl, pf = stat(simulate(df, side, adx, stop, rr, min_stop, harr))
                        if n >= MIN_N:
                            rows.append((pf, pnl, n, wr, adx, stop, rr))
            rows.sort(reverse=True)
            print(f"  [{label}] top by PF (n>={MIN_N}):  {'PF':>6}{'P&L%':>8}{'t':>5}{'WR':>5}  adx stop rr", flush=True)
            for pf, pnl, n, wr, adx, stop, rr in rows[:6]:
                print(f"            {pf:>6.2f}{pnl:>+7.2f}%{n:>5}{wr:>5.0f}%  {adx:>3} {stop:>4} {rr:>3}", flush=True)
            if not rows:
                print("            (no config cleared the trade-count floor)", flush=True)


if __name__ == "__main__":
    main()
