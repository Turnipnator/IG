#!/usr/bin/env python3
"""Indices MTF-pullback entry backtest (2026-06-15). Does deferring entry until a
pullback toward the EMA — the discipline we already gave Gold — beat the index
profiles' fire-immediately-on-the-signal-candle?

Motivation: 2026-06-15 (Monday) the bot took 5 trades, all losers (-£82.50). All
were with-trend pullback entries (price already beyond the fast EMA at entry, RSI
mid-range 43-53) that fired at market on a lagging EMA stack while momentum had
already turned — buying a dip / selling a bounce that was actually the start of a
reversal. Same failure family as the Gold early-entry problem and the Wall St
-£18.16 trade. Gold got a pullback-confirm fix (PF 1.91->3.28); does it help the
index book, or does it just give marginally better fills on a falling market?

Model (mirrors scripts/backtest_gold_pullback.py + main.py pending_pullback): on a
signal at close Cs, arm a resting order at Cs +/- frac*ATR (bounce for SELL, dip
for BUY); fill if price trades to it within `window` candles, else DROP the signal
(skip the runaway). window=0 = live's immediate entry (baseline). Stop/limit/HTF/
MACD-3-candle exit identical in both arms, so the ONLY variable is entry timing.

Faithful to each LIVE index profile (EMA 5/12/26, RSI 7, ADX 30/25, profile stop,
R:R 2.0, require_htf=True via hourly ema_9/21). Yahoo cash (^DJI/^FTSE/^NDX/^GSPC)
!= IG DFB, so this is a directional lead on whether the timing layer helps — not a
live P&L promise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_ema

logging.basicConfig(level=logging.ERROR)

# Yahoo cash tickers for the index book (added to the backtester's map at runtime).
TICKER_MAP.update({
    "Wall Street": "^DJI",
    "FTSE 100": "^FTSE",
    "NASDAQ 100": "^NDX",   # ^NDX already implied; explicit for clarity
    "S&P 500": "^GSPC",
})

# Per-market = the LIVE profile. (name, ticker-key, params, allowed_direction)
BASE = dict(
    ema_fast=5, ema_medium=12, ema_slow=26,
    rsi_period=7, rsi_overbought=70, rsi_oversold=30,
    rsi_buy_max=55, rsi_sell_min=45,
    atr_period=14, reward_risk_ratio=2.0,
)
MARKETS = [
    ("Wall Street", dict(BASE, adx_threshold=30, stop_atr_multiplier=1.5), None),   # indices
    ("FTSE 100",    dict(BASE, adx_threshold=30, stop_atr_multiplier=2.0), None),   # indices_tight
    ("NASDAQ 100",  dict(BASE, adx_threshold=30, stop_atr_multiplier=2.0), None),   # indices_wide
    ("S&P 500",     dict(BASE, adx_threshold=25, stop_atr_multiplier=1.5), "BUY"),  # indices_selective (long-only)
]
MIN_CONF = 0.55
RR = 2.0


def _enter(direction, entry_price, atr_fill, stop_mult, min_stop):
    sd = max(atr_fill * stop_mult, min_stop)
    ld = sd * RR
    if direction == "SELL":
        return {"dir": direction, "entry": entry_price, "stop": entry_price + sd, "limit": entry_price - ld}
    return {"dir": direction, "entry": entry_price, "stop": entry_price - sd, "limit": entry_price + ld}


def _exit_check(pos, row, df, i):
    close = row["close"]
    reason = price = None
    d = pos["dir"]
    if d == "SELL" and close >= pos["stop"]:
        reason, price = "Stop", pos["stop"]
    elif d == "BUY" and close <= pos["stop"]:
        reason, price = "Stop", pos["stop"]
    if d == "SELL" and close <= pos["limit"]:
        reason, price = "TP", pos["limit"]
    elif d == "BUY" and close >= pos["limit"]:
        reason, price = "TP", pos["limit"]
    if i >= 3:
        last3 = [df.iloc[i - j]["macd_hist"] for j in range(3)]
        if d == "SELL" and all((not pd.isna(h)) and h > 0 for h in last3):
            reason, price = "MACD", close
        elif d == "BUY" and all((not pd.isna(h)) and h < 0 for h in last3):
            reason, price = "MACD", close
    return reason, price


def simulate(df, htf, bt, stop_mult, allowed, window, frac, min_stop, cooldown):
    trades, skipped = [], 0
    pos = pending = None
    cooldown_until = -1
    for i in range(bt.params["ema_slow"], len(df)):
        row = df.iloc[i]
        close, atr = row["close"], row["atr"]
        if pos:
            reason, price = _exit_check(pos, row, df, i)
            if reason:
                if pos["dir"] == "SELL":
                    pnlp = (pos["entry"] - price) / pos["entry"] * 100
                else:
                    pnlp = (price - pos["entry"]) / pos["entry"] * 100
                trades.append({"dir": pos["dir"], "pnlp": pnlp, "reason": reason})
                if pnlp < 0:
                    cooldown_until = i + cooldown
                pos = None
            continue
        if i <= cooldown_until:
            pending = None
            continue
        if pending:
            if i > pending["deadline"]:
                skipped += 1
                pending = None
            else:
                if pending["dir"] == "SELL":
                    tgt = pending["sc"] + frac * pending["atr"]
                    if row["high"] >= tgt:
                        pos = _enter("SELL", tgt, atr, stop_mult, min_stop); pending = None
                else:
                    tgt = pending["sc"] - frac * pending["atr"]
                    if row["low"] <= tgt:
                        pos = _enter("BUY", tgt, atr, stop_mult, min_stop); pending = None
            if pos or pending:
                continue
        if pd.isna(atr):
            continue
        htf_trend = bt.calculate_htf_trend("", row["date"], htf)
        direction, conf, _ = bt.check_entry_signal(row, htf_trend, require_htf_alignment=True)
        if direction and conf >= MIN_CONF:
            if allowed and direction != allowed:
                continue
            if window == 0:
                pos = _enter(direction, close, atr, stop_mult, min_stop)
            else:
                pending = {"dir": direction, "sc": close, "atr": atr, "deadline": i + window}
    return trades, skipped


def stats(trades):
    n = len(trades)
    if not n:
        return "no trades"
    w = sum(1 for t in trades if t["pnlp"] > 0)
    p = sum(t["pnlp"] for t in trades)
    gp = sum(t["pnlp"] for t in trades if t["pnlp"] > 0)
    gl = -sum(t["pnlp"] for t in trades if t["pnlp"] < 0)
    pf = gp / gl if gl > 0 else 999
    return f"n={n:3d} WR={w/n*100:3.0f}% P&L={p:+7.2f}% PF={pf:.2f}"


def run_market(name, params, allowed, interval, days, cooldown):
    bt = Backtester(params=params.copy())
    raw = bt.fetch_data(name, days, interval)
    if raw is None or len(raw) < 100:
        print(f"\n{name} {interval}: NO/THIN data ({0 if raw is None else len(raw)})"); return
    df = bt.add_indicators(raw)
    # HTF = hourly ema_9/21 (calculate_htf_trend reads those columns)
    htf_raw = bt.fetch_data(name, min(days, 365), "1h")
    htf = htf_raw.copy()
    htf["ema_9"] = calculate_ema(htf["close"], 9)
    htf["ema_21"] = calculate_ema(htf["close"], 21)
    min_stop = float(df["close"].median()) * 0.0005  # tiny floor — let ATR drive the stop
    stop_mult = params["stop_atr_multiplier"]
    dirn = f" [{allowed}-only]" if allowed else ""
    print(f"\n{'='*78}\n{name} {interval}/{days}d{dirn} — EMA 5/12/26, ADX {params['adx_threshold']}, "
          f"stop {stop_mult}x, R:R {RR}, require_htf\n{'='*78}")
    base, _ = simulate(df, htf, bt, stop_mult, allowed, 0, 0, min_stop, cooldown)
    print(f"  IMMEDIATE (IG live, window=0):           {stats(base)}")
    print(f"  {'window':>6}{'frac':>6}  pullback-entry (DROP if no pullback)         dropped")
    for window in (3, 6, 12):
        for frac in (0.25, 0.5, 1.0):
            tr, sk = simulate(df, htf, bt, stop_mult, allowed, window, frac, min_stop, cooldown)
            print(f"  {window:>6}{frac:>6.2f}  {stats(tr):<44} {sk}")


def main():
    for name, params, allowed in MARKETS:
        run_market(name, params, allowed, "5m", 59, 12)   # IG's timeframe (12x5m = 1h cooldown)
    print()
    for name, params, allowed in MARKETS:
        run_market(name, params, allowed, "1h", 365, 1)   # bigger sample


if __name__ == "__main__":
    main()
