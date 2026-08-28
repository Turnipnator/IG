#!/usr/bin/env python3
"""Forex MTF-pullback entry backtest (2026-06-18).

Question (user): the "arm a resting order and wait for a retrace toward the EMA
before entering" discipline only runs on FTSE/NASDAQ/Gold. Forex enters
immediately on the signal candle. Does pullback-entry help the forex book too?

WHY A BESPOKE HARNESS (not src/backtest.Backtester.run): the stock engine ALWAYS
applies a 3-candle MACD exit and never models breakeven / ATR-trailing / ranging-
ADX exit. But the live forex profiles use `use_macd_exit=False` — their exits are
stop + TP + breakeven(entry) + ATR-trail + ranging-ADX-drop. Reusing the MACD
engine would answer the wrong question. So this models the REAL forex exit stack,
and — critically — runs the SAME exits in both arms (immediate vs pullback) so the
ONLY variable is entry TIMING.

Faithful to the live profiles:
  forex  (EUR/USD, GBP/USD): EMA 9/21/50, RSI7 70/30 buy<55/sell>45, ADX 30,
         stop 1.0xATR, R:R 2.0, require_htf (DAILY), BE 0.7, ATR-trail 1.5.
  usdjpy (USD/JPY):          EMA 9/21/50, RSI7 80/20, ADX 35, stop 1.5xATR,
         R:R 2.0, require_htf (HOURLY), BE 0.9, ATR-trail 1.5.

KEY: `window` is in CANDLES, and forex runs on bigger candles than the 5m
indices/Gold. window=6 means 6h on EUR/GBP (1h bars) and 90m on USD/JPY (15m
bars) vs 30m on the 5m markets — so this sweeps SHORT windows too, to see whether
a timeframe-appropriate window helps even though the off-the-shelf 6 likely won't.

Yahoo cash (EURUSD=X / GBPUSD=X / USDJPY=X) != IG DFB, ZERO IG API cost. Directional
lead on whether the TIMING layer helps, not a live-P&L promise. Caveats: no regime
filter, no trading-hours filter, no spread modelled (all apply equally to both arms,
so they don't bias the immediate-vs-pullback delta).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_ema

logging.basicConfig(level=logging.ERROR)

TICKER_MAP.update({
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
})

# (name, interval, htf_interval, days, params, min_conf, allowed_dir)
FOREX = dict(
    ema_fast=9, ema_medium=21, ema_slow=50,
    rsi_period=7, rsi_overbought=70, rsi_oversold=30,
    rsi_buy_max=55, rsi_sell_min=45,
    atr_period=14, adx_threshold=30,
)
USDJPY = dict(FOREX, rsi_overbought=80, rsi_oversold=20, adx_threshold=35)

MARKETS = [
    # name        interval htf   days params  conf  stop  be    trail rangingdrop
    ("EUR/USD",  "1h",  "1d", 365, FOREX,  0.65, 1.0, 0.7, 1.5, 10),
    ("GBP/USD",  "1h",  "1d", 365, FOREX,  0.55, 1.0, 0.7, 1.5, 10),
    ("USD/JPY",  "15m", "1h",  59, USDJPY, 0.55, 1.5, 0.9, 1.5, 10),
]
RR = 2.0


def _enter(direction, price, atr, stop_mult, min_stop):
    sd = max(atr * stop_mult, min_stop)
    ld = sd * RR
    if direction == "SELL":
        return dict(dir=direction, entry=price, stop=price + sd, limit=price - ld,
                    sd=sd, be=False)
    return dict(dir=direction, entry=price, stop=price - sd, limit=price + ld,
                sd=sd, be=False)


def _step_exit(pos, row, adx_threshold, ranging_drop, be_trig, trail_mult):
    """Faithful forex exit: stop/TP (intrabar) > ranging-ADX (close) > then update
    BE+trail for the NEXT candle. Returns (reason, price) or (None, None)."""
    hi, lo, close, atr, adx = row["high"], row["low"], row["close"], row["atr"], row["adx"]
    d = pos["dir"]
    # 1) stop / take-profit on the stop & limit carried in from prior candles.
    if d == "BUY":
        if lo <= pos["stop"]:
            return "Stop", pos["stop"]
        if hi >= pos["limit"]:
            return "TP", pos["limit"]
    else:
        if hi >= pos["stop"]:
            return "Stop", pos["stop"]
        if lo <= pos["limit"]:
            return "TP", pos["limit"]
    # 2) ranging exit — ADX collapses below threshold - drop (forex, no MACD exit).
    if ranging_drop and not pd.isna(adx) and adx < (adx_threshold - ranging_drop):
        return "Ranging", close
    # 3) update breakeven + ATR-trail (affect future candles only — no look-ahead).
    if not pd.isna(atr):
        profit = (close - pos["entry"]) if d == "BUY" else (pos["entry"] - close)
        if not pos["be"] and profit >= be_trig * pos["sd"]:
            pos["be"] = True
            pos["stop"] = max(pos["stop"], pos["entry"]) if d == "BUY" else min(pos["stop"], pos["entry"])
        if pos["be"]:
            if d == "BUY":
                pos["stop"] = max(pos["stop"], close - trail_mult * atr, pos["entry"])
            else:
                pos["stop"] = min(pos["stop"], close + trail_mult * atr, pos["entry"])
    return None, None


def simulate(df, htf, bt, conf_min, stop_mult, be_trig, trail_mult, ranging_drop,
             adx_threshold, allowed, window, frac, min_stop, cooldown):
    trades, dropped = [], 0
    pos = pending = None
    cooldown_until = -1
    for i in range(bt.params["ema_slow"], len(df)):
        row = df.iloc[i]
        close, atr = row["close"], row["atr"]
        if pos:
            reason, price = _step_exit(pos, row, adx_threshold, ranging_drop, be_trig, trail_mult)
            if reason:
                pnlp = ((price - pos["entry"]) if pos["dir"] == "BUY"
                        else (pos["entry"] - price)) / pos["entry"] * 100
                trades.append(dict(dir=pos["dir"], pnlp=pnlp, reason=reason))
                if pnlp < 0:
                    cooldown_until = i + cooldown
                pos = None
            continue
        if i <= cooldown_until:
            pending = None
            continue
        if pending:
            if i > pending["deadline"]:
                dropped += 1
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
        if direction and conf >= conf_min:
            if allowed and direction != allowed:
                continue
            if window == 0:
                pos = _enter(direction, close, atr, stop_mult, min_stop)
            else:
                pending = dict(dir=direction, sc=close, atr=atr, deadline=i + window)
    return trades, dropped


def stats(trades):
    n = len(trades)
    if not n:
        return "no trades"
    w = sum(1 for t in trades if t["pnlp"] > 0)
    p = sum(t["pnlp"] for t in trades)
    gp = sum(t["pnlp"] for t in trades if t["pnlp"] > 0)
    gl = -sum(t["pnlp"] for t in trades if t["pnlp"] < 0)
    pf = gp / gl if gl > 0 else 999
    return f"n={n:3d} WR={w/n*100:3.0f}% P&L={p:+7.2f}% PF={pf:5.2f}"


def _build_htf(bt, name, htf_interval, days):
    htf = bt.fetch_data(name, days, htf_interval)
    if htf is None or htf.empty:
        return None
    htf = htf.copy()
    htf["ema_9"] = calculate_ema(htf["close"], 9)
    htf["ema_21"] = calculate_ema(htf["close"], 21)
    return htf


def run_market(name, interval, htf_interval, days, params, conf, stop_mult,
               be_trig, trail_mult, ranging_drop, allowed=None):
    bt = Backtester(params=params.copy())
    raw = bt.fetch_data(name, days, interval)
    if raw is None or len(raw) < 100:
        print(f"\n{name} {interval}: NO/THIN data ({0 if raw is None else len(raw)})")
        return
    df = bt.add_indicators(raw)
    htf = _build_htf(bt, name, htf_interval, days)
    if htf is None:
        print(f"\n{name}: no HTF data"); return
    min_stop = float(df["close"].median()) * 0.0003   # tiny floor — let ATR drive
    adx_threshold = params["adx_threshold"]
    interval_min = {"15m": 15, "1h": 60}[interval]
    cooldown = max(1, round(60 / interval_min))        # live = 60min after a loss
    print(f"\n{'='*80}\n{name} {interval}/{days}d, HTF {htf_interval} — EMA 9/21/50, "
          f"ADX {adx_threshold}, stop {stop_mult}xATR, R:R {RR}, BE {be_trig}, "
          f"trail {trail_mult}x (no MACD)\n{'='*80}")
    base, _ = simulate(df, htf, bt, conf, stop_mult, be_trig, trail_mult, ranging_drop,
                       adx_threshold, allowed, 0, 0, min_stop, cooldown)
    print(f"  IMMEDIATE (live, window=0):                        {stats(base)}")
    print(f"  {'window':>6}{'=time':>7}{'frac':>6}   pullback-entry (DROP if no retrace)   dropped")
    for window in (2, 3, 6, 12):
        wt = f"{window*interval_min/60:g}h"
        for frac in (0.25, 0.5, 1.0):
            tr, dr = simulate(df, htf, bt, conf, stop_mult, be_trig, trail_mult,
                              ranging_drop, adx_threshold, allowed, window, frac,
                              min_stop, cooldown)
            print(f"  {window:>6}{wt:>7}{frac:>6.2f}   {stats(tr):<38} {dr}")


def main():
    for name, interval, htf_i, days, params, conf, stop_mult, be, trail, rdrop in MARKETS:
        run_market(name, interval, htf_i, days, params, conf, stop_mult, be, trail, rdrop)


if __name__ == "__main__":
    main()
