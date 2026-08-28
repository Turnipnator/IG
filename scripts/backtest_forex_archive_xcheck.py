#!/usr/bin/env python3
"""IG-native cross-check of the forex EXIT finding (2026-06-19). Run IN-CONTAINER.

The Yahoo focused test (scripts/backtest_forex_focused.py) found the BE + 1.5xATR
trail SCRATCHES winners before the 2.0 R:R target — pure stop/TP beat the live
exits on every pair. This re-tests that on the FREE IG candle archive (real DFB
prices), not Yahoo cash, to confirm it's not a proxy artefact.

CAVEAT: the archive is only ~9-11d, too short to build a faithful DAILY HTF for
EUR/GBP (needs 21 daily bars). So this DROPS the HTF gate and isolates the EXIT
mechanism: current-exits vs pure-stop/TP run on the SAME entry set, so the HTF
treatment cancels in the DELTA — which is exactly the thing we're cross-checking.
Direction split (short-only vs both) is also reported but is the SECONDARY,
regime-dependent finding; the exit delta is the robust one. Small n — directional.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester
from src.indicators import calculate_ema
from scripts.archive_loader import load_archive

logging.basicConfig(level=logging.ERROR)

FOREX = dict(ema_fast=9, ema_medium=21, ema_slow=50, rsi_period=7,
             rsi_overbought=70, rsi_oversold=30, rsi_buy_max=55, rsi_sell_min=45,
             atr_period=14, adx_threshold=30)
USDJPY = dict(FOREX, rsi_overbought=80, rsi_oversold=20, adx_threshold=35)

# epic, name, params, conf, stop_mult, be, trail, ranging, interval_min, htf_rule
PAIRS = [
    ("CS.D.EURUSD.TODAY.IP", "EUR/USD", FOREX,  0.65, 1.0, 0.7, 1.5, 10, 60, "4h"),
    ("CS.D.GBPUSD.TODAY.IP", "GBP/USD", FOREX,  0.55, 1.0, 0.7, 1.5, 10, 60, "4h"),
    ("CS.D.USDJPY.TODAY.IP", "USD/JPY", USDJPY, 0.55, 1.5, 0.9, 1.5, 10, 15, "1h"),
]
RR = 2.0


def _enter(direction, price, atr, stop_mult, min_stop):
    sd = max(atr * stop_mult, min_stop)
    ld = sd * RR
    if direction == "SELL":
        return dict(dir=direction, entry=price, stop=price + sd, limit=price - ld, sd=sd, be=False)
    return dict(dir=direction, entry=price, stop=price - sd, limit=price + ld, sd=sd, be=False)


def _step_exit(pos, row, adx_threshold, ranging_drop, be_trig, trail_mult):
    hi, lo, close, atr, adx = row["high"], row["low"], row["close"], row["atr"], row["adx"]
    d = pos["dir"]
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
    if ranging_drop and not pd.isna(adx) and adx < (adx_threshold - ranging_drop):
        return "Ranging", close
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
             adx_threshold, allowed, min_stop, cooldown, require_htf):
    trades, pos, cooldown_until = [], None, -1
    for i in range(bt.params["ema_slow"], len(df)):
        row = df.iloc[i]
        atr = row["atr"]
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
        if i <= cooldown_until or pd.isna(atr):
            continue
        htf_trend = bt.calculate_htf_trend("", row["date"], htf) if htf is not None else "NEUTRAL"
        direction, conf, _ = bt.check_entry_signal(row, htf_trend, require_htf_alignment=require_htf)
        if direction and conf >= conf_min:
            if allowed and direction != allowed:
                continue
            pos = _enter(direction, row["close"], atr, stop_mult, min_stop)
    return trades


def rstats(trades):
    n = len(trades)
    if not n:
        return None
    wins = [t for t in trades if t["pnlp"] > 0]
    losses = [t for t in trades if t["pnlp"] < 0]
    gp = sum(t["pnlp"] for t in wins)
    gl = -sum(t["pnlp"] for t in losses)
    avg_w = gp / len(wins) if wins else 0.0
    avg_l = -gl / len(losses) if losses else 0.0
    tp = sum(1 for t in trades if t["reason"] == "TP")
    trail_lock = sum(1 for t in trades if t["reason"] == "Stop" and t["pnlp"] >= 0)
    return dict(n=n, wr=len(wins) / n * 100, pnl=sum(t["pnlp"] for t in trades),
                pf=gp / gl if gl > 0 else 999.0, avg_w=avg_w, avg_l=avg_l,
                rr=(avg_w / -avg_l) if avg_l < 0 else 0.0, tp=tp, trail_lock=trail_lock)


def line(tag, s):
    if s is None:
        print(f"  {tag:<24} no trades"); return
    print(f"  {tag:<24} n={s['n']:>3} WR={s['wr']:>3.0f}% P&L={s['pnl']:>+7.2f}% PF={s['pf']:>5.2f}  "
          f"avgW={s['avg_w']:>+5.2f} avgL={s['avg_l']:>+6.2f} realR:R={s['rr']:>4.2f}  "
          f"[TP={s['tp']} trailLock={s['trail_lock']}]")


def build_htf(df, rule):
    idx = df.set_index("date")
    htf = (idx[["open", "high", "low", "close"]].resample(rule)
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
           .dropna().reset_index())
    if len(htf) < 22:
        return None  # not enough bars for ema_21 — caller drops the HTF gate
    htf["ema_9"] = calculate_ema(htf["close"], 9)
    htf["ema_21"] = calculate_ema(htf["close"], 21)
    return htf


def main():
    for epic, name, params, conf, stop, be, trail, rdrop, imin, htf_rule in PAIRS:
        raw = load_archive(epic)
        if raw is None or raw.empty:
            print(f"\n{name}: no archive"); continue
        if "date" not in raw.columns and "timestamp" in raw.columns:
            raw = raw.rename(columns={"timestamp": "date"})
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date").reset_index(drop=True)
        span = (raw["date"].iloc[-1] - raw["date"].iloc[0]).days or 1
        bt = Backtester(params=params.copy())
        df = bt.add_indicators(raw)
        htf = build_htf(raw, htf_rule)
        require_htf = htf is not None
        min_stop = float(df["close"].median()) * 0.0003
        cooldown = max(1, round(60 / imin))
        gate = f"HTF {htf_rule}-proxy gated" if require_htf else "NO HTF gate (archive too short)"
        print(f"\n{'='*86}\n{name} — IG archive {span}d, {len(df)} candles @ {imin}m, "
              f"ADX {params['adx_threshold']}, stop {stop}x — {gate}\n{'='*86}")
        print("  -- exit comparison (both directions, same entry set) --")
        line("current (BE+trail)", rstats(simulate(df, htf, bt, conf, stop, be, trail, rdrop,
             params["adx_threshold"], None, min_stop, cooldown, require_htf)))
        line("pure stop/TP", rstats(simulate(df, htf, bt, conf, stop, 999, 0.0, 0,
             params["adx_threshold"], None, min_stop, cooldown, require_htf)))
        print("  -- direction split (secondary, regime-dependent) --")
        for allowed, tag in [("SELL", "SELL-only (live exits)"), ("BUY", "BUY-only (live exits)")]:
            line(tag, rstats(simulate(df, htf, bt, conf, stop, be, trail, rdrop,
                 params["adx_threshold"], allowed, min_stop, cooldown, require_htf)))


if __name__ == "__main__":
    main()
