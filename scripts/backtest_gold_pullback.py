#!/usr/bin/env python3
"""Gold MTF-pullback entry backtest (2026-06-11). Does deferring entry until a
pullback toward the EMA — Oanda's MTF discipline — beat IG's fire-immediately-
on-the-signal-candle? Yahoo GC=F, zero IG cost.

Motivation: IG Gold 00:25 short entered at a local low and was stopped in 2 min
as price bounced +17pts; Oanda shorted Gold the same day at 15:00, 31pts higher,
and won. The strategy params are siblings — the gap is entry timing. Oanda waits
up to 8×M15 for a $0.50 pullback toward the EMA, then enters short into the
bounce (higher price = more room, better R:R), and SKIPS if no pullback comes.

Model: on a SELL signal at close Cs, place a limit to short at Cs + frac×ATR
(a bounce); fill if price trades up to it within `window` candles, else cancel
the signal (Oanda skip). BUY symmetric (dip of frac×ATR). Stop/limit recompute
from the fill price, so it's not just a better entry — the whole trade reframes.
window=0 = IG's immediate entry (baseline). Same exit logic both arms (stop/TP/
MACD-3-candle/1h-loss-cooldown) so the only variable is entry timing.

Faithful to the LIVE Gold profile (EMA 3/8/21, RSI 85/15, ADX 35, stop 1.5x,
R:R 3.0) — not the all-EPIC harness's generic 9/21/50. Yahoo GC=F continuous !=
IG Gold DFB, so this is a directional lead on whether the timing layer helps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from src.backtest import Backtester

logging.basicConfig(level=logging.ERROR)

GOLD_PARAMS = {
    "ema_fast": 3, "ema_medium": 8, "ema_slow": 21,
    "rsi_period": 7, "rsi_overbought": 85, "rsi_oversold": 15,
    "rsi_buy_max": 60, "rsi_sell_min": 40,
    "adx_threshold": 35, "atr_period": 14,
    "stop_atr_multiplier": 1.5, "reward_risk_ratio": 3.0,
}
MIN_CONF = 0.55
RR = 3.0


def _enter(direction, entry_price, atr_fill, min_stop):
    sd = max(atr_fill * GOLD_PARAMS["stop_atr_multiplier"], min_stop)
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


def simulate(df, bt, market, window, frac, min_stop, cooldown_candles):
    trades, skipped = [], 0
    pos = pending = None
    cooldown_until = -1
    for i in range(GOLD_PARAMS["ema_slow"], len(df)):
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
                    cooldown_until = i + cooldown_candles
                pos = None
            continue
        if i <= cooldown_until:
            pending = None
            continue
        # resolve a pending pullback order
        if pending:
            if i > pending["deadline"]:
                skipped += 1
                pending = None
            else:
                if pending["dir"] == "SELL":
                    tgt = pending["sc"] + frac * pending["atr"]
                    if row["high"] >= tgt:
                        pos = _enter("SELL", tgt, atr, min_stop); pending = None
                else:
                    tgt = pending["sc"] - frac * pending["atr"]
                    if row["low"] <= tgt:
                        pos = _enter("BUY", tgt, atr, min_stop); pending = None
            if pos or pending:
                continue
        # new signal (flat, no pending)
        if pd.isna(atr):
            continue
        direction, conf, _ = bt.check_entry_signal(row, "NEUTRAL", False)
        if direction and conf >= MIN_CONF:
            if window == 0:
                pos = _enter(direction, close, atr, min_stop)
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


def run_tf(interval, days, cooldown_candles):
    bt = Backtester(params=GOLD_PARAMS.copy())
    raw = bt.fetch_data("Gold", days, interval)
    if raw is None or len(raw) < 100:
        print(f"\n{interval}: NO/THIN data ({0 if raw is None else len(raw)})"); return
    df = bt.add_indicators(raw)
    min_stop = float(df["close"].median()) * 0.005
    print(f"\n{'='*72}\nGOLD {interval}/{days}d — GC=F, EMA 3/8/21, ADX 35, stop 1.5x, R:R 3.0 "
          f"(min_stop {min_stop:.1f})\n{'='*72}")
    base, _ = simulate(df, bt, "Gold", 0, 0, min_stop, cooldown_candles)
    print(f"  IMMEDIATE (IG live, window=0):           {stats(base)}")
    print(f"  {'window':>6}{'frac':>6}  pullback-entry (skip if no pullback)        skipped")
    for window in (3, 6, 12):
        for frac in (0.25, 0.5, 1.0):
            tr, sk = simulate(df, bt, "Gold", window, frac, min_stop, cooldown_candles)
            print(f"  {window:>6}{frac:>6.2f}  {stats(tr):<44} {sk}")


def main():
    run_tf("5m", 59, 12)    # IG's actual timeframe (12x5m = 1h loss cooldown)
    run_tf("1h", 365, 1)    # bigger sample / Oanda's H1 timeframe


if __name__ == "__main__":
    main()
