#!/usr/bin/env python3
"""
Crude Oil exit-variant A/B backtest (Yahoo CL=F, 5m).

Live `crude` profile bleeds: all-time PF 0.38, last-week PF 0.05, and the
"MACD histogram positive for 3 candles" exit accounts for -£45.6 of the loss.
Hypothesis: the 3-candle MACD exit is too twitchy for Crude's ATR (20-35pts)
— it bails on routine bounces before the down-leg resumes. This faithfully
replicates the LIVE crude ENTRY (EMA 9/21/50 + RSI7 band + ADX>=30 + pullback
+ HTF gate + MACD pre-check) and holds it FIXED while swapping only the EXIT,
to isolate whether a less-sensitive exit rescues the edge — or whether the
ENTRY itself has no edge (=> disable, don't retune).

Yahoo only — does NOT touch the IG 10k/week allowance.
"""
import argparse, os, sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators import calculate_ema, calculate_atr, calculate_adx, calculate_rsi, calculate_macd

# ---- Live `crude` profile (config.py) ----
EMA_F, EMA_M, EMA_S = 9, 21, 50
RSI_P = 7
RSI_OB, RSI_OS = 80, 20
RSI_BUY_MAX, RSI_SELL_MIN = 55, 45
ADX_T = 30
STOP_MULT = 1.0
RR = 2.0
PULLBACK_PCT = 0.3 / 100  # 0.3%


def fetch_5m(ticker="CL=F", days=58):
    import yfinance as yf
    start = datetime.now() - timedelta(days=days)
    d = yf.download(ticker, start=start, interval="5m", progress=False, auto_adjust=False)
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    df = pd.DataFrame({
        "date": pd.to_datetime(d.index).tz_localize(None),
        "open": d["Open"].values, "high": d["High"].values,
        "low": d["Low"].values, "close": d["Close"].values,
    }).dropna().reset_index(drop=True)
    return df


def add_ind(df):
    df = df.copy()
    df["ema_f"] = calculate_ema(df["close"], EMA_F)
    df["ema_m"] = calculate_ema(df["close"], EMA_M)
    df["ema_s"] = calculate_ema(df["close"], EMA_S)
    df["rsi"] = calculate_rsi(df["close"], RSI_P)
    df["adx"] = calculate_adx(df["high"], df["low"], df["close"], 14)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
    _, _, hist = calculate_macd(df["close"])
    df["macd_hist"] = hist
    # HTF: 1h EMA trend (resample) — mirrors require_htf
    h = df.set_index("date")["close"].resample("1h").last().dropna()
    hf = calculate_ema(h, 9); hs = calculate_ema(h, 21)
    htf = pd.Series(np.where(hf > hs, "BULLISH", "BEARISH"), index=h.index)
    df["htf"] = htf.reindex(df["date"], method="ffill").values
    return df.dropna().reset_index(drop=True)


def entry_signal(df, i):
    """Return 'BUY'/'SELL'/None replicating live crude entry at bar i (decision on closed bar)."""
    r = df.iloc[i]
    ema_f, ema_m, ema_s, close = r.ema_f, r.ema_m, r.ema_s, r.close
    rsi, adx = r.rsi, r.adx
    if adx < ADX_T:
        return None
    last3 = [df.iloc[i - k]["macd_hist"] for k in range(0, 3)]
    macd_bear = all(h < 0 for h in last3)
    macd_bull = all(h > 0 for h in last3)
    dist = (close - ema_f) / ema_f
    # bullish
    if ema_f > ema_m > ema_s and close > ema_s and RSI_OS < rsi < RSI_BUY_MAX:
        if dist <= PULLBACK_PCT and r.htf == "BULLISH" and not macd_bull:
            return "BUY"
    # bearish
    if ema_f < ema_m < ema_s and close < ema_s and RSI_SELL_MIN < rsi < RSI_OB:
        if dist >= -PULLBACK_PCT and r.htf == "BEARISH" and not macd_bear:
            return "SELL"
    return None


@dataclass
class Trade:
    direction: str; entry: float; exitp: float; reason: str; bars: int; r: float


def should_exit(df, i, direction, variant, entry, stop_dist, hh_track, ll_track):
    """Return (exit_bool, reason). Exit decided on bar i close (intrabar stop handled in run)."""
    r = df.iloc[i]
    rsi, adx = r.rsi, r.adx
    # RSI extreme exit — always active (live)
    if direction == "BUY" and rsi > RSI_OB:
        return True, "RSI extreme"
    if direction == "SELL" and rsi < RSI_OS:
        return True, "RSI extreme"

    if variant == "macd3":  # LIVE
        h = [df.iloc[i - k]["macd_hist"] for k in range(0, 3)]
        if direction == "SELL" and all(x > 0 for x in h): return True, "MACD3"
        if direction == "BUY" and all(x < 0 for x in h): return True, "MACD3"
    elif variant == "macd5":
        h = [df.iloc[i - k]["macd_hist"] for k in range(0, 5)]
        if direction == "SELL" and all(x > 0 for x in h): return True, "MACD5"
        if direction == "BUY" and all(x < 0 for x in h): return True, "MACD5"
    elif variant.startswith("donch"):  # donch10 / donch20 trailing channel
        M = int(variant[5:])
        if i > M:
            if direction == "SELL":
                hh = df["high"].iloc[i - M:i].max()
                if r.close > hh: return True, f"Donch{M}"
            else:
                ll = df["low"].iloc[i - M:i].min()
                if r.close < ll: return True, f"Donch{M}"
    elif variant.startswith("atrtrail"):  # atrtrail1.5 / atrtrail2.5 — handled in run() via trailing stop
        pass
    return False, ""


def run(df, variant, cost_pts=0.0):
    trades: List[Trade] = []
    i = EMA_S + 5
    n = len(df)
    while i < n - 1:
        sig = entry_signal(df, i)
        if not sig:
            i += 1; continue
        entry = df.iloc[i]["close"]
        atr = df.iloc[i]["atr"]
        stop_dist = max(atr * STOP_MULT, 1e-9)
        limit_dist = stop_dist * RR
        # ATR-trail param
        trail_mult = float(variant[8:]) if variant.startswith("atrtrail") else None
        if sig == "SELL":
            stop = entry + stop_dist; limit = entry - limit_dist
        else:
            stop = entry - stop_dist; limit = entry + limit_dist
        j = i + 1
        exit_reason = "EOD"; exitp = df.iloc[-1]["close"]
        best = entry  # for trailing
        while j < n:
            bar = df.iloc[j]
            # intrabar stop / limit (stop priority)
            if sig == "SELL":
                if bar["high"] >= stop: exitp, exit_reason = stop, "Stop"; break
                if bar["low"] <= limit: exitp, exit_reason = limit, "Limit"; break
                best = min(best, bar["low"])
                if trail_mult is not None:
                    new_stop = best + atr * trail_mult
                    stop = min(stop, new_stop)
            else:
                if bar["low"] <= stop: exitp, exit_reason = stop, "Stop"; break
                if bar["high"] >= limit: exitp, exit_reason = limit, "Limit"; break
                best = max(best, bar["high"])
                if trail_mult is not None:
                    new_stop = best - atr * trail_mult
                    stop = max(stop, new_stop)
            # signal exit on bar close
            ex, why = should_exit(df, j, sig, variant, entry, stop_dist, None, None)
            if ex:
                exitp, exit_reason = bar["close"], why; break
            j += 1
        gross = (entry - exitp) if sig == "SELL" else (exitp - entry)
        # cost_pts is in IG points; 1 IG point = $0.01 on the Yahoo CL=F dollar scale
        r_mult = (gross - cost_pts * 0.01) / stop_dist
        trades.append(Trade(sig, entry, exitp, exit_reason, j - i, r_mult))
        i = j + 1  # flat until trade closes, then look for next
    return trades


def summarize(name, trades):
    if not trades:
        print(f"{name:14} : no trades"); return
    rs = [t.r for t in trades]
    wins = [r for r in rs if r > 0]; losses = [r for r in rs if r <= 0]
    gp = sum(wins); gl = abs(sum(losses))
    pf = gp / gl if gl else float("inf")
    tot = sum(rs)
    wr = 100 * len(wins) / len(trades)
    longs = sum(1 for t in trades if t.direction == "BUY")
    print(f"{name:14} : n={len(trades):3d} totR={tot:+6.1f} WR={wr:4.0f}% PF={pf:4.2f} "
          f"avgWin={ (gp/len(wins) if wins else 0):+.2f}R avgLoss={ (-gl/len(losses) if losses else 0):+.2f}R "
          f"(L{longs}/S{len(trades)-longs})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=58)
    ap.add_argument("--ticker", default="CL=F")
    args = ap.parse_args()
    df = fetch_5m(args.ticker, args.days)
    if df is None or df.empty:
        print("No data"); return
    df = add_ind(df)
    print(f"Crude {args.ticker} 5m | {df.date.iloc[0]} -> {df.date.iloc[-1]} | {len(df)} bars")
    print(f"Entry FIXED = live crude (EMA9/21/50, RSI7 band, ADX>={ADX_T}, pullback{PULLBACK_PCT*100}%, HTF, MACD pre-check); stop {STOP_MULT}xATR, {RR}R limit")
    print("-" * 96)
    variants = ["macd3", "macd5", "donch10", "donch20", "atrtrail1.5", "atrtrail2.5"]
    print("### Frictionless (Yahoo mid-price, cost=0) ###")
    for v in variants:
        tag = v + (" (LIVE)" if v == "macd3" else "")
        summarize(tag, run(df, v, 0.0))
    print("-" * 96)
    print("### Spread-cost sweep on the two best exits (pts subtracted round-trip per trade) ###")
    print("    IG Crude DFB spread is typically ~2.8-6 IG pts; ATR~25 IG pts so 1R~25pts.")
    for v in ["macd3", "macd5"]:
        for c in [0, 3, 5, 7]:
            summarize(f"{v} {c}pt-spread", run(df, v, float(c)))
        print()
    print("R = multiples of initial 1xATR risk. Same entries throughout; only the EXIT differs.")


if __name__ == "__main__":
    main()
