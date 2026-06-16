#!/usr/bin/env python3
"""ADX-ceiling sweep for the SHARED `indices` profile (2026-06-16).

Trigger: 2026-06-16 a Wall Street BUY (ADX 34.9) won +£32 while an AI Index BUY 15
min later (ADX 47.1) was stopped in ~5min. Same profile, same gates, same stop —
the only discriminator was trend maturity (ADX). This echoes Monday's Hong Kong
loser (ADX 50.2, stopped in 6min) and the earlier S&P SELL @ ADX 57. Thesis: an
extreme ADX marks a climax about to mean-revert, not a trend to ride.

The shared `indices` profile is used by FOUR markets — Wall Street, Japan 225,
Hong Kong HS50, AI Index. A ceiling added to the profile applies to ALL of them,
so the decision must rest on the AGGREGATE, not on AI Index alone (esp. since AI
Index is our best-validated market's bedfellow — we must NOT hurt Wall Street).

Method: faithful to the live `indices` profile (EMA 5/12/26, RSI 7 70/30 buy_max
55/sell_min 45, ADX floor 30, stop 1.5x ATR, R:R 2.0, require_htf via hourly
ema_9/21, MACD-3-candle exit, immediate market entry). For each market sweep an
upper ADX bound; each ceiling row removes entries with ADX above it, so a positive
ΔP&L means that high-ADX band was a NET LOSER (evidence for the cap). Watch trade
attrition — a cap that helps P&L but halves the count is overfitting the tail.

Data: Wall St/Japan/Hong Kong = Yahoo cash (^DJI/^N225/^HSI 5m ~59d, ZERO IG API
cost). AI Index has no Yahoo proxy → loaded from the free IG candle archive (only
meaningful when run in-container on the VPS where the archive lives; thin until
~2-4wk of harvest accumulates). Yahoo cash != IG DFB, so this is a directional
lead, not a live-P&L promise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_ema

try:
    from scripts.archive_loader import load_archive
except Exception:  # pragma: no cover
    load_archive = None

logging.basicConfig(level=logging.ERROR)

# Yahoo cash tickers for the three `indices`-profile markets that have a proxy.
TICKER_MAP.update({
    "Wall Street": "^DJI",
    "Japan 225": "^N225",
    "Hong Kong HS50": "^HSI",
})

# Exactly the live shared `indices` profile.
PARAMS = dict(
    ema_fast=5, ema_medium=12, ema_slow=26,
    rsi_period=7, rsi_overbought=70, rsi_oversold=30,
    rsi_buy_max=55, rsi_sell_min=45,
    atr_period=14, adx_threshold=30,
    stop_atr_multiplier=1.5, reward_risk_ratio=2.0,
)
MIN_CONF = 0.55
RR = 2.0
COOLDOWN = 12          # 12x5m = 1h, matches live cooldown
CEILINGS = [None, 60, 55, 50, 45, 40]

# Yahoo markets (proxy) + the IG-only AI Index (archive).
YAHOO_MARKETS = ["Wall Street", "Japan 225", "Hong Kong HS50"]
AIIDX_EPIC = "IX.D.AIIDX.DAILY.IP"


def _enter(direction, entry_price, atr, min_stop):
    sd = max(atr * PARAMS["stop_atr_multiplier"], min_stop)
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


def simulate(df, htf, bt, ceiling, min_stop):
    """Immediate-entry sim faithful to live, with an optional ADX upper bound.
    Returns (trades, removed_high_adx) where removed counts signals skipped purely
    because ADX exceeded the ceiling (i.e. would-have-fired at no cap)."""
    trades = []
    pos = None
    cooldown_until = -1
    removed = 0
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
                trades.append({"dir": pos["dir"], "pnlp": pnlp, "adx": pos["adx"], "reason": reason})
                if pnlp < 0:
                    cooldown_until = i + COOLDOWN
                pos = None
            continue
        if i <= cooldown_until or pd.isna(atr):
            continue
        htf_trend = bt.calculate_htf_trend("", row["date"], htf)
        direction, conf, _ = bt.check_entry_signal(row, htf_trend, require_htf_alignment=True)
        if direction and conf >= MIN_CONF:
            adx = row["adx"]
            if ceiling is not None and not pd.isna(adx) and adx > ceiling:
                removed += 1
                continue
            p = _enter(direction, close, atr, min_stop)
            p["adx"] = adx
            pos = p
    return trades, removed


def stats(trades):
    n = len(trades)
    if not n:
        return dict(n=0, wr=0, pnl=0, pf=0)
    w = sum(1 for t in trades if t["pnlp"] > 0)
    p = sum(t["pnlp"] for t in trades)
    gp = sum(t["pnlp"] for t in trades if t["pnlp"] > 0)
    gl = -sum(t["pnlp"] for t in trades if t["pnlp"] < 0)
    pf = gp / gl if gl > 0 else 999.0
    return dict(n=n, wr=w / n * 100, pnl=p, pf=pf)


def _build_htf(bt, name, df, days, interval):
    """Hourly ema_9/21 for the HTF gate. Yahoo for proxies; resample 5m->1h for AIIDX."""
    if name in TICKER_MAP:
        htf = bt.fetch_data(name, min(days, 365), "1h").copy()
    else:
        idx = df.set_index("date")
        htf = (idx[["open", "high", "low", "close"]]
               .resample("1h").agg({"open": "first", "high": "max",
                                    "low": "min", "close": "last"})
               .dropna().reset_index())
    htf["ema_9"] = calculate_ema(htf["close"], 9)
    htf["ema_21"] = calculate_ema(htf["close"], 21)
    return htf


def run_market(name, raw, days, interval):
    if raw is None or len(raw) < 100:
        print(f"\n{name}: NO/THIN data ({0 if raw is None else len(raw)} candles) — skipped")
        return None
    bt = Backtester(params=PARAMS.copy())
    df = bt.add_indicators(raw)
    htf = _build_htf(bt, name, df, days, interval)
    min_stop = float(df["close"].median()) * 0.0005
    print(f"\n{'='*92}\n{name} — {interval}/{days}d, {len(df)} candles. "
          f"EMA 5/12/26, ADX floor 30, stop 1.5x, R:R 2.0, require_htf\n{'='*92}")
    print(f"{'ceiling':>8}{'Trades':>8}{'WR':>8}{'P&L%':>9}{'PF':>7}{'removed':>9}{'dP&L%':>9}  note")
    print("-" * 92)
    baseline = None
    per_ceiling = {}
    for ceil in CEILINGS:
        trades, removed = simulate(df, htf, bt, ceil, min_stop)
        per_ceiling[ceil] = trades   # re-simulated (cooldown-correct) for the aggregate
        s = stats(trades)
        if ceil is None:
            baseline = s
            print(f"{'none':>8}{s['n']:>8}{s['wr']:>7.0f}%{s['pnl']:>+8.2f}%{s['pf']:>7.2f}"
                  f"{'':>9}{'':>9}  <= BASELINE")
        else:
            dp = s["pnl"] - baseline["pnl"]
            note = "band NET LOSER" if dp > 0.01 else ("band net winner" if dp < -0.01 else "neutral")
            print(f"{ceil:>8}{s['n']:>8}{s['wr']:>7.0f}%{s['pnl']:>+8.2f}%{s['pf']:>7.2f}"
                  f"{removed:>9}{dp:>+8.2f}%  {note}")
    return per_ceiling


def main():
    days, interval = 59, "5m"
    # agg[ceiling] = concatenated re-simulated trades across all markets
    agg = {c: [] for c in CEILINGS}

    def absorb(per_ceiling):
        if per_ceiling:
            for c in CEILINGS:
                agg[c].extend(per_ceiling.get(c, []))

    for name in YAHOO_MARKETS:
        bt = Backtester(params=PARAMS.copy())
        raw = bt.fetch_data(name, days, interval)
        absorb(run_market(name, raw, days, interval))

    # AI Index from the IG candle archive (only present in-container on the VPS).
    if load_archive is not None:
        ai_raw = load_archive(AIIDX_EPIC)
        if ai_raw is not None and not ai_raw.empty:
            span_d = (ai_raw["date"].iloc[-1] - ai_raw["date"].iloc[0]).days
            absorb(run_market("AI Index", ai_raw, max(span_d, 1), "5m(archive)"))
        else:
            print("\nAI Index: no archive on this host (run in-container on the VPS).")

    # Aggregate across ALL shared-`indices` markets — the decision rests here,
    # because the ceiling is a profile-level change that hits every one of them.
    # Each ceiling's trades are the per-market RE-SIMULATIONS (cooldown-correct).
    if any(agg.values()):
        print(f"\n{'='*92}\nAGGREGATE (all shared-`indices` markets combined) — the profile-level decision\n{'='*92}")
        print(f"{'ceiling':>8}{'Trades':>8}{'WR':>8}{'P&L%':>9}{'PF':>7}{'dP&L%':>9}")
        print("-" * 92)
        base = stats(agg[None])
        for ceil in CEILINGS:
            s = stats(agg[ceil])
            dp = s["pnl"] - base["pnl"]
            tag = "  <= BASELINE" if ceil is None else ""
            print(f"{('none' if ceil is None else ceil):>8}{s['n']:>8}{s['wr']:>7.0f}%"
                  f"{s['pnl']:>+8.2f}%{s['pf']:>7.2f}{dp:>+8.2f}%{tag}")


if __name__ == "__main__":
    main()
