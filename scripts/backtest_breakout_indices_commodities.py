#!/usr/bin/env python3
"""Breakout edge sweep across the INDEX + COMMODITY book (2026-06-22, for the 06-26 review).

Question this answers: the user wants to know whether a Donchian breakout strategy is
worth running on the non-forex EPICs (as a toggleable alternative to the live EMA-
momentum book), the way GBP/USD breakout runs behind /forex. The honest first step is
DATA, not code — does a breakout edge even EXIST on these instruments, on WHICH ones,
and does it survive cost + regime change?

This reuses the exact forex breakout machinery (`breakout_sim` + `rstats` from
scripts/backtest_forex_breakout.py) and the exact walk-forward methodology that KILLED
USD/JPY (scripts/backtest_forex_breakout_walkforward.py): a FIXED config run unchanged
across full / 2 halves / 4 quarters of ~730d Yahoo 1h. The bar is SIGN CONSISTENCY
(every window >= breakeven), not raw P&L — a config that's +6% one quarter and red the
rest is a one-trending-year mirage, not an edge (exactly how USD/JPY failed).

Validated shape only (from the forex work): HTF-filtered Donchian-trail, k=2.0xATR; we
sweep N (the one real lever). Two cost passes per config: GROSS (0) and a realistic
per-market IG-DFB round-trip estimate. Also splits the full-period trades long vs short
so we can see whether breakout (like momentum) wants a restricted direction per EPIC.

ZERO IG API COST: Yahoo cash/futures only (^GSPC/^NDX/^DJI/^FTSE/^N225/^HSI/GC=F/CL=F/
HG=F/CC=F). Caveats baked in (read before trusting a number):
  - Yahoo cash/continuous-future != IG DFB price; the bps costs below are ESTIMATES to
    refine against live spread.
  - ~730d of 1h is still short for a breakout edge; quarter windows are thin (wide-N
    configs ~ 10-30 trades/window) — treat single-window P&L as low-confidence.
  - No BE/trail-management nuance, no live slippage beyond the flat bps haircut.
  - AI Index has NO Yahoo equivalent (IG-proprietary) — it is EXCLUDED here and handled
    by the separate archive-based AI Index sweep already on the 06-26 agenda.
  - Japan 225 (^N225) / Hong Kong HS50 (^HSI) are routinely thin on Yahoo 1h over 730d
    and may auto-skip ("NO/THIN Yahoo data"); when they do, run them off the harvested
    candle archive (scripts/archive_loader.py), same path as AI Index.
This is a CONCEPT/robustness gate, not final proof. A survivor earns a shadow trial next.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr, calculate_ema
from scripts.backtest_forex_breakout import breakout_sim, rstats

logging.basicConfig(level=logging.ERROR)

# Yahoo tickers for the non-forex book (some already in TICKER_MAP; add the rest).
TICKER_MAP.update({
    "Wall Street": "^DJI",
    "FTSE 100": "^FTSE",
    "Japan 225": "^N225",
    "Hong Kong HS50": "^HSI",
    "Copper": "HG=F",
    "NY Cocoa": "CC=F",
})

DAYS, INTERVAL, STOP_K = 730, "1h", 2.0
N_SWEEP = [20, 40, 55, 80]

# (name, live allowed_direction or None, approx IG-DFB round-trip cost in bps).
# allowed_direction mirrors the LIVE config so we can read the breakout result against
# how the market actually trades. bps are ROUGH IG DFB spread estimates — refine vs live.
MARKETS = [
    ("S&P 500",        "BUY",  4),
    ("NASDAQ 100",     None,   4),
    ("Wall Street",    None,   6),
    ("FTSE 100",       "BUY",  5),
    ("Japan 225",      None,   8),
    ("Hong Kong HS50", None,   8),
    ("Gold",           None,   4),
    ("Crude Oil",      None,   6),
    ("Copper",         "BUY",  8),
    ("NY Cocoa",       "SELL", 10),
]


def daily_htf(bt, name):
    """Daily BULLISH/BEARISH/NEUTRAL EMA trend, as-of-mergeable onto the 1h frame."""
    d = bt.fetch_data(name, DAYS, "1d")
    if d is None or d.empty:
        return None
    d = d.copy()
    d["ema_9"] = calculate_ema(d["close"], 9)
    d["ema_21"] = calculate_ema(d["close"], 21)
    def trend(r):
        if pd.isna(r["ema_9"]) or pd.isna(r["ema_21"]):
            return "NEUTRAL"
        if r["ema_9"] > r["ema_21"] and r["close"] > r["ema_21"]:
            return "BULLISH"
        if r["ema_9"] < r["ema_21"] and r["close"] < r["ema_21"]:
            return "BEARISH"
        return "NEUTRAL"
    d["htf"] = d.apply(trend, axis=1)
    return d[["date", "htf"]]


def load(name):
    bt = Backtester(params={"ema_fast": 9, "ema_medium": 21, "ema_slow": 50,
                            "rsi_period": 7, "atr_period": 14})
    raw = bt.fetch_data(name, DAYS, INTERVAL)
    if raw is None or len(raw) < 500:
        return None
    raw = raw.copy()
    raw["atr"] = calculate_atr(raw["high"], raw["low"], raw["close"], 14)
    hs = daily_htf(bt, name)
    df = (pd.merge_asof(raw.sort_values("date"), hs.sort_values("date"),
                        on="date", direction="backward")
          if hs is not None else raw.assign(htf="NEUTRAL"))
    return df


def windows(df, k):
    """k equal-time non-overlapping [lo, hi) slices over the data's date span."""
    lo, hi = df["date"].iloc[0], df["date"].iloc[-1]
    span = (hi - lo) / k
    out = []
    for i in range(k):
        a = lo + span * i
        b = hi + pd.Timedelta(seconds=1) if i == k - 1 else lo + span * (i + 1)
        out.append((a, b))
    return out


def cost_pips_for(df, bps):
    """Convert a round-trip bps cost into the absolute (pip=1.0) cost breakout_sim wants.
    With pip=1.0, breakout_sim charges cost_frac/entry*100 % per trade; setting
    cost_frac = median_close*bps/1e4 makes that ~= bps/100 % round-trip. Good enough
    (entry ~ median); the cost is a haircut, not a per-tick model."""
    return float(df["close"].median()) * bps / 1e4


def sim(df, lo, hi, n, bps):
    """Run the validated shape (HTF-filtered Donchian, k=2xATR) on a date slice.
    Returns the raw trade list (so the caller can split long/short) or None if thin."""
    sl = df[(df["date"] >= lo) & (df["date"] < hi)].reset_index(drop=True)
    if len(sl) < n + 50:
        return None
    min_stop = float(sl["close"].median()) * 0.0005
    cost = cost_pips_for(sl, bps)
    return breakout_sim(sl, n, STOP_K, "donchian", True, min_stop, cost_pips=cost, pip=1.0)


def show(tag, s):
    if s is None or s["n"] == 0:
        print(f"    {tag:<26} (thin/no trades)"); return
    sign = "OK " if s["pnl"] > 0 else ("~  " if s["pnl"] > -0.5 else "XX ")
    print(f"    {tag:<26} {sign} n={s['n']:>3} WR={s['wr']:>3.0f}% "
          f"P&L={s['pnl']:>+7.2f}% PF={s['pf']:>5.2f} R:R={s['rr']:>4.2f} maxW={s['maxw']:>+6.2f}")


def main():
    survivors = []   # (name, n, full PF/pnl) — positive in EVERY walk-forward window
    for name, allow, bps in MARKETS:
        df = load(name)
        if df is None:
            print(f"\n{name}: NO/THIN Yahoo data — skipped"); continue
        span_d = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        live = f"live allowed_direction={allow}" if allow else "live both-sided"
        print(f"\n{'='*86}\n{name} — {span_d}d of {INTERVAL} ({len(df)} candles), {bps}bps round-trip"
              f"  |  {live}\n{'='*86}")
        for n in N_SWEEP:
            full_lo = df["date"].iloc[0]
            full_hi = df["date"].iloc[-1] + pd.Timedelta(seconds=1)
            trades = sim(df, full_lo, full_hi, n, bps)
            full = rstats(trades) if trades is not None else None
            print(f"  N={n}  (HTF Donchian, {STOP_K}xATR):")
            show("FULL period", full)
            # long/short split on the full period — does breakout want a restricted side?
            if trades:
                longs = rstats([t for t in trades if t["dir"] == "BUY"])
                shorts = rstats([t for t in trades if t["dir"] == "SELL"])
                show("  longs only", longs)
                show("  shorts only", shorts)
            # walk-forward: sign-consistency is the real gate
            q_stats = [rstats(sim(df, a, b, n, bps) or []) for a, b in windows(df, 4)]
            for i, qs in enumerate(q_stats, 1):
                show(f"  Q{i}", qs)
            # survivor = full positive AND every quarter that has trades is >= breakeven
            quarters_ok = [qs for qs in q_stats if qs and qs["n"] > 0]
            if (full and full["pnl"] > 0 and len(quarters_ok) >= 3
                    and all(qs["pnl"] > -0.5 for qs in quarters_ok)):
                survivors.append((name, n, full["pnl"], full["pf"], allow))

    print(f"\n{'#'*86}\n# WALK-FORWARD SURVIVORS (full +ve AND >=3 quarters all >= breakeven)\n{'#'*86}")
    if not survivors:
        print("  NONE. No index/commodity breakout config is sign-consistent across regimes\n"
              "  on this data → breakout is NOT worth porting to the non-forex book as-is.\n"
              "  (Momentum already works on these EPICs; the bar was 'beat or complement' it.)")
    else:
        for name, n, pnl, pf, allow in survivors:
            side = f" [restrict {allow}]" if allow else ""
            print(f"  {name:<16} N={n:<3} full P&L={pnl:>+7.2f}% PF={pf:>5.2f}{side}")
        print("\n  NB survivor != deployable. Next gate: shadow trial behind a toggle, AND add the\n"
              "  news/calendar block to the breakout path FIRST (open follow-up from the forex work).")
    print("\n  Reminder: AI Index excluded (no Yahoo); Japan 225 / Hong Kong HS50 may auto-skip\n"
          "  (Yahoo-thin) — run those three off the candle archive. Compare each survivor's PF\n"
          "  against that EPIC's LIVE momentum PF before believing it.")


if __name__ == "__main__":
    main()
