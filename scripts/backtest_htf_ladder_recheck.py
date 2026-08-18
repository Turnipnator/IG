#!/usr/bin/env python3
"""Re-check v3 item 15(b): "longer HTF lookback is better" — across the breakout book.

The claim (2026-08-13) was that raising htf_resolution improved breakout PF in 8 of 9
markets, and it is what justified moving Gold and DXY from the inherited HOUR to DAY.
Re-measuring Gold on 2026-08-18 found the ladder is NOT monotonic there: slip-charged
HOUR 1.16 -> HOUR_4 1.12 -> DAY 1.44, i.e. 4h is the WORST rung, not the middle one.
DAY still wins, so the Gold decision stands, but "longer is better" is the wrong reading
of why. This runs the same ladder on the rest of the book to see which reading holds.

Two DIFFERENT claims are separated here, because the original conflates them:
    (A) DAY beats HOUR                      -> is the config change justified?
    (B) PF rises monotonically with lookback -> is "longer is better" a real principle?
A market can satisfy (A) and violate (B), which is exactly what Gold does.

Method, identical per market:
  - ONE Yahoo 730d 1h frame; ATR14; cost = 0.286 x median ATR (measured live entry slip)
    charged per trade to every arm, so arms differ ONLY in the HTF gate.
  - Live breakout config: N=55, stop 2.0xATR, Donchian-M trail, htf_filter=True.
  - HTF trend rule = live's: EMA9 vs EMA21 plus close-side, .shift(1) so a bar is gated
    by the last COMPLETED higher-timeframe bar (the look-ahead fixed in 5cba8b3).

METHOD CAVEAT, stated because it biases the result: DAY uses NATIVE 1d bars (what live's
update_htf_trends fetches), while HOUR/HOUR_4 are resampled from the 1h frame — Yahoo has
no native 4h. Resampling FLATTERS: on Gold a resampled DAY reads 1.57 vs 1.44 native. So
the DAY rung is measured on the stricter basis and any DAY win is, if anything, understated.

AI Index is absent: no Yahoo equivalent, and the IG archive only reaches 2026-06-12 —
too short for a 730d ladder. That market's rung is genuinely unmeasured, not omitted.

ZERO IG API cost.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.backtest import Backtester, TICKER_MAP
from src.indicators import calculate_atr, calculate_ema
from scripts.backtest_forex_breakout import breakout_sim, rstats, htf_series

logging.basicConfig(level=logging.ERROR)

TICKER_MAP.update({
    "Wall Street": "^DJI",
    "FTSE 100": "^FTSE",
    "Japan 225": "^N225",
    "Hong Kong HS50": "^HSI",
    "US Russell 2000": "^RUT",
    "Dollar Index (DXY)": "DX-Y.NYB",
})

DAYS, N, STOP_K = 730, 55, 2.0
SLIP_ATR_FRAC = 0.286

BOOK = ["Gold", "S&P 500", "NASDAQ 100", "Wall Street", "FTSE 100",
        "US Russell 2000", "Japan 225", "Hong Kong HS50", "Dollar Index (DXY)"]


def trend_col(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["ema_9"] = calculate_ema(d["close"], 9)
    d["ema_21"] = calculate_ema(d["close"], 21)

    def t(r):
        if pd.isna(r["ema_9"]) or pd.isna(r["ema_21"]):
            return "NEUTRAL"
        if r["ema_9"] > r["ema_21"] and r["close"] > r["ema_21"]:
            return "BULLISH"
        if r["ema_9"] < r["ema_21"] and r["close"] < r["ema_21"]:
            return "BEARISH"
        return "NEUTRAL"

    d["htf"] = d.apply(t, axis=1).shift(1).fillna("NEUTRAL")
    return d[["date", "htf"]]


def arm(raw: pd.DataFrame, hs: pd.DataFrame, cost: float):
    if hs is None or hs.empty:
        return None
    df = pd.merge_asof(raw.sort_values("date"), hs.sort_values("date"),
                       on="date", direction="backward")
    df["htf"] = df["htf"].fillna("NEUTRAL")
    return rstats(breakout_sim(df, N, STOP_K, "donchian", True, 0.0,
                               cost_pips=cost, pip=1.0))


def verdict(h, h4, d):
    """(A) does DAY beat HOUR?  (B) is the ladder monotonic?"""
    if not all([h, h4, d]):
        return "incomplete", "incomplete"
    a = "DAY>HOUR" if d["pf"] > h["pf"] else ("tie" if abs(d["pf"] - h["pf"]) < 0.02 else "HOUR>DAY")
    b = "monotonic" if h["pf"] <= h4["pf"] <= d["pf"] else "NOT monotonic"
    return a, b


def main() -> None:
    bt = Backtester(params={"ema_fast": 3, "ema_medium": 8, "ema_slow": 21,
                            "rsi_period": 7, "adx_threshold": 35})
    print(f"\n{'=' * 104}")
    print("v3 item 15(b) RE-CHECK — HTF ladder per market, live breakout config "
          f"(N{N}/{STOP_K}xATR/Donchian), Yahoo {DAYS}d 1h")
    print(f"{'=' * 104}")
    print(f"{'market':<20} {'HOUR':>14} {'HOUR_4':>14} {'DAY':>14}   "
          f"{'(A) DAY>HOUR?':<14} {'(B) ladder':<14}")
    print("-" * 104)

    rows = []
    for name in BOOK:
        # 730 trips Yahoo's "must be within the last 730 days" on some tickers
        # (second-precision drift between our start and their cutoff) — the same
        # trap backtest_forex_breakout.py works around with 725. Retry shorter.
        raw = None
        for days in (DAYS, 725, 700):
            try:
                raw = bt.fetch_data(name, days, "1h")
            except Exception:
                raw = None
            if raw is not None and len(raw) >= 800:
                if days != DAYS:
                    print(f"{name:<20}  (730d rejected by Yahoo; using {days}d)")
                break
        if raw is None or len(raw) < 800:
            print(f"{name:<20}  insufficient 1h data "
                  f"({0 if raw is None else len(raw)} bars) — SKIPPED")
            continue
        raw = raw.copy()
        raw["atr"] = calculate_atr(raw["high"], raw["low"], raw["close"], 14)
        cost = SLIP_ATR_FRAC * float(raw["atr"].median())

        h = arm(raw, trend_col(raw[["date", "open", "high", "low", "close"]]), cost)
        h4 = arm(raw, trend_col(raw.set_index("date").resample("4h")
                                .agg({"open": "first", "high": "max",
                                      "low": "min", "close": "last"})
                                .dropna().reset_index()), cost)
        try:
            d = arm(raw, htf_series(bt, name, days=DAYS), cost)
        except Exception:
            d = None

        def cell(s):
            return f"{s['pf']:>5.2f} (n={s['n']:>3})" if s else "     no data"

        va, vb = verdict(h, h4, d)
        print(f"{name:<20} {cell(h):>14} {cell(h4):>14} {cell(d):>14}   {va:<14} {vb:<14}")
        rows.append((name, h, h4, d, va, vb))

    print("-" * 104)
    done = [r for r in rows if all([r[1], r[2], r[3]])]
    if done:
        a_yes = sum(1 for r in done if r[4] == "DAY>HOUR")
        b_yes = sum(1 for r in done if r[5] == "monotonic")
        print(f"\n(A) DAY beats HOUR:        {a_yes}/{len(done)} markets")
        print(f"(B) ladder is monotonic:   {b_yes}/{len(done)} markets")
        print("\nper-market PF deltas (DAY - HOUR):")
        for name, h, h4, d, _, _ in sorted(done, key=lambda r: -(r[3]["pf"] - r[1]["pf"])):
            print(f"   {name:<20} {d['pf'] - h['pf']:>+6.2f}   "
                  f"(HOUR {h['pf']:.2f} -> HOUR_4 {h4['pf']:.2f} -> DAY {d['pf']:.2f})")
    print("\nNote: AI Index unmeasured (no Yahoo ticker, archive too short).")
    print("Read (A) to judge the config change; read (B) to judge the principle.\n")


if __name__ == "__main__":
    main()
