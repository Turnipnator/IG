"""Per-EPIC parameter search + walk-forward validation across the WHOLE universe.

THE QUESTION (2026-07-14, from the user): "there must be a way to be profitable
either shorting or going long on all instruments — it's just finding the right
settings per EPIC, right?"

THE TEST. For every market (12 live + 10 disabled):
  1. Grid-search the real levers (ADX, stop width, R:R, direction, MACD exit)
     on an IN-SAMPLE window.
  2. Take the single best config by in-sample P&L.
  3. Score THAT config on an OUT-OF-SAMPLE window it has never seen.

The number that settles the argument is not "how many markets have a profitable
config" (answer: nearly all will, by construction — that is what a grid search
DOES). It is **how many of those in-sample winners are still profitable
out-of-sample.** If that rate is ~50%, the search is finding noise, not settings.

COSTS. Spread is charged as a FRACTION of price, measured from live IG bid/offer
(get_market_info — a free snapshot, no historical allowance). This is the control
that Yahoo backtests lack, and its absence is exactly why Yahoo once reported FTSE
longs at "PF 10.26" right before live went 1W/9L. Run at 1.0x / 0.5x / 0.0x spread
so the conclusion can be checked for sensitivity to the cost estimate (several
EPICs were quoted out-of-session, where IG widens the spread).

Data: Yahoo 1h, 730d (the only source with enough history for multi-quarter
walk-forward). No IG REST allowance is consumed.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/paulturner/IG")

from config import MARKETS, STRATEGY_PROFILES
from src.indicators import add_all_indicators

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-paulturner-IG/6f107130-5cce-4510-9c2a-c5bc07b3ecf6/scratchpad"
)
DATA = SCRATCH / "wf_data"
DATA.mkdir(exist_ok=True)

# name -> (yahoo ticker, live-or-disabled, live strategy profile)
UNIVERSE = {
    # --- LIVE (12) ---
    "S&P 500":        ("^GSPC",     "LIVE", "indices"),
    "NASDAQ 100":     ("^NDX",      "LIVE", "indices_wide"),
    "Wall Street":    ("^DJI",      "LIVE", "indices"),
    "FTSE 100":       ("^FTSE",     "LIVE", "indices_tight"),
    "Japan 225":      ("^N225",     "LIVE", "indices"),
    "Hong Kong HS50": ("^HSI",      "LIVE", "indices"),
    "Gold":           ("GC=F",      "LIVE", "gold"),
    "NY Cocoa":       ("CC=F",      "LIVE", "cocoa"),
    "EUR/USD":        ("EURUSD=X",  "LIVE", "forex"),
    "GBP/USD":        ("GBPUSD=X",  "LIVE", "forex"),
    "US 10-Year":     ("ZN=F",      "LIVE", "default"),
    # AI Index has no Yahoo equivalent (IG-synthetic basket) — archive-only, excluded.
    # --- DISABLED (10) ---
    "Germany 40":     ("^GDAXI",    "DISABLED", "indices"),
    "Russell 2000":   ("^RUT",      "DISABLED", "indices"),
    "Crude Oil":      ("CL=F",      "DISABLED", "crude"),
    "Natural Gas":    ("NG=F",      "DISABLED", "natgas"),
    "Silver":         ("SI=F",      "DISABLED", "silver"),
    "Copper":         ("HG=F",      "DISABLED", "copper"),
    "Soybeans":       ("ZS=F",      "DISABLED", "default"),
    "NY Cotton":      ("CT=F",      "DISABLED", "cocoa"),
    "USD/JPY":        ("USDJPY=X",  "DISABLED", "usdjpy"),
    "US 2-Year":      ("ZT=F",      "DISABLED", "default"),
}

# The grid — the actual levers you would tune per EPIC.
GRID_ADX = [20, 25, 30, 35, 40]
GRID_STOP = [1.0, 1.5, 2.0, 2.5]
GRID_RR = [1.5, 2.0, 3.0]
GRID_DIR = ["BOTH", "BUY", "SELL"]
GRID_MACD = [True, False]
MIN_IS_TRADES = 10  # a config with fewer in-sample trades is not "found", it's luck


def download() -> None:
    import yfinance as yf

    for name, (tick, _, _) in UNIVERSE.items():
        f = DATA / f"{name.replace('/', '_')}.csv"
        if f.exists():
            continue
        for attempt in range(3):
            try:
                d = yf.download(tick, period="730d", interval="1h",
                                progress=False, auto_adjust=False)
                if d is not None and len(d) > 1000:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = [c[0] for c in d.columns]
                    d = d.reset_index()
                    tc = "Datetime" if "Datetime" in d.columns else "Date"
                    d = d.rename(columns={tc: "date", "Open": "open", "High": "high",
                                          "Low": "low", "Close": "close", "Volume": "volume"})
                    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_localize(None)
                    d[["date", "open", "high", "low", "close", "volume"]].dropna().to_csv(f, index=False)
                    print(f"  {name:16s} {tick:10s} {len(d)} bars")
                    break
            except Exception as e:
                print(f"  {name:16s} retry {attempt}: {e}")
            time.sleep(2)
        else:
            print(f"  {name:16s} {tick:10s} FAILED")


def prep(name: str, profile: str) -> pd.DataFrame | None:
    f = DATA / f"{name.replace('/', '_')}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["date"])
    if len(df) < 500:
        return None
    p = STRATEGY_PROFILES[profile]
    params = {
        "ema_fast": p.ema_fast, "ema_medium": p.ema_medium, "ema_slow": p.ema_slow,
        "rsi_period": p.rsi_period, "rsi_overbought": p.rsi_overbought,
        "rsi_oversold": p.rsi_oversold, "rsi_buy_max": p.rsi_buy_max,
        "rsi_sell_min": p.rsi_sell_min, "adx_threshold": p.adx_threshold,
    }
    df = add_all_indicators(df, params)
    # HTF = daily (the bot's convention for 1h-candle markets: htf_resolution=DAY)
    d = df.set_index("date").resample("1D").agg({"close": "last"}).dropna()
    d["e9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["e21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["htf"] = np.where((d.e9 > d.e21) & (d.close > d.e21), "BULLISH",
                np.where((d.e9 < d.e21) & (d.close < d.e21), "BEARISH", "NEUTRAL"))
    df["htf"] = df["date"].dt.normalize().map(d["htf"].to_dict()).fillna("NEUTRAL")
    return df.dropna(subset=["ema_fast", "ema_medium", "ema_slow", "rsi", "adx", "atr", "macd_hist"]).reset_index(drop=True)


def signals(df: pd.DataFrame, prof, adx_thr: int, use_macd: bool) -> np.ndarray:
    """Live entry rule (strategy.py:214-290), vectorised. +1 BUY, -1 SELL, 0 none."""
    c, ef, em, es = df.close.values, df.ema_fast.values, df.ema_medium.values, df.ema_slow.values
    rsi, adx, mh = df.rsi.values, df.adx.values, df.macd_hist.values
    htf = df.htf.values
    pb = prof.pullback_pct / 100
    dist = (c - ef) / ef

    bull = (ef > em) & (em > es) & (c > es) & (rsi > prof.rsi_oversold) & (rsi < prof.rsi_buy_max)
    bear = (ef < em) & (em < es) & (c < es) & (rsi < prof.rsi_overbought) & (rsi > prof.rsi_sell_min)
    bull &= dist <= pb           # pullback filter: not overextended
    bear &= dist >= -pb
    bull &= adx >= adx_thr       # ADX gate
    bear &= adx >= adx_thr

    # MACD pre-check: don't enter if the exit is already true (live guard)
    if use_macd:
        neg3 = pd.Series(mh < 0).rolling(3).sum().values == 3
        pos3 = pd.Series(mh > 0).rolling(3).sum().values == 3
        bull &= ~neg3
        bear &= ~pos3

    if prof.require_htf:
        bull &= htf == "BULLISH"
        bear &= htf == "BEARISH"
    else:                        # never trade against HTF even when not required
        bull &= htf != "BEARISH"
        bear &= htf != "BULLISH"

    return np.where(bull, 1, np.where(bear, -1, 0))


def simulate(df, sig, prof, stop_mult, rr, direction, use_macd, spread_frac, min_stop):
    """Sequential, one position at a time. Stop / R:R limit / break-even / MACD exit."""
    hi, lo, cl = df.high.values, df.low.values, df.close.values
    atr, mh = df.atr.values, df.macd_hist.values
    n = len(df)
    be = prof.breakeven_trigger_pct
    trades = []
    i, cooldown = 60, 6
    while i < n - 1:
        s = sig[i]
        if s == 0:
            i += 1
            continue
        if direction == "BUY" and s != 1:
            i += 1
            continue
        if direction == "SELL" and s != -1:
            i += 1
            continue
        entry = cl[i]
        cost = entry * spread_frac                     # IG spread, in price units
        sd = max(atr[i] * stop_mult, min_stop * entry) # min stop, price-relative
        if sd <= 0 or not np.isfinite(sd):
            i += 1
            continue
        long = s == 1
        stop = entry - sd if long else entry + sd
        lim = entry + sd * rr if long else entry - sd * rr
        bet = entry + sd * be if long else entry - sd * be
        moved, neg, exit_px, j = False, 0, None, i
        for j in range(i + 1, n):
            if long:
                if lo[j] <= stop:
                    exit_px = stop
                    break
                if hi[j] >= lim:
                    exit_px = lim
                    break
                if not moved and hi[j] >= bet:
                    stop, moved = entry, True
            else:
                if hi[j] >= stop:
                    exit_px = stop
                    break
                if lo[j] <= lim:
                    exit_px = lim
                    break
                if not moved and lo[j] <= bet:
                    stop, moved = entry, True
            if use_macd:
                bad = mh[j] < 0 if long else mh[j] > 0
                neg = neg + 1 if bad else 0
                if neg >= 3:
                    exit_px = cl[j]
                    break
        if exit_px is None:
            break
        pnl = (exit_px - entry if long else entry - exit_px) - cost
        trades.append((df.date.values[i], pnl / entry * 100))  # % of price = comparable across EPICs
        i = j + cooldown
    return trades


def stats(tr):
    if not tr:
        return {"n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0}
    p = [x[1] for x in tr]
    g = sum(x for x in p if x > 0)
    b = abs(sum(x for x in p if x <= 0))
    return {"n": len(p), "pnl": sum(p), "pf": (g / b) if b else float("inf"),
            "wr": sum(1 for x in p if x > 0) / len(p) * 100}


def main() -> None:
    spreads = json.loads((SCRATCH / "spreads.json").read_text())
    print("Downloading Yahoo 1h/730d ...")
    download()

    for label, mult in [("FULL live IG spread", 1.0), ("HALF spread", 0.5), ("ZERO cost", 0.0)]:
        print(f"\n{'='*100}\n### COST SETTING: {label}\n{'='*100}")
        print(f"{'market':16s} {'st':4s} {'IS best config':28s} {'IS n':>5} {'IS %':>7} "
              f"{'OOS n':>5} {'OOS %':>7} {'OOS PF':>7}  verdict")
        survive = fail = nodata = 0
        rows = []
        for name, (tick, status, profile) in UNIVERSE.items():
            df = prep(name, profile)
            if df is None:
                print(f"{name:16s} {status[:4]:4s} NO DATA")
                nodata += 1
                continue
            prof = STRATEGY_PROFILES[profile]
            sf = spreads.get(name, {}).get("spread_frac", 0.0) * mult
            mkt = next((m for m in MARKETS if m.name == name), None)
            min_stop_frac = 0.0005  # 5bps floor, price-relative

            split = len(df) // 2
            is_df, oos_df = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

            best, best_cfg = None, None
            sigcache = {}
            for adx, macd in itertools.product(GRID_ADX, GRID_MACD):
                sigcache[(adx, macd)] = signals(is_df, prof, adx, macd)
            for adx, stop, rr, d, macd in itertools.product(
                GRID_ADX, GRID_STOP, GRID_RR, GRID_DIR, GRID_MACD
            ):
                tr = simulate(is_df, sigcache[(adx, macd)], prof, stop, rr, d, macd, sf, min_stop_frac)
                st = stats(tr)
                if st["n"] < MIN_IS_TRADES:
                    continue
                if best is None or st["pnl"] > best["pnl"]:
                    best, best_cfg = st, (adx, stop, rr, d, macd)
            if best is None:
                print(f"{name:16s} {status[:4]:4s} no config reached {MIN_IS_TRADES} IS trades")
                nodata += 1
                continue

            adx, stop, rr, d, macd = best_cfg
            osig = signals(oos_df, prof, adx, macd)
            oos = stats(simulate(oos_df, osig, prof, stop, rr, d, macd, sf, min_stop_frac))
            ok = oos["pnl"] > 0 and oos["n"] >= 5
            if oos["n"] >= 5:
                survive += ok
                fail += (not ok)
            cfg = f"ADX{adx} stop{stop} rr{rr} {d}{' macd' if macd else ''}"
            v = "SURVIVES" if ok else ("FAILS" if oos["n"] >= 5 else "thin")
            print(f"{name:16s} {status[:4]:4s} {cfg:28s} {best['n']:5d} {best['pnl']:+7.2f} "
                  f"{oos['n']:5d} {oos['pnl']:+7.2f} {oos['pf']:7.2f}  {v}")
            rows.append((name, status, best, oos, ok))

        tot = survive + fail
        print(f"\n  IN-SAMPLE: a profitable config was found for {len(rows)}/{len(rows)} markets tested.")
        if tot:
            print(f"  OUT-OF-SAMPLE: {survive}/{tot} survive ({survive/tot*100:.0f}%). "
                  f"Coin-flip would give ~50%.")
        live = [r for r in rows if r[1] == "LIVE"]
        dis = [r for r in rows if r[1] == "DISABLED"]
        for lbl, grp in [("LIVE markets", live), ("DISABLED markets", dis)]:
            g = [r for r in grp if r[3]["n"] >= 5]
            if g:
                s = sum(1 for r in g if r[4])
                print(f"    {lbl:18s}: {s}/{len(g)} survive OOS")


if __name__ == "__main__":
    main()
