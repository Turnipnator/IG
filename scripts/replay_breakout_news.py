#!/usr/bin/env python3
"""Do 1h breakouts taken near scheduled high-impact releases do worse — or better —
than breakouts taken at the same weekday and hour in weeks with no release?

Pre-registered in research_notes.md ("PRE-REGISTRATION — Breakout news-proximity
replay", 2026-09-22, amendments B1-B5 and C1-C4). Read that before changing anything.

Two stages, run separately on purpose:

    venv/bin/python scripts/replay_breakout_news.py --count      # Step 1, outcome-blind
    venv/bin/python scripts/replay_breakout_news.py --outcomes   # only if Step 1's gate passed

--count prints entry counts, how many a ±30 min block would remove, and the
pre-registered power gate. It computes R (the engine can't avoid it) but prints
nothing derived from R. --outcomes refuses to run unless the saved Step 1 result
says the gate passed.

Method: ONE trade set from the live-config breakout engine (N=55 1h bars, 2×ATR
stop, Donchian-M trail, DAY HTF gate, hour-close fill), partitioned into BLOCKED
(decision time within ±30 min of a mapped event) and the rest — the
replay_htf_blocked_breaks.py design. Null: rotate the event calendar by whole
weeks (k = ±1..±26), which keeps weekday, hour and the US-data-hour regime, so
a time-of-day effect cannot masquerade as a news effect.

Data: Dukascopy 1h BID (data/news_events/dukascopy), events from
scripts/build_event_calendar.py. Offline; touches no bot state.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from config import MARKETS  # noqa: E402
from src.breakout import BREAKOUT_CONFIGS  # noqa: E402
from src.indicators import calculate_atr, calculate_ema  # noqa: E402

DATA = ROOT / "data" / "news_events"
DUKA = DATA / "dukascopy"
STEP1 = DATA / "step1_count.json"

# epic -> (dukascopy file glob, price scale to IG points, event currencies mapped)
UNIVERSE = {
    "CS.D.USCGC.TODAY.IP":  ("xauusd-h1-*.csv", 1.0, ("USD",)),
    "CS.D.GBPUSD.TODAY.IP": ("gbpusd-h1-*.csv", 1e4, ("USD", "GBP")),
    "CS.D.EURUSD.TODAY.IP": ("eurusd-h1-*.csv", 1e4, ("USD", "EUR")),
    "CC.D.CL.USS.IP":       ("wti/lightcmdusd-h1-*.csv", 100.0, ("USD",)),
}
# Measured trading-hours spreads in IG points (same table as replay_htf_blocked_breaks.py).
SPREAD = {"CS.D.USCGC.TODAY.IP": 0.4, "CS.D.GBPUSD.TODAY.IP": 0.9,
          "CS.D.EURUSD.TODAY.IP": 0.6, "CC.D.CL.USS.IP": 3.0}
ADMIN = 0.015            # FX / commodity DFB admin, measured 2026-09-22
GOLD_ALL_IN = 0.058      # Gold admin + interest, measured 2026-09-09
FAMILIES = {"core-USD": ("USD",), "core-GBP/EUR": ("GBP", "EUR")}
WINDOW = pd.Timedelta(minutes=30)
HALVES = [("2005-01-01", "2015-12-31"), ("2016-01-01", "2026-12-31")]
SIGMA_R_PRIOR = 2.0      # §3: the gate uses the PRE-REGISTERED σ, never one measured here
MIN_BOOK_EFFECT = 0.02   # R/trade
SHIFTS = [k for k in range(-26, 27) if k != 0]


# ----------------------------------------------------------------------------- data

def load_market(epic: str) -> pd.DataFrame:
    """Dukascopy UTC hourly bars → the engine's frame: `date` = bar START in naive
    Europe/London time (the container clock, and what the HTF module expects), plus
    `utc` = bar start in UTC for event matching. The duplicated 01:00 hour at each
    autumn clock change is dropped (keep first)."""
    pattern, scale, _ = UNIVERSE[epic]
    files = [f for f in sorted(glob.glob(str(DUKA / pattern))) if os.path.getsize(f) > 0]
    if not files:
        raise FileNotFoundError(f"no Dukascopy data for {epic}: {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["volume"] > 0].drop_duplicates("timestamp").sort_values("timestamp")
    utc = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    out = pd.DataFrame({"utc": utc.values,
                        "date": utc.dt.tz_convert("Europe/London").dt.tz_localize(None).values})
    for c in ("open", "high", "low", "close"):
        out[c] = df[c].values * scale
    out = out.drop_duplicates("date", keep="first").reset_index(drop=True)
    out["utc"] = pd.to_datetime(out["utc"], utc=True)
    return out


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(DATA / "events.csv")
    ev["utc"] = pd.to_datetime(ev["utc"], utc=True)
    return ev


# ----------------------------------------------------------------------------- HTF

def htf_day_fast(df: pd.DataFrame) -> pd.Series:
    """Vectorised equivalent of backtest_index_breakout_ignative.htf_trend_series
    (resolution DAY): the trend is refreshed at 22:30 London each day from COMPLETED
    daily bars (a daily bar dated D counts only once D + 1 day <= the refresh time,
    i.e. from day D+1's refresh), and HELD between refreshes. Asserted equal to the
    reference on a slice in main()."""
    h = df.set_index("date").resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h["ema9"], h["ema21"] = calculate_ema(h["close"], 9), calculate_ema(h["close"], 21)
    h["n"] = np.arange(1, len(h) + 1)
    bull = (h["ema9"] > h["ema21"]) & (h["close"] > h["ema21"])
    bear = (h["ema9"] < h["ema21"]) & (h["close"] < h["ema21"])
    ok = (h["n"] >= 21) & h[["ema9", "ema21", "close"]].notna().all(axis=1)
    lab = pd.Series(np.where(bull, "BULLISH", np.where(bear, "BEARISH", "NEUTRAL")), index=h.index)
    # refresh on day D uses the last daily bar dated <= D-1
    refresh_day = pd.Series(h.index + pd.Timedelta("1D"), index=h.index)
    table = pd.DataFrame({"refresh_day": refresh_day.values, "lab": lab.values, "ok": ok.values})
    # a refresh where the newest closed bar isn't "ok" leaves the trend unchanged —
    # carry the last ok label forward
    table["lab"] = table["lab"].where(table["ok"])
    table["lab"] = table["lab"].ffill().fillna("NEUTRAL")
    t = df["date"]
    first_refresh = t.min().normalize() + pd.Timedelta("22:30:00")
    # the refresh governing bar t is at D 22:30 with D = t.date if t >= 22:30 else t.date - 1
    D = (t - pd.Timedelta("22:30:00")).dt.normalize()
    q = pd.DataFrame({"D": D.values, "pos": np.arange(len(t))}).sort_values("D")
    m = pd.merge_asof(q, table.sort_values("refresh_day"), left_on="D", right_on="refresh_day",
                      direction="backward")
    m = m.sort_values("pos")
    out = pd.Series(m["lab"].fillna("NEUTRAL").values, index=df.index)
    out[t.values < first_refresh] = "NEUTRAL"
    return out


# ----------------------------------------------------------------------------- engine

def run_gated(df: pd.DataFrame, epic: str, financing: pd.Series | None = None,
              skip_mask: np.ndarray | None = None) -> pd.DataFrame:
    """The live breakout path as replay_htf_blocked_breaks.run_ungated models it,
    but WITH the live gate: a break is taken only inside trading hours and only when
    the DAY HTF trend agrees with it (NEUTRAL, including warm-up, blocks). One
    position at a time; stop, then the Donchian-M trail; fills at the level or the
    gapped open. R = (gross - spread - financing) / stop distance.
    `skip_mask` (per bar) refuses breaks at those bars too — the "block ON" run,
    whose sequencing differs from the partitioned one (secondary, descriptive)."""
    cfg = BREAKOUT_CONFIGS[epic]
    mk = next(m for m in MARKETS if m.epic == epic)
    N, M, k = cfg.n, cfg.m, cfg.stop_atr_mult
    h_lo, h_hi = (mk.trading_start + 1) % 24, (mk.trading_end + 1) % 24
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = calculate_atr(df["high"], df["low"], df["close"], 14).to_numpy(float)
    htf = df["htf"].to_numpy(object)
    dates = df["date"].to_numpy("datetime64[ns]")
    hours = df["date"].dt.hour.to_numpy()
    utc = df["utc"]
    cost = SPREAD[epic]
    out = []
    i, n = N + 1, len(df)
    while i < n:
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1; continue
        upper, lower = hi[i - N:i].max(), lo[i - N:i].min()
        d = "BUY" if hi[i] >= upper else "SELL" if lo[i] <= lower else None
        if d is None:
            i += 1; continue
        hh = hours[i]
        if (hh < h_lo or hh >= h_hi) if h_lo < h_hi else (h_hi <= hh < h_lo):
            i += 1; continue
        if htf[i] != ("BULLISH" if d == "BUY" else "BEARISH"):
            i += 1; continue          # the live DAY HTF gate refuses it
        if skip_mask is not None and skip_mask[i]:
            i += 1; continue
        entry = cl[i]
        sd = max(a * k, mk.min_stop_distance)
        stop = entry - sd if d == "BUY" else entry + sd
        j, xp, why = i + 1, None, None
        while j < n:
            if d == "BUY" and lo[j] <= stop:  xp, why = min(o[j], stop), "stop"; break
            if d == "SELL" and hi[j] >= stop: xp, why = max(o[j], stop), "stop"; break
            if j - M >= 0:
                if d == "BUY":
                    lvl = lo[j - M:j].min()
                    if lo[j] <= lvl: xp, why = min(o[j], lvl), "trail"; break
                else:
                    lvl = hi[j - M:j].max()
                    if hi[j] >= lvl: xp, why = max(o[j], lvl), "trail"; break
            j += 1
        if xp is None:
            xp, why, j = cl[-1], "open-at-end", n - 1
        g = (xp - entry) if d == "BUY" else (entry - xp)
        nights = int((pd.Timestamp(dates[j]).normalize() - pd.Timestamp(dates[i]).normalize()).days)
        rate = financing_rate(epic, d, pd.Timestamp(dates[i]), financing)
        fin = nights * rate / 365.0 * entry
        out.append(dict(epic=epic, market=mk.name, date=pd.Timestamp(dates[i]),
                        decision_utc=utc.iloc[i] + pd.Timedelta(hours=1), exit_date=pd.Timestamp(dates[j]),
                        d=d, entry=entry, why=why, nights=nights, sd=sd,
                        r=(g - cost - fin) / sd, r_gross=g / sd))
        i = j + 1
    return pd.DataFrame(out)


def financing_rate(epic: str, d: str, when: pd.Timestamp, diffs: pd.Series | None) -> float:
    """Annual cost as a fraction of notional (negative = a credit)."""
    if epic == "CS.D.USCGC.TODAY.IP":
        return GOLD_ALL_IN
    if epic in ("CS.D.GBPUSD.TODAY.IP", "CS.D.EURUSD.TODAY.IP") and diffs is not None:
        key = (epic, when.to_period("M"))
        diff = diffs.get(key, 0.0)       # i_base - i_USD, fraction
        carry = diff if d == "BUY" else -diff
        return ADMIN - carry
    return ADMIN                        # WTI: admin only (basis credit unmeasured)


def rate_diffs() -> pd.Series:
    """Monthly i_base - i_USD from the FX-carry cache (FRED OECD 3m), as fractions."""
    cache = ROOT / "data" / "fx_carry"
    out = {}
    try:
        def s(code):
            x = pd.read_csv(cache / f"IR3TIB01{code}M156N.csv")
            v = pd.to_numeric(x.iloc[:, 1], errors="coerce")
            return pd.Series(v.values, index=pd.to_datetime(x.iloc[:, 0]).dt.to_period("M")).dropna()
        us = s("US")
        for epic, code in (("CS.D.GBPUSD.TODAY.IP", "GB"), ("CS.D.EURUSD.TODAY.IP", "EZ")):
            b = s(code)
            idx = pd.period_range("2004-01", "2026-12", freq="M")
            dd = (b.reindex(idx).ffill() - us.reindex(idx).ffill()) / 100.0
            for p, v in dd.items():
                out[(epic, p)] = float(v) if np.isfinite(v) else 0.0
    except FileNotFoundError:
        pass
    return pd.Series(out, dtype=float)


# ----------------------------------------------------------------------------- tagging

def tag_blocked(decisions: pd.Series, event_times: np.ndarray, window: pd.Timedelta = WINDOW) -> np.ndarray:
    """True where a decision time is within ±window of any event (times only — no prices)."""
    if len(event_times) == 0 or len(decisions) == 0:
        return np.zeros(len(decisions), dtype=bool)
    ev = np.sort(event_times.astype("datetime64[ns]"))
    t = decisions.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
    w = np.timedelta64(int(window.total_seconds() * 1e9), "ns")
    left = np.searchsorted(ev, t - w, side="left")
    right = np.searchsorted(ev, t + w, side="right")
    return right > left


def shifted_events(real: np.ndarray, weeks: int, window: pd.Timedelta = WINDOW) -> np.ndarray:
    """Real events moved by `weeks` whole weeks, dropping any shifted time whose
    window would touch a REAL event (so a control slot is never a real news slot)."""
    real = np.sort(real.astype("datetime64[ns]"))
    sh = real + np.timedelta64(7 * weeks, "D")
    w2 = np.timedelta64(int(2 * window.total_seconds() * 1e9), "ns")
    left = np.searchsorted(real, sh - w2, side="left")
    right = np.searchsorted(real, sh + w2, side="right")
    return sh[right == left]


def family_events(ev: pd.DataFrame, epic: str, family: str) -> np.ndarray:
    ccys = [c for c in FAMILIES[family] if c in UNIVERSE[epic][2]]
    sel = ev[ev["currency"].isin(ccys)]
    return sel["utc"].dt.tz_localize(None).to_numpy("datetime64[ns]")


# ----------------------------------------------------------------------------- run

def build_trades(check_htf: bool = True) -> pd.DataFrame:
    diffs = rate_diffs()
    parts = []
    for epic in UNIVERSE:
        df = load_market(epic)
        df["htf"] = htf_day_fast(df)
        if check_htf:
            bt = importlib.import_module("backtest_index_breakout_ignative")
            tail = df[df["date"] >= df["date"].max() - pd.Timedelta(days=500)].reset_index(drop=True)
            ref = bt.htf_trend_series(tail[["date", "open", "high", "low", "close"]], "DAY",
                                      pd.Index(tail["date"]))
            mine = htf_day_fast(tail)
            agree = float((ref.values == mine.values).mean())
            assert agree == 1.0, f"{epic}: HTF fast path diverges from the reference ({agree:.4f})"
        t = run_gated(df, epic, diffs)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def yahoo_crosscheck() -> dict:
    """Hourly-return correlation, Dukascopy vs Yahoo, on the overlap (price only)."""
    import yfinance as yf
    tick = {"CS.D.USCGC.TODAY.IP": "GC=F", "CS.D.GBPUSD.TODAY.IP": "GBPUSD=X",
            "CS.D.EURUSD.TODAY.IP": "EURUSD=X", "CC.D.CL.USS.IP": "CL=F"}
    out = {}
    for epic, t in tick.items():
        y = yf.download(t, period="720d", interval="1h", progress=False, auto_adjust=False)["Close"]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y.index = pd.to_datetime(y.index, utc=True)
        dk = load_market(epic).set_index("utc")["close"]
        j = pd.concat([dk.pct_change(), y.pct_change()], axis=1, join="inner").dropna()
        out[epic] = {"corr": round(float(j.corr().iloc[0, 1]), 3), "n": int(len(j))}
    return out


def power(n_blocked: int, n_total: int) -> dict:
    f = n_blocked / n_total if n_total else 0.0
    mde = 2.487 * SIGMA_R_PRIOR / math.sqrt(n_blocked) if n_blocked else float("inf")
    need = MIN_BOOK_EFFECT / f if f else float("inf")
    return {"n_total": n_total, "n_blocked": n_blocked, "f": round(f, 4),
            "mde_R": round(mde, 3), "needed_delta_R": round(need, 3), "powered": bool(mde <= need)}


def step1() -> None:
    trades = build_trades()
    ev = load_events()
    res = {"entries_by_market": trades.groupby("market").size().to_dict(),
           "first_entry": str(trades["date"].min()), "last_entry": str(trades["date"].max())}
    fam = {}
    for family in FAMILIES:
        mask_all, blocked = np.zeros(len(trades), bool), np.zeros(len(trades), bool)
        for epic in UNIVERSE:
            evs = family_events(ev, epic, family)
            if len(evs) == 0:
                continue
            sel = (trades["epic"] == epic).to_numpy()
            mask_all |= sel
            blocked[sel] = tag_blocked(trades.loc[sel, "decision_utc"], evs)
        fam[family] = {**power(int(blocked[mask_all].sum()), int(mask_all.sum())),
                       "by_market": trades[blocked].groupby("market").size().to_dict()}
        trades[f"blocked_{family}"] = blocked
    res["families"] = fam
    res["gate_passes"] = all(v["powered"] for v in fam.values())
    res["gate_passes_any"] = {k: v["powered"] for k, v in fam.items()}
    try:
        res["yahoo_crosscheck"] = yahoo_crosscheck()
    except Exception as e:                       # network is not the replay's problem
        res["yahoo_crosscheck"] = f"skipped: {e}"
    trades.to_pickle(DATA / "replay_trades.pkl")
    STEP1.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))   # no R-derived number is printed


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--outcomes", action="store_true")
    args = ap.parse_args()
    if args.count:
        step1()
    else:
        s = json.loads(STEP1.read_text()) if STEP1.exists() else None
        if not s:
            sys.exit("run --count first")
        passing = [k for k, v in s["gate_passes_any"].items() if v]
        if not passing:
            sys.exit("Step 1 power gate FAILED for every family — the outcome replay is not run (pre-reg §3).")
        import replay_breakout_news_outcomes as rbo   # written only once the gate is known
        rbo.run(passing)


if __name__ == "__main__":
    main()
