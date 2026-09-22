#!/usr/bin/env python3
"""G10 FX carry backtest — the test pre-registered in research_notes.md on 2026-09-22
("PRE-REGISTRATION — FX carry backtest"). Read that section before changing anything
here: the strategy, costs, null and verdict rules are fixed there, and any departure
is an amendment that has to be logged with its reason.

What it tests. Each month, rank ten currencies by their 3-month interbank rate for
the PREVIOUS month; long the top 3, short the bottom 3, every leg against USD.
  A  equal weights (primary)
  B  inverse-vol weights + crash gate (the "carry-to-vol" idea from the LSEG
     fx-carry-trade skill in anthropics/financial-services)
  C  long USD/JPY whenever the differential exceeds the admin fee (descriptive only;
     underpowered by construction, cannot pass)

Costs are IG's, measured on this account in Step 0 (2026-09-22): admin 1.5%/yr on
every leg's notional every night whatever the direction; the interest leg passes
the differential through (model within 11% of the account's actual credits);
half-spread per side on changed weight, spreads captured from IG in-session.

Data is free and costs no IG allowance: FRED (OECD 3m interbank, SONIA, ECB deposit
rate) and Yahoo daily FX closes, cached under data/fx_carry/.

    .venv-bt/bin/python scripts/backtest_fx_carry.py            # full run
    .venv-bt/bin/python scripts/backtest_fx_carry.py --refresh  # re-download

Offline and read-only: it touches no bot state, no journal, no IG session.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "fx_carry"

CCYS = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "NOK", "SEK"]
FRED_CODE = {"USD": "US", "EUR": "EZ", "GBP": "GB", "JPY": "JP", "AUD": "AU",
             "NZD": "NZ", "CAD": "CA", "CHF": "CH", "NOK": "NO", "SEK": "SE"}
# Yahoo ticker and whether it quotes USD per unit of the currency (True) or the
# reverse (False, so it is inverted to USD value).
YAHOO = {"EUR": ("EURUSD=X", True), "GBP": ("GBPUSD=X", True), "JPY": ("JPY=X", False),
         "AUD": ("AUDUSD=X", True), "NZD": ("NZDUSD=X", True), "CAD": ("USDCAD=X", False),
         "CHF": ("USDCHF=X", False), "NOK": ("USDNOK=X", False), "SEK": ("USDSEK=X", False)}

ADMIN = 0.015            # per leg per year — measured 2026-09-22 on GBP/USD, EUR/USD, USD/JPY
# IG spread / mid, captured in-session 2026-09-22 (Step 0 item 3).
SPREAD = {"EUR": 0.6 / 11448.8, "GBP": 0.9 / 13342.85, "JPY": 1.0 / 15738.6,
          "AUD": 0.6 / 7114.8, "NZD": 1.8 / 5727.8, "CAD": 1.3 / 14063.25,
          "CHF": 1.5 / 8205.25, "NOK": 50.0 / 94418.5, "SEK": 25.0 / 98375.6}
# IG minDealSize (£/pt, demo) and price in points, for notional = size × price.
# GBP/USD reports its minimum as a PERCENTAGE (0.03) — not a stake; taken as 0.04
# like the other majors. Demo minimums may differ from live.
MIN_SIZE = {"EUR": 0.04, "GBP": 0.04, "JPY": 0.04, "AUD": 0.04, "NZD": 1.0,
            "CAD": 0.04, "CHF": 1.0, "NOK": 0.1, "SEK": 0.1}
PRICE_PTS = {"EUR": 11448.8, "GBP": 13342.85, "JPY": 15738.6, "AUD": 7114.8, "NZD": 5727.8,
             "CAD": 14063.25, "CHF": 8205.25, "NOK": 94418.5, "SEK": 98375.6}

START, END = pd.Period("2004-01", "M"), pd.Period("2026-08", "M")
HALVES = [(pd.Period("2004-01", "M"), pd.Period("2014-12", "M")),
          (pd.Period("2015-01", "M"), pd.Period("2026-08", "M"))]
DD_BUDGET_GBP = 250.0    # half the £500 hard stop


# ----------------------------------------------------------------------------- data

def _fred(series: str, refresh: bool) -> pd.Series:
    import requests
    path = CACHE / f"{series}.csv"
    if refresh or not path.exists():
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=30)
        r.raise_for_status()
        path.write_text(r.text)
    df = pd.read_csv(path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def load_rates(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Monthly 3m rates in %, PeriodIndex. GB and EZ are extended past the OECD
    series' end with SONIA / ECB deposit-rate month averages; every month still
    missing at the end is forward-filled. Both are reported, not hidden."""
    notes = {}
    cols = {}
    for c in CCYS:
        s = _fred(f"IR3TIB01{FRED_CODE[c]}M156N", refresh)
        s.index = s.index.to_period("M")
        cols[c] = s
    for c, daily_code in (("GBP", "IUDSOIA"), ("EUR", "ECBDFR")):
        d = _fred(daily_code, refresh)
        m = d.groupby(d.index.to_period("M")).mean()
        last = cols[c].index.max()
        ext = m[m.index > last]
        notes[f"{c}_splice"] = f"OECD to {last}, then {daily_code} month-average ({len(ext)} months)"
        cols[c] = pd.concat([cols[c], ext])
    rates = pd.DataFrame(cols).sort_index()
    idx = pd.period_range(rates.index.min(), END, freq="M")
    rates = rates.reindex(idx)
    ffilled = {c: [str(p) for p in rates.index[rates[c].isna() & (rates.index > rates[c].last_valid_index())]]
               for c in CCYS}
    notes["forward_filled"] = {c: v for c, v in ffilled.items() if v}
    return rates.ffill(), notes


def load_fx(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Daily USD value of one unit of each currency (USD = 1). AUD before AUDUSD=X
    starts is AUDJPY / USDJPY — the pre-committed rule."""
    import yfinance as yf
    notes = {}

    def closes(ticker: str) -> pd.Series:
        path = CACHE / f"{ticker.replace('=', '_')}.csv"
        if refresh or not path.exists():
            d = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.rename("close").to_csv(path)
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce")
        return s[s > 0].dropna()

    cols = {}
    for c, (t, direct) in YAHOO.items():
        s = closes(t)
        cols[c] = s if direct else 1.0 / s
    audjpy, usdjpy = closes("AUDJPY=X"), closes("JPY=X")
    synth = (audjpy / usdjpy).dropna()
    first = cols["AUD"].index.min()
    cols["AUD"] = pd.concat([synth[synth.index < first], cols["AUD"]])
    notes["AUD_splice"] = f"AUDJPY/USDJPY before {first.date()}"
    fx = pd.DataFrame(cols).sort_index()
    fx = fx[fx.index.dayofweek < 5]
    fx = fx.ffill(limit=3)
    fx["USD"] = 1.0
    return fx[CCYS], notes


# ----------------------------------------------------------------------------- signal

def decision_dates(fx_index: pd.DatetimeIndex, months: pd.PeriodIndex) -> dict:
    """Holding month m is decided at the last trading day of month m-1."""
    out = {}
    for m in months:
        prev_end = (m - 1).end_time
        eligible = fx_index[fx_index <= prev_end]
        out[m] = eligible[-1] if len(eligible) else None
    return out


def rank_weights(rates: pd.DataFrame, months: pd.PeriodIndex, n: int = 3, lag: int = 1) -> pd.DataFrame:
    """Variant A. The weight for holding month m uses rates for month m-lag only.
    +1/n on the n highest, -1/n on the n lowest. Ties break by CCYS order (stable)."""
    w = pd.DataFrame(0.0, index=months, columns=CCYS)
    for m in months:
        r = rates.loc[m - lag, CCYS]
        if r.isna().any():
            w.loc[m] = np.nan
            continue
        order = r.sort_values(kind="mergesort")
        w.loc[m, list(order.index[-n:])] = 1.0 / n
        w.loc[m, list(order.index[:n])] = -1.0 / n
    return w


def relative_vols(fx: pd.DataFrame, at: pd.Timestamp, lookback: int = 60, min_obs: int = 20) -> pd.Series:
    """Annualised vol of each currency's daily log return against the equal-weighted
    average of the other nine (amendment A1: USD has no vol against itself), using
    prices up to and including `at` only."""
    px = fx.loc[:at].tail(lookback + 1)
    lr = np.log(px).diff().dropna(how="all")
    if len(lr) < min_obs:
        return pd.Series(np.nan, index=CCYS)
    k = len(CCYS)
    rel = lr.mul(k / (k - 1)).sub(lr.sum(axis=1) / (k - 1), axis=0)   # r_c - mean_{j!=c} r_j
    return rel.std() * math.sqrt(252)


def daily_portfolio_returns(fx: pd.DataFrame, w: pd.Series, end: pd.Timestamp, days: int) -> pd.Series:
    """Constant-weight hypothetical daily spot returns of weights `w`, up to `end`."""
    px = fx.loc[:end].tail(days + 1)
    r = px.pct_change().dropna(how="all").fillna(0.0)
    return r[CCYS] @ w[CCYS]


def invvol_gated_weights(rates: pd.DataFrame, fx: pd.DataFrame, months: pd.PeriodIndex,
                         dec: dict, n: int = 3, lag: int = 1) -> pd.DataFrame:
    """Variant B. Same ranks as A; within each side weight ∝ 1/σ, each side summing to
    ±1 (A's gross). Crash gate: halve when the proposed portfolio's 20-day realised
    vol exceeds 2× the median of its rolling 20-day vol over the prior 252 days.
    The gate is inactive until 272 days of history exist (amendment A2)."""
    a = rank_weights(rates, months, n, lag)
    w = pd.DataFrame(0.0, index=months, columns=CCYS)
    for m in months:
        at = dec[m]
        if a.loc[m].isna().any() or at is None:
            w.loc[m] = np.nan
            continue
        vol = relative_vols(fx, at)
        row = pd.Series(0.0, index=CCYS)
        for sign in (1, -1):
            names = a.columns[np.sign(a.loc[m]) == sign]
            inv = 1.0 / vol[names]
            if inv.isna().any() or not np.isfinite(inv).all():
                inv = pd.Series(1.0, index=names)
            row[names] = sign * inv / inv.sum()
        hist = daily_portfolio_returns(fx, row, at, 272)
        if len(hist) >= 272:
            roll = hist.rolling(20).std().dropna()
            if roll.iloc[-1] > 2.0 * roll.iloc[-253:-1].median():
                row *= 0.5
        w.loc[m] = row
    return w


def usdjpy_weights(rates: pd.DataFrame, months: pd.PeriodIndex, lag: int = 1) -> pd.DataFrame:
    """Variant C: short JPY vs USD (= long USD/JPY) when i_USD - i_JPY > admin."""
    w = pd.DataFrame(0.0, index=months, columns=CCYS)
    for m in months:
        r = rates.loc[m - lag]
        if (r["USD"] - r["JPY"]) / 100 > ADMIN:
            w.loc[m, "JPY"] = -1.0
    return w


# ----------------------------------------------------------------------------- P&L

def month_components(fx: pd.DataFrame, rates: pd.DataFrame, months: pd.PeriodIndex, dec: dict):
    """Per-currency spot return and carry for each holding month (vs USD)."""
    spot = pd.DataFrame(index=months, columns=CCYS, dtype=float)
    for m in months:
        start = dec[m]
        end_eligible = fx.index[fx.index <= m.end_time]
        spot.loc[m] = fx.loc[end_eligible[-1]] / fx.loc[start] - 1.0
    carry = rates.loc[months, CCYS].sub(rates.loc[months, "USD"], axis=0) / 100.0 / 12.0
    return spot, carry


def net_returns(w: pd.DataFrame, spot: pd.DataFrame, carry: pd.DataFrame,
                admin: float = ADMIN, admin_scale: float = 1.0) -> pd.DataFrame:
    """Monthly gross / admin / spread / net for weights w. USD's column carries no
    leg (its return vs USD is zero), so costs are on the nine non-USD legs."""
    legs = [c for c in CCYS if c != "USD"]
    gross = (w[CCYS] * (spot[CCYS] + carry[CCYS])).sum(axis=1)
    adm = w[legs].abs().sum(axis=1) * admin / 12.0 * admin_scale
    prev = w[legs].shift(1).fillna(0.0)
    half = pd.Series({c: SPREAD[c] / 2 for c in legs})
    spr = ((w[legs] - prev).abs() * half).sum(axis=1)
    return pd.DataFrame({"gross": gross, "admin": adm, "spread": spr,
                         "net": gross - adm - spr})


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * math.sqrt(12)) if len(x) > 2 and x.std() > 0 else float("nan")


def rotation_null(w: pd.DataFrame, spot: pd.DataFrame, carry: pd.DataFrame, observed: float,
                  min_shift: int = 12) -> dict:
    """Circularly shift the weight series against the return series by every k in
    [12, T-12]; costs recomputed on the rotated weights. p = share of shifts with
    SR >= observed (one-sided)."""
    T = len(w)
    srs = []
    for k in range(min_shift, T - min_shift + 1):
        wr = pd.DataFrame(np.roll(w.values, k, axis=0), index=w.index, columns=w.columns)
        srs.append(sharpe(net_returns(wr, spot, carry)["net"]))
    srs = np.array(srs)
    return {"n_shifts": len(srs), "p": float((srs >= observed).mean()),
            "p_plus1": float(((srs >= observed).sum() + 1) / (len(srs) + 1)),
            "null_median": float(np.median(srs)), "null_p95": float(np.percentile(srs, 95))}


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, p * (len(items) - i)))
        out[k] = running
    return out


def daily_path(fx: pd.DataFrame, w: pd.DataFrame, months: pd.PeriodIndex, dec: dict,
               carry: pd.DataFrame, costs: pd.DataFrame) -> pd.Series:
    """Cumulative unit-notional P&L on DAILY closes (weights held through each
    month; carry/admin accrued evenly; spread booked on the first day). Used for the
    drawdown in the tradeability check — stricter than month-end sampling."""
    legs = [c for c in CCYS if c != "USD"]
    pieces = []
    for m in months:
        days = fx.loc[dec[m]:m.end_time]
        r = days[CCYS].pct_change().dropna(how="all").fillna(0.0)
        if r.empty:
            continue
        # daily return relative to the decision close, weights fixed in units at the start
        rel = days[CCYS].iloc[1:] / days[CCYS].iloc[0] - 1.0
        cum = rel @ w.loc[m, CCYS]
        step = cum.diff().fillna(cum.iloc[0])
        accr = ((w.loc[m, legs] * carry.loc[m, legs]).sum() - costs.loc[m, "admin"]) / len(step)
        step = step + accr
        step.iloc[0] -= costs.loc[m, "spread"]
        pieces.append(step)
    return pd.concat(pieces).cumsum()


def max_drawdown(cum: pd.Series) -> float:
    return float((cum.cummax() - cum).max())


def tradeability(w: pd.DataFrame, dd_unit: float) -> dict:
    """Size so the worst daily-path drawdown = £250; then every active leg's notional
    (|w| × N) must be >= IG's minimum (minDealSize × price) in every month."""
    if not dd_unit or not np.isfinite(dd_unit):
        return {"ok": False, "why": "no drawdown"}
    N = DD_BUDGET_GBP / dd_unit
    legs = [c for c in CCYS if c != "USD"]
    min_notional = pd.Series({c: MIN_SIZE[c] * PRICE_PTS[c] for c in legs})
    need = w[legs].abs() * N
    active = w[legs].abs() > 1e-12
    short = active & (need < min_notional)
    bad_months = int(short.any(axis=1).sum())
    worst = {c: int(short[c].sum()) for c in legs if short[c].any()}
    return {"unit_notional_gbp": round(N, 0), "months_with_leg_below_min": bad_months,
            "months": int(len(w)), "legs_below_min_by_ccy": worst,
            "min_leg_notional_gbp": {c: round(v, 0) for c, v in min_notional.items()},
            "ok": bad_months == 0}


# ----------------------------------------------------------------------------- run

def summarise(name: str, r: pd.DataFrame) -> dict:
    ann = lambda s: float(s.mean() * 12)
    return {"variant": name, "months": int(r["net"].notna().sum()),
            "sr_gross": round(sharpe(r["gross"]), 3), "sr_net": round(sharpe(r["net"]), 3),
            "ann_gross_%": round(100 * ann(r["gross"]), 2), "ann_admin_%": round(100 * ann(r["admin"]), 2),
            "ann_spread_%": round(100 * ann(r["spread"]), 3), "ann_net_%": round(100 * ann(r["net"]), 2),
            "ann_vol_%": round(100 * float(r["net"].std() * math.sqrt(12)), 2),
            "worst_month_%": round(100 * float(r["net"].min()), 2),
            "halves_sr_net": [round(sharpe(r.loc[a:b, "net"]), 3) for a, b in HALVES]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    rates, rnotes = load_rates(args.refresh)
    fx, fnotes = load_fx(args.refresh)
    months = pd.period_range(START, END, freq="M")
    dec = decision_dates(fx.index, months)
    spot, carry = month_components(fx, rates, months, dec)

    outliers = spot[CCYS[1:]].abs().stack()
    outliers = outliers[outliers > 0.12]
    print("data notes:", json.dumps({**rnotes, **fnotes}, indent=1))
    print(f"months {months[0]}..{months[-1]} (T={len(months)}); "
          f"monthly |spot| > 12% (inspect, not removed): "
          + (", ".join(f"{m}:{c} {v:+.1%}" for (m, c), v in
                       (spot[CCYS[1:]].stack()[outliers.index]).items()) or "none"))

    W = {"A": rank_weights(rates, months),
         "B": invvol_gated_weights(rates, fx, months, dec),
         "C": usdjpy_weights(rates, months)}
    R = {k: net_returns(w, spot, carry) for k, w in W.items()}
    S = {k: summarise(k, r) for k, r in R.items()}
    gated = int((W["B"].abs().sum(axis=1) < 1.99).sum())
    S["B"]["months_gated"] = gated

    nulls = {k: rotation_null(W[k], spot, carry, S[k]["sr_net"]) for k in ("A", "B")}
    padj = holm({k: nulls[k]["p"] for k in nulls})

    sens = {}
    for k in ("A", "B"):
        sens[k] = {f"admin_{a:.1%}": round(sharpe(net_returns(W[k], spot, carry, admin=a)["net"]), 3)
                   for a in (0.0, 0.010, 0.015, 0.020, 0.030)}
        sens[k]["crosses_half_admin"] = round(sharpe(net_returns(W[k], spot, carry, admin_scale=0.5)["net"]), 3)
    lag2 = {"A": rank_weights(rates, months, lag=2)}
    sens["A"]["signal_lag2"] = round(sharpe(net_returns(lag2["A"], spot, carry)["net"]), 3)

    trade = {}
    for k in ("A", "B"):
        cum = daily_path(fx, W[k], months, dec, carry, R[k])
        dd = max_drawdown(cum)
        trade[k] = {"max_dd_unit_daily": round(dd, 4),
                    "max_dd_unit_monthly": round(max_drawdown(R[k]["net"].cumsum()), 4),
                    **tradeability(W[k], dd)}

    verdict = {}
    for k in ("A", "B"):
        s = S[k]
        checks = {"sr_net>=0.40": s["sr_net"] >= 0.40,
                  "holm_p<=0.05": padj[k] <= 0.05,
                  "both_halves>0": all(h > 0 for h in s["halves_sr_net"]),
                  "step0_calibration": True,   # 2026-09-22: model £8.73 vs account £9.80 (−11%)
                  "tradeable": trade[k]["ok"]}
        edge = all(v for c, v in checks.items() if c != "tradeable")
        verdict[k] = {"checks": checks,
                      "verdict": "PASS" if edge and checks["tradeable"]
                      else "PASS-untradeable" if edge else "FAIL"}

    out = {"summary": S, "rotation_null": nulls, "holm_p": padj, "sensitivity": sens,
           "tradeability": trade, "verdict": verdict,
           "latest_weights": {k: {c: round(v, 3) for c, v in W[k].iloc[-1].items() if abs(v) > 0}
                              for k in W}}
    (CACHE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    R["A"].join(R["B"], lsuffix="_A", rsuffix="_B").to_csv(CACHE / "monthly_returns.csv")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
