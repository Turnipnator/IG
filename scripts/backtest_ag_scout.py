"""Four-grain ag scout (2026-07-30): Chicago Wheat / Corn / Coffee Arabica /
NY Sugar 11 as uncorrelated additions?

Reuses the per-EPIC walk-forward harness engine (grid on in-sample half ->
best config out-of-sample -> 8-quarter sign consistency, real-spread costs).
Yahoo 1h 730d, zero IG allowance.

Spreads: Coffee 12.2bps MEASURED live (CO.D.KC.Month3.IP, TRADEABLE);
Sugar 31.5bps from IG's standing 5.0pt quote (consistent across 3 months,
market OFFLINE at snapshot). Wheat/Corn were EDITS_ONLY/frozen -> spreads
ASSUMED (17/19bps, typical IG grain quotes) and flagged PROVISIONAL; re-snap
in-session before believing any pass.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/paulturner/IG")
sys.path.insert(0, "/Users/paulturner/IG/scripts")

import backtest_per_epic_walkforward as wf

CANDIDATES = {
    #  name              yahoo    spread_bps  measured?
    "Chicago Wheat":   ("ZW=F",   17.0, "ASSUMED"),
    "Corn":            ("ZC=F",   19.0, "ASSUMED"),
    "Coffee Arabica":  ("KC=F",   12.2, "MEASURED live"),
    "NY Sugar 11":     ("SB=F",   31.5, "standing quote"),
}
PROFILE = "default"


def download() -> None:
    import time

    import yfinance as yf
    for name, (tick, _, _) in CANDIDATES.items():
        f = wf.DATA / f"{name.replace('/', '_')}.csv"
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
                    print(f"  {name:16s} {tick:8s} {len(d)} bars")
                    break
            except Exception as e:
                print(f"  {name:16s} retry {attempt}: {e}")
            time.sleep(2)


def main() -> None:
    download()
    prof = wf.STRATEGY_PROFILES[PROFILE]
    for name, (tick, spread_bps, how) in CANDIDATES.items():
        df = wf.prep(name, PROFILE)
        if df is None:
            print(f"\n### {name}: NO DATA")
            continue
        sf = spread_bps / 1e4
        mid_i = len(df) // 2
        is_df, oos_df = df.iloc[:mid_i].reset_index(drop=True), df.iloc[mid_i:].reset_index(drop=True)
        print(f"\n### {name} ({tick})  spread {spread_bps}bps [{how}]  "
              f"{len(df)} bars  {df.date.iloc[0]:%Y-%m-%d} -> {df.date.iloc[-1]:%Y-%m-%d}")

        # grid on IS at full spread, pick best by IS pnl
        best, best_key = None, None
        for adx in wf.GRID_ADX:
            for use_macd in [False, True]:
                sig = wf.signals(is_df, prof, adx, use_macd)
                for stop in wf.GRID_STOP:
                    for rr in wf.GRID_RR:
                        for d in wf.GRID_DIR:
                            tr = wf.simulate(is_df, sig, prof, stop, rr, d, use_macd, sf, 0.0)
                            st = wf.stats(tr)
                            if st["n"] >= 8 and (best is None or st["pnl"] > best["pnl"]):
                                best, best_key = st, (adx, stop, rr, d, use_macd)
        if best is None:
            print("  no config with n>=8 in-sample — nothing to validate")
            continue
        adx, stop, rr, d, use_macd = best_key
        print(f"  best IS: ADX={adx} stop={stop} rr={rr} dir={d} macd={use_macd}  "
              f"n={best['n']} pnl={best['pnl']:+.2f}% PF={best['pf']:.2f}")

        # OOS at three cost levels
        for label, mult in [("1.0x", 1.0), ("0.5x", 0.5), ("0.0x", 0.0)]:
            sig = wf.signals(oos_df, prof, adx, use_macd)
            st = wf.stats(wf.simulate(oos_df, sig, prof, stop, rr, d, use_macd, sf * mult, 0.0))
            verdict = "SURVIVES" if st["pnl"] > 0 else "FAILS"
            print(f"  OOS spread {label}: n={st['n']:3d} pnl={st['pnl']:+7.2f}% "
                  f"PF={st['pf']:5.2f} WR={st['wr']:3.0f}%  {verdict}")

        # 8-quarter sign consistency at full spread
        sig = wf.signals(df, prof, adx, use_macd)
        tr = wf.simulate(df, sig, prof, stop, rr, d, use_macd, sf, 0.0)
        qs = []
        if tr:
            dates = pd.to_datetime([t[0] for t in tr])
            pnls = np.array([t[1] for t in tr])
            edges = pd.date_range(df.date.iloc[0], df.date.iloc[-1], periods=9)
            for k in range(8):
                m = (dates >= edges[k]) & (dates < edges[k + 1])
                qs.append(float(pnls[m].sum()) if m.any() else 0.0)
        marks = "".join("+" if q > 0 else ("0" if q == 0 else "-") for q in qs)
        green = sum(1 for q in qs if q > 0)
        print(f"  quarters [{marks}] {green}/8 green   full: n={len(tr)} "
              f"pnl={sum(t[1] for t in tr):+.2f}% ")


if __name__ == "__main__":
    main()
