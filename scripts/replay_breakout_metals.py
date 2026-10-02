"""Does the live Gold 1h breakout carry over to silver and platinum?
Pre-registered in research_notes.md (14b2bb5; platinum data-range deviation 1514d68).

  python scripts/replay_breakout_metals.py

Runs replay_breakout_news.run_gated UNCHANGED on Dukascopy 1h bars. Silver and
platinum are not in config.MARKETS, so in-memory copies of Gold's MarketConfig and
BreakoutConfig are registered under their IG epics for this process only — nothing
on disk or in the live bot changes. Gold is re-run first as a harness check.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_breakout_news as rbn  # noqa: E402

GOLD = "CS.D.USCGC.TODAY.IP"
SILVER = "CS.D.USCSI.TODAY.IP"
PLAT = "MT.D.PL.Month2.IP"
START = pd.Timestamp("2005-01-01")
CAP_GBP = 45.0
OUT = rbn.DATA / "metals_outcomes.json"

# epic -> (Dukascopy glob, scale to IG points, min stop, spreads low/base/high, financing %/yr)
METALS = {
    SILVER: ("xagusd-h1-*.csv", 100.0, 4.0, (2.0, 3.0, 4.0), rbn.GOLD_ALL_IN),
    PLAT:   ("xptcmdusd-h1-*.csv", 1.0, 8.0, (2.0, 3.0, 5.0), 0.0),
}
NAMES = {GOLD: "Gold", SILVER: "Silver", PLAT: "Platinum"}


def register() -> None:
    """Make silver/platinum look like Gold to run_gated (in memory only)."""
    gold_mk = next(m for m in rbn.MARKETS if m.epic == GOLD)
    for epic, (glob, scale, min_stop, spreads, fin) in METALS.items():
        rbn.UNIVERSE[epic] = (glob, scale, ("USD",))
        rbn.SPREAD[epic] = spreads[1]
        rbn.BREAKOUT_CONFIGS[epic] = rbn.BREAKOUT_CONFIGS[GOLD]
        if not any(m.epic == epic for m in rbn.MARKETS):
            rbn.MARKETS.append(dataclasses.replace(gold_mk, epic=epic, name=NAMES[epic],
                                                   min_stop_distance=min_stop))
    base_fin = rbn.financing_rate

    def fin_rate(epic, d, when, diffs):
        return METALS[epic][4] if epic in METALS else base_fin(epic, d, when, diffs)
    rbn.financing_rate = fin_rate


def run(epic: str, spread: float | None = None) -> pd.DataFrame:
    if spread is not None:
        rbn.SPREAD[epic] = spread
    df = rbn.load_market(epic)
    df["htf"] = rbn.htf_day_fast(df)
    t = rbn.run_gated(df, epic, None)
    t = t[t["date"] >= START].reset_index(drop=True)
    t["cost_r"] = t["r_gross"] - t["r"]
    return t


def verdict(t: pd.DataFrame, t_hi: pd.DataFrame, own_halves: bool) -> dict:
    r = t["r"].to_numpy(float)
    n = len(r)
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    tstat = mean / (sd / np.sqrt(n))
    if own_halves:
        cut = t["date"].median()
        halves = {"first": t[t["date"] < cut], "second": t[t["date"] >= cut]}
    else:
        halves = {"2005-2015": t[t["date"] <= "2015-12-31"], "2016-2026": t[t["date"] >= "2016-01-01"]}
    hmeans = {k: round(float(v["r"].mean()), 4) for k, v in halves.items()}
    drop10 = float(np.sort(r)[:-10].sum())
    cost = float(t["cost_r"].mean())
    crit = {"1_mean_pos_t_ge_2": mean > 0 and tstat >= 2.0,
            "2_both_halves_pos": all(v > 0 for v in hmeans.values()),
            "3_total_ex_top10_pos": drop10 > 0,
            "4_cost_le_0.10R": cost <= 0.10,
            "5_pos_at_high_cost": float(t_hi["r"].mean()) > 0}
    return {"n": n, "from": str(t["date"].min().date()), "to": str(t["date"].max().date()),
            "mean_r": round(mean, 4), "t": round(tstat, 2), "total_r": round(float(r.sum()), 1),
            "mean_gross_r": round(float(t["r_gross"].mean()), 4), "mean_cost_r": round(cost, 4),
            "halves": hmeans, "total_ex_top10": round(drop10, 1),
            "mean_r_high_cost": round(float(t_hi["r"].mean()), 4),
            "win_rate": round(float((r > 0).mean()), 3),
            "top3_share_of_total": round(float(np.sort(r)[-3:].sum() / r.sum()), 2) if r.sum() > 0 else None,
            "criteria": crit, "verdict": "PASS" if all(crit.values()) else "FAIL -> NO EDGE"}


def main() -> None:
    register()
    # Harness check: must reproduce the stored Gold replay exactly.
    stored = pd.read_pickle(rbn.DATA / "replay_trades.pkl")
    stored = stored[(stored["epic"] == GOLD) & (stored["date"] >= START)]
    g = run(GOLD)
    print(f"Harness: Gold n={len(g)} mean {g.r.mean():+.4f} vs stored n={len(stored)} mean {stored.r.mean():+.4f}")
    # Stored Gold used the FX-carry diffs; Gold's financing ignores them, so it must match.
    assert len(g) == len(stored) and abs(g.r.mean() - stored.r.mean()) < 1e-9, "harness does not reproduce Gold"

    res = {"gold_reference": verdict(g, g, own_halves=False)}
    for epic, (_, scale, _, spreads, _) in METALS.items():
        base = run(epic, spreads[1])
        hi = run(epic, spreads[2])
        lo = run(epic, spreads[0])
        v = verdict(base, hi, own_halves=(epic == PLAT))
        v["mean_r_low_cost"] = round(float(lo["r"].mean()), 4)
        # Tradeability at today's price and IG's min size.
        if epic == SILVER:
            gbp_risk = base["sd"] * 1.0                     # £1/pt minimum, 1 pt = 1 cent
            v["share_over_cap"] = round(float((gbp_risk > CAP_GBP).mean()), 3)
            last_px = rbn.load_market(epic)["close"].iloc[-1]
            # Express each trade's stop as % of ITS price, then at today's price.
            df = rbn.load_market(epic).set_index("date")["close"]
            pct = base["sd"].to_numpy() / df.reindex(base["date"]).to_numpy()
            v["share_over_cap_at_today_price"] = round(float((pct * last_px > CAP_GBP).mean()), 3)
            v["median_gbp_risk_today"] = round(float(np.median(pct * last_px)), 1)
        else:
            v["median_gbp_risk_min_size"] = round(float(np.median(base["sd"] * 0.04)), 2)
        res[NAMES[epic]] = v
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
