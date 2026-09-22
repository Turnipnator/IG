"""Outcome stage of the breakout news replay — called only by
`scripts/replay_breakout_news.py --outcomes`, after Step 1's power gate.

Implements pre-registration §6-§7 with amendments C1-C5 (research_notes.md):
  Δ = mean net R(BLOCKED) - mean net R(rest), per family, entries from 2005-01-01
  null = whole-week rotations of the event calendar (k = ±1..±26), two-sided
  Holm across the two families; halves; drop-the-extreme-trade robustness;
  top-10 tail veto. Everything after the verdict is descriptive.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import replay_breakout_news as rbn

START = pd.Timestamp("2005-01-01")


def _tags(trades: pd.DataFrame, ev: pd.DataFrame, family: str, weeks: int = 0,
          window: pd.Timedelta = rbn.WINDOW, extra: pd.DataFrame | None = None) -> np.ndarray:
    tag = np.zeros(len(trades), dtype=bool)
    for epic in rbn.UNIVERSE:
        real = rbn.family_events(ev, epic, family)
        if extra is not None:
            real = np.concatenate([real, rbn.family_events(extra, epic, family)])
        if len(real) == 0:
            continue
        sel = (trades["epic"] == epic).to_numpy()
        evs = real if weeks == 0 else rbn.shifted_events(real, weeks, window)
        tag[sel] = rbn.tag_blocked(trades.loc[sel, "decision_utc"], evs, window)
    return tag


def _delta(r: np.ndarray, tag: np.ndarray) -> float:
    if tag.sum() == 0 or (~tag).sum() == 0:
        return float("nan")
    return float(r[tag].mean() - r[~tag].mean())


def _rotation(r: np.ndarray, trades, ev, family, obs: float) -> dict:
    ds = np.array([_delta(r, _tags(trades, ev, family, k)) for k in rbn.SHIFTS])
    ds = ds[np.isfinite(ds)]
    return {"p_two_sided": float((np.abs(ds) >= abs(obs)).mean()), "n_shifts": int(len(ds)),
            "null_median": round(float(np.median(ds)), 4),
            "null_p2.5_p97.5": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)]}


def _holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, p * (len(items) - i)))
        out[k] = run
    return out


def run(families: list[str]) -> None:
    trades = pd.read_pickle(rbn.DATA / "replay_trades.pkl")
    trades = trades[trades["date"] >= START].reset_index(drop=True)
    ev = rbn.load_events()
    top10 = set(trades["r"].nlargest(10).index)
    res, pvals = {}, {}

    for fam in families:
        fam_mask = np.zeros(len(trades), dtype=bool)
        for epic in rbn.UNIVERSE:
            if len(rbn.family_events(ev, epic, fam)):
                fam_mask |= (trades["epic"] == epic).to_numpy()
        t = trades[fam_mask].reset_index(drop=True)
        r = t["r"].to_numpy(float)
        tag = _tags(t, ev, fam)
        n_b, n = int(tag.sum()), len(t)
        gate = rbn.power(n_b, n)
        obs = _delta(r, tag)
        rot = _rotation(r, t, ev, fam, obs)
        pvals[fam] = rot["p_two_sided"]

        halves = []
        for a, b in rbn.HALVES:
            m = ((t["date"] >= a) & (t["date"] <= b)).to_numpy()
            halves.append(round(_delta(r[m], tag[m]), 4))

        # robustness: drop the single most extreme blocked trade in the direction of Δ
        b_idx = np.flatnonzero(tag)
        extreme = b_idx[np.argmin(r[b_idx])] if obs < 0 else b_idx[np.argmax(r[b_idx])]
        keep = np.ones(n, dtype=bool); keep[extreme] = False
        t2, r2, tag2 = t[keep].reset_index(drop=True), r[keep], tag[keep]
        obs2 = _delta(r2, tag2)
        rot2 = _rotation(r2, t2, ev, fam, obs2)

        orig_index = trades.index[fam_mask][tag]
        tail_hits = int(len(set(orig_index) & top10))

        blocked_r = r[tag]
        res[fam] = {
            "gate_2005plus": gate,
            "mean_R_blocked": round(float(blocked_r.mean()), 4), "mean_R_rest": round(float(r[~tag].mean()), 4),
            "delta_R": round(obs, 4), "book_effect_R_per_trade": round(gate["f"] * abs(obs), 4),
            "rotation": rot, "halves_delta": halves,
            "without_extreme": {"delta_R": round(obs2, 4), "p_two_sided": rot2["p_two_sided"],
                                "book_effect": round(gate["f"] * abs(obs2), 4),
                                "dropped_trade_R": round(float(r[extreme]), 3)},
            "tail_top10_in_blocked": tail_hits,
            "blocked_detail": {"win_rate": round(float((blocked_r > 0).mean()), 3),
                               "stop_share": round(float((t.loc[tag, "why"] == "stop").mean()), 3),
                               "sum_R": round(float(blocked_r.sum()), 2),
                               "by_market": t[tag].groupby("market")["r"].agg(["count", "mean"]).round(3).to_dict("index")},
        }

    padj = _holm(pvals)
    for fam in families:
        x = res[fam]
        x["holm_p"] = round(padj[fam], 4)
        d, p, eff = x["delta_R"], padj[fam], x["book_effect_R_per_trade"]
        same_sign = all(np.sign(h) == np.sign(d) for h in x["halves_delta"])
        we = x["without_extreme"]
        # the same Holm multiplier this family's own p received (x2 for the smaller p)
        mult = len(families) if pvals[fam] == min(pvals.values()) else 1
        we["p_holm"] = round(min(1.0, we["p_two_sided"] * mult), 4)
        robust = (np.sign(we["delta_R"]) == np.sign(d) and we["p_holm"] <= 0.05
                  and we["book_effect"] >= rbn.MIN_BOOK_EFFECT)
        core = p <= 0.05 and eff >= rbn.MIN_BOOK_EFFECT and same_sign and robust
        if x["tail_top10_in_blocked"] >= 2:
            verdict = "BLOCK HURTS (tail veto)"
        elif core and d < 0:
            verdict = "BLOCK HELPS"
        elif core and d > 0:
            verdict = "BLOCK HURTS"
        else:
            verdict = "NO EFFECT"
        x["checks"] = {"holm_p<=0.05": p <= 0.05, "book_effect>=0.02": eff >= rbn.MIN_BOOK_EFFECT,
                       "halves_same_sign": bool(same_sign), "survives_extreme": bool(robust),
                       "tail_veto": x["tail_top10_in_blocked"] >= 2}
        x["verdict"] = verdict

    # ---------------------------------------------------------------- descriptive only
    desc = {}
    for fam in families:
        fam_mask = np.zeros(len(trades), dtype=bool)
        for epic in rbn.UNIVERSE:
            if len(rbn.family_events(ev, epic, fam)):
                fam_mask |= (trades["epic"] == epic).to_numpy()
        t = trades[fam_mask].reset_index(drop=True)
        r = t["r"].to_numpy(float)
        tag = _tags(t, ev, fam)
        d = {}
        for extra_R in (0.25, 0.50):
            rs = r.copy()
            rs[tag & (t["why"] == "stop").to_numpy()] -= extra_R
            d[f"news_slippage_{extra_R}R_delta"] = round(_delta(rs, tag), 4)
        for mins in (60, 90):
            w = pd.Timedelta(minutes=mins)
            d[f"window_{mins}min_delta"] = round(_delta(r, _tags(t, ev, fam, window=w)), 4)
        live = ((t["epic"].isin(["CS.D.USCGC.TODAY.IP", "CS.D.GBPUSD.TODAY.IP"])) & (t["d"] == "BUY")).to_numpy()
        if (tag & live).sum() and (~tag & live).sum():
            d["live_set_long_only"] = {"n_blocked": int((tag & live).sum()),
                                       "delta_R": round(_delta(r[live], tag[live]), 4)}
        if fam == "core-GBP/EUR":
            amb = ev[(ev["event"] == "UK CPI") & (ev["utc"] >= "2020-04-01") & (ev["utc"] < "2021-07-01")].copy()
            amb["utc"] = amb["utc"] + pd.Timedelta(hours=2, minutes=30)   # the 09:30 alternative
            d["uk_cpi_both_times_delta"] = round(_delta(r, _tags(t, ev, fam, extra=amb)), 4)
        desc[fam] = d

    # "block ON" run — different sequencing, reported as a book total
    diffs = rbn.rate_diffs()
    on, off = [], []
    for epic in rbn.UNIVERSE:
        df = rbn.load_market(epic)
        df["htf"] = rbn.htf_day_fast(df)
        evs = np.concatenate([rbn.family_events(ev, epic, f) for f in families])
        mask = rbn.tag_blocked(df["utc"] + pd.Timedelta(hours=1), evs)
        a = rbn.run_gated(df, epic, diffs)
        b = rbn.run_gated(df, epic, diffs, skip_mask=mask)
        off.append(a[a["date"] >= START]); on.append(b[b["date"] >= START])
    off, on = pd.concat(off), pd.concat(on)
    desc["block_on_run"] = {"off": {"n": int(len(off)), "sum_R": round(float(off["r"].sum()), 2),
                                    "mean_R": round(float(off["r"].mean()), 4)},
                            "on": {"n": int(len(on)), "sum_R": round(float(on["r"].sum()), 2),
                                   "mean_R": round(float(on["r"].mean()), 4)}}

    out = {"primary": res, "descriptive": desc,
           "book": {"n": int(len(trades)), "mean_R": round(float(trades["r"].mean()), 4),
                    "sd_R": round(float(trades["r"].std()), 3)}}
    (rbn.DATA / "replay_outcomes.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
