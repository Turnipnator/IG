#!/usr/bin/env python
"""Blind forecast test: can inclusionai/ling-3.0-flash-fin rank 5-day forward returns
from anonymised daily OHLC windows (no ticker, no dates, rebased to 100)?

Pre-registered in IG/research_notes.md (2026-09-15) BEFORE any forecast was requested.
Subcommands: build | run | score
"""
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/paulturner/IG")
OUT = REPO / "data" / "ling_forecast"
MODEL = "inclusionai/ling-3.0-flash-fin:free"
SEED = 20260915
WINDOW = 40
HORIZON = 5
PER_MARKET = 100
BATCH = 10
MIN_SPACING = 20
SOURCES = {
    "SPX": REPO / "data/backtest_cache/GSPC_1d_full.csv",
    "NDX": REPO / "tests/fixtures/ndx_daily.csv",
    "GOLD": REPO / "tests/fixtures/gc_f_daily.csv",
}
FEATURES = ["atr14", "sma200", "rsi14", "ret5", "ret20", "low5_prev", "fwd", "fwd_atr"]

SYSTEM = (
    "You are a quantitative market analyst. You will be shown anonymised daily price windows from liquid, "
    "exchange-traded financial instruments. Instrument names and dates are hidden and each window is rebased "
    f"so its first close = 100. Each window has {WINDOW} consecutive daily bars, oldest first, one per line "
    "as open,high,low,close."
)
USER_HEAD = (
    f"For EACH window below, estimate the probability that the close {HORIZON} trading days after the window's "
    "last bar will be HIGHER than the window's last close. Use whatever analysis you judge best. Be calibrated: "
    "0.50 means no edge either way. Return ONLY a JSON object, no prose, exactly in this shape: "
    '{"forecasts": [{"id": "<window id>", "p_up": <number between 0 and 1>}, ...]} with one entry per window.\n\n'
)


def load(path):
    df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["open", "high", "low", "close"])
    px = ["open", "high", "low", "close"]
    df = df[(df[px] > 0).all(axis=1)].reset_index(drop=True)
    df["high"] = df[px].max(axis=1)
    df["low"] = df[px].min(axis=1)
    c = df["close"]
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["sma200"] = c.rolling(200).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss)
    df["ret5"] = c / c.shift(5) - 1
    df["ret20"] = c / c.shift(20) - 1
    df["low5_prev"] = df["low"].shift(1).rolling(5).min()
    df["fwd"] = c.shift(-HORIZON) / c - 1
    df["fwd_atr"] = (c.shift(-HORIZON) - c) / df["atr14"]
    return df


def build():
    rng = random.Random(SEED)
    windows, used_ids = [], set()
    for mkt, path in SOURCES.items():
        df = load(path)
        cands = list(range(250, len(df) - HORIZON))
        rng.shuffle(cands)
        chosen = []
        for t in cands:
            if not np.isfinite(df.loc[t, FEATURES].astype(float)).all():
                continue
            if all(abs(t - s) >= MIN_SPACING for s in chosen):
                chosen.append(t)
            if len(chosen) == PER_MARKET:
                break
        for t in chosen:
            w = df.iloc[t - WINDOW + 1 : t + 1]
            base = w["close"].iloc[0]
            r = df.loc[t]
            while True:
                wid = f"W{rng.randrange(16**4):04x}"
                if wid not in used_ids:
                    used_ids.add(wid)
                    break
            windows.append(dict(
                id=wid, market=mkt, date=str(r.date.date()),
                bars=(w[["open", "high", "low", "close"]] / base * 100).round(2).values.tolist(),
                fwd=float(r.fwd), fwd_atr=float(r.fwd_atr), ret5=float(r.ret5), ret20=float(r.ret20),
                rsi14=float(r.rsi14), above_sma200=bool(r.close > r.sma200),
                pullback=bool(r.close > r.sma200 and r.close < r.low5_prev), atr_pct=float(r.atr14 / r.close),
            ))
    rng.shuffle(windows)
    batches = [[w["id"] for w in windows[i : i + BATCH]] for i in range(0, len(windows), BATCH)]
    OUT.mkdir(exist_ok=True)
    (OUT / "windows.json").write_text(json.dumps(windows))
    (OUT / "batches.json").write_text(json.dumps(batches))
    df = pd.DataFrame(windows)
    print(f"built {len(windows)} windows in {len(batches)} batches")
    print(df.groupby("market").agg(n=("id", "size"), first=("date", "min"), last=("date", "max"),
                                    up=("fwd", lambda s: (s > 0).mean()), pullback=("pullback", "sum")))
    sample = windows[0]
    print("example prompt block:\n" + window_text(sample)[:300])


def window_text(w):
    return f"### {w['id']}\n" + "\n".join(",".join(f"{x:.2f}" for x in bar) for bar in w["bars"])


def call(messages, timeout=600):
    key = os.environ["OPENROUTER_API_KEY"]
    body = {"model": MODEL, "messages": messages, "reasoning": {"enabled": True}, "seed": SEED}
    for attempt in range(3):
        t0 = time.time()
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=timeout,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                 json=body)
            d = resp.json()
        except Exception as e:  # network / JSON decode
            d = {"error": repr(e)}
            resp = None
        if resp is not None and resp.status_code == 200 and d.get("choices"):
            m = d["choices"][0]["message"]
            return dict(content=m.get("content") or "", reasoning=m.get("reasoning") or "",
                        usage=d.get("usage"), provider=d.get("provider"), elapsed=round(time.time() - t0, 1),
                        attempts=attempt + 1)
        print(f"  attempt {attempt + 1} failed: {str(d)[:300]}", flush=True)
        if resp is not None and resp.status_code == 429 and "per-day" in str(d).lower():
            raise SystemExit("daily free-tier quota exhausted")
        time.sleep(30)
    return None


def parse(content, ids):
    out = {}
    text = re.sub(r"```(?:json)?", "", content)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            for f in json.loads(m.group(0)).get("forecasts", []):
                out[str(f["id"])] = float(f["p_up"])
        except Exception:
            out = {}
    if not out:
        for wid, p in re.findall(r'"id"\s*:\s*"([^"]+)"\s*,\s*"p_up"\s*:\s*([0-9.]+)', text):
            out[wid] = float(p)
    return {k: v for k, v in out.items() if k in ids and 0 <= v <= 1}


def run_batch(i, ids, W):
    f = OUT / "raw" / f"batch_{i:02d}.json"
    if f.exists() and len(parse(json.loads(f.read_text())["content"], set(ids))) == len(ids):
        return i, "cached"
    user = USER_HEAD + "\n\n".join(window_text(W[w]) for w in ids)
    res = call([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    if res is None:
        return i, "FAILED"
    res["ids"] = ids
    f.write_text(json.dumps(res))
    got = len(parse(res["content"], set(ids)))
    return i, f"{got}/{len(ids)} parsed, {res['elapsed']}s, {res['usage'].get('completion_tokens')} out tok"


def run(limit=None):
    W = {w["id"]: w for w in json.loads((OUT / "windows.json").read_text())}
    batches = json.loads((OUT / "batches.json").read_text())[: limit or None]
    (OUT / "raw").mkdir(exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for i, status in ex.map(lambda ib: run_batch(ib[0], ib[1], W), enumerate(batches)):
            print(f"batch {i:02d}: {status}", flush=True)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def within_rank(df, col):
    return df.groupby("market")[col].rank(pct=True)


def perm_p(x, y, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    obs = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(rng.permutation(x), y)[0, 1] for _ in range(n)])
    return obs, float((null >= obs).mean())


def ols_t(y, X, names):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {n: (round(float(b), 4), round(float(b / s), 2)) for n, b, s in zip(["const"] + names, beta, se)}


def score():
    W = {w["id"]: w for w in json.loads((OUT / "windows.json").read_text())}
    rows, meta = [], []
    for f in sorted((OUT / "raw").glob("batch_*.json")):
        d = json.loads(f.read_text())
        preds = parse(d["content"], set(d["ids"]))
        meta.append((d["elapsed"], d["usage"].get("completion_tokens"), len(d["reasoning"])))
        rows += [{**{k: v for k, v in W[i].items() if k != "bars"}, "p": p} for i, p in preds.items()]
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"parsed {n}/{len(W)} forecasts from {len(meta)} batches; median latency "
          f"{np.median([m[0] for m in meta]):.0f}s, median out tokens {np.median([m[1] for m in meta]):.0f}")
    print("\n== forecast distribution ==")
    print(df["p"].describe().round(3).to_string(), f"\nunique values: {df.p.nunique()}, share exactly 0.50: "
          f"{(df.p == 0.5).mean():.2f}")

    y = (df.fwd > 0).astype(float)
    df["rp"], df["rf"] = within_rank(df, "p"), within_rank(df, "fwd_atr")
    rho, p = perm_p(df.rp.values, df.rf.values)
    print(f"\n== PRIMARY: within-market rank corr(p_up, 5d fwd ATR move) ==\nrho={rho:+.3f}  one-sided perm p={p:.3f}  "
          f"(n={n}; 2-sigma detectable |rho| ~ {2 / np.sqrt(n):.2f})")
    for mkt, g in df.groupby("market"):
        print(f"  {mkt:5s} n={len(g):3d} rho={spearman(g.p, g.fwd_atr):+.3f}")

    df["terc"] = pd.qcut(df.p.rank(method="first"), 3, labels=["low", "mid", "high"])
    t = df.groupby("terc", observed=True).agg(n=("p", "size"), p_mean=("p", "mean"), fwd_atr=("fwd_atr", "mean"),
                                              up_rate=("fwd", lambda s: (s > 0).mean()))
    print("\n== terciles of p_up ==\n" + t.round(3).to_string())
    hi, lo = df[df.terc == "high"].fwd_atr, df[df.terc == "low"].fwd_atr
    welch = (hi.mean() - lo.mean()) / np.sqrt(hi.var() / len(hi) + lo.var() / len(lo))
    print(f"high-minus-low fwd ATR = {hi.mean() - lo.mean():+.3f}  Welch t = {welch:+.2f}")

    z = lambda s: (s - s.mean()) / s.std()
    dummies = [(df.market == m).astype(float).values for m in sorted(df.market.unique())[1:]]
    print("\n== OLS fwd_atr ~ p_up + ret5 + ret20 + market (coef, t) ==")
    print(ols_t(df.fwd_atr.values, [z(df.p).values, z(df.ret5).values, z(df.ret20).values] + dummies,
                ["p_up_z", "ret5_z", "ret20_z"] + [f"mkt_{m}" for m in sorted(df.market.unique())[1:]]))

    longrun = {m: float((load(SOURCES[m]).fwd.dropna() > 0).mean()) for m in SOURCES}
    clim = df.market.map(longrun)
    decided = df.p != 0.5
    print("\n== calibration / hit rate ==")
    print(f"Brier model={np.mean((df.p - y) ** 2):.4f}  long-run climatology={np.mean((clim - y) ** 2):.4f}  "
          f"coin 0.5={np.mean((0.5 - y) ** 2):.4f}")
    print(f"hit rate (p!=0.5, n={decided.sum()}): {((df.p[decided] > 0.5) == (df.fwd[decided] > 0)).mean():.3f}  "
          f"vs always-up: {(df.fwd[decided] > 0).mean():.3f}   long-run up-rates {dict((k, round(v, 3)) for k, v in longrun.items())}")

    print("\n== trivial baselines: within-market rank corr with 5d fwd ATR move ==")
    for name, col, sign in [("momentum ret20", "ret20", 1), ("reversal -ret5", "ret5", -1), ("-RSI14", "rsi14", -1),
                            ("above SMA200", "above_sma200", 1), ("pullback flag", "pullback", 1)]:
        df["_b"] = sign * df[col].astype(float)
        print(f"  {name:16s} rho={np.corrcoef(within_rank(df, '_b'), df.rf)[0, 1]:+.3f}")

    print("\n== what is the model keying on? within-market rank corr(p_up, feature) ==")
    for col in ["ret20", "ret5", "rsi14", "above_sma200", "pullback", "atr_pct"]:
        df["_f"] = df[col].astype(float)
        print(f"  {col:13s} rho={np.corrcoef(df.rp, within_rank(df, '_f'))[0, 1]:+.3f}")
    df.drop(columns=["_b", "_f", "rp", "rf"]).to_csv(OUT / "scored.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "score":
        score()
