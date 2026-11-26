"""Forward scoring of alternative breakout exits on the bot's OWN entries.

Pre-registered in research_notes.md (2026-11-27, "Forward scoring of breakout exits").
Read that before changing an arm or a rule.

Every breakout entry the bot actually made (live `trades`) or observed (shadow
`benched_outcomes`) is replayed on the IG candle archive under each exit arm. The entry,
its price and its initial stop are the journal's; only the exit differs. Read-only: the
journal is opened mode=ro and nothing is written anywhere.

Run inside the container (prints a report):
    docker exec -i ig-trading-bot python3 - < scripts/score_breakout_exits.py
    docker exec -i ig-trading-bot python3 - --since 2026-11-27 < scripts/score_breakout_exits.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.breakout import BREAKOUT_CONFIGS

DB = "/app/data/trade_journal.db"
ARCHIVE = Path("/app/data/candle_archive")
FORWARD_FROM = "2026-11-27"          # entries from here on are the forward sample
TIGHT_M = 8
DEFAULT_N = 55                       # retired epics (old futures contracts) ran N55 too
# arm -> (kind, value). R = the journal's initial stop distance.
ARMS = {
    "trail (now)":      (None, 0.0),
    "target 1R":        ("tp_r", 1.0),     # ~ 0.5 x 40-day ATR (measured 0.99-1.15R)
    "target 2R":        ("tp_r", 2.0),     # ~ 1.0 x 40-day ATR
    "target 50pts":     ("tp_pts", 50.0),  # Gold only
    "break-even at 1R": ("be", 1.0),
    "break-even at 2R": ("be", 2.0),
    "tighten at 1R":    ("tight", 1.0),    # trail channel M -> 8 once +1R is seen
}
GOLD = "CS.D.USCGC.TODAY.IP"

since = FORWARD_FROM
if "--since" in sys.argv:
    since = sys.argv[sys.argv.index("--since") + 1]


def load_archive(epic):
    path = ARCHIVE / f"{epic}.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if len(rows) < 100:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    step = df["date"].diff().median()
    hourly = (df.set_index("date").resample("1h")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    cfg = BREAKOUT_CONFIGS.get(epic)
    m = cfg.m if cfg else max(2, DEFAULT_N // 2)
    # Level in force during hour H = extreme of the M completed hours before H.
    lv = pd.DataFrame({
        "lo_m": hourly["low"].rolling(m).min().shift(1), "hi_m": hourly["high"].rolling(m).max().shift(1),
        "lo_t": hourly["low"].rolling(TIGHT_M).min().shift(1), "hi_t": hourly["high"].rolling(TIGHT_M).max().shift(1)})
    df = df.join(lv, on=df["date"].dt.floor("1h"))
    return df, step


def replay(df, t0, direction, entry, risk, kind, val):
    """Returns (R, exit reason) or None if the trade is still open at the end of the data."""
    s = 1.0 if direction == "BUY" else -1.0
    start = int(df["date"].searchsorted(t0))
    if start >= len(df):
        return None
    o, hi, lo = (df[c].to_numpy(float) for c in ("open", "high", "low"))
    lvl_m = df["lo_m" if s > 0 else "hi_m"].to_numpy(float)
    lvl_t = df["lo_t" if s > 0 else "hi_t"].to_numpy(float)
    stop = entry - s * risk
    tp = entry + s * val * risk if kind == "tp_r" else entry + s * val if kind == "tp_pts" else None
    mfe, armed = 0.0, False
    for j in range(start, len(df)):
        if kind in ("be", "tight") and not armed and mfe >= val * risk:
            armed = True                               # seen on bars closed before j
            if kind == "be":
                stop = max(stop, entry) if s > 0 else min(stop, entry)
        level = lvl_t[j] if (armed and kind == "tight") else lvl_m[j]
        if np.isfinite(level):
            stop = max(stop, level) if s > 0 else min(stop, level)
        adverse, favourable = (lo[j], hi[j]) if s > 0 else (hi[j], lo[j])
        if s * adverse <= s * stop:                    # stop checked before the target
            px = min(o[j], stop) if s > 0 else max(o[j], stop)
            return s * (px - entry) / risk, "stop/trail"
        if tp is not None and s * favourable >= s * tp:
            return s * (tp - entry) / risk, "target"
        mfe = max(mfe, s * (favourable - entry))
    return None


def entries():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("""SELECT id, epic, market_name, direction, entry_price, entry_time, stop_distance,
                                 exit_price, status FROM trades WHERE strategy='breakout'"""):
        actual = None
        if r["status"] == "CLOSED" and r["exit_price"] and r["stop_distance"]:
            sgn = 1 if r["direction"] == "BUY" else -1
            actual = sgn * (r["exit_price"] - r["entry_price"]) / r["stop_distance"]
        out.append(dict(src="live", id=r["id"], epic=r["epic"], market=r["market_name"], d=r["direction"],
                        entry=r["entry_price"], risk=r["stop_distance"],
                        when=pd.to_datetime(r["entry_time"]), actual=actual))
    for r in c.execute("""SELECT id, epic, market_name, direction, entry_price, benched_at, stop_distance,
                                 r_multiple, status FROM benched_outcomes
                          WHERE bench_type IN ('breakout-shadow','breakout-shadow-bf') AND status != 'VOID'"""):
        out.append(dict(src="shadow", id=r["id"], epic=r["epic"], market=r["market_name"], d=r["direction"],
                        entry=r["entry_price"], risk=r["stop_distance"],
                        when=pd.to_datetime(r["benched_at"]) + pd.Timedelta(hours=1), actual=r["r_multiple"]))
    e = pd.DataFrame(out)
    e = e[(e["risk"] > 0) & e["entry"].notna()].sort_values("when").reset_index(drop=True)
    # The observer also logs breaks on markets that are live: keep the live row.
    live = e[e.src == "live"]
    dup = e.apply(lambda r: r.src == "shadow" and bool(
        ((live.epic == r.epic) & (live.d == r.d) & ((live.when - r.when).abs() <= pd.Timedelta(hours=2))).any()), axis=1)
    return e[~dup].reset_index(drop=True), int(dup.sum())


def boot_ci(x, n=10_000, seed=0):
    if len(x) < 5:
        return None
    rng = np.random.default_rng(seed)
    sums = x[rng.integers(0, len(x), size=(n, len(x)))].sum(axis=1)
    return round(float(np.percentile(sums, 2.5)), 1), round(float(np.percentile(sums, 97.5)), 1)


def table(t, title):
    if t.empty:
        print(f"\n{title}: no scored entries yet"); return
    base = t["trail (now)"].to_numpy(float)
    print(f"\n{title}  (entries {len(t)}: live {int((t.src == 'live').sum())}, shadow {int((t.src == 'shadow').sum())})")
    print(f"  {'exit':18s} {'n':>4s} {'total R':>8s} {'vs now':>8s} {'95% range':>16s} {'wins':>5s} {'best':>6s}")
    for arm in ARMS:
        if arm not in t or t[arm].isna().all():
            continue
        ok = t[arm].notna().to_numpy()
        x, b = t[arm].to_numpy(float)[ok], base[ok]
        ci = boot_ci(x - b) if arm != "trail (now)" else None
        print(f"  {arm:18s} {len(x):4d} {x.sum():+8.1f} {(x - b).sum():+8.1f} {str(ci) if ci else '':>16s} "
              f"{100 * (x > 0).mean():4.0f}% {x.max():+6.1f}")


def main():
    e, ndup = entries()
    frames, rows, skipped, still_open = {}, [], 0, 0
    for r in e.itertuples():
        if r.epic not in frames:
            frames[r.epic] = load_archive(r.epic)
        if frames[r.epic] is None:
            skipped += 1; continue
        df, step = frames[r.epic]
        t0 = r.when.floor(step) if r.src == "live" else r.when
        if t0 > df["date"].iloc[-1] or t0 < df["date"].iloc[0]:
            skipped += 1; continue
        res = {}
        for arm, (kind, val) in ARMS.items():
            if kind == "tp_pts" and r.epic != GOLD:
                res[arm] = np.nan; continue
            out = replay(df, t0, r.d, r.entry, r.risk, kind, val)
            if out is None:
                res = None; break
            res[arm] = out[0]
        if res is None:                      # open in at least one arm: scored once it has closed in all
            still_open += 1; continue
        rows.append(dict(src=r.src, id=r.id, epic=r.epic, market=r.market, d=r.d, when=r.when, actual=r.actual, **res))
    t = pd.DataFrame(rows)
    print(f"Breakout exit scoring — data to {max(f[0]['date'].iloc[-1] for f in frames.values() if f)}")
    print(f"entries {len(e)} (dropped {ndup} shadow rows that duplicate a live trade); scored {len(t)}, "
          f"still open {still_open}, no candles {skipped}")
    chk = t.dropna(subset=["actual"])
    for src, g in chk.groupby("src"):
        d = g["trail (now)"] - g["actual"]
        print(f"fidelity {src}: replayed trail vs recorded result, n={len(g)}, total {g['trail (now)'].sum():+.1f}R vs "
              f"{g['actual'].sum():+.1f}R, median |diff| {d.abs().median():.2f}R, corr {g['trail (now)'].corr(g['actual']):.2f}")
    for label, sub in ((f"FORWARD SAMPLE (entries from {since})", t[t.when >= since]),
                       ("BACKGROUND (all entries since June; already seen, not for decisions)", t)):
        print(f"\n################ {label}")
        table(sub, "All breakout markets, both directions")
        table(sub[sub.src == "live"], "Live trades only")
        for d in ("BUY", "SELL"):
            table(sub[(sub.epic == GOLD) & (sub.d == d)], f"Gold {d}")
            table(sub[(sub.epic != GOLD) & (sub.d == d)], f"Other markets {d}")


main()
