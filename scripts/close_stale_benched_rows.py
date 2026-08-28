"""One-off: resolve the six benched_outcomes rows left OPEN at the archive edge.

None of these six failed. Each was left OPEN because its resolver ran out of
FUTURE DATA, not because it timed out — backfill_breakout_shadow.replay() is
explicit about the difference, returning EXPIRED only at MAX_BARS and OPEN
otherwise ("still running at the archive edge"). That data now exists, so the
true per-model outcome is computable:

  #94 #95  breakout-shadow-bf  benched 2026-08-07, 366+ 1h bars now available
  #96 #97 #98  breakout-shadow-bf  benched 2026-08-10, 337+ 1h bars available
  #2   shadow (DXY momentum)  benched 2026-07-27, 2075 15m bars available

The five -bf rows are additionally outside the LIVE resolver by design:
get_open_breakout_shadow filters bench_type='breakout-shadow' exactly, so
nothing in the running bot will ever pick them up. #2 is a momentum row for a
market that has since moved to breakout mode, so its resolver no longer runs
for it either. Both need a deliberate pass; this is it.

Instrumentation only — benched_outcomes never feeds an order. It DOES feed the
pooled breakout-shadow readout the v3 go-live decision leans on, which is why
this replays the real models rather than stamping a terminal status.

Dry run (prints proposed outcomes, writes nothing):
    docker exec -i ig-trading-bot python3 - < scripts/close_stale_benched_rows.py
Apply:
    docker exec -i ig-trading-bot python3 - --apply < scripts/close_stale_benched_rows.py
"""
import json
import sqlite3
import sys
from datetime import datetime

import pandas as pd

from src import breakout
from src.indicators import add_all_indicators
from config import MARKETS, get_strategy_for_market

APPLY = "--apply" in sys.argv
ARCHIVE = "/app/data/candle_archive"
MAX_BARS = 240   # breakout model, matches main._resolve_breakout_shadow
MAX_FWD = 96     # momentum model, matches main._resolve_benched

# Addressed by id so this can touch nothing else, even if another row goes OPEN
# between the dry run and the apply.
TARGETS = {94: "breakout", 95: "breakout", 96: "breakout", 97: "breakout",
           98: "breakout", 2: "momentum"}

db = sqlite3.connect("/app/data/trade_journal.db", timeout=30)
db.execute("PRAGMA busy_timeout = 30000")  # the live bot holds this file open
db.row_factory = sqlite3.Row


# --- Frame + breakout replay: copied VERBATIM from scripts/backfill_breakout_shadow.py
# (lines 51 and 75). Copied rather than imported because that script executes at
# module scope — it opens the DB, counts existing rows and can SystemExit — so
# importing it would run the backfill. Any edit there must be mirrored here; these
# six rows must be resolved by the same arithmetic as the other 85 or they are not
# comparable with them.

def hourly(epic: str):
    """1h frame from the durable archive (identical to _breakout_frame_1h's output:
    verified 0.0 max OHLC difference against archive+stream on 2026-08-10)."""
    rows = []
    try:
        with open(f"{ARCHIVE}/{epic}.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return None
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["timestamp"])
    d = (d[["date", "open", "high", "low", "close"]]
         .drop_duplicates(subset="date").sort_values("date").set_index("date"))
    h = d.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return h.iloc[:-1].reset_index()


def native(epic: str):
    """Archive at its stored interval, for the momentum model (DXY is 15m)."""
    rows = []
    try:
        with open(f"{ARCHIVE}/{epic}.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return None
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["timestamp"])
    keep = [c for c in ("date", "open", "high", "low", "close", "volume") if c in d.columns]
    return (d[keep].drop_duplicates(subset="date").sort_values("date")
            .reset_index(drop=True).iloc[:-1])


def replay(frame, epic, direction, entry, risk, start):
    """Byte-for-byte main._resolve_breakout_shadow: k×ATR stop ratcheted to the
    Donchian-M trail, checked against each subsequent CLOSED 1h bar."""
    is_buy = direction == "BUY"
    fwd = frame[frame["date"] > pd.to_datetime(start)].reset_index(drop=True)
    if fwd.empty:
        return None
    base = len(frame) - len(fwd)
    stop = entry - risk if is_buy else entry + risk
    initial = stop
    ncdl = 0
    for i in range(len(fwd)):
        ncdl = i + 1
        bar = fwd.iloc[i]
        if (is_buy and float(bar["low"]) <= stop) or \
           ((not is_buy) and float(bar["high"]) >= stop):
            r = ((stop - entry) if is_buy else (entry - stop)) / risk
            return ("WIN" if r > 0 else "LOSS",
                    "stop" if stop == initial else "trail", stop, r, ncdl)
        lvl = breakout.exit_channel(frame.iloc[:base + i + 1], epic, direction)
        if lvl is not None:
            stop = max(stop, lvl) if is_buy else min(stop, lvl)
        if ncdl >= MAX_BARS:
            break
    px = float(fwd.iloc[-1]["close"])
    r = ((px - entry) if is_buy else (entry - px)) / risk
    if ncdl >= MAX_BARS:
        return ("EXPIRED", "expired", px, r, ncdl)
    return ("OPEN", None, None, None, ncdl)  # still running at the archive edge


def replay_momentum(row, market):
    """Transcribed from main._resolve_benched (main.py:2818-2859), unchanged:
    broker barriers first (stop assumed to win a same-candle tie), then the
    RSI-extreme exit, then MACD-3 or ADX-ranging-3 depending on the profile."""
    strat = get_strategy_for_market(market)
    df = native(market.epic)
    if df is None or df.empty:
        return None, None, None, 0, None, None
    ip = {"ema_fast": strat.ema_fast, "ema_medium": strat.ema_medium,
          "ema_slow": strat.ema_slow, "rsi_period": strat.rsi_period}
    ind = add_all_indicators(df.copy(), ip)
    use_macd = getattr(strat, "use_macd_exit", True)
    ob = getattr(strat, "rsi_overbought", 70)
    os_ = getattr(strat, "rsi_oversold", 30)
    adx_exit = getattr(strat, "adx_threshold", 25) - 10

    b_at = pd.to_datetime(row["benched_at"])
    fwd = ind[ind["date"] > b_at]
    if fwd.empty:
        return None, None, None, 0, None, ind
    entry, stop, lim = row["entry_price"], row["stop_distance"], row["limit_distance"]
    is_buy = row["direction"] == "BUY"
    hh, ll, cc = fwd["high"].values, fwd["low"].values, fwd["close"].values
    mh = fwd["macd_hist"].values if "macd_hist" in fwd.columns else None
    rs = fwd["rsi"].values if "rsi" in fwd.columns else None
    ax = fwd["adx"].values if "adx" in fwd.columns else None

    status = outcome = rmult = None
    ncdl = 0
    for i in range(len(fwd)):
        ncdl = i + 1
        if is_buy:
            if ll[i] <= entry - stop:
                status, outcome, rmult = "LOSS", "stop", -1.0; break
            if hh[i] >= entry + lim:
                status, outcome, rmult = "WIN", "limit", lim / stop; break
        else:
            if hh[i] >= entry + stop:
                status, outcome, rmult = "LOSS", "stop", -1.0; break
            if ll[i] <= entry - lim:
                status, outcome, rmult = "WIN", "limit", lim / stop; break
        if rs is not None:
            if is_buy and rs[i] > ob:
                d = (cc[i] - entry) / stop
                status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "rsi", d; break
            if (not is_buy) and rs[i] < os_:
                d = (entry - cc[i]) / stop
                status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "rsi", d; break
        if use_macd and mh is not None and i >= 2:
            win3 = mh[i - 2:i + 1]
            if is_buy and all(h < 0 for h in win3):
                d = (cc[i] - entry) / stop
                status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "macd", d; break
            if (not is_buy) and all(h > 0 for h in win3):
                d = (entry - cc[i]) / stop
                status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "macd", d; break
        if (not use_macd) and ax is not None and i >= 2:
            if all(a < adx_exit for a in ax[i - 2:i + 1]):
                d = ((cc[i] - entry) if is_buy else (entry - cc[i])) / stop
                status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "ranging", d; break

    if status:
        return status, outcome, cc[ncdl - 1], ncdl, rmult, ind
    if len(fwd) >= MAX_FWD:
        d = ((cc[-1] - entry) if is_buy else (entry - cc[-1])) / stop
        return "EXPIRED", "expired", cc[-1], len(fwd), d, ind
    return "OPEN", None, None, ncdl, None, ind


def owns_frame(frame, entry):
    """Frame-ownership guard — the #134 lesson. The row and the frame arrive from
    two independent sources (a DB read and an archive read) and nothing else checks
    they describe the same instrument. An entry price must lie inside the range its
    own instrument actually traded; a Gold frame against a NASDAQ entry does not."""
    lo, hi = float(frame["low"].min()), float(frame["high"].max())
    return lo * 0.9 <= float(entry) <= hi * 1.1, lo, hi


rows = db.execute(
    "SELECT * FROM benched_outcomes WHERE id IN (%s) ORDER BY id"
    % ",".join(str(i) for i in TARGETS)
).fetchall()

print(f"{len(rows)} target rows ({'APPLY' if APPLY else 'DRY RUN'})\n")
print(f"{'id':>4} {'market':<16}{'type':<20}{'dir':<5}{'entry':>10}{'stop':>8}"
      f"  -> {'status':<9}{'outcome':<9}{'exit':>10}{'R':>8}{'bars':>6}")
print("-" * 108)

updates, skipped = [], []
for row in rows:
    if row["status"] != "OPEN":
        skipped.append((row["id"], f"already {row['status']} — not touching it"))
        continue
    market = next((m for m in MARKETS if m.epic == row["epic"]), None)
    if market is None:
        skipped.append((row["id"], f"epic {row['epic']} not in MARKETS"))
        continue

    kind = TARGETS[row["id"]]
    if kind == "breakout":
        frame = hourly(row["epic"])
        if frame is None or frame.empty:
            skipped.append((row["id"], "no archive frame"))
            continue
        ok, lo, hi = owns_frame(frame, row["entry_price"])
        if not ok:
            skipped.append((row["id"], f"FRAME MISMATCH entry={row['entry_price']} "
                                       f"outside {lo:.1f}..{hi:.1f} — refusing"))
            continue
        out = replay(frame, row["epic"], row["direction"], row["entry_price"],
                     row["stop_distance"], pd.to_datetime(row["benched_at"]))
        if out is None:
            skipped.append((row["id"], "no forward bars"))
            continue
        status, outcome, exit_px, r, ncdl = out
    else:
        status, outcome, exit_px, ncdl, r, ind = replay_momentum(row, market)
        if status is None:
            skipped.append((row["id"], "no archive frame"))
            continue
        ok, lo, hi = owns_frame(ind, row["entry_price"])
        if not ok:
            skipped.append((row["id"], f"FRAME MISMATCH entry={row['entry_price']} "
                                       f"outside {lo:.1f}..{hi:.1f} — refusing"))
            continue

    if status == "OPEN":
        skipped.append((row["id"], f"STILL open after {ncdl} bars — archive edge again"))
        continue

    print(f"{row['id']:>4} {row['market_name']:<16}{row['bench_type']:<20}"
          f"{row['direction']:<5}{row['entry_price']:>10.2f}{row['stop_distance']:>8.1f}"
          f"  -> {status:<9}{str(outcome):<9}{exit_px:>10.2f}{r:>+8.2f}{ncdl:>6}")
    updates.append((status, outcome, round(float(r), 3), round(float(exit_px), 2),
                    int(ncdl), row["id"]))

if skipped:
    print("\nskipped:")
    for rid, why in skipped:
        print(f"  #{rid}: {why}")

pooled = db.execute(
    """SELECT bench_type, COUNT(*) n, ROUND(SUM(r_multiple), 2) r
       FROM benched_outcomes WHERE r_multiple IS NOT NULL GROUP BY bench_type"""
).fetchall()
print("\npooled R before:")
for p in pooled:
    print(f"  {p['bench_type']:<20} n={p['n']:<5} R={p['r']}")
delta = sum(u[2] for u in updates)
print(f"\nthese {len(updates)} rows contribute {delta:+.2f}R")

if APPLY and updates:
    with db:  # one transaction; nothing half-written on SQLITE_BUSY
        db.executemany(
            """UPDATE benched_outcomes
               SET status=?, outcome=?, r_multiple=?, exit_price=?,
                   candles_to_resolve=?, resolved_at=?
               WHERE id=? AND status='OPEN'""",
            [(s, o, r, x, n, datetime.now().isoformat(), i) for s, o, r, x, n, i in updates],
        )
    print(f"\n{len(updates)} rows resolved.")
    for p in db.execute(
        """SELECT bench_type, COUNT(*) n, ROUND(SUM(r_multiple), 2) r
           FROM benched_outcomes WHERE r_multiple IS NOT NULL GROUP BY bench_type"""):
        print(f"  {p['bench_type']:<20} n={p['n']:<5} R={p['r']}")
elif not APPLY:
    print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
