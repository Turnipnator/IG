#!/usr/bin/env python3
"""Resolve `rejected_signals` rows blocked by MarketConfig.allowed_direction into
counterfactual outcomes against the IG candle archive.

Answers v3 agenda item 18: "is the direction restriction still earning its place?"
Those rows record a signal that passed EVERY strategy gate and was then discarded
solely because of its side. They carry epic/direction/timestamp/confidence/adx/rsi
but NO entry, stop or limit -- so the trade has to be rebuilt from the archive bar
at each timestamp using the market's live profile, then walked forward.

FAITHFULNESS. The direction gate (main.py:1582) fires EARLY -- after analyze() and
the HOLD return, but BEFORE the pullback arm, the screener gate, the per-market
regime check, the balance check and validate_trade. So a rejected row is NOT a
trade the bot would have taken; it is a signal that had not yet met those gates.
Replaying the rows naively therefore OVERSTATES participation. This script models
the three gates that materially change the answer and are reconstructable offline:

  * trading hours   -- live refuses entries outside trading_start..trading_end (UTC)
  * pullback entry  -- indices_tight/indices_wide/gold ARM and wait <=N candles for
                       a frac*ATR retrace, entering at the retraced price, and DROP
                       the signal outright if it never comes (runaway avoided).
                       This is the mechanism that protects against shorting into an
                       uptrend, so omitting it flatters the counterfactual badly.
  * one at a time   -- no new counterfactual entry while one is still open

NOT modelled (all shared with the live benched resolver, so the shadow record is
comparable): break-even stop, ATR trail, the price-relative stop ceiling, the
correlation-cluster filter, re-entry cooldown, screener activity, and the risk
manager's position cap. BE/trail in particular would convert some losers toward 0R
and cap some winners, so the spread of outcomes here is WIDER than live would be.

⚠️ BAR ALIGNMENT. A signal stamped 00:25:01 was computed on the bar that CLOSED at
00:25 -- i.e. the one timestamped 00:20 -- NOT the bar timestamped 00:25, which had
only just opened. Selecting `date <= timestamp` therefore reads a bar that did not
exist yet: a look-ahead. This is not cosmetic; it moved the FTSE verdict materially.
The alignment is VERIFIED, not assumed: at this offset the archive reproduces live's
logged RSI to 0.000 (median and p90) and ADX to 0.093, versus 5.388/11.542 RSI at
the naive offset. The join-validation block below re-proves it on every run -- if
those residuals are ever not ~0, STOP, because the join is wrong again.

The exit ladder mirrors main._resolve_benched exactly (barriers stop-first on a
same-candle tie, then RSI extreme, then MACD-N, then ADX-ranging for non-MACD
profiles) so numbers are directly comparable to benched_outcomes.

Costs are SWEPT rather than assumed -- a single spread guess is the fragile part of
any such verdict (see scripts/backtest_crude_exit.py, which set this precedent).

Usage:
  python scripts/resolve_direction_restricted.py --epic IX.D.FTSE.DAILY.IP
  python scripts/resolve_direction_restricted.py --market "FTSE 100" --macd-bars 3
  python scripts/resolve_direction_restricted.py --market "FTSE 100" --naive
"""
import argparse
import os
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.archive_loader import load_archive          # noqa: E402
from src.indicators import add_all_indicators            # noqa: E402
from config import MARKETS, STRATEGY_PROFILES            # noqa: E402

DB = Path("/app/data/trade_journal.db") if os.path.exists("/app") \
    else Path("data/trade_journal.db")

# The archive and rejected_signals.timestamp are BOTH written by the container,
# whose TZ is Europe/London -- so the JOIN needs no conversion. The trading-hours
# gate however is defined in UTC (utc_hour(), c1e244f), so the bar's local stamp
# must be converted before it is compared. Getting this backwards silently shifts
# the window by an hour through the whole BST half of the year.
LONDON = ZoneInfo("Europe/London")


def utc_hour_of(ts: pd.Timestamp) -> int:
    return ts.tz_localize(LONDON, nonexistent="shift_forward",
                          ambiguous=True).astimezone(ZoneInfo("UTC")).hour


def resolve(mc, strat, rows, ind, macd_bars, cost_r, use_pullback, use_hours,
            dedupe, be_trail=False):
    """Walk each rejected signal forward. Returns (trades, drops).

    be_trail=True additionally models the live break-even stop and ATR trail
    (main.py:778-890), which the benched resolver omits. Live runs both PER TICK;
    this replay is per BAR, so it is a conservative approximation:

      * stop is tested against the level as it stood at the START of the bar,
        BEFORE this bar's BE/trail update. Within one bar OHLC cannot say whether
        the 0.7R profit spike or the adverse reversal came first, so we assume the
        adverse one did -- the same stop-first tie rule _resolve_benched uses.
        This UNDERSTATES the benefit of BE in exactly the spike-then-reverse case
        where BE helps most, so a BE gain measured here is a floor, not a ceiling.
      * live trails on every tick and can ratchet several times inside a 5m bar;
        here it ratchets at most once per bar, off that bar's extreme. Understates
        the trail slightly in fast moves.
    """
    ob, os_ = strat.rsi_overbought, strat.rsi_oversold
    adx_exit = strat.adx_threshold - 10
    use_macd = getattr(strat, "use_macd_exit", True)
    pb_frac = getattr(strat, "pullback_entry_atr_frac", 0.0) or 0.0
    pb_win = getattr(strat, "pullback_entry_window", 0) or 0
    pullback_on = use_pullback and pb_frac > 0 and pb_win > 0
    be_pct = getattr(strat, "breakeven_trigger_pct", 0.0) or 0.0
    lock_pct = getattr(strat, "breakeven_lock_pct", 0.0) or 0.0
    trail_mult = getattr(strat, "atr_trail_mult", 0.0) or 0.0
    MAX_FWD = 96  # ~8h at 5m -- matches _resolve_benched's EXPIRED horizon

    dates = ind["date"].values
    trades, drops = [], []
    busy_until = None

    for r in rows:
        ts = pd.to_datetime(r["timestamp"]).floor("min")
        # The signal fires ON a completed bar; find that bar.
        idx = ind.index[ind["date"] <= ts]
        if len(idx) == 0:
            drops.append((r, "before-archive")); continue
        # -1: the signal fired on the bar that CLOSED at `ts`, not the one that
        # opened at it. See the BAR ALIGNMENT note in the module docstring.
        i0 = idx[-1] - 1
        if i0 < 0:
            drops.append((r, "before-archive")); continue
        bar = ind.loc[i0]
        if pd.isna(bar.get("atr")) or not bar["atr"] > 0:
            drops.append((r, "no-atr")); continue

        if dedupe and busy_until is not None and bar["date"] <= busy_until:
            drops.append((r, "position-open")); continue

        if use_hours and mc.trading_start is not None:
            h = utc_hour_of(bar["date"])
            if not (mc.trading_start <= h < mc.trading_end):
                drops.append((r, "out-of-hours")); continue

        is_buy = r["direction"] == "BUY"
        atr = float(bar["atr"])
        stop = max(strat.stop_atr_mult * atr, mc.min_stop_distance or 0.0)
        lim = stop * strat.reward_risk
        entry_i = i0
        entry = float(bar["close"])

        # Pullback arm: wait <=pb_win candles for a frac*ATR retrace AWAY from the
        # trade (up for a SELL, down for a BUY); enter at that price, else DROP.
        if pullback_on:
            offset = pb_frac * atr
            target = entry + offset if not is_buy else entry - offset
            hit = None
            for j in range(i0 + 1, min(i0 + 1 + pb_win, len(ind))):
                b = ind.loc[j]
                if (not is_buy and b["high"] >= target) or (is_buy and b["low"] <= target):
                    hit = (j, target); break
            if hit is None:
                drops.append((r, "pullback-expired")); continue
            entry_i, entry = hit[0], float(hit[1])

        fwd = ind.loc[entry_i + 1:]
        if fwd.empty:
            drops.append((r, "no-forward-bars")); continue

        hh, ll, cc = fwd["high"].values, fwd["low"].values, fwd["close"].values
        mh = fwd["macd_hist"].values if "macd_hist" in fwd else None
        rs = fwd["rsi"].values if "rsi" in fwd else None
        ax = fwd["adx"].values if "adx" in fwd else None
        aa = fwd["atr"].values if "atr" in fwd else None
        # Absolute stop level, which BE/trail ratchet. Starts at the broker stop.
        stop_lvl = (entry - stop) if is_buy else (entry + stop)
        be_done = False

        status = outcome = None
        rmult = None
        n = 0
        for i in range(len(fwd)):
            n = i + 1
            if is_buy:
                if ll[i] <= stop_lvl:
                    d = (stop_lvl - entry) / stop
                    status, outcome = ("WIN" if d > 0 else "LOSS"), ("stop" if not be_done else "be-trail")
                    rmult = d; break
                if hh[i] >= entry + lim:
                    status, outcome, rmult = "WIN", "limit", lim / stop; break
            else:
                if hh[i] >= stop_lvl:
                    d = (entry - stop_lvl) / stop
                    status, outcome = ("WIN" if d > 0 else "LOSS"), ("stop" if not be_done else "be-trail")
                    rmult = d; break
                if ll[i] <= entry - lim:
                    status, outcome, rmult = "WIN", "limit", lim / stop; break

            # BE / ATR trail, applied AFTER this bar's stop test (see docstring).
            if be_trail:
                if not be_done and be_pct > 0:
                    prof = (hh[i] - entry) if is_buy else (entry - ll[i])
                    if prof >= stop * be_pct:
                        off = stop * lock_pct
                        stop_lvl = (entry + off) if is_buy else (entry - off)
                        be_done = True
                        # live does BE then `continue` -- no trail on the same pass
                elif be_done and trail_mult > 0 and aa is not None and aa[i] > 0:
                    td = aa[i] * trail_mult
                    nt = (hh[i] - td) if is_buy else (ll[i] + td)
                    better = (nt > stop_lvl) if is_buy else (nt < stop_lvl)
                    # 20% minimum-move throttle, as live (main.py:869)
                    if better and abs(nt - stop_lvl) >= td * 0.2:
                        stop_lvl = nt
            if rs is not None:
                if is_buy and rs[i] > ob:
                    d = (cc[i] - entry) / stop
                    status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "rsi", d; break
                if (not is_buy) and rs[i] < os_:
                    d = (entry - cc[i]) / stop
                    status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "rsi", d; break
            if use_macd and mh is not None and i >= macd_bars - 1:
                w = mh[i - macd_bars + 1:i + 1]
                if is_buy and all(h < 0 for h in w):
                    d = (cc[i] - entry) / stop
                    status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "macd", d; break
                if (not is_buy) and all(h > 0 for h in w):
                    d = (entry - cc[i]) / stop
                    status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "macd", d; break
            if (not use_macd) and ax is not None and i >= 2:
                if all(a < adx_exit for a in ax[i - 2:i + 1]):
                    d = ((cc[i] - entry) if is_buy else (entry - cc[i])) / stop
                    status, outcome, rmult = ("WIN" if d >= 0 else "LOSS"), "ranging", d; break
            if n >= MAX_FWD:
                d = ((cc[i] - entry) if is_buy else (entry - cc[i])) / stop
                status, outcome, rmult = "EXPIRED", "expired", d; break

        if status is None:
            drops.append((r, "unresolved-tail")); continue

        net = float(rmult) - cost_r
        trades.append({
            "ts": bar["date"], "dir": r["direction"], "conf": r["confidence"],
            "entry": entry, "stop": stop, "n": n,
            "status": status, "outcome": outcome,
            "r_gross": float(rmult), "r_net": net,
        })
        busy_until = fwd["date"].values[n - 1]

    return trades, drops


def report(trades, drops, label):
    if not trades:
        print(f"  {label}: no resolvable trades ({len(drops)} dropped)")
        return None
    w = [t for t in trades if t["r_net"] > 0]
    l = [t for t in trades if t["r_net"] <= 0]
    tot = sum(t["r_net"] for t in trades)
    gw = sum(t["r_net"] for t in w)
    gl = -sum(t["r_net"] for t in l)
    pf = (gw / gl) if gl > 0 else float("inf")
    print(f"  {label}: n={len(trades):3d}  {len(w)}W/{len(l)}L  "
          f"WR={100*len(w)/len(trades):4.1f}%  totR={tot:+7.2f}  PF={pf:5.2f}")
    return {"n": len(trades), "w": len(w), "l": len(l), "totR": tot, "pf": pf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epic")
    ap.add_argument("--market", default="FTSE 100")
    ap.add_argument("--macd-bars", type=int, default=3,
                    help="MACD exit window IN FORCE during the sample. All rows "
                         "predate 9e2dd0c (2026-09-01, 3->5), so 3 is correct for "
                         "a historical replay; pass 5 to ask the forward question.")
    ap.add_argument("--naive", action="store_true",
                    help="disable hours/pullback/dedupe -- the OVERSTATED read")
    ap.add_argument("--be-trail", action="store_true",
                    help="also model the live break-even stop + ATR trail "
                         "(main.py:778-890), which benched_outcomes omits")
    args = ap.parse_args()

    mc = next((m for m in MARKETS
               if m.epic == args.epic or m.name == args.market), None)
    if mc is None:
        sys.exit(f"no such market: {args.epic or args.market}")
    strat = STRATEGY_PROFILES[mc.strategy]

    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT * FROM rejected_signals WHERE epic=? "
        "AND reject_reason LIKE 'Direction-restricted%' ORDER BY timestamp",
        (mc.epic,)).fetchall()

    df = load_archive(mc.epic)
    print(f"=== {mc.name} ({mc.epic}) ===")
    print(f"strategy={mc.strategy} stop={strat.stop_atr_mult}xATR RR={strat.reward_risk} "
          f"allowed={mc.allowed_direction} hours={mc.trading_start}-{mc.trading_end}Z "
          f"macd_exit={getattr(strat,'use_macd_exit',True)}/{args.macd_bars} "
          f"pullback={getattr(strat,'pullback_entry_atr_frac',0)}/"
          f"{getattr(strat,'pullback_entry_window',0)}")
    print(f"blocked signals: {len(rows)}   archive: {len(df)} bars "
          f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    if not rows or df.empty:
        sys.exit("nothing to resolve")

    ind = add_all_indicators(df.copy(), {
        "ema_fast": strat.ema_fast, "ema_medium": strat.ema_medium,
        "ema_slow": strat.ema_slow, "rsi_period": strat.rsi_period})

    # --- JOIN VALIDATION -------------------------------------------------
    # The row logged the ADX/RSI live saw. Recomputing them from the archive at
    # the same timestamp must reproduce them. If it does not, the join or the
    # indicator params are wrong and every number below is meaningless.
    print("\n--- join validation (live-logged vs archive-recomputed) ---")
    da, dr, matched = [], [], 0
    for r in rows:
        ts = pd.to_datetime(r["timestamp"]).floor("min")
        idx = ind.index[ind["date"] <= ts]
        if len(idx) == 0 or idx[-1] - 1 < 0:
            continue
        b = ind.loc[idx[-1] - 1]
        if pd.notna(b.get("adx")) and r["adx"] is not None:
            da.append(abs(b["adx"] - r["adx"])); matched += 1
        if pd.notna(b.get("rsi")) and r["rsi"] is not None:
            dr.append(abs(b["rsi"] - r["rsi"]))
    if matched:
        print(f"  matched {matched}/{len(rows)} rows to an archive bar")
        print(f"  |ADX diff|  median={pd.Series(da).median():.3f}  "
              f"p90={pd.Series(da).quantile(.9):.3f}  max={max(da):.3f}")
        print(f"  |RSI diff|  median={pd.Series(dr).median():.3f}  "
              f"p90={pd.Series(dr).quantile(.9):.3f}  max={max(dr):.3f}")
        print("  (near-zero => the archive reproduces what live saw; large => STOP)")
    else:
        print("  NO rows matched an archive bar -- cannot proceed")
        sys.exit(1)

    faithful = not args.naive
    print(f"\n--- outcomes ({'FAITHFUL: hours+pullback+one-at-a-time' if faithful else 'NAIVE: no gates'}{' +BE/TRAIL' if args.be_trail else ''}) ---")
    for cost in (0.0, 0.02, 0.05, 0.10):
        tr, dp = resolve(mc, strat, rows, ind, args.macd_bars, cost,
                         faithful, faithful, faithful, args.be_trail)
        report(tr, dp, f"cost={cost:.2f}R")

    tr, dp = resolve(mc, strat, rows, ind, args.macd_bars, 0.0,
                     faithful, faithful, faithful, args.be_trail)
    from collections import Counter
    print(f"\n  dropped {len(dp)}: {dict(Counter(d[1] for d in dp))}")
    if tr:
        print(f"  exit mix: {dict(Counter(t['outcome'] for t in tr))}")
        print(f"  median hold: {pd.Series([t['n'] for t in tr]).median():.0f} bars")
        print(f"  by side: {dict(Counter(t['dir'] for t in tr))}")


if __name__ == "__main__":
    main()
