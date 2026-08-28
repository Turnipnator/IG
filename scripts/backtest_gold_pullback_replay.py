"""Gold pullback-entry A/B replay on the IG candle archive (2026-07-30).

Question: does Gold's pullback-entry (wait <=6 candles for a 0.5xATR retrace,
else drop the signal) ADD or DESTROY edge vs entering immediately on signal?
Trigger: 36h in which the filter dropped 6 would-be winners and filled the two
losers (07-29/07-30). Prior evidence FOR the filter: 2026-06-11 Yahoo GC=F
backtest (PF 1.91->3.28) and the 2026-06-30 archive check ("PB+55 beats
NOPB+55", 18d). This is the re-test on the full ~7wk IG-native archive with the
CURRENT exit params (be 0.7 / lock 0.25 / trail 2.0 — deployed 07-24).

Design: collect every Gold BUY/SELL signal the live analyze() raises (gold
profile, HTF + regime context from trailing hourly candles, hours gate,
effective confidence gate = max(config, regime floor)). Arm cadence mimics
live (one pending arm at a time; a fill blocks new arms until the trade
exits). For each ARM, two paired paths:
  A (live):     wait for the 0.5xATR retrace; fill at target or drop at 6.
  B (no-filter): enter at the signal candle close.
Identical exits both paths: stop 1.5xATR (from the armed signal), limit 3R,
BE at 0.7 (stop -> entry +/- 0.25xstop), ATR-trail 2.0x after BE (per-candle
ratchet; live is per-tick), RSI 85/15 momentum exit on candle close. Spread
charged flat per round trip, swept 0 / 0.4 / 0.8 pts.

Known gaps (bias both paths equally): no "market turned ranging" exit (2/100
journal trades), no intrabar stop-vs-limit ordering (stop checked first =
conservative), archive hole 07-17..07-20 (polling incident killed harvesting).

NOTE ON THE CLOCK: archive timestamps are BST (container TZ); converted to UTC
before the 23-21 UTC gold hours gate.
"""
import json
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, ".")

from config import MARKETS, get_strategy_for_market
from src.indicators import add_all_indicators
from src.regime import classify_regime, get_regime_params
from src.strategy import Signal, TradingStrategy

ARCHIVE = ("/private/tmp/claude-501/-Users-paulturner-IG/"
           "6f107130-5cce-4510-9c2a-c5bc07b3ecf6/scratchpad/gold_archive.jsonl")
SPREAD = 0.4
BST_OFFSET = 1
WARMUP = 200

gold = next(m for m in MARKETS if m.epic == "CS.D.USCGC.TODAY.IP")
prof = get_strategy_for_market(gold)
strategy = TradingStrategy()


def load() -> pd.DataFrame:
    rows = []
    with open(ARCHIVE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"]) - timedelta(hours=BST_OFFSET)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["volume"] = 0
    return df[["date", "open", "high", "low", "close", "volume"]]


def hourly_context(df: pd.DataFrame) -> dict:
    """HTF trend + regime per hour from trailing 30 hourly candles (mimics live)."""
    h = (
        df.set_index("date")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    params = {"ema_fast": prof.ema_fast, "ema_medium": prof.ema_medium,
              "ema_slow": prof.ema_slow, "rsi_period": prof.rsi_period,
              "adx_threshold": prof.adx_threshold}
    ctx = {}
    for i in range(21, len(h)):
        window = h.iloc[max(0, i - 29): i + 1].copy()
        ema9 = window["close"].ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = window["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        close = window["close"].iloc[-1]
        if ema9 > ema21 and close > ema21:
            trend = "BULLISH"
        elif ema9 < ema21 and close < ema21:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        try:
            reg = classify_regime(add_all_indicators(window, params))
            rp = get_regime_params(reg)
            ctx[h["date"].iloc[i]] = (trend, reg.code, reg.is_tradeable, rp.min_confidence)
        except Exception:
            ctx[h["date"].iloc[i]] = (trend, "UNKNOWN", True, 0.5)
    return ctx


def ctx_at(ctx: dict, keys, ts):
    import bisect
    i = bisect.bisect_right(keys, ts) - 1
    return ctx[keys[i]] if i >= 0 else None


def in_gold_hours(ts) -> bool:
    # trading_start=23, trading_end=21 (UTC, wrap-around): closed only 21-23.
    return not (gold.trading_start > gold.trading_end
                and gold.trading_end <= ts.hour < gold.trading_start) \
        if gold.trading_start > gold.trading_end else \
        gold.trading_start <= ts.hour < gold.trading_end


def collect_signals(df: pd.DataFrame, ctx: dict) -> list:
    keys = sorted(ctx)
    out = []
    for i in range(WARMUP, len(df)):
        ts = df["date"].iloc[i]
        if not in_gold_hours(ts):
            continue
        c = ctx_at(ctx, keys, ts)
        if c is None:
            continue
        htf, code, tradeable, regime_min = c
        if not tradeable:
            continue
        window = df.iloc[max(0, i - 250): i + 1].copy()
        sig = strategy.analyze(window, gold, float(df["close"].iloc[i]), htf)
        if sig.signal not in (Signal.BUY, Signal.SELL):
            continue
        gate = max(gold.min_confidence, regime_min)
        out.append({
            "i": i, "date": ts,
            "dir": 1 if sig.signal == Signal.BUY else -1,
            "conf": sig.confidence, "gate": gate,
            "stop": sig.stop_distance,
            "atr": float(window["atr"].iloc[-1]),
            "adx": float(window["adx"].iloc[-1]),
            "regime": code,
        })
    return out


def sim_exit(df, entry_i: int, entry: float, d: int, stop_dist: float,
             spread: float):
    """Walk forward from entry: stop / limit / BE(0.7, lock 0.25) /
    ATR-trail 2.0 after BE / RSI 85-15 close exit. d=+1 BUY, -1 SELL."""
    stop_px = entry - d * stop_dist
    limit_px = entry + d * prof.reward_risk * stop_dist
    be_at = entry + d * prof.breakeven_trigger_pct * stop_dist
    be_done = False
    for j in range(entry_i + 1, len(df)):
        hi, lo = float(df["high"].iloc[j]), float(df["low"].iloc[j])
        cl = float(df["close"].iloc[j])
        rsi = float(df["rsi"].iloc[j])
        atr_j = float(df["atr"].iloc[j])
        # stop first (conservative), then limit
        if (d > 0 and lo <= stop_px) or (d < 0 and hi >= stop_px):
            reason = "trail/be" if be_done else "stop"
            return _res(df, entry_i, j, entry, stop_px, d, stop_dist, spread, reason)
        if (d > 0 and hi >= limit_px) or (d < 0 and lo <= limit_px):
            return _res(df, entry_i, j, entry, limit_px, d, stop_dist, spread, "limit")
        if (d > 0 and rsi >= prof.rsi_overbought) or (d < 0 and rsi <= prof.rsi_oversold):
            return _res(df, entry_i, j, entry, cl, d, stop_dist, spread, "rsi")
        if not be_done and ((d > 0 and hi >= be_at) or (d < 0 and lo <= be_at)):
            stop_px = entry + d * prof.breakeven_lock_pct * stop_dist
            be_done = True
        if be_done and atr_j == atr_j and atr_j > 0:
            cand = cl - d * prof.atr_trail_mult * atr_j
            if (d > 0 and cand > stop_px) or (d < 0 and cand < stop_px):
                stop_px = cand
    return None  # still open at archive end


def _res(df, entry_i, exit_i, entry, exit_px, d, stop_dist, spread, reason):
    pts = d * (exit_px - entry) - spread
    return {"pts": pts, "R": pts / stop_dist, "reason": reason,
            "bars": exit_i - entry_i, "exit_date": df["date"].iloc[exit_i]}


def build_arms(df, signals: list) -> list:
    """Live arm cadence: one pending arm at a time; a fill blocks new arms
    until path-A's trade exits; a drop frees the slot at window end."""
    arms = []
    blocked_until = -1
    for s in signals:
        i = s["i"]
        if i <= blocked_until:
            continue
        if s["conf"] < s["gate"]:
            continue
        target = float(df["close"].iloc[i]) - s["dir"] * prof.pullback_entry_atr_frac * s["atr"]
        fill_j = None
        for j in range(i + 1, min(i + 1 + prof.pullback_entry_window, len(df))):
            hi, lo = float(df["high"].iloc[j]), float(df["low"].iloc[j])
            if (s["dir"] > 0 and lo <= target) or (s["dir"] < 0 and hi >= target):
                fill_j = j
                break
        arm = dict(s)
        arm["target"] = target
        if fill_j is None:
            arm["A"] = None
            blocked_until = i + prof.pullback_entry_window
        else:
            outcome = sim_exit(df, fill_j, target, s["dir"], s["stop"], SPREAD)
            arm["A"] = outcome
            arm["fill_j"] = fill_j
            if outcome is None:
                blocked_until = len(df)
            else:
                blocked_until = fill_j + int((outcome["exit_date"] - df["date"].iloc[fill_j])
                                             .total_seconds() // 300)
        arm["B"] = sim_exit(df, i, float(df["close"].iloc[i]), s["dir"], s["stop"], SPREAD)
        arms.append(arm)
    return arms


def agg(outs: list) -> str:
    outs = [o for o in outs if o]
    if not outs:
        return "n=0"
    n = len(outs)
    wins = [o for o in outs if o["pts"] > 0]
    gl = abs(sum(o["pts"] for o in outs if o["pts"] <= 0))
    gp = sum(o["pts"] for o in wins)
    pf = gp / gl if gl else float("inf")
    return (f"n={n:3d} WR={len(wins) / n * 100:3.0f}% pts={sum(o['pts'] for o in outs):+8.1f} "
            f"sumR={sum(o['R'] for o in outs):+6.2f} PF={pf:5.2f}")


def main() -> None:
    df = load()
    print(f"{len(df)} 5m candles  {df['date'].iloc[0]} -> {df['date'].iloc[-1]} (UTC)")
    days = df["date"].dt.date.nunique()
    print(f"{days} distinct days (archive hole 07-17..07-20 expected)")

    params = {"ema_fast": prof.ema_fast, "ema_medium": prof.ema_medium,
              "ema_slow": prof.ema_slow, "rsi_period": prof.rsi_period,
              "rsi_overbought": prof.rsi_overbought, "rsi_oversold": prof.rsi_oversold,
              "rsi_buy_max": prof.rsi_buy_max, "rsi_sell_min": prof.rsi_sell_min,
              "adx_threshold": prof.adx_threshold}
    df = add_all_indicators(df, params)
    ctx = hourly_context(df)
    signals = collect_signals(df, ctx)
    print(f"raw signals: {len(signals)} "
          f"(BUY {sum(1 for s in signals if s['dir'] > 0)} / "
          f"SELL {sum(1 for s in signals if s['dir'] < 0)})")

    arms = build_arms(df, signals)
    fills = [a for a in arms if a["A"] is not None]
    drops = [a for a in arms if a["A"] is None and a.get("fill_j") is None]
    print(f"\narms (gate-passed, live cadence): {len(arms)}  "
          f"filled {len(fills)}  dropped {len(drops)}  "
          f"fill-rate {len(fills) / len(arms) * 100:.0f}%")

    print("\n=== HEADLINE (spread 0.4/rt) ===")
    print(f"  A live pullback (fills only):   {agg([a['A'] for a in fills])}")
    print(f"  B immediate-entry (ALL arms):   {agg([a['B'] for a in arms])}")
    print(f"  B on the FILLED subset:         {agg([a['B'] for a in fills])}")
    print(f"  B on the DROPPED subset:        {agg([a['B'] for a in drops])}")

    mid = df["date"].iloc[len(df) // 2]
    for label, sel in [("half1", lambda a: a["date"] < mid),
                       ("half2", lambda a: a["date"] >= mid)]:
        sub = [a for a in arms if sel(a)]
        f = [a["A"] for a in sub if a["A"]]
        print(f"  {label}: A {agg(f)}   |   B {agg([a['B'] for a in sub])}")

    print("\n=== by ADX at signal ===")
    for lo, hi in [(0, 40), (40, 45), (45, 99)]:
        sub = [a for a in arms if lo <= a["adx"] < hi]
        if sub:
            print(f"  ADX {lo}-{hi}: A {agg([a['A'] for a in sub if a['A']])}  "
                  f"|  B {agg([a['B'] for a in sub])}")

    print("\n=== by direction ===")
    for d, name in [(1, "BUY "), (-1, "SELL")]:
        sub = [a for a in arms if a["dir"] == d]
        print(f"  {name}: A {agg([a['A'] for a in sub if a['A']])}  "
              f"|  B {agg([a['B'] for a in sub])}")

    print("\n=== direction x fate (the asymmetry question) ===")
    for d, name in [(1, "BUY "), (-1, "SELL")]:
        dsub = [a for a in arms if a["dir"] == d]
        dfills = [a for a in dsub if a["A"]]
        ddrops = [a for a in dsub if a["A"] is None and a.get("fill_j") is None]
        print(f"  {name} drops:        {agg([a['B'] for a in ddrops])}   ({len(ddrops)}/{len(dsub)} arms dropped)")
        print(f"  {name} fills A vs B: A {agg([a['A'] for a in dfills])}  |  B {agg([a['B'] for a in dfills])}")

    print("\n=== halves x direction (robustness) ===")
    for d, name in [(1, "BUY "), (-1, "SELL")]:
        for label, sel in [("h1", lambda a: a["date"] < mid),
                           ("h2", lambda a: a["date"] >= mid)]:
            sub = [a for a in arms if a["dir"] == d and sel(a)]
            print(f"  {name} {label}: A {agg([a['A'] for a in sub if a['A']])}  |  B {agg([a['B'] for a in sub])}")

    print("\n=== exit-reason mix (A fills / B all) ===")
    for tag, outs in [("A", [a["A"] for a in fills if a["A"]]),
                      ("B", [a["B"] for a in arms if a["B"]])]:
        mix = {}
        for o in outs:
            mix[o["reason"]] = mix.get(o["reason"], 0) + 1
        print(f"  {tag}: {mix}")

    print("\n=== spread robustness (sumR) ===")
    for sp in [0.0, 0.4, 0.8]:
        ra = rb = 0.0
        for a in arms:
            if a["A"]:
                ra += (a["A"]["pts"] + SPREAD - sp) / a["stop"]
            if a["B"]:
                rb += (a["B"]["pts"] + SPREAD - sp) / a["stop"]
        print(f"  spread {sp:.1f}:  A {ra:+6.2f}R   B {rb:+6.2f}R")

    print("\n=== validation: arms 07-28 .. 07-30 (compare vs live logs) ===")
    for a in arms:
        if a["date"] >= pd.Timestamp("2026-07-28"):
            oa = a["A"]
            tag = (f"FILL -> {oa['reason']} {oa['R']:+.2f}R" if oa else
                   ("OPEN" if a.get("fill_j") else "DROP"))
            ob = a["B"]
            btag = f"{ob['reason']} {ob['R']:+.2f}R" if ob else "open"
            print(f"  {a['date']:%m-%d %H:%M} {'BUY' if a['dir'] > 0 else 'SELL'} "
                  f"conf={a['conf']:.2f} adx={a['adx']:.0f}  A: {tag:24s} B: {btag}")


if __name__ == "__main__":
    main()
