"""FTSE confidence-threshold sweep on the IG candle archive.

Question: FTSE's effective entry gate is max(market_config.min_confidence=0.55,
regime_params.min_confidence) — which is 0.60 in TRENDING_HIGH / RANGING_NORMAL.
Is that gate cutting net-winners (the Gold 487959f shape) or correctly filtering
junk? Sweep the EFFECTIVE threshold and look for a sweet spot.

Uses the LIVE TradingStrategy.analyze() so the confidence values are the real
ones. Source is the IG candle archive (the live instrument) — no REST allowance
is consumed.

NOTE ON THE CLOCK: archive timestamps are written with datetime.now() inside a
TZ=Europe/London container, so they are BST (UTC+1) over this window. They are
converted to UTC before the 08:00-17:00 UTC trading-hours gate is applied.
"""
import json
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, ".")

from config import MARKETS, STRATEGY_PARAMS, get_strategy_for_market
from src.indicators import add_all_indicators
from src.regime import classify_regime, get_regime_params
from src.strategy import Signal, TradingStrategy

ARCHIVE = "/private/tmp/claude-501/-Users-paulturner-IG/6f107130-5cce-4510-9c2a-c5bc07b3ecf6/scratchpad/ftse_archive.jsonl"
SPREAD = 1.0          # live IG FTSE DFB spread, verified via get_market_info
BST_OFFSET = 1        # whole archive window (2026-06-12 .. 07-14) is BST
WARMUP = 200

ftse = next(m for m in MARKETS if m.epic == "IX.D.FTSE.DAILY.IP")
prof = get_strategy_for_market(ftse)
strategy = TradingStrategy()


YAHOO = "/private/tmp/claude-501/-Users-paulturner-IG/6f107130-5cce-4510-9c2a-c5bc07b3ecf6/scratchpad/ftse_yahoo.csv"


def load(source: str = "archive") -> pd.DataFrame:
    """IG archive (live instrument, ~1mo) or Yahoo ^FTSE 5m (cash index, ~60d).

    Yahoo is the independent corroborator: ^FTSE is the CASH index and trades
    08:00-16:30 UTC, which is essentially FTSE's configured 08-17 UTC window, so
    unlike the Wall St indices there is no out-of-hours blind spot here.
    """
    if source == "yahoo":
        df = pd.read_csv(YAHOO, parse_dates=["date"])
        return df[["date", "open", "high", "low", "close", "volume"]]
    rows = []
    with open(ARCHIVE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"]) - timedelta(hours=BST_OFFSET)  # BST -> UTC
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["volume"] = 0
    return df[["date", "open", "high", "low", "close", "volume"]]


def hourly_context(df: pd.DataFrame) -> dict:
    """HTF trend + regime per hour, from the trailing 30 hourly candles (mimics live)."""
    h = (
        df.set_index("date")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    ctx = {}
    for i in range(21, len(h)):
        window = h.iloc[max(0, i - 29) : i + 1].copy()
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
            reg = classify_regime(add_all_indicators(window, STRATEGY_PARAMS))
            rp = get_regime_params(reg)
            ctx[h["date"].iloc[i]] = (trend, reg.code, reg.is_tradeable, rp.min_confidence)
        except Exception:
            ctx[h["date"].iloc[i]] = (trend, "UNKNOWN", True, 0.5)
    return ctx


def ctx_at(ctx: dict, keys, ts):
    """Most recent hourly context strictly at or before ts."""
    import bisect

    i = bisect.bisect_right(keys, ts) - 1
    return ctx[keys[i]] if i >= 0 else None


def collect_signals(df: pd.DataFrame, ctx: dict) -> list:
    """Every FTSE BUY signal the live engine would raise, with its confidence."""
    keys = sorted(ctx)
    out = []
    for i in range(WARMUP, len(df)):
        ts = df["date"].iloc[i]
        if not (8 <= ts.hour < 17):          # FTSE 08-17 UTC
            continue
        c = ctx_at(ctx, keys, ts)
        if c is None:
            continue
        htf, code, tradeable, regime_min = c
        if not tradeable:                     # RANGING_HIGH: live skips entirely
            continue
        window = df.iloc[max(0, i - 250) : i + 1].copy()
        sig = strategy.analyze(window, ftse, float(df["close"].iloc[i]), htf)
        if sig.signal != Signal.BUY:          # FTSE is allowed_direction=BUY
            continue
        out.append(
            {
                "i": i,
                "date": ts,
                "conf": sig.confidence,
                "stop": sig.stop_distance,
                "regime": code,
                "regime_min": regime_min,
                "effective_gate": max(ftse.min_confidence, regime_min),
                "atr": float(window["atr"].iloc[-1]) if "atr" in window else None,
            }
        )
    return out


def simulate(df: pd.DataFrame, signals: list, threshold: float, spread: float = SPREAD) -> dict:
    """Sequentially take signals with conf >= threshold. One position at a time.

    Faithful to live: pullback-entry arm/wait, profile stop (2.0xATR), R:R 2.0
    limit, break-even at 0.7R, MACD-histogram-negative-3 exit (not on entry
    candle), 6-candle re-entry cooldown.
    """
    frac = prof.pullback_entry_atr_frac
    win = prof.pullback_entry_window
    rr = prof.reward_risk
    be = prof.breakeven_trigger_pct
    cooldown = 6

    trades = []
    open_until = -1
    cooldown_until = -1

    for s in signals:
        if s["conf"] < threshold:
            continue
        i = s["i"]
        if i <= open_until or i <= cooldown_until:
            continue
        stop_dist = s["stop"]
        atr = s["atr"]
        if not stop_dist or not atr or stop_dist <= 0:
            continue

        # --- pullback entry: wait <= win candles for a frac*ATR retrace ---
        target = float(df["close"].iloc[i]) - frac * atr
        entry_i, entry = None, None
        for j in range(i + 1, min(i + 1 + win, len(df))):
            if float(df["low"].iloc[j]) <= target:
                entry_i, entry = j, target
                break
        if entry_i is None:
            continue  # runaway — live drops it

        stop_px = entry - stop_dist
        limit_px = entry + stop_dist * rr
        be_trigger = entry + stop_dist * be
        moved_be = False
        neg = 0
        exit_i, exit_px, reason = None, None, None

        for j in range(entry_i + 1, len(df)):
            hi, lo = float(df["high"].iloc[j]), float(df["low"].iloc[j])
            if lo <= stop_px:
                exit_i, exit_px, reason = j, stop_px, ("breakeven" if moved_be else "stop")
                break
            if hi >= limit_px:
                exit_i, exit_px, reason = j, limit_px, "limit"
                break
            if not moved_be and hi >= be_trigger:
                stop_px = entry
                moved_be = True
            # MACD exit (never on the entry candle itself — live min-hold guard)
            if prof.use_macd_exit and j > entry_i:
                mh = df["macd_hist"].iloc[j]
                neg = neg + 1 if mh < 0 else 0
                if neg >= 3:
                    exit_i, exit_px, reason = j, float(df["close"].iloc[j]), "macd"
                    break
        if exit_i is None:
            continue  # still open at end of archive — exclude

        pts = (exit_px - entry) - spread
        trades.append(
            {
                "entry_date": df["date"].iloc[entry_i],
                "conf": s["conf"],
                "regime": s["regime"],
                "pts": pts,
                "R": pts / stop_dist,
                "reason": reason,
                "bars": exit_i - entry_i,
            }
        )
        open_until = exit_i
        cooldown_until = exit_i + cooldown

    return metrics(trades)


def metrics(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0, "pts": 0.0, "pf": 0.0, "avg_r": 0.0, "trades": []}
    wins = [t for t in trades if t["pts"] > 0]
    losses = [t for t in trades if t["pts"] <= 0]
    gp = sum(t["pts"] for t in wins)
    gl = abs(sum(t["pts"] for t in losses))
    return {
        "n": n,
        "wr": len(wins) / n * 100,
        "pts": sum(t["pts"] for t in trades),
        "pf": (gp / gl) if gl > 0 else float("inf"),
        "avg_r": sum(t["R"] for t in trades) / n,
        "trades": trades,
    }


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "archive"
    df = load(source)
    print(f"[{source}] {len(df)} 5m candles  {df['date'].iloc[0]} -> {df['date'].iloc[-1]} (UTC)")

    # indicators for MACD exit lookups
    params = {
        "ema_fast": prof.ema_fast, "ema_medium": prof.ema_medium, "ema_slow": prof.ema_slow,
        "rsi_period": prof.rsi_period, "rsi_overbought": prof.rsi_overbought,
        "rsi_oversold": prof.rsi_oversold, "rsi_buy_max": prof.rsi_buy_max,
        "rsi_sell_min": prof.rsi_sell_min, "adx_threshold": prof.adx_threshold,
    }
    df = add_all_indicators(df, params)

    ctx = hourly_context(df)
    signals = collect_signals(df, ctx)
    print(f"\nFTSE BUY signals raised (in-hours, regime-tradeable): {len(signals)}")
    if not signals:
        return

    confs = sorted(s["conf"] for s in signals)
    print(f"Confidence range: {confs[0]:.2f} - {confs[-1]:.2f}   median {confs[len(confs)//2]:.2f}")
    dist = {}
    for s in signals:
        dist[s["conf"]] = dist.get(s["conf"], 0) + 1
    print("Confidence histogram: " + "  ".join(f"{k:.2f}:{v}" for k, v in sorted(dist.items())))
    gates = {}
    for s in signals:
        gates[(s["regime"], s["effective_gate"])] = gates.get((s["regime"], s["effective_gate"]), 0) + 1
    print("\nRegime -> effective gate (signal counts):")
    for (code, g), c in sorted(gates.items(), key=lambda x: -x[1]):
        print(f"  {code:16s} gate={g:.2f}  n={c}")

    # ---- is confidence informative at all? (take everything, bucket by conf) ----
    allt = simulate(df, signals, 0.0)
    print(f"\n=== ALL signals taken (no confidence gate): "
          f"n={allt['n']} WR={allt['wr']:.0f}% pts={allt['pts']:+.1f} PF={allt['pf']:.2f} avgR={allt['avg_r']:+.2f}")
    print("\nOutcome by confidence band (does confidence predict anything?):")
    bands = [(0.0, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    for lo, hi in bands:
        sel = [t for t in allt["trades"] if lo <= t["conf"] < hi]
        if sel:
            m = metrics(sel)
            print(f"  conf {lo:.2f}-{hi:.2f}: n={m['n']:2d} WR={m['wr']:3.0f}% "
                  f"pts={m['pts']:+7.1f} PF={m['pf']:.2f} avgR={m['avg_r']:+.2f}")
        else:
            print(f"  conf {lo:.2f}-{hi:.2f}: n=0")

    # ---- the sweep ----
    print("\n=== THRESHOLD SWEEP (cumulative: take every signal >= T) ===")
    print(f"{'T':>5} {'n':>3} {'WR%':>5} {'points':>8} {'PF':>6} {'avgR':>6}   half1 / half2 (pts)")
    mid = df["date"].iloc[len(df) // 2]
    for t in [x / 100 for x in range(50, 76, 1)]:
        m = simulate(df, signals, t)
        if m["n"] == 0:
            print(f"{t:5.2f}   0     -        -      -      -")
            continue
        h1 = metrics([x for x in m["trades"] if x["entry_date"] < mid])
        h2 = metrics([x for x in m["trades"] if x["entry_date"] >= mid])
        star = "  <-- LIVE (regime 0.60)" if abs(t - 0.60) < 1e-9 else (
               "  <-- config 0.55" if abs(t - 0.55) < 1e-9 else "")
        print(f"{t:5.2f} {m['n']:3d} {m['wr']:5.0f} {m['pts']:+8.1f} {m['pf']:6.2f} {m['avg_r']:+6.2f}"
              f"   {h1['pts']:+7.1f} / {h2['pts']:+7.1f}{star}")

    # ---- cost robustness ----
    print("\n=== spread robustness (total points) ===")
    for sp in [0.0, 1.0, 2.0]:
        row = "  ".join(
            f"T={t:.2f}:{simulate(df, signals, t, spread=sp)['pts']:+.0f}"
            for t in [0.50, 0.55, 0.60, 0.65, 0.70]
        )
        print(f"  spread {sp:.1f}: {row}")


if __name__ == "__main__":
    main()
