#!/usr/bin/env python3
"""Gold MOMENTUM vs BREAKOUT head-to-head at the CURRENT live configs (2026-08-18).

Why now: Gold was flipped breakout -> momentum -> breakout on 08-18 off the back of
three bad breakout trades. The July pullback-replay concluded "Gold 5m momentum edge
~= 0" and left momentum-vs-breakout as the open question; this answers it.

The live record is thin and era-split, so it cannot settle it on its own:
    momentum ("gold" profile)  59 trades  PF 1.00  +GBP  0.44   (mostly pre-07-24)
    breakout (N55)              7 trades  PF 0.84  -GBP 11.80   (all post-08-03)
Different eras, different regimes, and breakout's 7 hang on one +GBP52 winner.

WHAT MAKES THIS HARD, STATED UP FRONT: the two strategies do not run on the same
frame. Momentum trades Gold's native 5m candles; breakout trades a 1h resample
(main._breakout_frame_1h). Yahoo serves 5m for only ~60d, so there is NO source
that gives a long-window comparison at both live configs. Therefore:

  PRIMARY   IG candle archive, ~67d, the live instrument, both at live config.
            Momentum on native 5m, breakout on a 1h resample of the SAME candles,
            both gated by the SAME DAY HTF series. Apples-to-apples but SHORT.
  CONTEXT   Yahoo GC=F 730d 1h breakout, to say whether the 67d archive window is
            a representative breakout regime or a bad patch. Momentum has no
            730d equivalent — that gap is not closable and is not papered over.

HTF: Gold moved to htf_resolution="DAY" on 08-13 (52e97bd). The July replay built
its context from HOURLY bars, which is no longer live. Here BOTH sides read the
same DAY series, built with the .shift(1) that stops a day being gated by its own
close (the look-ahead fixed in 5cba8b3). Regime classification stays hourly, as in
the replay — only the HTF trend field changes, which is how the 08-13 call was made.

COSTS are asymmetric on purpose, because the entries are: momentum enters at market
on a signal and pays spread (0.4pt round trip, as the July replay); breakout enters
on a stop through a channel and pays the measured live slip of 0.286xATR
(project-breakout-entry-fill-gap-2026-08). A spread-only sensitivity for breakout is
printed too, so the verdict can be read against both.

ZERO IG API cost: archive + Yahoo only.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from src.indicators import add_all_indicators, calculate_atr, calculate_ema
from src.backtest import Backtester, TICKER_MAP

import scripts.backtest_gold_pullback_replay as replay
from scripts.backtest_forex_breakout import breakout_sim, rstats

logging.basicConfig(level=logging.ERROR)

SCRATCH = ("/private/tmp/claude-501/-Users-paulturner-IG/"
           "6f107130-5cce-4510-9c2a-c5bc07b3ecf6/scratchpad")
ARCHIVE = f"{SCRATCH}/gold_archive.jsonl"

N, STOP_K = 55, 2.0          # live BREAKOUT_CONFIGS["CS.D.USCGC.TODAY.IP"]
SLIP_ATR_FRAC = 0.286        # measured live breakout entry slip
SPREAD_PTS = 0.4             # momentum round-trip, same as the July replay
YAHOO_DAYS = 730


# ---------------------------------------------------------------- shared data

def load_archive() -> pd.DataFrame:
    rows = []
    with open(ARCHIVE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"])
    # Archive timestamps are container-local BST; the gold hours gate is UTC.
    df["date"] = df["date"] - pd.Timedelta(hours=replay.BST_OFFSET)
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values(
        "date").reset_index(drop=True)


def day_htf(df: pd.DataFrame) -> pd.DataFrame:
    """DAY BULLISH/BEARISH/NEUTRAL from the archive itself, gated one bar back.

    Same rule as live update_htf_trends (EMA9 vs EMA21 plus close-side) and the
    same .shift(1) as htf_series: a day is judged by the last COMPLETED day, not
    by its own close."""
    d = (df.set_index("date").resample("1D")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
         .dropna().reset_index())
    d["ema_9"] = calculate_ema(d["close"], 9)
    d["ema_21"] = calculate_ema(d["close"], 21)

    def trend(r):
        if pd.isna(r["ema_9"]) or pd.isna(r["ema_21"]):
            return "NEUTRAL"
        if r["ema_9"] > r["ema_21"] and r["close"] > r["ema_21"]:
            return "BULLISH"
        if r["ema_9"] < r["ema_21"] and r["close"] < r["ema_21"]:
            return "BEARISH"
        return "NEUTRAL"

    d["htf"] = d.apply(trend, axis=1).shift(1).fillna("NEUTRAL")
    return d[["date", "htf"]]


# ---------------------------------------------------------------- momentum

def run_momentum(df: pd.DataFrame, htf_day: pd.DataFrame):
    """Live 'gold' profile via the July replay's faithful analyze() walk, with the
    HTF trend swapped from HOURLY to DAY to match the current config."""
    prof = replay.prof
    params = {"ema_fast": prof.ema_fast, "ema_medium": prof.ema_medium,
              "ema_slow": prof.ema_slow, "rsi_period": prof.rsi_period,
              "rsi_overbought": prof.rsi_overbought, "rsi_oversold": prof.rsi_oversold,
              "rsi_buy_max": prof.rsi_buy_max, "rsi_sell_min": prof.rsi_sell_min,
              "adx_threshold": prof.adx_threshold}
    ind = add_all_indicators(df.copy(), params)

    ctx = replay.hourly_context(ind)          # regime + tradeable + regime floor
    # Swap the trend element for the DAY value in force at that hour.
    hd = htf_day.sort_values("date")
    keys = sorted(ctx)
    merged = pd.merge_asof(pd.DataFrame({"date": keys}).sort_values("date"),
                           hd, on="date", direction="backward")
    day_at = dict(zip(merged["date"], merged["htf"].fillna("NEUTRAL")))
    for k in keys:
        trend, code, tradeable, rmin = ctx[k]
        ctx[k] = (day_at.get(k, "NEUTRAL"), code, tradeable, rmin)

    signals = replay.collect_signals(ind, ctx)
    arms = replay.build_arms(ind, signals)
    filled = [a["A"] for a in arms if a["A"] is not None]
    return arms, filled


def mom_stats(outs):
    outs = [o for o in outs if o]
    if not outs:
        return None
    wins = [o for o in outs if o["pts"] > 0]
    losses = [o for o in outs if o["pts"] <= 0]
    gp = sum(o["pts"] for o in wins)
    gl = abs(sum(o["pts"] for o in losses))
    return dict(n=len(outs), wr=len(wins) / len(outs) * 100,
                pts=sum(o["pts"] for o in outs), r=sum(o["R"] for o in outs),
                pf=gp / gl if gl else 999.0,
                avg_w=gp / len(wins) if wins else 0.0,
                avg_l=-gl / len(losses) if losses else 0.0,
                maxw=max((o["pts"] for o in outs), default=0.0))


# ---------------------------------------------------------------- breakout

def hourly_frame(df: pd.DataFrame, htf_day: pd.DataFrame) -> pd.DataFrame:
    """1h resample + ATR + the same DAY htf column — mirrors _breakout_frame_1h."""
    h = (df.set_index("date").resample("1h")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum"})
         .dropna().reset_index())
    h["atr"] = calculate_atr(h["high"], h["low"], h["close"], 14)
    return pd.merge_asof(h.sort_values("date"), htf_day.sort_values("date"),
                         on="date", direction="backward").assign(
        htf=lambda x: x["htf"].fillna("NEUTRAL"))


def run_breakout(h: pd.DataFrame, cost_pts: float):
    return rstats(breakout_sim(h, N, STOP_K, "donchian", True, 0.0,
                               cost_pips=cost_pts, pip=1.0))


# ---------------------------------------------------------------- reporting

def bline(tag, s):
    if s is None:
        print(f"  {tag:<34} no trades"); return
    print(f"  {tag:<34} n={s['n']:>3} WR={s['wr']:>3.0f}% P&L={s['pnl']:>+7.2f}% "
          f"PF={s['pf']:>5.2f} avgW={s['avg_w']:>+5.2f} avgL={s['avg_l']:>+6.2f} "
          f"maxW={s['maxw']:>+5.2f}")


def mline(tag, s, median):
    if s is None:
        print(f"  {tag:<34} no trades"); return
    pct = s["pts"] / median * 100
    print(f"  {tag:<34} n={s['n']:>3} WR={s['wr']:>3.0f}% P&L={pct:>+7.2f}% "
          f"PF={s['pf']:>5.2f} avgW={s['avg_w'] / median * 100:>+5.2f} "
          f"avgL={s['avg_l'] / median * 100:>+6.2f} "
          f"maxW={s['maxw'] / median * 100:>+5.2f}  ({s['pts']:+.1f}pts, {s['r']:+.2f}R)")


def main() -> None:
    df = load_archive()
    median = float(df["close"].median())
    print(f"\n{'=' * 96}")
    print("GOLD — MOMENTUM (5m, live 'gold' profile) vs BREAKOUT (1h, live N55) "
          "— IG archive, both at live config")
    print(f"{'=' * 96}")
    print(f"{len(df)} 5m candles  {df['date'].iloc[0]} -> {df['date'].iloc[-1]} (UTC)  "
          f"{df['date'].dt.date.nunique()} days")

    htf_day = day_htf(df)
    counts = htf_day["htf"].value_counts().to_dict()
    print(f"DAY HTF over the window: {counts}")

    h = hourly_frame(df, htf_day)
    atr_med = float(h["atr"].median())
    slip = SLIP_ATR_FRAC * atr_med
    print(f"1h frame: {len(h)} bars, median ATR {atr_med:.2f}pt "
          f"-> breakout slip charge {slip:.2f}pt/trade; momentum spread "
          f"{SPREAD_PTS:.1f}pt/trade\n")

    arms, filled = run_momentum(df, htf_day)
    ms = mom_stats(filled)
    bs = run_breakout(h, slip)
    bs_spread = run_breakout(h, SPREAD_PTS)
    bs_free = run_breakout(h, 0.0)

    print("=== HEADLINE — same 67d window, same instrument, same DAY HTF ===")
    mline("MOMENTUM live (pullback+0.4pt)", ms, median)
    bline("BREAKOUT live (0.286xATR slip)", bs)
    bline("  breakout @ spread-only 0.4pt", bs_spread)
    bline("  breakout @ zero cost", bs_free)
    print(f"\n  momentum arms {len(arms)}, filled {len(filled)}, "
          f"fill-rate {len(filled) / max(1, len(arms)) * 100:.0f}%")

    # --- thirds, for sign-consistency on a short window -----------------
    print("\n=== thirds (sign-consistency; ~22d each, small n) ===")
    third = len(df) // 3
    hthird = len(h) // 3
    for k in range(3):
        dsl = df.iloc[k * third:(k + 1) * third].reset_index(drop=True)
        hsl = h.iloc[k * hthird:(k + 1) * hthird].reset_index(drop=True)
        _, mf = run_momentum(dsl, htf_day)
        m = mom_stats(mf)
        b = run_breakout(hsl, slip)
        mp = m["pts"] / median * 100 if m else 0.0
        bp = b["pnl"] if b else 0.0
        print(f"   T{k + 1} ({dsl['date'].iloc[0].date()} -> {dsl['date'].iloc[-1].date()}):  "
              f"momentum {mp:>+6.2f}% (n={m['n'] if m else 0:>2})    "
              f"breakout {bp:>+6.2f}% (n={b['n'] if b else 0:>2})")

    # --- long-window regime context for breakout only -------------------
    print(f"\n=== CONTEXT: is the 67d archive window a fair patch for breakout? ===")
    print(f"    (Yahoo GC=F {YAHOO_DAYS}d 1h, same N55/2.0xATR/DAY-HTF config. "
          f"Momentum has NO 730d 5m equivalent — gap, not an omission.)")
    try:
        TICKER_MAP.update({"Gold": "GC=F"})
        bt = Backtester(params={"ema_fast": 3, "ema_medium": 8, "ema_slow": 21,
                                "rsi_period": 7, "adx_threshold": 35})
        from scripts.backtest_forex_breakout import htf_series
        raw = bt.fetch_data("Gold", YAHOO_DAYS, "1h")
        if raw is None or len(raw) < 500:
            print("    Yahoo fetch returned too little data — SKIPPED")
        else:
            raw = raw.copy()
            raw["atr"] = calculate_atr(raw["high"], raw["low"], raw["close"], 14)
            hs = htf_series(bt, "Gold", days=YAHOO_DAYS)
            ydf = pd.merge_asof(raw.sort_values("date"), hs.sort_values("date"),
                                on="date", direction="backward")
            ydf["htf"] = ydf["htf"].fillna("NEUTRAL")
            yslip = SLIP_ATR_FRAC * float(ydf["atr"].median())
            print(f"    {len(ydf)} 1h bars, median ATR {float(ydf['atr'].median()):.2f}pt "
                  f"-> slip {yslip:.2f}pt")
            bline("BREAKOUT 730d (live cfg)", run_breakout(ydf, yslip))
            q = len(ydf) // 4
            for k in range(4):
                s = run_breakout(ydf.iloc[k * q:(k + 1) * q].reset_index(drop=True), yslip)
                print(f"       Q{k + 1}: P&L={s['pnl'] if s else 0:>+7.2f}% "
                      f"PF={s['pf'] if s else 0:>5.2f} (n={s['n'] if s else 0:>2})")
    except Exception as e:
        print(f"    Yahoo context failed: {e}")

    print("\nRead: the headline decides only if BOTH sides agree in sign across thirds.")
    print("A 67d window is ~1 regime. Treat a narrow gap as 'no difference shown'.\n")


if __name__ == "__main__":
    main()
