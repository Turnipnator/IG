"""
Long-only backtest of an EMA-20 trend filter + RSI oversold entry on daily SPY.

Data: a local "SPY Historical Price Data.csv" export (investing.com format:
DD/MM/YYYY, Price=close, Open/High/Low, Vol., Change %; newest row first).

The requested strategy is arm A. Arms B-D are a SMALL, PRE-DECLARED family of
minimal repairs, fixed before any result was inspected, because arm A produces
zero trades on this sample (see the tear sheet). Every arm is reported whatever
it scores -- no arm is selected on its own backtest result. That is the
anti-curve-fit contract for this script; do not add an arm after seeing output.

Conventions shared by every arm (also pre-declared, not tuned):
  Entry  : next session's OPEN after a qualifying CLOSE. Never the signal bar.
  Exits  : whichever fires first, checked in this order each bar after entry
             1. hard stop  - 2.0 x ATR(14)-at-entry below entry, intraday on Low
                             (gap-through fills at the Open)
             2. signal exit - RSI(14) CROSSES back up through 55, filled next
                             OPEN. A bare level test is degenerate here: any
                             entry taken while close > EMA20 already has
                             RSI(14) ~ 50+, so it exits on the next bar and
                             measures nothing. The exit only arms once RSI has
                             actually been below the level.
             3. time stop   - 15 trading days held, filled next OPEN
  Costs  : 1 bp per side (2 bp round trip) -- SPY spread + commission.
           Never run this with costs off; uncosted runs flatter the arms that
           trade most often.
  Sizing : fixed $10,000 notional, one position at a time, no pyramiding.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indicators import calculate_atr, calculate_ema, calculate_rsi  # noqa: E402

CSV_PATH = Path("/Users/paulturner/Downloads/SPY Historical Price Data.csv")
TEARSHEET = Path(__file__).resolve().parents[1] / "SPY_EMA_RSI_TEARSHEET.md"

NOTIONAL = 10_000.0
COST_BPS_PER_SIDE = 1.0
STOP_ATR_MULT = 2.0
RSI_EXIT_LEVEL = 55.0
MAX_HOLD_BARS = 15


# ---------------------------------------------------------------- data

def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", thousands=",")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("Date").reset_index(drop=True)

    df = df.rename(columns={"Price": "close", "Open": "open",
                            "High": "high", "Low": "low"})
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""),
                                errors="raise")

    bad = df[(df["high"] < df["low"]) |
             (df["high"] < df[["open", "close"]].max(axis=1)) |
             (df["low"] > df[["open", "close"]].min(axis=1))]
    if not bad.empty:
        raise ValueError(f"{len(bad)} OHLC-inconsistent rows, e.g. {bad.iloc[0].to_dict()}")
    if df["Date"].duplicated().any():
        raise ValueError("duplicate dates in source CSV")

    return df[["Date", "open", "high", "low", "close"]]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema20"] = calculate_ema(c, 20)
    df["ema50"] = calculate_ema(c, 50)
    df["rsi14"] = calculate_rsi(c, 14)
    df["rsi2"] = calculate_rsi(c, 2)
    df["atr14"] = calculate_atr(df["high"], df["low"], c, 14)
    df["low5"] = df["low"].rolling(5).min()
    return df


# ---------------------------------------------------------------- signals

def signal_a(df):  # as specified
    return (df["close"] > df["ema20"]) & (df["rsi14"] < 35)


def signal_b(df):  # repair 1: push the trend filter to a slower horizon
    return (df["close"] > df["ema50"]) & (df["rsi14"] < 35)


def signal_c(df):  # repair 2: keep EMA20 as trend, move the dip to a fast RSI
    return (df["ema20"].diff() > 0) & (df["rsi2"] < 15)


def signal_d(df):  # repair 3: price-based dip (mirrors the bot's live pullback arm)
    return (df["close"] > df["ema20"]) & (df["close"] < df["low5"].shift(1))


def signal_e(df):  # POST-HOC DIAGNOSTIC, not a candidate strategy -- see tear sheet
    return df["rsi14"] < 35


ARMS = {
    "A_spec_ema20_rsi14_lt35": ("As specified: close > EMA20 AND RSI(14) < 35", signal_a),
    "B_ema50_rsi14_lt35": ("Repair 1: close > EMA50 AND RSI(14) < 35", signal_b),
    "C_ema20_rising_rsi2_lt15": ("Repair 2: EMA20 rising AND RSI(2) < 15", signal_c),
    "D_ema20_5day_low": ("Repair 3: close > EMA20 AND close < prior 5-session low", signal_d),
    "E_rsi14_lt35_notrend": ("Diagnostic (post-hoc): RSI(14) < 35, no trend filter", signal_e),
}


# ---------------------------------------------------------------- engine

def run_arm(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """Walk the bars once. Returns the trade log."""
    trades = []
    i = 0
    n = len(df)
    cost = COST_BPS_PER_SIDE / 10_000.0

    while i < n - 1:
        if not bool(signal.iloc[i]) or not np.isfinite(df["atr14"].iloc[i]):
            i += 1
            continue

        entry_i = i + 1
        entry_px = df["open"].iloc[entry_i] * (1 + cost)
        stop_px = df["close"].iloc[i] - STOP_ATR_MULT * df["atr14"].iloc[i]

        exit_i = exit_px = reason = None
        armed = False  # signal exit arms only after RSI has been below the level
        for j in range(entry_i, n):
            bar = df.iloc[j]
            if bar["rsi14"] < RSI_EXIT_LEVEL:
                armed = True

            if bar["low"] <= stop_px:
                exit_i = j
                exit_px = min(bar["open"], stop_px)  # gap-through fills at the open
                reason = f"Stop -{STOP_ATR_MULT:g}xATR"
                break

            if armed and bar["rsi14"] >= RSI_EXIT_LEVEL and j + 1 < n:
                exit_i = j + 1
                exit_px = df["open"].iloc[j + 1]
                reason = f"RSI(14) >= {RSI_EXIT_LEVEL:g}"
                break

            if j - entry_i + 1 >= MAX_HOLD_BARS and j + 1 < n:
                exit_i = j + 1
                exit_px = df["open"].iloc[j + 1]
                reason = f"Time stop {MAX_HOLD_BARS}d"
                break

        if exit_i is None:  # still open at the end of the sample
            exit_i = n - 1
            exit_px = df["close"].iloc[exit_i]
            reason = "Open at sample end"

        exit_px *= (1 - cost)
        shares = NOTIONAL / entry_px
        trades.append({
            "entry_date": df["Date"].iloc[entry_i],
            "entry_px": entry_px,
            "exit_date": df["Date"].iloc[exit_i],
            "exit_px": exit_px,
            "reason": reason,
            "rsi_entry": df["rsi14"].iloc[i],
            "bars_held": exit_i - entry_i,
            "pnl": (exit_px - entry_px) * shares,
            "ret_pct": (exit_px / entry_px - 1) * 100,
            "entry_i": entry_i,
            "exit_i": exit_i,
        })
        i = exit_i + 1  # flat-only: no re-entry before the exit clears

    return pd.DataFrame(trades)


def equity_curve(df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Daily mark-to-market equity, so drawdown includes open-trade heat."""
    eq = pd.Series(0.0, index=df.index)
    realised = 0.0
    for _, t in trades.iterrows():
        shares = NOTIONAL / t["entry_px"]
        for j in range(int(t["entry_i"]), int(t["exit_i"])):
            eq.iloc[j] = realised + (df["close"].iloc[j] - t["entry_px"]) * shares
        realised += t["pnl"]
        eq.iloc[int(t["exit_i"]):] = realised
    return eq


def metrics(df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_win = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())

    eq = equity_curve(df, trades)
    peak = (eq + NOTIONAL).cummax()
    max_dd = ((eq + NOTIONAL) / peak - 1).min() * 100

    exposure = sum(int(t["exit_i"]) - int(t["entry_i"]) for _, t in trades.iterrows())

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "net_pnl": trades["pnl"].sum(),
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy": trades["pnl"].mean(),
        "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,
        "best": trades["pnl"].max(),
        "worst": trades["pnl"].min(),
        "avg_bars": trades["bars_held"].mean(),
        "max_dd": max_dd,
        "exposure_pct": exposure / len(df) * 100,
    }


def buy_and_hold(df: pd.DataFrame) -> dict:
    cost = COST_BPS_PER_SIDE / 10_000.0
    entry = df["open"].iloc[0] * (1 + cost)
    exit_px = df["close"].iloc[-1] * (1 - cost)
    shares = NOTIONAL / entry
    eq = (df["close"] - entry) * shares + NOTIONAL
    peak = eq.cummax()
    return {
        "net_pnl": (exit_px - entry) * shares,
        "ret_pct": (exit_px / entry - 1) * 100,
        "max_dd": ((eq / peak) - 1).min() * 100,
    }


# ---------------------------------------------------------------- report

def fmt_trades(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "_No trades generated._\n"
    rows = ["| # | Entry Date | Entry $ | RSI@sig | Exit Date | Exit $ | Bars | Exit Reason | PnL $ | PnL % |",
            "|--:|---|--:|--:|---|--:|--:|---|--:|--:|"]
    for k, (_, t) in enumerate(trades.iterrows(), 1):
        rows.append(
            f"| {k} | {t['entry_date']:%Y-%m-%d} | {t['entry_px']:.2f} "
            f"| {t['rsi_entry']:.1f} "
            f"| {t['exit_date']:%Y-%m-%d} | {t['exit_px']:.2f} | {t['bars_held']} "
            f"| {t['reason']} | {t['pnl']:+.2f} | {t['ret_pct']:+.2f}% |"
        )
    return "\n".join(rows) + "\n"


def fmt_metrics(m: dict) -> str:
    if m["trades"] == 0:
        return ("| Metric | Value |\n|---|--:|\n| Trades | 0 |\n\n"
                "_No trades — every other metric is undefined._\n")
    pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
    return "\n".join([
        "| Metric | Value |",
        "|---|--:|",
        f"| Trades | {m['trades']} |",
        f"| Win rate | {m['win_rate']:.1f}% ({m['wins']}W / {m['losses']}L) |",
        f"| Profit factor | {pf} |",
        f"| Max drawdown | {m['max_dd']:.2f}% |",
        f"| Net P&L (on ${NOTIONAL:,.0f}) | {m['net_pnl']:+,.2f} |",
        f"| Expectancy / trade | {m['expectancy']:+,.2f} |",
        f"| Avg win / avg loss | {m['avg_win']:+,.2f} / {m['avg_loss']:+,.2f} |",
        f"| Best / worst | {m['best']:+,.2f} / {m['worst']:+,.2f} |",
        f"| Avg bars held | {m['avg_bars']:.1f} |",
        f"| Time in market | {m['exposure_pct']:.1f}% |",
    ]) + "\n"


def main() -> None:
    raw = load_prices(CSV_PATH)
    df = add_indicators(raw)
    bh = buy_and_hold(df)

    out = [
        "# SPY Tear Sheet — EMA-20 Trend Filter + RSI Oversold (Long Only)",
        "",
        f"**Generated:** {pd.Timestamp.today():%Y-%m-%d} · "
        f"**Source:** `{CSV_PATH.name}` · "
        f"**Script:** `scripts/backtest_spy_ema_rsi.py`",
        "",
        f"**Sample:** {df['Date'].iloc[0]:%Y-%m-%d} → {df['Date'].iloc[-1]:%Y-%m-%d}, "
        f"{len(df)} daily bars (~{len(df)/21:.1f} months — **not** 12 months).",
        f"**Regime:** close {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f} "
        f"({bh['ret_pct']:+.2f}%), buy-and-hold max DD {bh['max_dd']:.2f}%. "
        "One-directional uptrend — every long-only arm below is measured inside a bull sample.",
        "",
        "## Rules common to all arms",
        "",
        f"- **Entry:** next session's OPEN after a qualifying close (no look-ahead).",
        f"- **Exit:** first of — hard stop {STOP_ATR_MULT:g}×ATR(14) below entry "
        f"(intraday, gap-through fills at the open) · RSI(14) crossing back up through "
        f"{RSI_EXIT_LEVEL:g} (filled next open; arms only once RSI has been below it) · "
        f"{MAX_HOLD_BARS}-bar time stop.",
        f"- **Costs:** {COST_BPS_PER_SIDE:g} bp per side ({2*COST_BPS_PER_SIDE:g} bp round trip).",
        f"- **Sizing:** ${NOTIONAL:,.0f} notional, one position at a time.",
        "",
        "Arms B–D were fixed **before** any result was read, and all are reported "
        "regardless of outcome. Nothing here was selected on its own backtest score. "
        "Arm E was added **after** seeing that A and B were empty, and is labelled a "
        "post-hoc diagnostic — it is not a candidate strategy.",
        "",
    ]

    # --- why arm A is empty: measure the collision rather than assert it
    v = pd.DataFrame({"rsi": df["rsi14"], "above": df["close"] > df["ema20"]}).dropna()
    rsi_min_above = v.loc[v["above"], "rsi"].min()
    rsi_max_below = v.loc[~v["above"], "rsi"].max()
    n_above, n_below = int(v["above"].sum()), int((~v["above"]).sum())

    out += [
        "## Why arm A is empty (the finding)",
        "",
        "`close > EMA20` and `RSI(14) < 35` are close to **mutually exclusive on daily "
        "bars**, because both are functions of net price change over a similar lookback. "
        "Measured on this sample:",
        "",
        "| Condition | Bars | RSI(14) range |",
        "|---|--:|---|",
        f"| close **above** EMA20 | {n_above} | min **{rsi_min_above:.1f}** |",
        f"| close **below** EMA20 | {n_below} | max **{rsi_max_below:.1f}** |",
        "",
        f"RSI(14) never drops below **{rsi_min_above:.1f}** while price is above its 20-EMA — "
        f"it is {rsi_min_above - 35:.1f} points clear of the 35 threshold, and the two "
        "conditions partition the sample almost exactly at RSI 50. Loosening the "
        "threshold does not rescue it: `close > EMA20 AND RSI(14) < 45` is also empty. "
        "This is structural, not a quirk of 2026 — the filter and the trigger are "
        "measuring the same thing with opposite signs.",
        "",
        "The same collision hits the **exit**. A bare `RSI(14) >= 55` level test exits on "
        "the bar after entry for any arm whose trend filter is `close > EMA20`, because "
        "RSI is already above 55 at entry. The engine therefore arms the signal exit only "
        "after RSI has actually traded below the level (a crossing, not a level test). "
        "That is a correctness fix to a degenerate rule, not a tuned parameter.",
        "",
    ]

    summary = []
    for key, (label, fn) in ARMS.items():
        sig = fn(df).fillna(False)
        trades = run_arm(df, sig)
        m = metrics(df, trades)

        out += [f"## Arm {key.split('_')[0]} — {label}", ""]
        out += [f"Signal bars in sample: **{int(sig.sum())}** of {len(df)}.", ""]
        out += ["### Trade log", "", fmt_trades(trades), "### Performance", "", fmt_metrics(m), ""]

        if m["trades"] == 0:
            pf = win = dd = pnl = "—"
        else:
            pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
            win = f"{m['win_rate']:.1f}%"
            dd = f"{m['max_dd']:.2f}%"
            pnl = f"{m['net_pnl']:+,.2f}"
        summary.append(
            f"| {key.split('_')[0]} | {label} | {m['trades']} | {win} | {pf} | {dd} | {pnl} |"
        )

    out += [
        "## Summary",
        "",
        "| Arm | Rule | Trades | Win rate | Profit factor | Max DD | Net P&L |",
        "|---|---|--:|--:|--:|--:|--:|",
        *summary,
        f"| BH | Buy & hold | 1 | — | — | {bh['max_dd']:.2f}% | {bh['net_pnl']:+,.2f} |",
        "",
        "## Verdict",
        "",
        "**The strategy as specified cannot be tested on this data, and no repair of it "
        "produced a sample large enough to judge.** Confidence labels follow the repo "
        "research protocol.",
        "",
        "- **HIGH — arm A is unimplementable as written.** Zero signals, and the gap to "
        "the threshold is wide and structural (table above). This is not a sample-size "
        "problem; it would hold on most daily equity-index series.",
        "- **HIGH — this sample cannot support any long-only verdict.** "
        f"SPY ran {bh['ret_pct']:+.2f}% with a {bh['max_dd']:.2f}% worst drawdown over "
        f"{len(df)} bars. A one-directional bull window flatters every long-only rule and "
        "starves every oversold trigger of instances.",
        "- **HIGH — no arm clears the repo's evidence bar.** The best arm fired twice. "
        "`GO_LIVE_CRITERIA.md` requires ≥30 trades on the exact strategy-market pair. "
        "Win rate, profit factor and max drawdown are reported above because they were "
        "asked for, but at n≤2 **none of them carry information** — arm D's `inf` "
        "profit factor is one winning trade, not an edge.",
        "- **MEDIUM — the oversold signal itself is one episode, not a pattern.** Every "
        "RSI(14) < 35 bar in the sample falls in a single 3-week drawdown "
        "(2026-03-13 → 2026-03-30). Arm E's +163 is that one episode's bounce. Treat it "
        "as an anecdote.",
        "- **LOW — nothing here identifies a 'most consistent profitable pattern'.** "
        "Searching 175 bars for one would be the curve-fit the brief rules out; a rule "
        "found that way would not survive out-of-sample.",
        "",
        "### What would disprove this",
        "",
        "- A multi-year sample containing bear and range regimes: if arm A still yields "
        "~zero signals there, the structural claim is confirmed; if it yields trades, the "
        "collision is regime-dependent and this conclusion narrows to trending markets.",
        "- A shorter EMA or a longer RSI would weaken the collision by separating the two "
        "lookbacks. Untested here — deliberately, since picking the pair that trades most "
        "is the curve-fit.",
        "",
        "### Suggested next actions",
        "",
        "1. **Re-run on 10+ years of SPY** before drawing any conclusion about the rule. "
        "One bull year cannot qualify or disqualify a dip-buying strategy.",
        "2. **Separate the timescales** if the EMA20+RSI shape is still wanted: a trend "
        "filter must be materially slower than the oscillator (SMA200 + RSI(14), or "
        "EMA20 + RSI(2)). The 200-day SMA needs 200 bars — this file has "
        f"{len(df)}, so it could not be tested.",
        "3. **Note the overlap with live work:** the bot already runs a daily pullback arm "
        "on S&P 500 and NASDAQ 100 (close > SMA200, buy a 5-session low) deployed "
        "2026-09-09. Arm D is its close sibling. Extending that arm's existing evidence "
        "base is a better use of effort than qualifying a new EMA20+RSI rule from scratch.",
        "",
    ]

    TEARSHEET.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n[written] {TEARSHEET}")


if __name__ == "__main__":
    main()
