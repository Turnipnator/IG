# SPY Tear Sheet — EMA-20 Trend Filter + RSI Oversold (Long Only)

**Generated:** 2026-09-15 · **Source:** `SPY Historical Price Data.csv` · **Script:** `scripts/backtest_spy_ema_rsi.py`

**Sample:** 2026-01-02 → 2026-09-14, 175 daily bars (~8.3 months — **not** 12 months).
**Regime:** close 683.17 → 760.88 (+10.94%), buy-and-hold max DD -9.13%. One-directional uptrend — every long-only arm below is measured inside a bull sample.

## Rules common to all arms

- **Entry:** next session's OPEN after a qualifying close (no look-ahead).
- **Exit:** first of — hard stop 2×ATR(14) below entry (intraday, gap-through fills at the open) · RSI(14) crossing back up through 55 (filled next open; arms only once RSI has been below it) · 15-bar time stop.
- **Costs:** 1 bp per side (2 bp round trip).
- **Sizing:** $10,000 notional, one position at a time.

Arms B–D were fixed **before** any result was read, and all are reported regardless of outcome. Nothing here was selected on its own backtest score. Arm E was added **after** seeing that A and B were empty, and is labelled a post-hoc diagnostic — it is not a candidate strategy.

## Why arm A is empty (the finding)

`close > EMA20` and `RSI(14) < 35` are close to **mutually exclusive on daily bars**, because both are functions of net price change over a similar lookback. Measured on this sample:

| Condition | Bars | RSI(14) range |
|---|--:|---|
| close **above** EMA20 | 102 | min **48.0** |
| close **below** EMA20 | 60 | max **52.1** |

RSI(14) never drops below **48.0** while price is above its 20-EMA — it is 13.0 points clear of the 35 threshold, and the two conditions partition the sample almost exactly at RSI 50. Loosening the threshold does not rescue it: `close > EMA20 AND RSI(14) < 45` is also empty. This is structural, not a quirk of 2026 — the filter and the trigger are measuring the same thing with opposite signs.

The same collision hits the **exit**. A bare `RSI(14) >= 55` level test exits on the bar after entry for any arm whose trend filter is `close > EMA20`, because RSI is already above 55 at entry. The engine therefore arms the signal exit only after RSI has actually traded below the level (a crossing, not a level test). That is a correctness fix to a degenerate rule, not a tuned parameter.

## Arm A — As specified: close > EMA20 AND RSI(14) < 35

Signal bars in sample: **0** of 175.

### Trade log

_No trades generated._

### Performance

| Metric | Value |
|---|--:|
| Trades | 0 |

_No trades — every other metric is undefined._


## Arm B — Repair 1: close > EMA50 AND RSI(14) < 35

Signal bars in sample: **0** of 175.

### Trade log

_No trades generated._

### Performance

| Metric | Value |
|---|--:|
| Trades | 0 |

_No trades — every other metric is undefined._


## Arm C — Repair 2: EMA20 rising AND RSI(2) < 15

Signal bars in sample: **2** of 175.

### Trade log

| # | Entry Date | Entry $ | RSI@sig | Exit Date | Exit $ | Bars | Exit Reason | PnL $ | PnL % |
|--:|---|--:|--:|---|--:|--:|---|--:|--:|
| 1 | 2026-05-20 | 735.78 | 62.1 | 2026-06-11 | 728.69 | 15 | Time stop 15d | -96.45 | -0.96% |
| 2 | 2026-08-19 | 770.44 | 56.6 | 2026-08-28 | 771.68 | 7 | RSI(14) >= 55 | +16.17 | +0.16% |

### Performance

| Metric | Value |
|---|--:|
| Trades | 2 |
| Win rate | 50.0% (1W / 1L) |
| Profit factor | 0.17 |
| Max drawdown | -5.05% |
| Net P&L (on $10,000) | -80.28 |
| Expectancy / trade | -40.14 |
| Avg win / avg loss | +16.17 / -96.45 |
| Best / worst | +16.17 / -96.45 |
| Avg bars held | 11.0 |
| Time in market | 12.6% |


## Arm D — Repair 3: close > EMA20 AND close < prior 5-session low

Signal bars in sample: **1** of 175.

### Trade log

| # | Entry Date | Entry $ | RSI@sig | Exit Date | Exit $ | Bars | Exit Reason | PnL $ | PnL % |
|--:|---|--:|--:|---|--:|--:|---|--:|--:|
| 1 | 2026-08-19 | 770.44 | 56.6 | 2026-08-28 | 771.68 | 7 | RSI(14) >= 55 | +16.17 | +0.16% |

### Performance

| Metric | Value |
|---|--:|
| Trades | 1 |
| Win rate | 100.0% (1W / 0L) |
| Profit factor | inf |
| Max drawdown | -1.02% |
| Net P&L (on $10,000) | +16.17 |
| Expectancy / trade | +16.17 |
| Avg win / avg loss | +16.17 / +0.00 |
| Best / worst | +16.17 / +16.17 |
| Avg bars held | 7.0 |
| Time in market | 4.0% |


## Arm E — Diagnostic (post-hoc): RSI(14) < 35, no trend filter

Signal bars in sample: **6** of 175.

### Trade log

| # | Entry Date | Entry $ | RSI@sig | Exit Date | Exit $ | Bars | Exit Reason | PnL $ | PnL % |
|--:|---|--:|--:|---|--:|--:|---|--:|--:|
| 1 | 2026-03-16 | 668.45 | 33.1 | 2026-03-27 | 641.94 | 9 | Stop -2xATR | -396.59 | -3.97% |
| 2 | 2026-03-31 | 639.00 | 27.1 | 2026-04-09 | 674.77 | 6 | RSI(14) >= 55 | +559.76 | +5.60% |

### Performance

| Metric | Value |
|---|--:|
| Trades | 2 |
| Win rate | 50.0% (1W / 1L) |
| Profit factor | 1.41 |
| Max drawdown | -4.30% |
| Net P&L (on $10,000) | +163.16 |
| Expectancy / trade | +81.58 |
| Avg win / avg loss | +559.76 / -396.59 |
| Best / worst | +559.76 / -396.59 |
| Avg bars held | 7.5 |
| Time in market | 8.6% |


## Summary

| Arm | Rule | Trades | Win rate | Profit factor | Max DD | Net P&L |
|---|---|--:|--:|--:|--:|--:|
| A | As specified: close > EMA20 AND RSI(14) < 35 | 0 | — | — | — | — |
| B | Repair 1: close > EMA50 AND RSI(14) < 35 | 0 | — | — | — | — |
| C | Repair 2: EMA20 rising AND RSI(2) < 15 | 2 | 50.0% | 0.17 | -5.05% | -80.28 |
| D | Repair 3: close > EMA20 AND close < prior 5-session low | 1 | 100.0% | inf | -1.02% | +16.17 |
| E | Diagnostic (post-hoc): RSI(14) < 35, no trend filter | 2 | 50.0% | 1.41 | -4.30% | +163.16 |
| BH | Buy & hold | 1 | — | — | -9.13% | +1,094.02 |

## Verdict

**The strategy as specified cannot be tested on this data, and no repair of it produced a sample large enough to judge.** Confidence labels follow the repo research protocol.

- **HIGH — arm A is unimplementable as written.** Zero signals, and the gap to the threshold is wide and structural (table above). This is not a sample-size problem; it would hold on most daily equity-index series.
- **HIGH — this sample cannot support any long-only verdict.** SPY ran +10.94% with a -9.13% worst drawdown over 175 bars. A one-directional bull window flatters every long-only rule and starves every oversold trigger of instances.
- **HIGH — no arm clears the repo's evidence bar.** The best arm fired twice. `GO_LIVE_CRITERIA.md` requires ≥30 trades on the exact strategy-market pair. Win rate, profit factor and max drawdown are reported above because they were asked for, but at n≤2 **none of them carry information** — arm D's `inf` profit factor is one winning trade, not an edge.
- **MEDIUM — the oversold signal itself is one episode, not a pattern.** Every RSI(14) < 35 bar in the sample falls in a single 3-week drawdown (2026-03-13 → 2026-03-30). Arm E's +163 is that one episode's bounce. Treat it as an anecdote.
- **LOW — nothing here identifies a 'most consistent profitable pattern'.** Searching 175 bars for one would be the curve-fit the brief rules out; a rule found that way would not survive out-of-sample.

### What would disprove this

- A multi-year sample containing bear and range regimes: if arm A still yields ~zero signals there, the structural claim is confirmed; if it yields trades, the collision is regime-dependent and this conclusion narrows to trending markets.
- A shorter EMA or a longer RSI would weaken the collision by separating the two lookbacks. Untested here — deliberately, since picking the pair that trades most is the curve-fit.

### Suggested next actions

1. **Re-run on 10+ years of SPY** before drawing any conclusion about the rule. One bull year cannot qualify or disqualify a dip-buying strategy.
2. **Separate the timescales** if the EMA20+RSI shape is still wanted: a trend filter must be materially slower than the oscillator (SMA200 + RSI(14), or EMA20 + RSI(2)). The 200-day SMA needs 200 bars — this file has 175, so it could not be tested.
3. **Note the overlap with live work:** the bot already runs a daily pullback arm on S&P 500 and NASDAQ 100 (close > SMA200, buy a 5-session low) deployed 2026-09-09. Arm D is its close sibling. Extending that arm's existing evidence base is a better use of effort than qualifying a new EMA20+RSI rule from scratch.
