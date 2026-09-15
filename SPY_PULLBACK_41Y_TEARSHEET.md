# S&P 500 Long-Only — 41-Year Replication, Out-of-Sample, Walk-Forward

**Generated:** 2026-09-15 · **Script:** `scripts/spy_pullback_walkforward.py` · **Data:** `^GSPC` daily cash bars, 10,505 sessions, 1985-01-02 → 2026-09-14

Subject: the pullback-in-uptrend rule live on demo since `7ebd4e8` — long-only, `close > SMA200`, enter on a close below the prior 5-session low, exit above the prior 5-session high or after 10 sessions, 3×ATR20 stop.

**Costs charged throughout:** 0.6 pt spread round trip + (bench + 2.5%)/365 of notional per calendar night. R = 2 × ATR20 at entry.

## Engine parity

This engine was written independently of `src/pullback.replay`. Over all 363 trades on the full 41 years the two trade lists are **identical** (entry date, exit date, both fills, exit reason). The live engine is confirmed by independent reimplementation, not just by its own golden test.

## R1–R2 · Replication, and the period the rule was never fitted on

| Window | n | mean R | z | PF | hit | yrs+ | total R | max DD | cost/trade |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2004–2026 (the sweep's window) | 192 | **+0.219** | +2.84 | 1.58 | 70% | 73% | +42.1 | -9.2R | 0.079R |
| **1985–2003 (never fitted)** | 165 | **+0.207** | +2.75 | 1.65 | 69% | 76% | +34.1 | -4.9R | 0.162R |
| 1985–2026 (all) | 363 | **+0.203** | +3.79 | 1.57 | 69% | 72% | +73.8 | -9.2R | 0.118R |

The 2026-09-09 sweep reported n 183 · +0.22R · z 2.4 · PF 1.57 · hit 72% · 73% of years positive. This run reproduces it on a fresh data pull with an independent engine.

**The 1985–2003 row is the new evidence.** It is 17 years the rule was never tuned on, it carries a *higher* modelled cost (0.16R/trade against 0.08R, from the higher rates of that era), and it holds. Short-term reversal in US indices is widely held to have changed character around 2000; this says the edge predates that.

## R3 · Walk-forward (8-year train → 2-year trade, rolling, 17 windows)

Each window re-selects the best of 72 parameter cells on the prior 8 years, then trades it blind for 2. The control runs the **live defaults** over the identical spans, so the only variable is whether re-optimising helps.

| Arm | n | mean R | z | PF | hit | yrs+ | total R | max DD |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Re-optimised each window | 227 | +0.288 | +3.93 | 1.79 | 70% | 80% | +65.4 | -7.2R |
| **Live defaults, untouched** | 276 | +0.230 | +3.70 | 1.65 | 71% | 73% | +63.4 | -9.2R |

**Both survive walk-forward** — that is the headline, and it is the test most strategies fail. Re-optimising scored +0.058R/trade more, but the standard error on that difference is 0.096R (t = +0.60), and the two arms share most of their trades. That gap is noise. **Leave the parameters alone.**

What the optimiser reached for is still worth reading:

- `n_in` — live is **5**; picked 1×4, 6×5, 10×7 (modal 7)  ← *drifts away from live*
- `n_out` — live is **5**; picked 3×5, 14×7 (modal 7)  ← *drifts away from live*
- `max_hold` — live is **10**; picked 2×5, 4×10, 11×15 (modal 15)  ← *drifts away from live*
- `sma_n` — live is **200**; picked 3×100, 14×200 (modal 200)

`sma_n=200` and a **wider exit** (`n_out=7`, `max_hold=15`) get picked far more often than the live 5/10. That is a hint the exit is a little tight, not a mandate — the difference is inside the noise band above, and the original 72-cell grid was already positive in every cell. Logged as an observation; no change proposed.

## R4 · Rotation null — is this entry timing, or just being long a rising market?

The entry signals are circularly rotated against price 2,000 times, keeping the regime filter and the exit rules. Trade count and clustering survive; alignment with price does not.

| | mean R |
|---|--:|
| **Observed rule** | **+0.203** |
| Rotation null, mean (n≈322/run) | +0.024 |
| Rotation null, 95th percentile | +0.088 |
| p (null ≥ observed) | **0.0000** |

The real rule sits **4.5 standard deviations** above randomly-timed copies of itself. A randomly-timed long in the same uptrend regime earns +0.024R — that is the drift, and it is real but small. The entry is doing the work.

## Benchmark vs buy & hold

| | Points captured | Worst drawdown (points) | Time in market |
|---|--:|--:|--:|
| Buy & hold | 7,455 | -1,220  (-56.8%) | 100% |
| Pullback (net of costs) | 2,936 | **-628** | **22%** |

The strategy captures **39% of buy-and-hold's points in 22% of the calendar time**. Being long a random 22% of the time would capture about 22%.

That reframes the earlier 'buy & hold wins' comparison. On raw total return over 41 years buy & hold still wins — it is exposed four times as long. But it paid 1,220 points of drawdown (57%, 2007–09) to get there, against 628 points for the rule. Per unit of exposure and per unit of pain this rule is ahead, and on a leveraged spread bet that is the axis that matters: the capital is free three-quarters of the time, and market risk is only carried inside the regime where the edge was measured. A 56% drawdown on a leveraged account is not a drawdown, it is a closed account.

## R5 · The EMA20 + RSI<35 question, settled on 41 years

| Condition | Signal bars in 10,505 |
|---|--:|
| A  close>EMA20 & RSI(14)<35  (as specified) | 0 |
|    close>EMA20 & RSI(14)<45  (loosened) | 0 |
| B  close>EMA50 & RSI(14)<35 | 0 |
| C  EMA20 rising & RSI(2)<15 | 237 |
| D  close>SMA200 & RSI(2)<15  (timescales separated) | 966 |
| E  RSI(14)<35, no trend filter | 530 |
|    min RSI(14) while close>EMA20 | 46.5 |

**Zero in 41 years.** The collision found in the 8-month file was not a small-sample artefact: a close above the 20-EMA and RSI(14) below 35 are mechanically incompatible on daily bars, because both measure net change over a similar lookback. Separating the timescales (`SMA200` + `RSI(2)`) is what makes the shape tradeable — and that is structurally the same idea as the pullback rule already running.

## Verdict

**The strategy already live on S&P 500 is the answer to "the most consistent long-only pattern", and it should be left alone.** Four independent checks agree:

1. **Replicates** — independent engine, fresh data pull, byte-identical trade list to `src/pullback.replay`, and the sweep's headline numbers reproduce (+0.219R vs +0.22R published).
2. **Holds out-of-sample on 17 years it was never fitted on** (+0.207R, z +2.75, 76% of years positive, at double the cost).
3. **Survives walk-forward** (+0.230R, z +3.70 across 17 blind 2-year windows), and re-optimising each window does **not** reliably beat it.
4. **Beats its own rotation null by 4.5σ** — the entry timing carries the result, not the market's drift.

**Confidence: HIGH** that this is a real, costed, positive-expectancy edge on S&P 500 daily bars. That is a stronger statement than the 09-09 sweep could make, because the evidence base has roughly doubled and now includes a walk-forward and a null model.

### What to do

- **Change nothing in the parameters.** The walk-forward's preference for a wider exit is inside the noise band. `MEMORY.md`'s standing instruction — stop tuning, review with data — applies exactly here.
- **The one observation worth carrying to the next review:** `n_out=7` was picked in 14 of 17 windows and `max_hold=15` in 11 of 17, both against live's 5/10. Pre-register a single test of the wider exit at that review rather than changing it now.
- **This strengthens the `GO_LIVE_CRITERIA.md` §7 case, it does not bypass it.** G3 still wants 30 IG-native trades; at ~8/yr that is ~4 years. The gate is about live execution risk on IG's own prices, which no amount of Yahoo history can settle.
- **Open item unchanged:** index DFB financing has still never been observed on this account. One overnight index position settles it. Every number above uses IG's published formula, not a measurement.

### What this does not say

- It says nothing about IG execution: Yahoo cash closes are not IG DFB fills, and the rule needs a market order at the US cash close.
- 41 years is one price path. The 2008 and 2020 bears are each a single event, and the regime filter's behaviour in them rests on very few decisions.
- The pre-2004 financing rates are assumed, not the repo's measured table. They make that window harsher, so the direction of the error is safe.

