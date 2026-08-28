# Research: Underperforming Markets Investigation

## Question
The journal shows 3 markets (Spot Silver, Germany 40, Gold) as the largest
live losers. Which need fixing, and what's the right fix?

(US Russell 2000 added to the batch since it shares the same recent
ADX-30 revert as Germany 40.)

## Per-market summary

### Spot Silver — already disabled
- 2 live losses totalling -£71.04 (single bad day on 2026-03-11).
- Already removed from config (commit `0a8b952`). No further action.

### Germany 40 — RAISE ADX 30 → 35
**Live**: 7 trades, 43% WR, **-£46.90** (avg loss -£13.35 vs avg win +£2.17).

**60d backtest, 5m candles, HTF aligned:**

| Config            | Trades | WinRate | P&L%   | PF   |
|-------------------|-------:|--------:|-------:|-----:|
| Current (ADX 30)  |     17 |   47.1% | -0.06% | 0.96 |
| **ADX 35**        |    **8** | **62.5%** | **+0.30%** | **1.87** |
| ADX 40            |      6 |   66.7% | +0.22% | 1.81 |
| ADX 35 + 1.0× stop|      8 |   62.5% | +0.30% | 1.87 |
| ADX 35 + 2.0× stop|      8 |   62.5% | +0.30% | 1.87 |

Stop multiplier is irrelevant — 5m DAX ATR ≈ 12-15pt and the min-stop floor
of 25pt dominates. The lever is ADX threshold. ADX 40 over-filters.

**Confidence: HIGH.** Clean signal across all variants tested.

### US Russell 2000 — MARGINAL CASE
**Live**: 13 trades, 38% WR, **-£11.50** (mostly MACD exits, fast).

**60d backtest:**

| Config           | Trades | WinRate | P&L%   | PF   |
|------------------|-------:|--------:|-------:|-----:|
| Current (ADX 30) |      9 |   33.3% | -0.22% | 0.71 |
| ADX 35           |      5 |   40.0% | -0.19% | 0.69 |
| ADX 40           |      2 |   50.0% | -0.16% | 0.50 |

**Every config is negative.** Russell on 5m doesn't have a clean edge with
this strategy. Marginal improvements only.

Options:
1. Move to ADX 35 anyway — slightly better and halves trade count (less risk).
2. Disable until we find a real edge (slower timeframe? mean-reversion?).
3. Status quo — bleeding is small (-£11/57d), low priority.

**Confidence: MEDIUM** that ADX 35 helps; **HIGH** that Russell is the
weakest market in the universe and may need a fundamental rethink.

### Gold — DO NOT TOUCH
**Live overall**: 30 trades, 43% WR, -£23.86. But split by era:

| Era                            | Trades | WinRate | P&L     |
|--------------------------------|-------:|--------:|--------:|
| Pre-tightening (before Apr 7)  |     18 |   27.8% | **-£94.29** |
| Post-tightening (Apr 7+)       |     12 |   66.7% | **+£70.43** |

The 2026-04-07 commit (`1b18cc2`: stop 1.5×, ADX 35, R:R 3.0) flipped Gold
from severe bleed to clear winner. Live evidence agrees with backtest.

**60d backtest of current Gold profile: +6.00% PF 1.78** — large positive,
strategy is working.

Backtest hints that ADX 30 (+8.16%) and R:R 4.0 (+7.47%) might be
marginally better, but post-tightening live is only 12 trades. Premature
to re-tune. Let it run another 30+ trades before considering.

**Confidence: HIGH** on "leave alone for now."

## Recommendations
1. **Germany 40**: ADX 30 → 35 (HIGH confidence)
2. **US Russell 2000**: ADX 30 → 35 (MEDIUM confidence — marginal but cuts
   trade count, reduces drawdown exposure)
3. **Gold**: status quo — recent tightening is paying off
4. **Spot Silver**: no action — already disabled

Cleanest implementation: add a new `"indices_adx35"` strategy profile
between `"indices"` (ADX 30) and `"indices_selective"` (ADX 40), point
Germany 40 + Russell 2000 at it.

## Self-critique
- 60d is the max yfinance allows for 5m. Single window — no walk-forward.
- Yahoo `^GDAXI` is cash DAX; live IG epic `IX.D.DAX.DAILY.IP` is the
  daily-rolling future. They diverge slightly especially overnight.
- Russell 2000 backtest used `^RUT` cash; IG's `IX.D.RUSSELL.DAILY.IP` is
  the daily future. Same caveat.
- Backtest doesn't model BE/ATR-trail/pullback filter or live spread, so
  P&L numbers are upper bounds for entry-quality only.
- Germany 40 live sample (7 trades) is too small to be confident the live
  -£47 is signal vs noise. ADX 35 fix targets the entry quality, which is
  the upstream lever regardless.

## Open questions / next actions
- After 30 days of ADX 35 data on DAX/Russell, re-evaluate.
- Russell 2000 may need a structural rethink (15m candles? different
  strategy entirely?). Out of scope this turn.
- Gold post-tightening sample is still small (12 trades). Watch for
  another 30-50 trades before considering R:R 4 or ADX 30 tweaks.

---

# 2026-06-22 — Breakout strategy for non-forex EPICs + screener expansion?

## Question (decomposed)
Q1. Add a runtime momentum↔breakout strategy toggle to index/commodity EPICs
    (like /forex), switchable by the user on macro news / discretionary info.
    Feasible? Worth it?
Q2. Should we add more instruments to the screener universe?

## Hypotheses (Q1)
H1. A discretionary macro toggle adds edge (flip to breakout ahead of a known
    catalyst to catch the directional move).
H2. It adds operational risk + unbacktestable human calls; index momentum book
    already captures trends, so switching is net-negative/noise.
H3. A structural breakout edge may exist on SOME index/commodity EPICs (worth
    finding with data), but the macro-NEWS-EVENT trigger specifically is wrong.

## Evidence / reasoning
- Plumbing is feasible (HIGH): /forex toggle pattern + breakout.py (EPIC-keyed)
  + _execute_breakout_entry/_update_breakout_trail exist; exit routing already
  keys on breakout_deals (deal-id set), not sector, so mixed per-position
  strategy already works. A /strategy <mkt> momentum|breakout generalization is
  a moderate, well-understood refactor.
- Index momentum ALREADY WORKS (HIGH): unlike forex (momentum net-losing →
  breakout was a rescue), the index book is profitable (Wall St/NASDAQ/FTSE/Gold
  lead). So breakout here is a speculative ALTERNATIVE, not a hole-filler → bar
  is "beat or complement what works," much higher.
- Macro-NEWS-EVENT trigger is the weakest version (HIGH): breakouts INTO
  high-impact news = spike-and-reverse, slippage, gap-through-stop. Momentum
  path HAS a calendar/news block; the breakout path explicitly does NOT (logged
  open follow-up). "Flip to breakout for the CPI print" asks the system to do
  the one thing it's least protected against.
- Discretionary switching cuts against the validated discipline (MEDIUM-HIGH):
  backtest-first / observational-then-enforce / per-EPIC / "stop tuning". A
  human macro toggle is unbacktestable by construction → can't be validated →
  must never run real size on a hunch.
- Cheap first step is DATA not CODE: Donchian breakout can be backtested on
  every index/commodity EPIC at ZERO IG cost (Yahoo + candle archive). Same
  playbook as forex (walk-forward → only GBP/USD survived). Owe the indices the
  same test before believing in it.

## Confidence
- Toggle technically feasible: HIGH
- Macro-news-EVENT breakout risky without a news block first: HIGH
- Index momentum already works → breakout bar is high: HIGH
- Whether a structural breakout edge exists on indices: LOW / UNKNOWN (the gap)
- Q2 add-for-breadth not worth it: MEDIUM-HIGH (just culled 17→15; per-EPIC
  profitability principle; correlated indices add cluster risk)

## Most supported
H3 + H2. Build NOTHING discretionary yet. First run a zero-cost breakout sweep
across the index/commodity EPICs (does the edge exist, on which, vs momentum).
Only if an EPIC shows a robust walk-forward breakout edge, consider a toggle —
as a REGIME tool, not a news-event tool, and only after the breakout path gets
a news block. Reframe of "react to macro news": higher-value/lower-risk =
(a) the existing calendar/news block (pause around events) + (b) a discretionary
per-market direction-bias toggle (extend allowed_direction) to inject a macro
view, both backtestable-ish and aligned with existing machinery.

## Q2 (screener instruments)
- Universe just CULLED 17→15 on per-EPIC profitability. Adding for breadth cuts
  against that. New EPICs must (a) be SPREADBET-streaming-compatible (CC.D.*
  "Invalid account type" lesson) and (b) earn the slot via Yahoo backtest first.
- Edge concentrates in a few less-efficient indices; correlated additions mostly
  add cluster risk (cluster-filter exists for FTSE/WallSt/S&P co-movement).
- AI Index sweep (first-ever, off harvested archive) already on the 06-26 agenda
  — that's the live "new instrument" evaluation. Don't add breadth blindly.

## Next actions
- [ ] Run scripts/backtest_breakout_*.py-style Donchian sweep over index/
      commodity EPICs (Yahoo, zero IG cost): N/stop/HTF grid + walk-forward +
      cost (DFB spread). Compare PF/WR vs the live momentum profile per EPIC.
- [ ] Fold the result into the 2026-06-26 two-week review alongside AI Index.
- [ ] If building any toggle later: add the news/calendar block to the breakout
      path FIRST (shared open follow-up from the forex breakout work).

## UPDATE 2026-06-22 — sweep written + first run (scripts/backtest_breakout_indices_commodities.py)
Script done, validated, ZERO IG cost (Yahoo 1h/730d). First-run headline (LOW-MED
confidence — Yahoo cash != IG DFB, ~2yr only, thin quarter windows):
- ONLY survivor of the strict sign-consistency gate: **S&P 500 N=20 long-only**
  (full +14.45% PF1.45; +ve in all 4 quarters). Notably the EPIC already capped
  long-only live — breakout agrees with momentum's read there.
- ROBUST cross-market pattern (HIGH within-sample): on EVERY index, **longs make
  money, shorts lose** at every N (e.g. NASDAQ N55 longs +19.6%/PF2.39 vs shorts
  -15.1%/PF0.37; Wall St, FTSE same shape). Breakout SHORTS on indices are a
  structural loser — mirrors the live long-bias culls (S&P/FTSE BUY-only).
- Commodities poor: Gold/Crude weak, **Cocoa catastrophic** (N20 -197%); no
  commodity is sign-consistent.
- Q3 (a chop regime) is red across most indices even where the full period is green
  → confirms breakout's regime-fragility; the strict gate correctly fails those.
- Japan 225 / Hong Kong HS50 auto-skipped (Yahoo 1h thin) → run off the candle
  archive for the review, same as AI Index (no Yahoo).
TAKEAWAY (provisional): breakout does NOT broadly beat the momentum book on indices;
at best it's a LONG-ONLY tool on a couple of EPICs, and momentum already captures
those. Not worth a discretionary toggle on current evidence — but finish the archive
legs (N225/HSI/AI) + a head-to-head vs live momentum PF at the 06-26 review before
concluding. Shorts-lose is the cleanest takeaway and is already how we trade live.

---

## 2026-06-23 — Gold IG-long-loss vs Oanda-short-win (same time)

**Question:** 3 losers today (Crude −£13.04, FTSE −£3.20, Gold −£14.40, all small, ranging session). User: Oanda gold bot SHORTED and won ~same time IG went LONG and lost. Anything missing with IG on Gold?

**Gold trade facts (from live logs + journal):**
- 11:40 BUY signal 77% (ADX 39.4, RSI 59, HTF=BULLISH). Pullback armed → 11:50 retrace reached, entered @4124.4 (~4pt BETTER than 4128.7 signal candle). Stop 7.27pt (1.5xATR, ATR~4.85). 12:02 stopped −£14.40 (12 min).
- ADX trace: 28–30 ALL morning, single-candle spike to 39.6 @11:38, fired @11:40, collapsed back to 33.7 @12:00 → 27 by 12:40. Bought a one-candle ADX poke at the leg-exhaustion peak in an otherwise RANGING session.

**Hypotheses / confidence:**
- H1 entry-timing (Oanda fills better): REFUTED — pullback fix gave us the better fill. Timing wasn't the problem.
- H2 timeframe selectivity: MOST SUPPORTED (MED-HIGH). IG Gold = 5m signals; Oanda = H1 + 2h cooldown. 5m frame catches false trend-pokes an H1 frame never sees.
- H3 Oanda short-bias (20/21 shorts 79%WR): MEDIUM but regime-dependent — May investigation found IG Gold shorts lost MORE historically (−£40 vs −£14 longs). Not a clean "short gold" rule. Daily-HTF gate forced IG long today.
- H4 one-trade noise: HIGH that today was a ranging whipsaw; LOW that one stop is signal.

**Conclusion:** No bug. Pullback fix working (better fill). Real gap = TIMEFRAME: Gold on 5m nibbles marginal trend signals at leg-exhaustion (ADX peak→decay); Oanda's H1+cooldown sidesteps the class. Pullback fixes FILLS not SIDE.

**Next (06-26 review, zero IG cost):**
1. ADX-rising / N-consecutive-≥35 gate on Gold (would've skipped today's spike) — leg-exhaustion thesis applied to Gold. NB book-wide ADX didn't separate winners (38.3 vs 38.4) — test on Gold specifically.
2. Slower Gold signal frame (15m/1h) to match Oanda selectivity — bigger lever, backtest properly off candle archive.
Do NOT change Gold off one trade. Links: project-gold-oanda-comparison, project-leg-filter-exhaustion.

---

## 2026-06-25 — Crude Oil: why it bleeds live (PF 0.38) — exit retune vs disable

**Question:** Live `crude` is a chronic loser (all-time n=12 PF 0.38 net −£49.55; since 06-20 n=5 PF 0.05 net −£62.59), with the "MACD histogram positive 3 candles" exit alone = −£45.6. Is the exit too twitchy (retune) or is there no real edge (disable)?

**Method:** `scripts/backtest_crude_exit.py` — Yahoo CL=F 5m, 58d (10,941 bars, n=318–356). Entry held FIXED = faithful live crude (EMA9/21/50 align + RSI7 band + ADX≥30 + 0.3% pullback + 1h-HTF gate + MACD pre-check; stop 1.0×ATR, 2.0R limit). Only the EXIT swapped: macd3 (LIVE) / macd5 / donch10 / donch20 / atrtrail1.5 / atrtrail2.5. Then a spread-cost sweep (1 IG pt = $0.01 on CL=F; ATR median $0.246 ≈ 24.6 IG pts, matches live journal ATR 20–35).

**Hypotheses & confidence:**
- **H1 — exit too twitchy (MACD-3 bails on noise):** SUPPORTED but INSUFFICIENT (MEDIUM→HIGH). macd5 beats macd3 at every cost level (frictionless PF 1.24 vs 1.11; +21.5R vs +10R) — less-sensitive exit genuinely helps. But it does NOT lift the strategy above costs.
- **H2 — stop too tight:** REFUTED-ish. atrtrail/donch "let-winners-run" variants (avgWin ~2R) don't beat macd5 net and die to costs the same way.
- **H3 — no real edge / too thin to clear IG spread:** CONFIRMED, HIGH. Frictionless edge is only ~0.03–0.07R/trade. Spread sweep:
  - macd3 (LIVE): PF 1.11 → **0.72** @3pt → **0.54** @5pt → 0.41 @7pt
  - macd5 (best): PF 1.24 → **0.85** @3pt → **0.67** @5pt → 0.53 @7pt
  IG Crude DFB spread ~2.8–6 pts. At a realistic ~5pt, even the BEST exit is PF 0.67 and LIVE macd3 is PF 0.54 — **the live PF 0.38 sits squarely in the 5–7pt-spread+slippage band.** The backtest REPRODUCES the live bleed from costs alone.

**Self-critique:** Yahoo CL=F (continuous front-month, mid-price) ≠ IG Crude DFB (real spread, financing on holds, different contract). Flat per-trade points deduction ignores variable spread + slippage + overnight financing on multi-bar holds → my cost model is if anything OPTIMISTIC. 58d/one regime, but agrees with live across two independent samples (live n=12 vs backtest n=356). Did not replicate the live BE+trail exactly; the trail-style variants approximate it and also fail on costs.

**Conclusion (most-supported):** Crude has **no edge that survives IG's spread** — it's the Cocoa / Germany-40 pattern (thin Yahoo-cash edge eaten by live costs). A better exit (macd5) is real but cosmetic; it does not clear costs. **Recommend DISABLE Crude** (bench like Germany 40, keep profile commented for re-enable), NOT retune. Retuning the exit is rearranging deck chairs on a sub-marginal entry.

**Next:** await user nod to disable. If kept, the only thing worth live-testing is macd5 — but expect it to still bleed. Re-validate on the IG-native candle archive once Crude has enough harvested history (Yahoo-independent check).

---

## 2026-06-25 — Forex breakout expansion: which other pairs clear the bar? (answer: none)

**Question:** We run Donchian breakout on EUR/USD + GBP/USD only. Are there other liquid majors worth adding?

**Method:** Two-gate, Yahoo-only, ZERO IG cost.
1. `scripts/backtest_forex_breakout_candidates.py` — 11 pairs through the LIVE breakout config (N55, 2.0×ATR, HTF-filtered, Donchian-trail) + spread-cost sweep (0 / realistic IG spread / 2×). Incumbents as references, USD/JPY as negative control.
2. `scripts/backtest_forex_breakout_walkforward.py` — survivors across full/halves/quarters of 730d + 3-pip cost, looking for SIGN CONSISTENCY (the gate that killed USD/JPY).

**Gate 1 (single 365d window) results:** AUD/USD ✅ (PF 1.45 after spread, R:R 3.52 — best of basket, beat GBP/USD), AUD/JPY ✅ (PF 1.15, fragile). All others FAIL after spread: USD/CAD 0.66, NZD/USD 0.96, USD/CHF 0.88, EUR/GBP 0.57, EUR/JPY 0.83, GBP/JPY 0.44. Calibration caveat: EUR/USD (live winner) shows ❌ here (sim is Donchian-only, misses the RSI-extreme exit that banks EUR's live wins) → the sim is a CONSERVATIVE/noisy proxy; trust relative ranking + spread-robustness, not absolute levels.

**Gate 2 (walk-forward) — the decider — KILLS both candidates:**
- **AUD/USD: REJECTED.** Full 730d −12.27% PF 0.61; NEGATIVE in 3 of 4 quarters (Q2 −10.99% PF 0.16); only the recent half/Q4 positive. The 365d ✅ was a fluke that caught one good recent leg — the exact overfitting trap. HIGH confidence reject.
- **AUD/JPY: REJECTED.** Full breakeven (−0.12% PF 1.00), quarters alternate ✓✗✓✗, no sign-consistency.
- **Controls validate the method:** GBP/USD PASSES ALL 8 WINDOWS (full +8.41% PF 1.51, every quarter ✓ 1.24/2.21/1.31/1.14) = genuinely regime-stable. USD/JPY control FAILS (full −4.89%, 3/4 quarters ✗) = matches its −£93 live death. EUR/USD net-positive but weaker (full +4.76% PF 1.27, H2 + Q4 negative).

**Bonus — resolved the GBP/USD live-vs-sim contradiction.** Rolling-100 showed GBP/USD 0W/5L −£51 (looked like a losing breakout pair). Journal by era: **strat=breakout n=2 net −£2.79 (flat, and one was #199 closed by the restart bug we fixed today); strat=forex (RETIRED momentum) n=6 net −£42.64 (1W/5L)**. The −£51 is the dead momentum era, NOT breakout. GBP/USD breakout belongs (walk-forward robust); live sample just too small (n=2) to show it yet.

**Conclusion (HIGH confidence):** ADD NOTHING. Keep forex breakout at EUR/USD + GBP/USD. No major survives the walk-forward — AUD/USD (the scout's top pick) is a 2-year net loser that faked an edge on one window; everything else fails on spread alone. Process worked: the single-window scout tempted us toward AUD/USD, the walk-forward saved us from importing a −12% pair.

**Next:** none for deployment. Re-scout only if (a) we later stream a candidate to build IG-native archive data for a Yahoo-independent check, or (b) far more breakout-era live samples accumulate on the two incumbents to confirm forward. New harnesses: `backtest_forex_breakout_candidates.py` (+ AUD legs added to `backtest_forex_breakout_walkforward.py`).

---

## 2026-06-29 — "Should we broaden the EPIC count? Only the same ones fire."

**Question (decomposed):** (Q1) Is it true only the same EPICs fire? (Q2) If concentrated, is the binding constraint the EPIC *count*, the *caps*, or the *strategy gate*? (Q3) Would adding EPICs diversify firing, or just add correlated/break-even noise?

**Hypotheses:**
- H1 — too few EPICs configured; broadening would spread the firing. 
- H2 — count isn't the constraint; the ADX gate concentrates fires into whatever's trending, and the firing universe is one correlated equity-index cluster, so more indices = same bet N×.
- H3 — the caps (screener 11 / MAX_POSITIONS 8) bind, so adding EPICs just reshuffles, not broadens.

**Evidence (live journal + logs, this date):**
- **Firing is broader than it feels: 11 distinct markets fired in ≤14d, 14 across the rolling-100.** But VOLUME concentrates in the top names — Wall St (10 in 14d), Gold (6), Crude (5, now disabled), S&P (5), FTSE (4). Perception "same ones fire" = volume concentration, not narrow participation. (HIGH)
- **Dormancy is overwhelmingly ADX-too-low: 10,973 `ADX too low` HOLDs vs ~2,640 for ALL other reasons combined** (RSI 983+878, EMA-unaligned 779). Markets sit out because they're not trending — NOT because too few exist. Adding more EPICs of similar character just adds more ADX-too-low dormancy. (HIGH)
- **The firing universe is a correlated equity cluster:** Wall St, S&P, NASDAQ, FTSE, HK, Japan, Germany, AI Index are ALL equity indices; the high-volume firers (Wall St/S&P/NASDAQ) are co-moving US large-cap. "Same ones firing" = the US-equity cluster trending together. More indices ≠ diversification — it's the cluster-amplifier risk the existing correlation filter watches. (HIGH)
- **Count is not the binding constraint:** 13 configured, 11 active (screener cap 11), MAX_POSITIONS 8, and 0 positions open at check time → not cap-saturated. With cap 11<configured 13, adding EPICs without raising the cap just rotates the active set. To truly broaden firing you'd raise BOTH caps → MORE simultaneous correlated equity positions in an equity trend = concentration, the opposite of the goal. (HIGH)

**Self-critique / what would disprove:** If non-firing markets were held out by position-cap saturation (not ADX), then count/cap WOULD bind — but 0 open positions + ADX-too-low dominating the HOLD log refutes that. Simpler explanation for the perception: Telegram-alert salience (the big US names fire most and are most memorable) — the data says 11 markets actually fired in 14d. Caveat: the rolling-100 spans the disabled-market era (Crude/USD-JPY inflate the "14 fired" count); genuine active universe ≈ 11.

**Conclusion (most supported: H2, HIGH):** The constraint is NOT the number of EPICs — it's (a) the ADX gate correctly concentrating fires into trending markets (the edge working as designed; the recurring lesson says don't clip it) and (b) the firing universe being one correlated equity cluster. **Broadening the COUNT for its own sake does not help and adds cluster risk.** "More fills" is the wrong objective — the ADX-too-low dormancy is the bot correctly avoiding chop. RULED OUT: H1 (count) and H3 (caps) as binding.

The ONLY broadening with a real rationale = adding genuinely *uncorrelated* edges (commodity / rates / non-USD FX that don't co-move with US equities), and every candidate must clear the per-EPIC backtest bar FIRST (per `feedback_per_epic_profitability` + the 06-22/06-25 "add nothing" findings above — both prior expansion scouts were killed by walk-forward). Adding more indices is explicitly NOT that.

**Next actions (zero IG cost, deferred to the 2026-07-24 v2 review unless user wants sooner):** if diversification (not "more action") is the real goal, run a per-EPIC archive/Yahoo backtest on a shortlist of UNCORRELATED candidates (e.g. Silver, NatGas — profiles already exist but disabled; a non-USD-bloc FX cross; a rate) and only adopt any that clear PF + walk-forward sign-consistency. Otherwise: no change — the concentration is the gate + correlation, both working as intended.

---

# Research: Other Forex breakout candidates? (2026-07-08)

## Question
EUR/USD breakout just flipped to shadow (`909c43d`). GBP/USD is the sole live
forex breakout. Are there OTHER liquid pairs whose Donchian breakout edge is
robust enough to run live?

## Hypotheses
- H1 GBP-crosses / JPY-crosses trend well (GBP/JPY "dragon", AUD/JPY, EUR/JPY).
- H2 Commodity-dollar bloc trends on rate/commodity cycles (AUD/USD, NZD/USD, USD/CAD).
- H3 Haven pairs (USD/CHF).
- H4 (null) No pair beats GBP/USD after IG spread — the forex edge is
  idiosyncratic to GBP/USD (per-EPIC-profitability prior; the EUR/USD lesson).

## Evidence
Reused live breakout engine (N55 / 2.0×ATR / HTF-filtered / Donchian-trail),
Yahoo 1h, ZERO IG API cost. Two gates: (1) 365d spread-cost scout; (2) 730d
walk-forward sign-consistency (the gate that killed USD/JPY & exposed EUR/USD).

**Gate 1 — 365d scout (PF after realistic IG spread):**
- ✅ GBP/USD 1.54 · AUD/USD 1.57 · NZD/USD 1.14 (marginal)
- ❌ EUR/USD 1.00 · USD/JPY 1.01 · USD/CAD 0.95 · USD/CHF 0.89 · AUD/JPY 1.07
     · EUR/JPY 0.73 · EUR/GBP 0.47 · GBP/JPY 0.45
- Controls behaved: USD/JPY, EUR/GBP, GBP/JPY all fail.

**Gate 2 — 730d walk-forward (FULL + 4 quarters):**
- **GBP/USD ✓✓✓✓** — every window green, PF 1.51 full (1.36/1.73/1.30/1.25). Robust.
- **AUD/USD ✗** — FULL −10.22% PF 0.68; older half −13.33% PF 0.37; Q2 −10.93%
  PF 0.12. Only recent half/Q4 green → its 365d pass was RECENCY BIAS (scout
  window = its one good stretch). Reject.
- **NZD/USD ✗** — FULL −3.50% PF 0.88, Q2 −5.14%. Not sign-consistent. Reject.
- **AUD/JPY ✗** — FULL flat PF 1.02, half-1 negative, Q2 −4.40%. Reject.
- **EUR/USD** — FULL +4.40% but half-2 & Q4 negative → independently CONFIRMS
  the shadow flip (edge decayed recently).

## Confidence
**HIGH** that no tested pair joins GBP/USD live. Every non-incumbent candidate
collapses in walk-forward; GBP/USD is the only regime-stable forex breakout.
Caveats (don't over-read levels): Yahoo cash ≠ IG DFB; flat per-trade spread;
730d is still short for a breakout edge. But sign-consistency is unambiguous.

## Verdict / Next steps
H4 (null) WINS — the forex breakout edge lives in GBP/USD alone. Do NOT add any
pair. AUD/USD is the cautionary case: a single-window scout will flag it, only
walk-forward exposes it. Watch GBP/USD for its own fade at the 2026-07-24 review
(its half-2 PF 1.27 < half-1 1.72 — still green but softening). If ever revisited,
go straight to 730d walk-forward, skip the 365d scout (it snoops).

---

# Research: The 2026-07 losing streak — signals or conditions? (2026-07-13)

## Question
Last 15 closed = -£79.36, 4W/11L. Are the signals/entries faulty, or is it market conditions?

## Hypotheses
- H1 Market conditions: chop. Signals fire correctly; the regime is unfavourable.
- H2 Signal quality: ADX threshold too loose (entering marginal trends).
- H3 Structural flaw: Wall Street traded OUTSIDE US cash hours (thin liquidity).
- H4 Normal variance in a ~40%-WR system.

## Evidence
**H2 REFUTED (HIGH).** ADX-at-entry vs outcome, last 25 momentum trades: EVERY bucket
negative — <30 -£4.06 · 30-35 -£32.62 · 35-40 -£18.63 · **40+ -£55.44 (WORST)**.
Raising the ADX gate would NOT have helped; the highest-ADX entries lost the most.
(Another tick for the recurring "intuitive filter" lesson.)

**H1 SUPPORTED (HIGH).** Exit-reason mix, last 20 closed: fast momentum deaths dominate —
"MACD histogram negative" 6 trades -£51.75, "Stop/limit hit" 12 trades -£73.47. Entries
dying in 5-30 min = whipsaw. RSI-at-entry clustered 46-54 (no conviction). IG-archive
run of the LIVE analyze() over Wall St (5952 candles, Jun12-Jul13) is negative in BOTH
sessions -> the last month genuinely is a poor momentum regime. Losses are mostly SMALL
(risk mgmt cutting fast) -> -£79 ~ -0.85% of balance. Not a crisis.

**H3 CONFIRMED (HIGH) — the real finding.** Wall Street split at the 14:30 UTC US cash open:
- JOURNAL (full history): IN cash n=18 **+£126.78** 9W/9L (50% WR) ·
  OUTSIDE n=15 **-£77.22** 2W/13L (**13% WR**). Outside is negative in EVERY month
  with data (Mar ✗ -17.46, Jun ✗ -27.07, Jul ✗ -32.69) = SIGN-CONSISTENT.
- IG ARCHIVE (independent, live analyze()): OUTSIDE n=22 PF **0.30**, 18% WR, negative
  both months · IN cash n=13 PF 0.85, 38% WR. Outside is ~3x worse by PF.
- Wall Street's entire lifetime profit is its cash-session trades; out-of-hours has given
  back 60% of it. S&P corroborates (IN +£6.33 / OUT -£33.48). NASDAQ does NOT (OUT +£32.88)
  -> the effect is Wall-St-specific (+weakly S&P), NOT a blanket "no US index out of hours".

**WHY THIS SURVIVED SO LONG — methodological blind spot (HIGH).** `src/backtest.py`
TICKER_MAP maps indices to Yahoo CASH indices (^GSPC, ^NDX). A cash index has NO
out-of-hours bars — it only prints 14:30-21:00 UTC. But IG's Wall Street DFB quotes
~24/5 off Dow futures, so the LIVE bot takes trades (08:30, 09:25, 09:35 UTC) in a
session our Yahoo backtests literally cannot see. Every index backtest we've run has been
implicitly cash-only. The IG candle archive is the ONLY source that can see this.

## Verdict / Next steps
BOTH, but they're separable. The streak = mostly genuine chop (H1) + variance (H4), and
the signals are functioning as designed (no bug; H2 refuted). BUT the drawdown exposed a
real, sign-consistent structural leak (H3): Wall Street out-of-hours.
PROPOSED: session filter on Wall Street — only trade 14:30-21:00 UTC. Qualifies under the
"restrict only when data condemns it" rule (condemned in journal AND archive, every month).
Prefer config (`MarketConfig.trading_hours_utc`) over architecture. Do NOT generalise to
NASDAQ. Also: index backtests should be re-run off the ARCHIVE, not Yahoo, for anything
session-sensitive.

## ⚠️ CORRECTION (same session, 2026-07-13) — H3 RETRACTED

The H3 "Wall Street out-of-hours" finding above is **WRONG** and is retracted. I read journal
`entry_time` as UTC. The container runs **TZ=Europe/London**, so `datetime.now()` — and hence
journal times AND streamed-candle timestamps — are **BST (UTC+1)** in summer.

Re-run on the correct clock (US cash 14:30-21:00 UTC = 15:30-22:00 BST):
- Journal: IN cash n=12 +£70.95 (58% WR) · OUTSIDE n=21 **-£21.39** (19% WR) — a much smaller
  leak, and **NOT sign-consistent** (Mar ✗-20.03, Jun **✓+40.09**, Jul ✗-41.45).
- The two biggest winners I'd credited to the cash session are actually PRE-OPEN on the correct
  clock (#206 +£45.20 @13:55 UTC, #172 +£32.36 @14:00 UTC). A cash-hours-only filter would have
  KILLED both — the classic clip-the-tail error.
- **IG ARCHIVE (independent, live analyze(), 35 signals) REFUTES the split:** before 12:00 UTC
  -316.7 pts (PF 0.41) vs from 12:00 UTC -316.1 pts (PF 0.66) — both ~equally negative, neither
  sign-consistent. Per-hour n=1-4 → any boundary drawn here is data snooping.

**Corrected verdict:** NO session filter is justified. Wall Street momentum is unprofitable
across ALL hours in the recent window → this is the REGIME (chop), i.e. H1, not a session flaw.
H1 + H4 stand; H2 stays refuted; **H3 is dead.**

**REAL BUG found en route (undeployed):** `MarketConfig.trading_start/trading_end` are documented
as UTC hours (and the per-market comments describe UTC cash sessions) but both gates
(main.py ~810, ~1231) compare against `datetime.now().hour` = **BST**. So every market's window
runs **1h early Mar-Oct**, correct Nov-Feb. Fix = gate on UTC; note it shifts every market's
effective window by an hour in summer.

**META-LESSON: verify the clock before any time-of-day analysis.**

---

## 2026-07-14 — FTSE min_confidence sweep: is 0.60 vs 0.55 the lever? (VERDICT: NO — don't change it)

**QUESTION.** Healthcheck found FTSE arming a pullback at 56% confidence, waiting for the
retrace, getting a good fill level — then binning it on `Confidence too low: 56% < 60%`.
Same shape as the Gold `487959f` finding (a gate cutting its own net-winners). Asked: sweep
all confidence levels, find FTSE's sweet spot.

### Finding 0 — the premise was wrong (HIGH confidence, code-verified)
FTSE's `min_confidence` is **already 0.55** (config.py:693). The 60% came from the REGIME layer:
`main.py:1223-1227` → `min_confidence = max(market_config.min_confidence, regime_params.min_confidence)`.
FTSE was in `TRENDING_HIGH` (ADX 46, high vol), whose `RegimeParams.min_confidence = 0.6`.
Consequences:
- **Lowering FTSE's config value is a NO-OP.** `max()` means the regime floor wins. To let a 56%
  signal through in TRENDING_HIGH you must edit the GLOBAL `REGIME_PARAMS` table — which moves
  all 12 markets, not FTSE. Blast radius is the opposite of what "tweak FTSE" implies.
- Only *raising* FTSE's config above 0.60 binds in every regime.
This is the same misread that nearly shipped as "option A" in the Gold work. Check the effective
gate, not the config field.

### Finding 1 — there is NO sweet spot to find; the parameter is not continuous (HIGH)
FTSE confidence is **bimodal: 0.56-0.57 or 0.72-0.78, nothing in between.** Cause is structural,
not FTSE-specific — `_calculate_confidence` (strategy.py:512) sums a **binary** macd_factor
(0.0 or 0.2) with a quantised htf_factor (0/0.1/0.2) and two small continuous terms. HTF alignment
is mandatory (`require_htf`), so htf is pinned at 0.2 and **macd is the only thing that moves the
number materially.** Verified on all 11 archive signals: every 0.56 has macd_hist < 0; every
0.74-0.76 has macd_hist > 0. Therefore:
> **`min_confidence = 0.60` for FTSE is not a confidence gate at all — it is a
> "require MACD histogram already positive" gate wearing a disguise.**
Every threshold in (0.58, 0.72) is IDENTICAL. A "sweep for the sweet spot" is meaningless: there
are exactly TWO reachable policies — take MACD-unconfirmed entries (T<=0.57), or don't (T>=0.58).

### Finding 2 — the two data sources DISAGREE on which policy is better, and both are underpowered (LOW)
| source | conf < 0.60 (macd-neg) | conf >= 0.60 (macd-pos) |
|---|---|---|
| Yahoo ^FTSE 5m, 60d (n=8 trades) | −30.1 pts, PF **0.02** | +78.0 pts, PF **33.2** |
| IG archive 5m, ~1mo (n=2 trades) | +6.4 pts | **0 trades** (pullback dropped all 4 signals) |
| LIVE journal, FTSE all-era (n=17) | **+£4.46** (n=7) | **−£36.38** (n=10) |
| LIVE journal, FTSE current-era (n=5) | −£5.02 (n=2) | −£32.68 (n=3) |
Yahoo says MACD-confirmed is far better. Live says the opposite. Archive says nothing. PF 33.2 and
PF 0.02 on n=4 each are not statistics, they are anecdotes. **Do not pick the source that fits the
story.**

**Yahoo is the LEAST trustworthy source here, and we already knew that.** config.py:696-705 records
the precedent verbatim: Yahoo FTSE gave "long is robust (PF 10.26)" — live FTSE then went 1W/9L.
Yahoo ^FTSE is the CASH index: no IG DFB spread, no funding, no gaps. It systematically overstates
FTSE edge. Note this run reproduces exactly that failure: Yahoo shows FTSE longs **+48 to +78 pts
profitable** over the window in which live FTSE **lost £82**.

### Finding 3 — confidence does not discriminate ANYWHERE, era-controlled (MEDIUM-HIGH, best-powered result)
Whole book, current era only (post-06-15, so FTSE long-only + pullback are in force):
- conf < 0.60 : n=17, −£26.73, **PF 0.73**
- conf >= 0.60: n=52, −£119.07, **PF 0.75**
Indistinguishable. (A naive all-era split showed 0.88 vs 0.75 and looked like a real effect — that
was an **era artifact**; it evaporates once you control for config era. Textbook confounding.)
**Confidence carries no information about outcome in this bot.** Not for FTSE, not for the book.

### Finding 4 — the real FTSE problem is not the gate (HIGH)
FTSE **current era: 5 trades, 0W/5L, −£37.70.** All-time −£82.29 (1W/9L over 12t at healthcheck;
17 closed trades in full history). Both confidence buckets are negative in the current era. No
threshold rescues a market with no edge — you cannot filter your way out of −EV. Per the per-EPIC
profitability principle, **FTSE's live question is shadow/disable, not tuning.**

### Finding 5 — pullback-entry and the confidence gate select OPPOSITE populations (MEDIUM)
MACD-confirmed (high-conf) = momentum already moving = price does NOT retrace. In the IG archive
**all 4 high-conf signals were dropped by pullback-entry as runaways** (never retraced 0.5xATR
within 6 candles) → raising FTSE to 0.60 would have produced **zero trades that month**. Yahoo saw
4 of 8 survive. So "raise min_confidence to 0.60" plausibly means "FTSE stops trading", which is a
disable-by-side-effect, not a tuned gate. If we ever DO want 0.60, the pullback window must be
revisited in the same change.

### VERDICT — change nothing about min_confidence.
1. No sweet spot exists (bimodal → the knob has two settings, not a curve).
2. The two sources disagree in DIRECTION and both are n<10.
3. The source favouring a change (Yahoo) is the one with a documented history of overstating FTSE
   edge, and it overstates it again in this very window.
4. The best-powered test (era-controlled, n=69) says confidence is uninformative.
5. This is precisely the recurring lesson: an intuitive filter tweak, refuted by data.

### Open items for review (2026-07-24)
- **FTSE live viability** — 0W/5L current era, −£82 all-time, worst market in the book. Shadow or
  disable? This is the real decision. (Do NOT pre-empt: bake phase.)
- **Regime/config incoherence (structural, worth fixing regardless of FTSE):** whether a market
  requires MACD confirmation is currently an emergent side-effect of `max(config, regime)`, flipping
  on a regime classification refreshed once a day from 30 hourly candles. The same signal is gated
  or not depending on a stale read. If MACD-confirmation is wanted, it should be an EXPLICIT
  per-market flag, not an accident of two tables colliding.
- **Wasted pullback arm:** main.py arms a pullback for signals that cannot clear `min_confidence`
  (checked later), so `rejected_signals` also misattributes out-of-hours rejects as confidence
  rejects. Cheap fix: test the gate at arm time.

Scripts: `scripts/backtest_ftse_confidence_sweep.py` (archive + yahoo; live `analyze()`, live profile
stop, 1.0pt IG spread verified via `get_market_info`, BST->UTC clock correction applied).

---

## 2026-07-14 — "Can EVERY instrument be made profitable with the right per-EPIC settings?" (VERDICT: NO — but it found 1 real re-enable candidate)

**QUESTION (user).** "We're disabling/shadowing quite a few EPICs, yet there must be a way to be
profitable either shorting or going long on all instruments. It's just finding the right settings
per EPIC, right?"

**THE TEST.** 21 markets (11 live w/ a Yahoo proxy + 10 disabled). For each: grid-search the real
levers (ADX 20-40, stop 1.0-2.5xATR, R:R 1.5-3.0, direction BOTH/BUY/SELL, MACD-exit on/off = 360
configs) on an IN-SAMPLE half; take the single best config by IS P&L; score THAT config on an
OUT-OF-SAMPLE half it never saw; then check quarter-by-quarter sign consistency across all 8
quarters. Yahoo 1h/730d. **Spread charged as a fraction of price from LIVE IG bid/offer** (free
`get_market_info` snapshots) — the control Yahoo backtests lack. Run at 1.0x / 0.5x / 0.0x cost.
Scripts: `scripts/backtest_per_epic_walkforward.py`, quarter check in scratchpad.

### Result 1 — finding profitable settings is TRIVIAL and therefore MEANINGLESS (HIGH)
**A profitable in-sample config was found for 21/21 markets. Every single one.** Including Natural
Gas, Cotton, Soybeans, USD/JPY — everything already disabled for bleeding money live. This is not
evidence of edge; it is what a 360-config grid search *does* to a finite sample. **The user's premise
is true and worthless in the same breath: yes, settings exist that make any instrument profitable —
in the past.**

### Result 2 — out-of-sample, the search is a coin flip (HIGH)
Best-IS configs that stayed profitable OOS: **12/20 (60%)**. Binomial P(>=12/20 | p=0.5) = **0.25**
— **statistically indistinguishable from chance.** The grid is mostly harvesting noise.

### Result 3 — the LIVE vs DISABLED split is the real signal (MEDIUM, p=0.065)
| group | OOS survival |
|---|---|
| **LIVE markets** | **9/11 (82%)** |
| **DISABLED markets** | **3/9 (33%)** |
Fisher exact **p = 0.065** — suggestive, not conclusive at 0.05, but directionally clear:
**the markets the bot disabled are precisely the ones where tuning does NOT transfer.** The disable
decisions were correct, not defeatist. Edge is a property of the MARKET, not of the settings.

### Result 4 — COST is the mechanism, and it is decisive (HIGH — the money finding)
Natural Gas, best IS config, OOS:
- **zero cost: +41.91%, PF 1.49 ("SURVIVES")**
- **real IG spread: −31.34%, PF 0.49 ("FAILS")**
**The spread flips the sign.** Same for US 2-Year (+0.30 -> −0.13). NatGas spread is **34.8 bps**;
NASDAQ's is **0.68 bps** — a 50x difference. You cannot tune your way past a cost floor: no ADX,
stop or R:R setting recovers 35bps a round trip. **This is exactly the Yahoo fantasy, quantified** —
and exactly why Yahoo once reported FTSE longs at "PF 10.26" right before live went 1W/9L.
Disabled-market survival: 5/10 at zero cost -> **3/9 at real cost.** Cost does the killing.

### Result 5 — quarter-by-quarter (the AUD/USD test) separates edge from tails (MEDIUM-HIGH)
| market | PF | quarters green | read |
|---|---|---|---|
| **Russell 2000** (disabled) | **1.84** | **7/8**, evenly spread | **REAL — best re-enable candidate** |
| **Copper** (disabled) | **1.98** | **7/8**, evenly spread | **REAL edge — but STRUCTURALLY INACCESSIBLE** |
| NASDAQ 100 (live) | 1.70 | 7/8 | real — validates keeping it |
| S&P 500 (live) | 1.51 | 7/8 | real |
| Silver (disabled) | 1.51 | 5/8 | **tail/recency-dependent** — Q7+Q8 do all the work (the AUD/USD shape) |
| Gold (live) | 2.05 | 5/8 | tail-dependent (Q6 = +15.2 of +35.9) |
| NY Cocoa (live) | 2.23 | 5/8 | **Q1 alone = +49.6 of +69.7 total** — one quarter is 71% of the P&L |
| FTSE 100 (live) | 1.18 | 6/8 | green but **+3.12% over TWO YEARS** = nothing, on the source that flatters FTSE |
| **US 10-Year** (LIVE) | 1.15 | **3/8** | **FAILS OOS at every cost level — disable candidate** |
| Natural Gas (disabled) | 1.09 | 3/8, last 4 quarters red | correctly disabled |

### VERDICT
**The premise is refuted.** It is not "just settings". Two hard constraints sit above the parameters:
1. **Cost.** A market whose spread eats its ATR cannot be tuned into profit (NatGas, Cotton, Crude).
2. **Whether an edge exists to find at all.** 21/21 tuneable in-sample; only ~half transfer, and the
   ones that transfer cluster in the markets already live.
**But the exercise PAID FOR ITSELF** — it surfaced Russell 2000, and it exposed a live market
(US 10-Year) that is failing.

### ACTIONS (review 2026-07-24 — bake phase, nothing deployed today)
- **Russell 2000 — strongest re-enable candidate.** PF 1.84, 7/8 quarters green, survives all three
  cost levels, OOS +10.5%. Confirm on the IG archive (live instrument) before enabling.
- **US 10-Year — disable candidate.** LIVE, 3/8 quarters green, fails OOS at 1.0x/0.5x/0.0x cost,
  and it has the worst cost structure of any live market bar none (3.69 bps, ATR/spread 1.8x).
- **Copper — leave disabled, but record WHY it is special:** its edge is REAL (PF 1.98, 7/8 green).
  It is blocked by STRUCTURE, not by edge — IG min deal size 1.0 means stop x size always breaches
  the £45 cap (see `28dde62`). If IG's min size ever drops, Copper is the first thing to revisit.
  **This is the cleanest proof that "right settings" is not the only binding constraint.**
- **Cocoa — watch.** Live, but 71% of its 2y backtest P&L is a single quarter. Tail-dependent.
- **FTSE — reinforces the same-day confidence finding:** +3.12% over 2 years on the source known to
  flatter it. Shadow/disable remains the honest call.

### CAVEATS (do not skip)
- **1h timeframe; live trades 5m** for most of these. **The CONFIGS DO NOT PORT to live as-is.** This
  answers "is edge transferable?", NOT "here are your new settings". (Internal sanity check: the grid
  chose ADX 20 — the floor — almost everywhere, which independently reproduces the known
  "1h wants LOWER ADX than 5m" fact. The harness behaves; the numbers are timeframe-specific.)
- Single IS/OOS split + an 8-quarter sign check, not a full rolling walk-forward.
- Yahoo cash != IG DFB: no overnight funding modelled; FTSE flattery documented.
- Spread snapshots for HK/Cocoa/NatGas/2Y/10Y were taken OUT of session, where IG widens the quote —
  so those markets are charged a CONSERVATIVE (too-wide) spread. HK "FAILS" at 1.0x but SURVIVES at
  0.5x, which is the likely truth; it is fine, and its live +£68 agrees.

---

# 2026-07-24 — v2 Review (full agenda run; deployed a41013e)

## Question
Four-week scheduled deep review. Book since 06-26: −£190.08 / 51 trades; rolling-100 −£238.61.
Where is the bleed, which agenda decisions does the accumulated data support, and is go-live green?

## Headline finding: Wall Street 0W/11L −£125.35 since 06-26 — and the mechanism
Hypotheses considered: (H1) chop-session variance, (H2) session/hours leak, (H3) structural
entry/exit incoherence. **H3 CONFIRMED (HIGH):** 5 of the 11 losses (−£86.54) were conf≈0.56
entries (the bimodal MACD-not-positive mode) exiting via "MACD histogram negative for 3 candles"
within 5–11 min. The entry pre-check blocked only 3/3-armed (exit already true); entering at 2/3
armed self-destructs one candle later. Counterfactual on REAL post-cap trades (archive MACD at
entry): the 2/3 gate blocks 11 trades netting **−£92.85** (9 losers, 2 small wins) and touches
none of the kept winners (+£249). Sim A/B: PF up on 5/7 index markets (DOW 0.73→0.75, S&P
1.20→1.23, NDX 1.72→1.78, FTSE 1.07→3.07, HK 0.83→0.87; cost: Japan 1.74→1.61, AI n-too-thin).
**DEPLOYED** (strategy.py 2/3 pre-check). NOT confidence-gate tuning — the FTSE-sweep refutation
of conf-tuning stands; this is entry/exit coherence.

## Gold (item 10): exit params, not entry gates (HIGH on direction)
Live since 06-26: 71% WR yet −£13.94 (wins avg +£4.53, losses avg −£13.66). IG-archive sweep
(n=27, 6wk): be_trigger 0.7 + trail 2.0 → PF 1.21→1.50, sumR +1.85→+4.13. The 0.5 trigger was
Yahoo-derived (05-31) — another proxy artifact. ADX-sustained gate REFUTED (PF 0.85); ranging-exit
neutral (kept). **DEPLOYED** (gold profile be 0.7, trail 2.0).

## Shadow mode (new MarketConfig.shadow_only) — reuses E1 benched_outcomes plumbing
Full live pipeline (all gates incl. regime floors, hours, cooldowns, calendar), no orders;
signals → rejected_signals 'Shadow-only' + benched_outcomes bench_type='shadow', auto-resolved.
- **FTSE → shadow** (agenda 13): post-cap −£73.81 2W/9L; 3rd condemnation = IG-archive replay n=3 PF 1.07.
- **AI Index → shadow** (agenda 4/17): live 0W/2L −£52.36 BUT both losses predate long-only;
  BUY-only archive replay n=4 PF 1.93 (+28) at assumed 3pt spread. Mixed → measure, don't guess.
- **Russell 2000 → re-added as shadow** (agenda 14 + E2): WF PF 1.84 @1h but configs don't port
  and NO 5m archive exists (unsubscribed while disabled). Shadow builds the archive AND runs the
  memoryless-thesis test. Thesis predicts it survives; WF agrees; live pipeline will arbitrate.
- CAVEAT: shadow rows skip order-mechanics gates (IG stop clamp, sizing, spread/cluster) — mild
  optimism vs live fills; compare in R-multiples, subtract spread mentally at readout.

## Disables (HIGH)
- **US 10-Year** (agenda 15): WF 3/8 quarters, fails OOS all cost levels, ATR ~1.8× spread;
  last 3 CLOSED = −£23.52/−£24.00/−£24.00, inert since 06-22. DISABLED.
- **NY Cocoa** (agenda 6): 4 trades ever, inert 15wk (rule said drop at 8wk). DISABLED.

## Forex (agenda 11 + EUR/USD decision)
- WF re-run (730d Yahoo, 3-pip cost): GBP/USD full PF 1.44 but quarters 1.49→1.70→1.01→**0.89**
  (first sub-1.0); live breakout since 07-08 −£18.66/3t (all PROVISIONAL/UNMATCHED — reconciler
  handles multi-day forex holds badly, flag for next review). **GBP → breakout_shadow_only.**
- EUR/USD shadow replay off 1h archive: 07-08 episode −1.04R (stop), 07-13 −1.05R (stop),
  07-23 still open +0.68R. Confirms demotion; stays shadow. **Zero live forex now.**
- MEDIUM confidence on GBP demotion (full-period edge real; this is a pause not a burial).

## Section A — 06-26 deploys all HELD (HIGH)
Cooldown: no Wall St re-entry-cooldown fires (rare-event, no regression). Re-adopt: 0 re-adopts,
6 debounce deferrals each confirmed next cycle, no false closes. Min-hold: no sub-1-candle
momentum exits since 06-26 (min observed ≈5m = 1 candle, by design).

## Section C leftovers
- Cluster filter: ALREADY ENFORCED since 07-06 (memory agenda was stale). Post-06-26 would-block
  bucket: 5 matched trades −£7.51 net (1 winner +£10.43); 07-21 blocks genuinely prevented 2
  entries. Evidence still mildly pro-enforcement. No change.
- E1: near-inert confirmed — exactly 1 bench in 4wk (HK quality-bench 07-22, resolved WIN +0.5R).
  Instrumentation works; keep accumulating. No cap change.
- ADX-ceilings/leg-filter: 3+1 would-blocks in 4wk, inert; left as dead-cheap observational.
- S&P tight-stop/low-ATR hypothesis (agenda 16): DISSOLVED — the tightest-stop trade (#250,
  4.52pt stop, full size) was the +£45.72 limit winner; interaction cuts both ways. No floor.

## Go-live gate (Section D): FAILED — stay DEMO
(1) fixes clean ✓; (2) rolling-100 positive ✗ (−£238.61); (3) infra clean ✗ (2.5-day silent
polling degradation 07-17→07-20, fixed a564583). Re-gate at next review ~2026-08-21.

## Open questions / next review
- Wall Street is the WATCH item: does the coherence gate stop the bleed? If still net-negative
  by ~08-21 → demote to shadow (it's the 3rd US index; NDX+SPX cover the exposure).
- Shadow readouts: FTSE / AI / Russell benched_outcomes R-sums; Russell 5m archive depth.
- GBP/USD quarterly WF refresh: did Q-current recover above 1.0?
- Reconciler: 15 stale PROVISIONAL/UNMATCHED rows (all forex/Gold, incl. #228/#242 multi-day
  GBP holds) — closed-trade queries undercount forex P&L. Fix candidate.
- Japan 225 puzzle: sim PF 1.74 (+90R×£23.5 ≈ +£90 over 6wk) but ~never trades live (n=2 ever).
  Why does live generate so few Japan signals vs sim? (Sim lacks regime floors/screener — but HK
  trades fine on the same profile.) Worth one forensic pass.

---

# 2026-07-30 — Gold pullback-entry replay: keep or kill? (VERDICT: KEEP — the streak was real but unrepresentative)

## Question
07-29/07-30: pullback filter dropped 6 would-be Gold winners and filled the 2
losers. Is pullback-entry (0.5xATR retrace, <=6 candles) destroying Gold edge?

## Hypotheses
H1 filter helps (June evidence: Yahoo PF 1.91->3.28; 06-30 archive "PB+55 beats NOPB+55").
H2 filter hurts — selection effect (drops runaways, fills reversals).
H3 regime-dependent — hurts only in strong trends; recent streak is that cell.

## Method
`scripts/backtest_gold_pullback_replay.py` — IG archive 5m (9,336 candles,
06-12->07-30 UTC-corrected, hole 07-17..20), live analyze() + gold profile,
live arm cadence, paired per-arm A/B: A = pullback fill at target; B =
immediate entry at signal close. Identical exits (stop 1.5xATR, 3R limit,
BE 0.7/lock 0.25, trail 2.0 post-BE, RSI 85/15), spread 0/0.4/0.8.
63 gate-passed arms: 45 filled (71%), 18 dropped.

## Results (spread 0.4)
- **A (live, pullback): −3.66R PF 0.66** vs **B (no filter): −5.97R PF 0.67**
  → REMOVING the filter would have been WORSE over the full 7 weeks. H1 holds.
- Paired fills (n=45): A −3.66R vs B −13.28R — retrace entry saves ~+9.6R of
  entry-price damage on the SAME trades. This is the filter's real value.
- Drops (n=18): B +7.31R, 89% WR forgone. **SELL drops: 11/11 winners +7.79R**
  (BUY drops: 7, −0.48R — correctly dropped). The whole cost is one cell:
  runaway SELL trends. H2's mechanism is real but SMALLER than the fill benefit.
- Halves x direction: BUY A>B in BOTH halves (robust). SELL: h1 A>B, h2 A<B
  (B PF 2.09 in h2 only) → the SELL-side pain is RECENT-regime-specific (July's
  sharp selloffs). H3 confirmed for the streak; not sign-consistent.
- ADX 45+ at entry: negative BOTH policies (A −5.66R / B −8.67R) — blow-off
  entries lose regardless of entry mechanics.
- **Both policies NET-NEGATIVE at real spread** — echoes live (Gold −£7.50
  rolling-100). Gold 5m momentum edge this window is ~zero; entry mechanics are
  second-order. The 06-30 "PB+55 beats NOPB+55" direction reproduces, on a
  weaker book.

## Self-critique / caveats
- **Partial tautology in the drop stat**: a drop happens BECAUSE price ran
  0.5xATR+ without retracing = the trade instantly working → dropped trades are
  conditioned winners (esp. with BE 0.7 + trail). "11/11 drops won" overstates
  the loss; the NET (A vs B, all arms) is the honest metric and favors A.
- Sim arms don't 1:1 match live arms (250-bar window vs live 100-candle deque;
  hourly-fresh HTF/regime vs live daily refresh) — paired A/B is controlled,
  absolute levels are MEDIUM confidence.
- Tempting hybrid "pullback BUY / immediate SELL" = +1.71R full-window BUT
  loses in h1 (−3.24R vs A −0.87R) — pure h2 recency, the AUD/USD trap shape.
  DO NOT ship without a longer walk-forward. Archive only 7wk; can't WF yet.

## Verdict & actions
- **KEEP pullback-entry on Gold. No config change.** (H1, MEDIUM-HIGH.)
- The 36h streak was the known, bounded cost of the filter hitting its worst
  cell (runaway SELLs in a selloff regime), not a broken mechanism.
- v3 (~08-21) additions: re-run this replay with +3wk archive (h2 SELL flip =
  watch item; hybrid only if it survives sign-consistency); note ADX 45+
  entries lose both ways (ties to refuted-ADX-ceiling lesson — observe only).
- Bigger open question than entry mechanics: Gold 5m momentum edge ~zero for
  7wk while the breakout observer would-have-caught both big moves (07-29,
  07-30 readouts logged). The v3 Gold momentum-vs-breakout readout is the real
  decision; this replay says don't rearrange the momentum deck chairs meanwhile.

---

# 2026-07-30 — Invesco DB Agriculture Fund (DBA DFB) as an ag play? (VERDICT: NO — cost-blocked)

User proposed adding IG's "Invesco DB Agriculture Fund (DFB)". IG app quote:
SELL 2744.9 / BUY 2753.1 = 8.2pts on 2749 = **29.8 bps round trip**.
Yahoo DBA ATR check (scratchpad dba_cost_check.py): median ATR = 6.8 bps (5m),
28.4 bps (1h), 101 bps (daily) → **ATR/spread 0.23x @5m, 0.95x @1h, 3.4x @daily**.
References: US 10-Year was DISABLED at 1.8x; DXY (worst tolerated, shadow-only)
is ~0.5x. A diversified low-vol ETF wrapped in a 30bps spread cannot clear the
cost floor at any timeframe the bot trades — the NatGas sign-flip mechanism.
HIGH confidence. Do not revisit this vehicle.

The diversification MOTIVE is sound (06-29 finding: uncorrelated edges are the
only broadening worth doing). Open next step (not yet run): cost-first scout of
IG's INDIVIDUAL ag futures DFBs never yet tested — Chicago Wheat, Corn, Sugar,
Coffee (ZW=F/ZC=F/SB=F/KC=F) — via free get_market_info spreads + the two-gate
Yahoo walk-forward that killed AUD/USD. Softs priors are bad (Soybeans/Cotton/
Cocoa/NatGas all culled), so expectations low; only a survivor earns shadow.

---

# 2026-07-30 — Four-grain ag scout: Wheat / Corn / Coffee / Sugar (VERDICT: ADD NOTHING)

Follow-on from the DBA rejection (same day, above). `scripts/backtest_ag_scout.py`
reusing the per-EPIC walk-forward engine: grid on IS half -> best config OOS ->
8-quarter sign consistency, Yahoo 1h 730d, real-spread costs. Spreads: Coffee
12.2bps MEASURED live (CO.D.KC.Month3.IP TRADEABLE); Sugar 31.5bps standing
5.0pt quote; Wheat 17 / Corn 19bps ASSUMED (frozen EDITS_ONLY at snapshot).

| candidate | IS (trap) | OOS @full spread | quarters | verdict |
|---|---|---|---|---|
| Chicago Wheat | +25.8% PF 2.78 | −4.19% FAILS (fails even at ZERO cost) | 3/8 | REJECT |
| Corn | +13.2% PF 5.14 (n=11) | −0.40% FAILS | 2/8 | REJECT |
| Coffee Arabica | +61.7% PF 2.89 | −0.54% FAILS (0.5x: +0.62% marginal) | 6/8 | REJECT |
| NY Sugar 11 | +15.0% PF 2.02 | −11.26% FAILS (fails at zero cost too) | 4/8 | REJECT |

Coffee is the near-miss worth remembering: genuinely cheap to trade (12bps,
ATR/spread ~10x at 1h — Gold-class cost structure) and 6/8 green — BUT the two
red quarters are the most RECENT two, so the OOS (recent) half is dead at
measured cost. Full-period +61% is all old-half trend (2025 coffee bull run).
The AUD/USD shape inverted: edge existed, then died. If coffee trends again
it's the first ag to re-scout (cost structure is fine; the edge is the issue).

Every IS grid looked spectacular (PF 2-5) and every OOS failed — Results 1+2 of
the 07-14 per-EPIC study reproduced on 4 fresh markets. Softs/ags record now
0-for-8 (Soybeans, Cotton, Cocoa, NatGas culled live; DBA, Wheat, Corn, Sugar,
Coffee rejected at scout). Fourth consecutive expansion scout to end ADD NOTHING.
The uncorrelated-sleeve goal remains legitimate but ags don't currently offer an
edge that survives costs + walk-forward. HIGH confidence; don't re-run before v4
unless coffee's trend visibly resumes.

## 2026-08-06 — Gold breakout cap-skip: did it cost money? + reconciler UNMATCHED root cause

**Q1: Did the £45-cap skip of Gold breakout entries cost money?**
- Only ONE skip episode ever: 2026-08-06 04:20–05:00 BST, 9× "Min size 1.0 × stop 46.0 risks £46.02 > £45.00" (ATR 23.0 → 2×ATR stop 46pt; skips whenever Gold ATR > 22.5).
- Replay vs candle archive: skipped signal = BUY @ ~4267.4 (re-entry minutes after the +£4.24 trail-stop exit at 4273). As of 12:50 BST: −10.6pts (−£10.62 at size 1.0) and still open, trail never ratcheted.
- **Verdict: skip SAVED ~£10.6 so far (position unresolved). n=1 — no evidence the cap is costing money. HIGH confidence on the replay, LOW on generalisation.**

**Q2: Why do breakout closes go UNMATCHED?**
- ROOT CAUSE (HIGH, probe-confirmed): reconciler 3h age-out keyed on ENTRY time (main.py ~2188) and get_provisional_rows windows on entry_time too. Any trade held >3h is "stale" the minute it closes. Momentum trades (minutes) reconcile; breakout holds (hours) never do.
- Probe: IG txn history CONTAINS exact matches for all 3 recent UNMATCHED rows (Gold open=4195.25 pnl £52.05 vs cached 53.45; Gold open=4268.76 pnl £4.24 vs cached 6.37; DXY open=9957 pnl −£3.70 exact). Matcher (openLevel±1.0 + direction) would have hit.
- Competing hypotheses ruled out: price-scale mismatch (levels match exactly), pageSize crowding (only 11 txns/72h, zero non-DEAL rows), matching-logic bug (matches found manually).
- Side-finding: get_recent_trades/get_stats_* filter status='CLOSED' → UNMATCHED rows (~+£76 net) invisible in all reporting.

---

# Gold: MOMENTUM vs BREAKOUT at live configs (2026-08-18)

**Trigger:** Gold flipped breakout → momentum → breakout on 08-18 after three poor
breakout trades (#289 −£27.17, #311 −£35.60). The 2026-07 pullback replay concluded
"Gold 5m momentum edge ≈ 0" and left momentum-vs-breakout explicitly open.

## Question
Decomposed into four sub-questions:
1. What does the LIVE record say, separated by strategy and config era?
2. On identical data at current live configs, which wins?
3. Is the comparison window a representative regime, or a bad patch?
4. Does either survive realistic costs?

## Hypotheses (formed before looking)
- **H1** Breakout is better; the recent losses are a bad patch in a real edge.
- **H2** Momentum is better; breakout's apparent edge was a DAY-HTF backtest artefact.
- **H3** Neither has an edge on Gold that survives costs; the flip-flopping is noise-chasing.

## Evidence

**Live journal, all time (zero modelling) — HIGH confidence**
| strategy | n | W/L | PF | total |
|---|---|---|---|---|
| `gold` (momentum) | 59 | 30/25 | **1.00** | +£0.44 |
| `breakout` (N55)  | 7  | 4/3   | 0.84 | −£11.80 |
| `default` (old era) | 7 | 1/5 | 0.01 | −£62.78 |

Momentum is break-even to three decimal places over the largest sample the bot has.
Eras barely overlap (momentum mostly pre-07-24, breakout all post-08-03), so this
does not by itself rank them — but it does establish momentum has no edge either.

**Head-to-head, IG archive 67d (56 trading days), both at live config — MEDIUM**
Momentum on native 5m; breakout on a 1h resample of the SAME candles; both gated by
the SAME DAY HTF (`.shift(1)`, no look-ahead). `scripts/backtest_gold_momentum_vs_breakout.py`.

    MOMENTUM live (pullback + 0.4pt)   n=40 WR=57% +0.53% PF=1.14  (+21.8pt, +2.83R)
    BREAKOUT live (0.286xATR slip)     n=12 WR=17% -4.49% PF=0.44
      breakout @ spread only           n=12 WR=17% -3.29% PF=0.53
      breakout @ zero cost             n=12 WR=17% -3.17% PF=0.54

Thirds: momentum +0.56 / −0.34 / +0.49 (2 of 3 positive); breakout 0.00 / −3.33 / −0.28
(0 of 3 positive). n=12 is far too few for a low-WR/high-payoff distribution.

**Regime context, Yahoo GC=F 730d 1h, same breakout config — MEDIUM**

⚠️ CORRECTED 2026-08-18 (same day). The first pass of this section reported PF 1.17
and "ex-top3 −14.98%". Both were WRONG: that run hit a transient Yahoo response with a
different price series (median 1h ATR 15.56 vs the stable 13.70 seen on three
consecutive later fetches), which inflated the ATR-scaled cost. Re-run twice with
identical output, native 1d HTF bars (live-faithful):

    zero cost        n=109 +36.67% PF=1.77
    spread only 0.4  n=109 +35.42% PF=1.74
    0.286xATR slip   n=109 +24.37% PF=1.44   <- live-equivalent
    2x slip          n=109 +12.07% PF=1.19
    Q1 0.52 | Q2 1.69 | Q3 3.03 | Q4 1.17
    total +24.37% | top1 +13.16% | top3 +26.16% | ex-top3 -1.78% | ex-top5 -10.49%

**Lesson worth keeping: a Yahoo-sourced backtest is not reproducible by assumption.**
An ATR-scaled cost turns a bad fetch into a plausible-looking verdict rather than an
obvious error. Re-fetch and re-run before quoting any number that drives a decision.

## Findings
1. **config.py's "DAY → PF 1.51" is essentially SOUND — HIGH.** Re-measured 1.44
   slip-charged; the gap is the 730d window rolling 5 days. The earlier claim in this
   file that it was a gross-vs-net error is retracted.
2. **The HTF ladder is NOT monotonic — HIGH.** Slip-charged: HOUR 1.16 (n=193),
   HOUR_4 **1.12** (n=166), DAY 1.44 (n=109). HOUR_4 is the worst rung, not the middle.
   Gold supports "DAY specifically wins", NOT "longer lookback is monotonically better".
   v3 item 15(b) leans on the monotonic reading and needs re-checking on the other 8.
   Method caveat: DAY uses native 1d bars, HOUR/HOUR_4 are resampled (no native 4h).
   Resampling flatters — resampled DAY reads 1.57 vs 1.44 native — so the DAY-vs-HOUR
   gap is if anything understated.
3. **Gold breakout is concentrated but less extremely than the profit-protection note
   suggests — MEDIUM.** Top 3 of 109 trades = +26.16% against a +24.37% total, so
   ex-top3 is −1.78%: near-flat, not deeply negative. Directionally consistent with the
   recorded "84% of two years' profit is three trades", but the tail is not the whole
   edge; ex-top1 is still +11.21%.
4. **Cost-sensitive but not cost-fragile — MEDIUM.** PF 1.77 → 1.44 → 1.19 across
   0 / 1x / 2x measured slip. Still positive at double the measured cost.
5. **On the last ~2 months momentum beat breakout on the same data — MEDIUM** (window),
   **LOW** (generality). n=12 breakout trades in the archive window; Q4 of the 730d run
   reads PF 1.17, so the archive window is a weak patch inside a mildly positive quarter.
6. **No long-window momentum comparator exists — gap, not an omission.** Yahoo serves
   5m for ~60d; the IG archive starts 2026-06-12. Not constructible from any source.

## Confidence in the hypotheses
- **H1 (breakout better) is the best supported on the LONG window, H3 (neither has an
  edge) on the SHORT one — and they are not in conflict.** Over 730d breakout is
  PF 1.44 slip-charged and survives 2x cost; over the last 67d it is PF 0.44 on 12
  trades while momentum is 1.14 on 40. Momentum has NO long-window evidence at all,
  and its long-run live record is PF 1.00 over 59 trades. So: breakout has the only
  demonstrated edge on Gold, and is currently in a bad patch.
- **H2 (breakout's edge was a DAY-HTF artefact) REFUTED.** The 1.51 reproduces at 1.44.
  My initial support for H2 came from the bad fetch and is withdrawn.
- **The practical read is still H3-flavoured for DECISIONS taken on a few trades:**
  a PF-1.44 process with 109 trades in 2 years produces long losing stretches by
  construction, so three bad trades carry no information.

## Self-critique
- Momentum's 5m vs breakout's 1h frames is a real asymmetry; a 5m-native strategy gets
  many more shots. Mitigated by charging each its own measured cost and comparing PF/R
  rather than trade counts, but not eliminated.
- Survivorship/data-snooping: the archive window was chosen by data availability, not
  by me — no cherry-picking. But it is ONE regime.
- Simpler explanation checked: is breakout's n=12 a harness bug? No — 110 trades/730d
  scales to ~10 per 67d, so 12 is the expected rate.
- What would disprove finding 4: another 100+ breakout trades on the live instrument.
  At ~10 per 67 days that is ~2 years away. This question cannot be settled by waiting.

## Next steps
- Do NOT re-derive Gold's strategy from the next few trades either way; both live
  samples are inside the noise band of a PF ≈ 1.0 process.
- Correct the PF 1.51 claim in `config.py:939-947` to the cost-charged 1.17.
- v3 agenda: the real Gold question is not momentum-vs-breakout but whether Gold earns
  a live slot at all, given £35.60 forced risk/trade from the 1.0 min deal size against
  a PF ≈ 1.0-1.17 process.

---

# v3 item 15(b) re-check: "longer HTF lookback is better" (2026-08-18)

**Trigger:** re-measuring Gold's ladder found it non-monotonic (HOUR 1.16 → HOUR_4 1.12
→ DAY 1.44). The recorded claim was "longer lookback won in 8 of 9 markets, FTSE the
sole exception", and it is what justified moving Gold and DXY to DAY on 08-13.

**Method:** `scripts/backtest_htf_ladder_recheck.py`. One Yahoo 730d 1h frame per market
(725d fallback where Yahoo's boundary rejects 730 — the trap the forex script already
works around); cost = 0.286×median ATR charged to every arm so arms differ ONLY in the
HTF gate; live breakout config N55/2.0×ATR/Donchian/htf_filter; live trend rule with
`.shift(1)`. Two claims separated deliberately, because the original conflates them:
**(A)** does DAY beat HOUR — justifies the config change; **(B)** is the ladder
monotonic — justifies the *principle*.

| market | HOUR | HOUR_4 | DAY | (A) | (B) |
|---|---|---|---|---|---|
| Gold | 1.16 (193) | 1.12 (166) | **1.44** (109) | DAY | not mono |
| US Russell 2000 | 0.98 (73) | 1.03 (64) | **1.47** (44) | DAY | monotonic |
| NASDAQ 100 | 0.77 (69) | 0.78 (63) | 0.99 (49) | DAY | monotonic |
| Japan 225 | 0.92 (70) | 0.97 (58) | 1.10 (48) | DAY | monotonic |
| Wall Street | 0.75 (72) | 0.91 (59) | 0.89 (48) | DAY | not mono |
| Dollar Index | 0.89 (224) | 0.88 (187) | 0.99 (118) | DAY | not mono |
| S&P 500 | 0.86 (70) | 0.97 (63) | 0.88 (54) | DAY | not mono |
| FTSE 100 | 1.13 (89) | 1.11 (79) | 1.10 (55) | **HOUR** | not mono |
| Hong Kong HS50 | **1.34** (61) | 1.13 (60) | 1.09 (48) | **HOUR** | not mono |

    (A) DAY beats HOUR:      7/9      (B) ladder monotonic:   3/9

## Findings
1. **(A) reproduces, but at 7/9 not 8/9 — HIGH.** FTSE is still an exception (−0.03,
   inside noise) and **Hong Kong is a NEW one and material: −0.25** (HOUR 1.34 → DAY
   1.09). HK's HOUR rung is the best HOUR reading in the whole book.
2. **(B) is REFUTED — HIGH. Only 3/9 ladders are monotonic.** HOUR_4 is frequently the
   worst or best rung rather than the middle. "Longer lookback is better" is not what
   this book shows; "DAY specifically tends to win" is. The distinction matters because
   the monotonic reading invites the next step (WEEK), which nothing here supports.
3. **Robustness: the exceptions are not a basis artefact — HIGH.** DAY is built from
   native 1d bars while HOUR/HOUR_4 are resampled, and resampling flatters (Gold DAY
   reads 1.57 resampled vs 1.44 native). Re-running the DAY rung BOTH ways flips no
   verdict: FTSE 1.13 vs 1.10/1.11, HK 1.34 vs 1.09/1.06, Gold 1.16 vs 1.44/1.57,
   Russell 0.98 vs 1.47/1.47, DXY 0.89 vs 0.99/0.95.
4. **Most of the book has no edge at ANY rung — HIGH.** S&P (0.86/0.97/0.88), NASDAQ
   (0.77/0.78/0.99), Wall Street (0.75/0.91/0.89) and DXY (0.89/0.88/0.99) are sub-1.0
   on every rung. For these "DAY beats HOUR" means "loses less". Choosing an HTF for a
   market with no edge is not a strategy decision.
5. **DXY runs LIVE breakout at DAY with a 730d PF of 0.99 — HIGH.** Independently
   corroborates the recorded live PF 1.01. It is also the source of 100% of the
   stop-amendment refusals (min stop 10.0 vs 20.0 entry stop, 8.0 spread).
6. **Russell is shadow-only at HOUR (0.98) when its DAY rung is 1.47 — MEDIUM.** The
   shadow is therefore measuring the weaker of the two configs. NB the recorded
   "Russell PF 1.11 at live HOUR" does not reproduce here (0.98) — different cost model.

## Self-critique
- n is small for the indices (44–73 trades over 2y); a 0.1 PF difference there is noise.
  Only Gold (109–193) and DXY (118–224) have counts worth ranking finely.
- The exceptions could be regime artefacts of one 2y window; no walk-forward run here.
- AI Index is unmeasured — no Yahoo ticker, archive starts 2026-06-12. 9 of 10 breakout
  markets covered, not all.
- What would disprove (B)'s refutation: monotonic ladders appearing under a different
  cost model. Cost is common to all arms per market, so it cannot create non-monotonicity
  on its own, but it can move rungs past each other where gaps are ~0.05.

## Next steps
- Restate item 15(b): "DAY beats HOUR in 7/9; the ladder is NOT a monotonic lookback
  effect (3/9)." Do NOT extrapolate to WEEK.
- Hong Kong must stay on HOUR — it is currently on the inherited HOUR default, so no
  action, but flag it against any future book-wide "move everything to DAY" sweep.
- Russell: set htf_resolution="DAY" so its shadow measures the config that would go live.
- DXY: PF 0.99 at its live config strengthens the case to bench it (v3 item 3/16).

---

# Go-Live Verdict — Full-Book Synthesis (2026-08-19)

**Scope.** Four measurement arms (breakout, momentum, instrument/cost structure, horizon, mean-reversion) plus three adversarial reviews (cost-model, sample-size, data-validity). Where a reviewer refuted or weakened a claim, the corrected version is what appears below. Zero IG REST calls; all work on Yahoo + the local IG 5m candle archive.

---

## 1. VERDICT

**Neither answer the question was framed around is correct on its own, and the honest composite is uncomfortable: there *is* a real, statistically significant GROSS breakout edge when the book is pooled (+0.162R/trade at N=20, cluster-robust t = +4.32, n=1348; +0.191R at N=55, t = +3.85, n=676), and the measured cost base of ~0.14R/trade sits *inside* the 90% confidence interval of that edge at every horizon tested — so cost is the binding constraint in the precise sense that it consumes essentially all of the edge, leaving a residual of +0.019R (N=20) to +0.048R (N=55) per trade that is statistically indistinguishable from zero (t = +0.50 and +0.97).** The strategy choice is not the limit — momentum and mean-reversion were tested and mean-reversion is *adequately powered and conclusively dead* (symmetric-barrier hit rate 49.50%, 95% CI 48.08–50.93%, n=4737), while breakout is the only arm with a defensible gross signal. But "cost is the binding constraint" does not translate into "fix the cost and go live", because the per-market triage that would tell you *where* to trade cannot be done with this data: no single market on the book is significantly above PF 1.0 at 90% confidence, the family-wise p-value for the observed best market (Russell, net PF 1.47) is **0.72**, and the flat 0.286×ATR cost convention is wrong per-market by ±0.207 PF — larger than the entire 0.11–0.33 PF effect it was being used to decompose. **So: the bot is not "just breaking even" in the sense of having no edge — it has a gross edge roughly the same size as its costs. It is breaking even in the sense that the post-cost residual is unmeasurable, and the paper-trading year did not and could not measure it** (the −£180.16 / 100-trade live result gives t = −0.39, 90% CI [−£945, +£585]; a genuine PF-1.44 edge posts a losing 100-trade run 14% of the time). Going live now is a bet on a quantity measured with ±100% relative error against a drag measured with near-certainty.

---

## 2. Gross vs Net — the tables that carry the argument

### 2a. BREAKOUT, 730d, live `htf_resolution`, live profile stop (Yahoo)

Cost columns: **flat** = the established 0.286×ATR-at-median convention (= 0.143R); **own-ATR** = each trade charged its own entry ATR (data-validity B8 / cost-model #11); **live-faithful** = cost-model's explicit model of live entry mechanics (market entry at candle close + half spread, broker stop filled at level, gap-through at open, daytime spreads).

| Market | HTF | n | Gross PF | Net (flat) | Net (own-ATR) | Live-faithful | Gap-fill fixed | 90% CI on net (flat) |
|---|---|---|---|---|---|---|---|---|
| **Gold** | DAY | 109 | 1.775 | **1.443** | 1.445 | **1.847** | 1.424 | [0.86, 2.22] |
| **GBP/USD** | DAY | 103 | 1.613 | **1.330** | 1.288 | 1.072 | 1.317 | [0.84, 1.99] |
| US Russell 2000 | DAY | 44 | 1.749 | 1.471 | 1.420 | 1.137 | 1.343 | [0.71, 2.76] |
| Hong Kong HS50 | HOUR | 61 | 1.568 | 1.341 | 1.304 | 1.262 | 1.198 | [0.50, 2.52] |
| FTSE 100 | HOUR | 89 | 1.340 | 1.129 | 1.089 | 0.984 | — | [0.63, 1.81] |
| EUR/USD | DAY | 107 | 1.297 | 1.083 | 1.038 | 1.240 | — | [0.66, 1.66] |
| Dollar Index (DXY) | DAY | 116–118 | 1.189–1.219 | 0.990–1.015 | 0.953 | **0.631** | 0.989 | [0.64, 1.52] |
| Japan 225 | HOUR | 70 | 1.082 | 0.921 | 0.868 | 0.784 | 0.849 | — |
| S&P 500 | HOUR | 70 | 1.021 | 0.856 | — | **1.195** | — | — |
| NASDAQ 100 | HOUR | 69 | 0.917 | 0.773 | — | 0.831 | 0.688 | — |
| Wall Street | HOUR | 72 | 0.903 | 0.747 | — | 0.867 | — | — |
| Crude Oil | DAY | 117 | 0.821 | 0.705 | — | 0.777 | 0.645 | — |
| AI Index | HOUR | — | **no Yahoo ticker — UNMEASURED at 730d** | | | | | |

Three independent harnesses reproduced the gross/net(flat) columns digit-for-digit, so the arithmetic is not in dispute. The *spread between the cost models* — up to 0.404 PF (Gold) and 0.384 PF (DXY) — is.

**Correction to the brief's own baseline (HIGH, reproduced by two reviewers):** five of the nine quoted "730d at live configs" figures were DAY-HTF measurements of HOUR-HTF markets. At their real live resolution: Japan **0.92** (not 1.10), NASDAQ **0.77** (not 0.99), Wall St **0.75** (not 0.89), S&P **0.86** (not 0.88), HK **1.34** (not 1.09). Gold, Russell and DXY were quoted correctly.

### 2b. BREAKOUT, POOLED, in RISK units — the only adequately-framed test

10 markets, 730d, clusters = markets resampled as units.

| N | n | Mean GROSS R/trade | t (cluster-robust) | Cost R/trade | Mean NET R | t (net) | 90% CI on gross R | Cost inside CI? |
|---|---|---|---|---|---|---|---|---|
| 20 | 1348 | **+0.162** | **+4.32** | 0.143–0.146 | +0.019 | +0.50 | [+0.107, +0.229] | **YES** |
| **55 (live)** | 676 | **+0.191** | **+3.85** | 0.139 | **+0.048** | +0.97 | [+0.105, +0.269] | **YES** |
| 100 | 490 | +0.124 | +1.19 | 0.135 | −0.019 | −0.18 | [−0.037, +0.306] | YES |
| 300 | 254 | +0.454 | +1.31 | 0.130 | +0.311 | +0.90 | [−0.033, +1.094] | YES |

This is the single most important table in the package. It is the one place the "strategy or cost?" question has a determinate answer: **the gross edge is real; the cost is the same size as it; the residual is not measurable.**

### 2c. MOMENTUM (IG 5m archive preferred where n ≥ 13; Yahoo 5m/59d secondary)

| Market | Archive n | Archive gross | Archive net | Yahoo n | Yahoo gross | Yahoo net | Read |
|---|---|---|---|---|---|---|---|
| **Gold** | 44 | 1.62 | 1.47 | 40 | 1.45 | 1.33 | see reconciliation below — true value ≈ **1.0–1.14** |
| NASDAQ 100 | 33 | 2.41 | 1.81 | 16 | 1.39 | 1.05 | best in book; 90% CI **[1.00, 7.17]** touches 1.0; family-wise p = **0.666** |
| Wall Street | 26 | 1.53 | 1.02 | 8 | 1.67 | 1.27 | coin flip after cost, P(net>1)=0.50 |
| S&P 500 | 19 | 0.49 | 0.24 | 8 | 0.32 | 0.22 | both sources agree; but p = 0.83-class evidence, see §5 F4 |
| Japan 225 | 13 | 0.63 | 0.33 | 9 | 0.61 | 0.42 | both sources agree, negative |
| Hong Kong | 11 | 1.94 | 0.67 | 7 | 1.15 | 0.54 | unmeasurable |
| Russell / FTSE / AI Index / DXY | 7 / 5 / 5 / 6 | 2.12 / 3.23 / 1.06 / 0.38 | 0.71 / — / — / — | 3 / 6 / — / 11 | 2.60 / 0.41 / — / 0.29 | | **unmeasurable; sources contradict** |
| Crude 1h (not live arm) | — | — | — | 35 | 0.81 | 0.81* | negative |
| EUR/USD 1h (not live arm) | — | — | — | 23 | 1.16 | 0.59 | * |
| GBP/USD 1h (not live arm) | — | — | — | 24 | 1.36 | 0.40 | * |
| **POOLED live momentum** | **146** | **1.54** | **1.11** | **88** | **1.22** | **0.97** | straddles breakeven |

\* net repeats gross where the dealing-rules capture read bid == offer (spread MISSING, not zero): Crude, DXY, AI Index. Forex net columns use out-of-hours-widened spreads (GBP/USD 16.9 vs ~0.9 daytime) and are hard pessimistic bounds.

**Gold momentum reconciliation (three independent routes converge on break-even):** naive Yahoo 5m 1.45/1.33 → corrected for the engine's close-only stop check (intrabar pierces re-priced at −1R) **1.05 gross / 0.91 net, n=65** → IG archive with the live pullback-entry filter and intrabar exits **1.28 gross / 1.14 net, n=69** → **live paper record PF 1.00 over 59 trades.** The gap between backtest and live is a *backtest bug* (missed stop hits), not an execution problem.

### 2d. MEAN REVERSION (the only adequately powered arm)

| Pass | n | Gross PF | Net PF | Test statistic |
|---|---|---|---|---|
| IG archive 5m, 7 deep markets × 2 specs | 5807 | **0.864** | 0.535 | mean −0.935 bps/trade; market-clustered t = **−3.00** |
| Yahoo 59d 5m (source cross-check) | 2935 | 0.855 | 0.592 | t = −3.21 (uncorrected) |
| Yahoo 725d 1h (2-year premise test) | 3338 | 0.816 | 0.732 | t = −4.67 (uncorrected) |
| **Symmetric 1R-barrier direction test** | **4737** | — | — | **WR 49.50%, 95% CI 48.08–50.93% → max PF 1.04** |

Even under maximally conservative clustering (one market = one observation), the symmetric-barrier upper bound is 53.27% → **max PF 1.14**. A PF-1.3 mean-reversion edge is genuinely ruled out. Four deliberately generous variants (optimistic intrabar ties, no stop, symmetric barriers, 6h time stop) never produced anything above 1.16 on any arm.

### 2e. Statistical power — the frame for everything above

| Market | n (730d) | Net PF | Smallest net PF this n could detect (80% power, α=.05) | n needed to certify observed PF | ≈ years at current rate |
|---|---|---|---|---|---|
| Gold | 109 | 1.44 | **2.33** | 482 | 8.8 |
| GBP/USD | 103 | 1.33 | **2.12** | 600 | 11.6 |
| Russell | 44 | 1.47 | **3.59** | 335 | 15.2 |
| Hong Kong | 61 | 1.34 | 6.13 | 1047 | 34.3 |
| FTSE | 89 | 1.13 | 2.60 | 3770 | 84.7 |
| DXY | 116 | 1.02 | 2.08 | 202544 | 3492 |
| NASDAQ | 69 | 0.77 | 3.82 | 934 | 27.1 |

**Not one market has an observed PF within reach of its own detection threshold.** Per-trade kurtosis 5.4–47.8, skew 1.7–5.9.

---

## 3. Go-live shortlist

**By the stated standard — "PF > 1 NET at realistic cost with n large enough to believe" — the shortlist is EMPTY.** No market on this book meets all three conditions. That is the honest answer and I am not going to dress it up.

What can be said, in descending order of evidential quality:

**Gold breakout (`CS.D.USCGC.TODAY.IP`, DAY HTF, N=55) — the only market that survives every lens.**
- n=109, the second-largest clean breakout sample.
- Net PF 1.443 (flat charge) / 1.445 (own-ATR) / 1.449 (de-drifted for GC=F futures carry, measured at +1.61%/yr vs GLD) / 1.424 (gap-fill corrected) / **1.847 (live-faithful cost model, P(PF>1) = 0.98, 90% CI [1.14, 2.88] — the only market whose CI excludes 1.0 from above).**
- Total honest execution cost measured at **0.012R**, i.e. **8% of the flat charge** — Gold is the cheapest market on the book to trade, and the only one where the entry-ATR correction goes the *right* way (entry/median ATR ratio 0.941).
- 100% session coverage on Yahoo (15.7 bars/day vs IG's 17.0); archive OHLC fidelity confirmed (close correlation 0.99309, Donchian-55 channel width within 1.5% of IG's).
- **Against it:** one-sided p = 0.088 (fails α=0.10); drop-top-3 → 0.97, which is *exactly the median* of what a genuine PF-1.44 edge produces at n=109 (so that test is uninformative, not damning); n=109 can only detect PF ≥ 2.33; and it carries a real sizing defect (§4).

**GBP/USD breakout (`CS.D.GBPUSD.TODAY.IP`, DAY HTF) — second, with a wider cost-model spread.**
- n=103; net PF **1.072 to 1.330** depending on cost model — the cost-model reviewer's live-faithful run costs it most of its headroom (measured entry gap +0.107R vs the +0.0397R pooled). One-sided p = 0.130.
- Methodologically the cleanest FX row: 100% session coverage, IG/Yahoo hourly-return correlation 0.987, channel width within 1.6%.
- Its "0.06 PF recent collapse" in the archive is a **measurement artefact** and should be struck (see §5 F9).

**Everything else fails on at least one hard criterion:**
- **Russell 2000** — highest headline headroom (1.471) and the *worst-evidenced row on the book*: n=44, 44.6% session coverage, −0.128 PF from the gap-fill fix, 1.137 under the live-faithful model, and a 22-day archive that cannot even seed a daily EMA21. Do not promote.
- **All seven index rows (S&P, NASDAQ, Wall St, FTSE, Japan, HK, Russell), breakout AND momentum — STRIKE, do not adjust.** Yahoo sees 43.3–89.0% of the bars inside the bot's own trading window and 56.1–93.0% of the price movement; on IG-native data **42.6% of breakout entries fire in hours Yahoo cannot see**, and those entries are *worse* (gross PF 0.43 vs 0.57). The rows undercount live trades ~3× (134 vs 42 in the ground-truth test). N_eff halves the error but does not close it.
- **DXY** — the widest disagreement in the package: 0.990–1.015 flat, 0.953 own-ATR, **0.631 live-faithful with P(PF>1) = 0.04, CI [0.37, 0.96]** (significantly *below* 1). But the 7.1-pt spread driving that is *derived by inverting a screener ratio, never measured*; at zero spread it reads 1.134. **One free mid-session `get_market_info` call decides whether DXY is the book's best marginal case or its worst market. This is the highest-leverage missing measurement anywhere.**
- **AI Index** — genuinely unmeasured (no Yahoo ticker; n=10 on archive; 18.7% of its resampled 1h bars are stray out-of-session flats). Keep shadow.
- **Crude** — 0.705–0.777, n=117. Its `breakout-shadow` demotion stands as a *decision rule*, though not as a proven finding (§5 F4).

---

## 4. What the cost base would have to become

### 4a. Pooled — the arithmetic requirement

At N=55 the pooled gross edge is **+0.191R/trade** and cost is **0.139R/trade** — cost is **73% of the gross edge**. For the net residual to reach even half the gross edge, cost would have to fall to ≈0.095R/trade, a **~32% reduction**. To reach a residual of +0.10R/trade (the level the power table says needs 2,400 trades to certify), cost would have to fall to ≈0.09R.

**Is that reachable?** Partly, and the decomposition says where:
- The **entry-timing gap** — the component the 0.286×ATR charge was built to represent — measures **+0.0397R pooled (SE 0.0120, 90% CI [0.020, 0.059]R = [0.040, 0.119]×ATR)** over 1,034 mechanically-computed entries. The adopted charge sits ~9 SE above that. The n=21 live sample that produced 0.286 has its own 90% CI of **[0.0, 0.58]×ATR** — the two estimates are statistically indistinguishable (t = 1.21, p ≈ 0.23), but the point estimate may be roughly **double the truth**.
- The **gap-through exit slip** measures **0.000–0.097R (mean 0.038R)**.
- The **spread** is fixed-in-price and therefore ranges **51× across the book** as a fraction of risk: half-spread/(2×ATR) runs **0.0035R (NASDAQ) to 0.1789R (DXY)**.

So honest per-market execution cost, in R:

| Gold | Wall St | S&P | NASDAQ | EUR | Crude | FTSE | GBP | HK | Russell | Japan | **DXY** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.012 | 0.009 | 0.054 | 0.077 | 0.084 | 0.091 | 0.145 | 0.146 | 0.157 | 0.219 | 0.239 | **0.372** |
| 0.08× | 0.06× | 0.37× | 0.54× | 0.59× | 0.64× | 1.01× | 1.02× | 1.09× | 1.53× | 1.67× | **2.60×** the flat charge |

**A 41× range against a flat 0.143R charge.** Median 0.118R. The flat charge over-charges Gold by 12× and Wall Street by 16×, and under-charges DXY by 2.6×.

### 4b. Concrete per-market requirements

| Market | The binding cost/structure problem | What would have to change |
|---|---|---|
| **DXY** | Round-trip spread ≈7.1 pts on a ~19.9 pt stop = **35.8% of risk** (the single worst on the book). Separately, the stop is **floor-bound on 98.8% of bars** (20-pt effective floor vs 1.8×ATR(15m) ≈ 6.4; 2×ATR(1h) = 14.1) — the ATR risk model is *inert*. | Spread would need to be **~3× tighter (7.1 → ~2.3 pts)** to reach the book median cost. But the spread is derived, not measured — **measure it first**. The floor-bound stop is unfixable without a wider frame. |
| **Gold breakout** | Not a spread problem — a **size** problem. `min_deal_size = 1.0` forces **£32.34 risk = 1.38× the £23.50 target and 72% of the £45 cap**; the cap trips whenever ATR(1h) > 22.5, which is **10.3% of bars — by construction the highest-volatility setups**, i.e. adverse selection against exactly the trades the 1.44 PF was earned on. | Either accept 1.38× intended risk, or raise `max_risk_gbp` above ~£45 to stop the adverse-selected skips. IG's minimum cannot go below 1.0, so sizing correctly at £23.50 is arithmetically impossible. **Gold momentum is fine** (floors to size 3.0 at £20.25, 0% skip). |
| **Japan 225** | **Unresolved and decisive.** `dealing_rules.csv` says `minDealSize = 0.50` → forced risk **£96.99 = 2.2× the cap → 97.1% of momentum bars and 100% of breakout bars silently cap-skip.** Commit `3478456` (2026-06-29) records IG's live value as **0.04** → £23.28, viable. | **Verify the field before anything else.** Caveat from data-validity: the archive window is a Japan volatility outlier (90th–97.9th percentile); at median vol forced risk ≈**£48**, still 7% over the cap, not 116%. The failure mode is silent — the market just goes quiet. |
| **Hong Kong** | Min stop floor 40.5 = **1.44× the median 5m ATR**, so **72% of a 2×ATR stop is floor, not choice**; binds on 47.2% of bars. Spread unmeasured in session (30.0 capture is 13h outside its window). | Even charging the full out-of-session 30.0 spread, live-faithful reads 1.206 — so spread is probably *not* the problem; the stop floor is, and it is IG's, not config's. |
| **Momentum, all markets** | The problem is **frequency, not per-trade size**. Gold momentum costs 0.050–0.063R/trade but fires ~376 times/yr = **19–24R/yr of cost**; Gold breakout N=55 costs 0.176R/trade but only 54.5 times/yr = **9.6R/yr**. The cheaper trade is a 2–3× more expensive year. | Momentum needs a *selectivity* fix, not a spread fix. |
| **Five markets** | `config.min_stop_distance` is **below** IG's live minimum (FTSE 1.0 vs 8.0; Wall St 4.0 vs 12.0; NASDAQ 4.0 vs 8.0; Japan 20 vs 40; HK 20 vs 40; Russell 1.0 vs 2.0). Only the order-time clamp at `main.py:1699` prevents `ATTACHED_ORDER_LEVEL_ERROR` rejections. | Any sweep or sizing calc using `config.min_stop_distance` understates the floor by up to 8×. Fix the config to match reality. |
| **Everything held overnight** | **DFB overnight financing on notional is charged by no backtest in this repo.** Implied notional/risk leverage: 1304× (S&P momentum), 1279× (Wall St), 1244× (EUR/USD); on breakout stops 619× (EUR), 517× (GBP), 498× (DXY). At an assumed 7.5%/yr this is **2.7–12.7% of 1R per calendar day**, i.e. 20–60% of 1R over a typical 5-day breakout hold, versus 14.3% for the one-off entry slip. | This is the largest unmodelled cost in the package and it biases every long-hold result **optimistically**. Rate is assumed; mechanism is certain. |

### 4c. Where cost genuinely is *not* the answer

**Horizon extension does not amortise cost.** Under an ATR-scaled stop, cost/risk is a mathematical constant (0.286 ÷ 2.0 = 0.143R) at every N — measured 0.146 / 0.139 / 0.135 / 0.130 / 0.130 across N=20…300. Stretching N to 300 saves 163.1R of cost but gives up 103.0R of gross; holding cost/trade fixed at the N=20 level changes net PF by ~0.02. And time-in-market *rises* with N (33.0% → 43.1%), so "trade less" does not mean "be exposed less". The apparent N=300 advantage is refuted twice over: fraction-matched trade removal reverses the direction (at a matched 3.9% removal, N=300 reads 0.75 vs N=20's 0.66), and splitting by data quality shows **the entire long-horizon signal lives in the seven session-blind index markets** (faithful-data pooled ladder: 0.89 / 1.07 / 1.01 / 1.03 / **0.88 worst at N=300**; session-blind: 0.91 / 1.00 / 0.87 / 1.12 / **1.06**).

---

## 5. Findings

### HIGH confidence

**F1. Mean reversion in the ranging regime is dead, and this is the only adequately-powered conclusion in the package.** Symmetric-1R-barrier hit rate 49.50%, 95% CI 48.08–50.93%, n=4737 — geometry-free, immune to every data mismatch elsewhere. Even one-market-one-observation clustering caps the achievable PF at 1.14. Pooled gross 0.864 (archive 5m), 0.855 (Yahoo 5m), 0.816 (Yahoo 725d 1h) — three passes, two sources, two timeframes. **Do not build the mean-reversion arm.** The bot's ~73% "ADX too low" HOLD rate (independently confirmed: median 74.0% of archive bars below the market's own ADX floor; and 72.0% of GBP/USD bars blocked by ADX alone in the harness) is correct behaviour, not a missed opportunity.

**F2. The pooled breakout gross edge is real and significant; the post-cost residual is not.** +0.162R (t=+4.32, n=1348) and +0.191R (t=+3.85, n=676) gross; +0.019R and +0.048R net (t=+0.50, +0.97). Cost sits inside the gross 90% CI at every horizon. Twelve individually-insignificant markets all leaning the same way are not twelve null results — but neither is the residual a measured profit.

**F3. No individual market is significantly profitable net of cost at 90% confidence.** Reproduced independently by two reviewers to ~1pp. Best one-sided p: Gold 0.088, GBP/USD 0.130, Russell 0.157. Under a global null of zero edge everywhere, the **median best-of-12 net PF is 1.62 — higher than the observed best (1.47)**; family-wise p for Russell = **0.72**.

**F4. The HIGH-confidence *negative* verdicts ("NASDAQ / Wall St / Crude have no gross edge, unsaveable by execution") are not supported.** Gross-side p-values: NASDAQ **0.831**, Wall St **0.753**, Crude **0.607**, S&P **0.952**; every 90% CI contains both 1.0 *and* 1.4. Under a global null the median *worst*-of-12 t is −2.12; the observed worst is −0.55 (family-wise p = 0.99) — the book's negative tail is *less* negative than chance produces. Separately, at live entry mechanics with zero spread, S&P reads **1.243** and NASDAQ **1.043**. "Don't trade unproven markets" remains a sound *decision rule*; "these markets have no edge" is not a *finding*.

**F5. The flat 0.286×ATR charge is unbiased pooled but reorders the book per-market.** Mean Δ vs the live-faithful model = **−0.015 PF**; mean |Δ| = **0.207 PF**; Spearman ρ between rankings = **0.65**. S&P moves +5 ranks, DXY −5, Russell −4. The *error in the cost model* exceeds the entire 0×→1× cost effect (0.11–0.33 PF) the package was decomposing. **The pooled conclusion survives; the per-market triage does not.**

**F6. Execution cost is three separable things, not one scalar.** (a) ATR-proportional entry gap, **0.079×ATR pooled** (not 0.286); (b) ATR-proportional gap-through exit slip, 0.000–0.097R; (c) fixed-in-price spread ranging **51×** as a fraction of risk (0.0035R NASDAQ → 0.1789R DXY).

**F7. Four code-level defects, all confirmed, all worth fixing regardless of the go-live decision.**
- `src/backtest.py` checks stops on the **CLOSE only**, so bars that pierced the stop intrabar and recovered are booked as still-open. Re-pricing: Gold 5m gross 1.52 → **1.05**; Gold 1h 2.40 → **1.53**. This is the term that reconciles Gold momentum's backtest to its live PF 1.00.
- `src/backtest.py:363` selects the HTF row with `date <= current_time` — the bar still *forming*. Look-ahead. Effect on PF is unsigned (median +0.03) but it changes trade counts materially (EUR/USD 30 vs 23; Crude 39 vs 35). Still shipping.
- `breakout_sim` gap-adjusts **entries** but fills stop/trail **exits at the exact level even when the bar opened through it** — a free lunch on gappy cash indices. NASDAQ 0.773→**0.688**, HK 1.341→**1.198**, Russell 1.471→**1.343**, Japan 0.921→**0.849**; Gold/DXY/GBP barely move (2–3 affected trades each).
- The momentum arm's "re-price stop exits at the candle close" correction is **wrong**: the bot attaches a *non-guaranteed broker stop at order time* (`main.py:1817-1821` → `client.open_position(stop_distance=…)`, `guaranteed_stop=False`), which fills at the level and only slips on gaps. Measured bar-to-bar gap on IG-native 5m: mean 0.028–0.063×ATR, exceeding 0.5×ATR on 0.4–0.9% of bars — **8–30× smaller than charged**. Gold momentum's "1.45 → 1.23 for realistic fills" should read ≈**1.45 → 1.43**.

**F8. The seven index rows cannot be settled with Yahoo data and should be struck, not adjusted.** Yahoo cash proxies deliver 4.6–6.2 bars/calendar-day vs IG's ~17. Yahoo sees only **43.3% (S&P/NASDAQ/Wall St), 44.6% (Russell), 74.8% (Japan), 75.0% (HK), 89.0% (FTSE)** of the bars inside the bot's own trading window, and **42.6% of IG-native breakout entries fire in hours Yahoo cannot see** — and those are *worse* entries, so the rows are optimistically selected as well as under-powered. Ground-truth test: session-only data undercounts live trades **3×** (42 vs 134) and produces a mean absolute per-market PF error of **1.04** vs truth. N_eff cuts that to 0.15 but still misses per-market (FTSE truth 0.27 vs N_eff 0.09).

**F9. "The strategy has been gross-negative through the paper-trading period" is a measurement artefact.** Every DAY-HTF archive figure was produced by resampling a 67-day intraday archive into daily bars for the HTF gate. With a real daily series: **GBP/USD 0.06 → 1.14**, EUR/USD 0.28 → 1.00, Gold 0.49 → 0.81. Russell and DXY have **22 calendar days** of archive and cannot seed a daily EMA21 at all. Yahoo over the identical window gives **Gold 1.21 and GBP/USD 2.07** (the breakout arm reported 0.25 and 0.20 — those do not reproduce in any of 16 configuration cells). Mechanism: at n≈10 both sources fire the same trades on the same dates and differ on *one exit* (Gold 06-17 18h SHORT: archive +2.00% vs Yahoo +6.16%, which is the entire 0.81-vs-1.21 difference).

**F10. The IG candle archive is faithful, and the "streaming clips highs/lows" worry is refuted.** 3,194 matched 5m bars: archive/Yahoo median range ratio S&P 1.066, NASDAQ 1.128, Gold 0.995; close correlation 0.99998/0.99998/0.99309. No synthetic flat filler at any London hour for the 7 deep epics. The brief's "known hole 07-17..07-20" **does not exist** (07-17 has 266 bars, 07-20 has 265; only the Sunday-evening session of 07-19 is missing, 13 bars).

**F11. Instrument mismatch (Yahoo proxy vs IG DFB) is NOT a material problem for Gold, FX, DXY or Crude.** GC=F-vs-spot carry measured at +1.61%/yr (≈+1.2% net of GLD's fee); de-drifting moves Gold breakout **1.443 → 1.449**, and at a deliberately excessive 4%/yr **1.459**. IG-vs-Yahoo 1h channel-width ratios 0.986–1.019; hourly-return correlations 0.962–0.987. **The index problem is 100% session coverage and 0% instrument.**

**F12. Cost cannot amortise with holding period under an ATR-scaled stop.** 0.286 ÷ 2.0 = 0.143R at every N — arithmetic, not statistics.

**F13. Structural constraints, not spread, bind on this book.** At daytime spreads where measured or documented, momentum round-trip cost is **3.6–15.2% of 1R** (NASDAQ 3.6, Crude 4.1, Gold 7.6, GBP 7.6, EUR 10.8, AI 15.2) — survivable. Meanwhile min-stop floors consume **280% (DXY), 72% (HK), 64% (Russell), 50% (FTSE)** of a 2×ATR stop, binding on 98.8% / 47.2% / 36.9% / 0.2% of in-window bars, and min deal size forces **£96.99 (Japan) and £32.34 (Gold breakout)** against a £23.50 target and £45 cap.

**F14. The "drop the top-3 trades" test has essentially no discriminating power at these sample sizes and should stop being cited as evidence.** Drawing n from a population whose *true* PF equals the market's own observed value (edge REAL by construction): Gold's observed drop-3 of 0.97 sits at the **median (0.98)** of that distribution; GBP/USD's 0.99 at the median (1.00). P(drop-3 < 1.0 | edge is real) = **0.52 (Gold), 0.51 (GBP), 0.64 (Russell), 0.80 (FTSE)**; P(drop-top-1 < 1.0 | edge real) = **0.79 at every N**. It is a property of fat tails at n≈100, not of these strategies.

**F15. `min_stop_distance` does not contaminate the breakout results.** DXY tested explicitly: gross/net 1.189/0.990 at min_stop 0.0, identical at IG's live 0.10 floor, 1.177/0.988 at the config's 20.0 (which binds on 50.7% of bars but moves PF by 0.01). Elsewhere 2×ATR exceeds the config floor by 1–2 orders of magnitude.

**F16. Where n is large the two data sources agree; where n is small they contradict.** Agree (n≥13): S&P 0.32/0.49, Japan 0.61/0.63, Wall St 1.67/1.53, Gold 1.45/1.62. Contradict (n≤7): FTSE 0.41/3.23, HK 1.15/1.94, Russell 2.60/2.12. This is itself the strongest evidence that the small-n rows are noise.

### MEDIUM confidence

**F17. Gold breakout is the single best-evidenced market and the only one that survives every cost model tried** (0.012R total execution cost; live-faithful 1.847, P(PF>1)=0.98, CI [1.14, 2.88]; carry-corrected 1.449; gap-fill-corrected 1.424). MEDIUM rather than HIGH because it still fails a conventional significance test on the flat model (p=0.088) and n=109 cannot detect anything below PF 2.33.

**F18. DXY is either the book's best marginal case or its worst market, and one free measurement decides which.** Live-faithful 0.631 with CI [0.37, 0.96] (significantly below 1) — but driven entirely by a **derived, never-measured** 7.1-pt spread. At zero spread it reads 1.134.

**F19. Yahoo "re-fetch jitter" is not jitter — it is the 730d→725d fallback firing non-deterministically.** Three fresh fetches: EUR/USD gross 1.297 / 1.297 / **1.448** (n=107/107/106); GBP/USD 1.613 / 1.613 / 1.585. **Dropping 5 days (1% of bars) moves EUR/USD by 0.151 PF** — 5× the ±0.03 tolerance the breakout arm quoted, and EUR/USD sits 0.08 from breakeven.

**F20. The archive window's volatility regime is not representative for several markets, and two arms disagree on how badly.** DXY / EUR/USD / GBP/USD sit at the **21st percentile (instruments arm) or the 0th (data-validity arm)** of 730d 1h ATR — i.e. their cost ratios are pessimistic, possibly maximally so. Japan sits at the **88.5th (instruments) / 90th on daily range / 97.9th on 1h (data-validity)** — its cap-skip magnitude is inflated. Others: NASDAQ 72.9–85.3, Crude 77.2–78.9, FTSE 70.5–71.8, HK 55.9–56.4, Gold 56.8–63.3, S&P 50.7–51.6, Wall St 43.2–47.1, Russell 36.8–44.3.

**F21. The paper-trading year has not measured the bot.** −£180.16 over 100 trades at the measured pooled per-trade dispersion (sd = 1.98R on 676 N=55 breakout trades) and £23.50 risk → **t = −0.39, 90% CI on the 100-trade total = [−£945, +£585]**. Halving the dispersion assumption still gives t = −0.77. A genuine PF-1.44 edge posts a losing 100-trade run **14%** of the time.

**F22. Restricting the IG archive to Yahoo-visible hours removes 32–52% of momentum trades and lowers gross PF in 5 of 6 clean cases** (NASDAQ 2.41 n=33 → 1.97 n=16; Wall St 1.53 n=26 → 1.26 n=16; Japan 0.63 → 0.51; S&P 0.49 → 0.50; HK 1.94 → 1.94; Russell 2.12 → 2.00). The Yahoo momentum rows for three of the six live momentum markets are drawn from 43% of the live window's bars, and the bias runs **low**, not high.

**F23. The 67-day archive has no power to confirm or deny a 730-day result.** Rolling 67d gross PF is below 1.0 in **43–87% of the last two years even for markets whose 730d PF exceeds 1** (S&P median 0.23 / 76% below 1; HK 0.33 / 87%; FTSE 1.20 / 43%; Gold 1.77 / 24%; GBP 1.73 / 24%). The observed archive readings sit at incoherent percentiles (S&P's 0.38 at the 67th, HK's 0.88 at the 87th, GBP's and EUR's at the 0th). There is no "recent regime" story in these data — there is a 10-trade-per-market estimator.

**F24. `min_confidence` sweeps, per-EPIC parameter search, ADX ceilings, leg filters, swing proximity and breakout profit-protection remain refuted** (pre-existing findings, not re-tested here, but nothing in this work contradicts them). The recurring pattern — intuitive "improvements" that clip the tail — is corroborated by F14's demonstration that tail trades carry the observed edge.

### LOW confidence

**F25. The 0.286×ATR charge may under-charge the Asian/UK index markets.** HK's captured spread alone is 86% of the whole charge, FTSE 60%, Japan 50%. Arithmetically correct, but the capture is 13h out of session and even charging it in full HK still reads 1.206 live-faithful. **The genuinely under-charged market is DXY at 2.60× the flat charge, and it is not on that list.**

**F26. Sub-hourly entry delay (Gold at :05, DXY at :15) is not a cost.** Archive-measured drift from the 1h close to the actual entry minute: signed mean **+0.003 to +0.012R** (unbiased), but |mean| 0.17–0.22R, p90 0.36–0.48R — large *noise*, no systematic cost.

**F27. Crude's 2026-06-25 cost-based condemnation does not transfer to its current 1h config** (1h ATR 68 pts vs a documented 2.8-pt daytime spread = 4.1% of 1R). But the Crude archive is **two disjoint segments with a 650.5-hour hole (2026-06-25 → 2026-07-22) across which the price jumps 6888 → 8735 (+26.81%)**, so every rolling ATR and Donchian channel straddling the join is invalid, and the archive-derived ATR is not well defined. The 730d Yahoo verdict (0.705–0.777) stands on edge grounds regardless.

---

## 6. GAPS — what could not be measured

**Measurement gaps that would change verdicts if closed:**

1. **DXY's daytime spread is DERIVED, never measured** (inverted from a screener ratio of unstated timeframe). It swings DXY's live-faithful PF from 0.631 to 1.134. **One free mid-session `get_market_info` call. Highest-leverage missing datum in the package.**
2. **Japan 225's `minDealSize` is contradicted**: `dealing_rules.csv` says 0.50, commit `3478456` says 0.04. The verdict flips between "97% of signals silently cap-skipped, effectively untradeable" and "fully viable". Resolve from the live container's journal / `rejected_signals` table.
3. **Daytime spreads are unmeasured for 6 of 13 markets** (S&P, Russell, Wall St, FTSE, Japan, HK). All 13 CSV rows were captured at 21:50 UTC, outside every market's own window bar AI Index (which returned bid == offer). Both sanctioned derivation routes failed: the archive stores MIDs only (`src/streaming.py` builds from (bid+offer)/2), and `logs/ig_bot.log` locally is an 8.8KB stub from 2026-02. Fix: grep the container for `[ACTIVE] … ATR/Spread=` lines that `main.py:651` already writes.
4. **AI Index is genuinely unmeasured** — no Yahoo ticker, n=10 on archive, and 18.7% of its resampled 1h bars are stray out-of-session flats. Do not act on its "0.04 PF" in either direction.
5. **The live trade journal was never reachable** (it lives on the VPS). Nothing here confirms that predicted Japan cap-skips or Gold's £32.34 breakout risks actually appear in the live record, and the −£180.16/100 baseline could not be bucketed by market or by config era.

**Structural gaps that bound the whole exercise:**

6. **Statistical power is the binding constraint, not cost accounting.** The study is under-powered by 4–40× in n. Resolving a +0.10R/trade edge at 80% power needs **~2,400 trades**; at N=55 the book generated 676 in 730 days (≈338/yr), i.e. **~7 years**.
7. **The seven index rows cannot be settled with any locally available data.** Settling them needs an IG-native 730d frame (impossible without burning the REST allowance) or a far longer shadow-observer record.
8. **DFB overnight financing is modelled nowhere.** 2.7–12.7% of 1R per calendar day at an assumed 7.5%/yr; forex is tom-next and can be positive on one side. It biases every long-hold result optimistically and is the largest single unmodelled cost.
9. **The backtest engine implements a simpler strategy than the bot runs**: no break-even stop (`breakeven_trigger_pct` / `breakeven_lock_pct`), no ATR trailing stop (`atr_trail_mult`), no Gold pullback-entry arming in the Yahoo path. Direction of bias is not determinable a priori.
10. **The correlation-cluster filter is enforced live** and blocks 2nd+ same-direction entries across the six grouped equity indices within 15 minutes. It is cross-market and cannot be applied per-market, so index trade counts are an upper bound. `MAX_POSITIONS=8` slot competition is likewise unmodelled.
11. **In-sample contamination.** Strategy parameters were read from `config.py`, but they were fitted by the owner over months on largely this same data — most acutely Gold (EMA 3/8/21, ADX 35, RSI 85/15). This biases gross PF **upward** and *strengthens* the "gross ≈ 1 means no edge" readings.
12. **Cost is charged at the sample-median ATR, not each trade's own.** Breakout entries fire at 1.08–1.54× median ATR (median 1.24), so the flat charge under-charges by 0.03–0.05 PF in 12 of 13 markets. Gold is the sole exception (ratio 0.941).
13. **Forex mean-reversion was never tested** (archive holds only ~1,270 1h bars, no 5m). It is plausibly the most mean-reverting and cheapest corner of the book — the one place a ranging edge could still hide.
14. **A 5m out-of-sample period does not exist locally.** The Yahoo 59d 5m pass *overlaps* the archive window; it is a source cross-check, not an independent period.
15. **Bootstrap CIs resample trades i.i.d.**, ignoring serial dependence and shared macro regimes across the ten markets. True CIs are **wider** than reported, so every non-significance verdict is if anything understated.
16. **RESEARCH.md step 5 was not executed by the measurement agents** (their harness forbade writing .md files). This document is the persistence step; paste it into `research_notes.md`.

---

## 7. RECOMMENDATION ON GOING LIVE

### The recommendation

**Do not go live on the book as it stands. Do not shut it down either.** The correct action is a narrow, pre-committed, size-limited live experiment — or continued paper trading — depending on whether the owner is prepared to *pay* for data he cannot otherwise obtain.

### What the numbers actually imply, in pounds

Taking the pooled N=55 figures at face value and scaling by the current £23.50 risk-per-trade and 338 trades/yr *(derived arithmetic from the measured pooled R-values; see caveats)*:

| | Per trade | Per year | On a £9.4k account |
|---|---|---|---|
| Gross edge | +0.191R = +£4.49 | +£1,517 | +16.1% |
| Measured cost | −0.139R = −£3.27 | −£1,104 | −11.7% |
| **Net (point estimate)** | **+0.048R = +£1.13** | **+£381** | **+4.1%** |
| **Net (90% interval)** | **[−£0.80, +£3.06]** | **[−£270, +£1,033]** | **[−2.9%, +11.0%]** |

Caveats that all push the same way: the seven session-blind index rows undercount live trades ~3×, so the trade count is uncertain; DFB overnight financing (2.7–12.7% of 1R/day) is subtracted from none of the above; `MAX_POSITIONS=8` and the correlation filter would reduce the realised count; and the parameters are in-sample. **The central estimate is a low-single-digit-percent annual return with an interval that includes meaningful loss.**

### Honest framing of the bet

- **What is known with near-certainty:** the cost drag, ~0.14R/trade (the entry-timing component is arithmetic; the spread component is fixed and measurable).
- **What is known with ±100% relative error:** the gross edge, +0.16 to +0.19R/trade with a standard error of ±0.08.
- **What is not known at all:** the residual, +0.02 to +0.05R/trade with the same error bar — which *is* the entire P&L.
- **What no further analysis can fix:** only more trades resolve this. The requirement is ~2,400 trades ≈ 7 years at the current rate. No amount of additional cost accounting, parameter work, or backtesting will settle it, and the four arms plus three reviews here are the demonstration of that.

### If going live anyway — the minimum defensible version

1. **Take the three free measurements first.** They are zero-cost and two of them flip verdicts: (a) a mid-session `get_market_info` sweep for daytime spreads on DXY, S&P, Russell, Wall St, FTSE, Japan, HK; (b) Japan's real `minDealSize`; (c) an audit of the live journal for Japan cap-skips and Gold's actual per-trade risk. **Do not go live before (a) and (b).**
2. **Trade Gold breakout only, possibly plus GBP/USD breakout.** These are the only two markets that survive session-coverage, instrument-fidelity, cost-model and gap-fill scrutiny, and they are the two with the best (though still not significant) evidence. Gold: net PF 1.42–1.85 across every cost model, 0.012R execution cost, n=109, p=0.088. GBP/USD: 1.07–1.33, n=103, p=0.130.
3. **Strike the seven index rows from the decision entirely.** They are measured on 43–89% of the bot's own trading window, undercount live trades ~3×, and carry a gap-fill subsidy worth up to −0.14 PF. Russell in particular — the highest headline headroom on the book — is the single worst-evidenced row (n=44, 44.6% coverage, 22-day archive). Leave them shadow.
4. **Fix Gold breakout's sizing before trading it.** `min_deal_size = 1.0` forces £32.34 = 1.38× intended risk and cap-skips the highest-ATR 10.3% of setups — adverse selection against exactly the trades the edge lives on. Either accept the 1.38× or raise the risk cap; do not leave it silently skipping.
5. **Size the experiment as a research budget, not as an investment.** Pre-commit a hard drawdown stop (e.g. 30 trades' worth of risk) and a review date, and understand that even at completion the sample will not distinguish the edge from zero.
6. **Fix the four backtest defects (F7) regardless.** Every number the repo produces going forward depends on them, and one of them (close-only stop checking) is the term that reconciles Gold's backtest to its live record.

### What would change this recommendation

**Upward:** a measured DXY daytime spread near 2 pts (DXY goes from worst market to a live candidate); Japan's `minDealSize` confirmed at 0.04 (restores a market); an IG-native 730d index frame or 12+ months of shadow-observer record on the seven index markets (they are currently *unknown*, not *bad*).

**Downward:** the live-faithful cost model being closer to truth than the flat charge for GBP/USD (1.330 → 1.072) and Russell (1.471 → 1.137); DFB financing being charged properly on multi-day breakout holds; or Gold's `min_deal_size` skips proving more adversely selected in the live record than the 10.3%-of-bars estimate.

### The one thing to hold onto

The mean-reversion result is the only conclusion in this entire package with adequate statistical power, and it is a **negative**: there is no edge in the 73% of the time the bot sits out. That is a real, valuable answer — it tells you the idle time is correct behaviour and closes off an entire line of development. Everything else here is a statement about what cannot yet be known.
---

## POST-SYNTHESIS CORRECTIONS (main session, 2026-08-19, from the live journal)

**GAP 2 RESOLVED — Japan 225 is NOT cap-skipping.** The synthesis flagged a contradiction
between `dealing_rules.csv` (minDealSize 0.50 → £96.99 forced risk → "97% of signals
silently skipped, effectively untradeable") and commit `3478456` (0.04 → viable). The live
journal settles it: **Japan has traded 9 times at sizes 0.04–0.16 with risk £10.34–£22.15**,
all inside the £23.50 target.

    #309 SELL size=0.08 stop 176.1pt risk £14.08     #287 SELL size=0.08 stop 276.9pt risk £22.15
    #305 BUY  size=0.08 stop 229.4pt risk £18.35     #286 BUY  size=0.12 stop 160.3pt risk £19.23
    #304 BUY  size=0.16 stop 131.1pt risk £20.97     #269 BUY  size=0.04 stop 475.4pt risk £19.01

The 0.5 minimum IS real but intermittent — `rejected_signals` holds exactly **2** Japan
cap-skips ("Min size 0.5 × stop 160.6 / 228.4"), against 9 completed trades. So the effect
is occasional, not structural. **Japan's viability verdict reverts to VIABLE**; the
"untradeable" branch is refuted. Same pattern for Hong Kong (3 skips at min size 0.5).

**Gold breakout's sizing defect CONFIRMED live**, and slightly worse than modelled. All 7
Gold breakout trades ran at size 1.0 (the floor) with risk **£33.39–£37.68**, mean ≈£35.0 —
the synthesis modelled £32.34. That is **1.49× the £23.50 target** and 78% of the £45 cap,
on every single trade, not just high-ATR ones. Gold momentum is unaffected (sizes float).

**Cap-skips are rare book-wide** — the whole `rejected_signals` size/cap population is ~16
rows across FTSE (3), NASDAQ (5), AI Index (4), HK (3), Japan (2), Gold (1). The "silent
cap-skip" failure mode is real but small, and it does NOT invalidate any market.

**GAP 1 (DXY daytime spread) remains OPEN** — it needs a mid-session `get_market_info`
call and the market was closed at the time of writing. It is still the highest-leverage
missing datum: it swings DXY between live-faithful PF 0.631 and 1.134.

**Correction owed to the workflow's own input:** the ground-truth context I supplied listed
"breakout 730d at live configs" using the DAY column for all nine markets, when only Gold,
Russell and DXY are on DAY. The reviewers caught it. Correct live-config figures are
Japan **0.92**, NASDAQ **0.77**, Wall St **0.75**, S&P **0.86**, HK **1.34** (not 1.10 /
0.99 / 0.89 / 0.88 / 1.09). This makes the index rows *worse* than briefed, except HK
which is better — and it does not change the verdict, since F8 strikes those rows anyway.

---

## 2026-08-19 — Alorse pinescript-strategies: catalogue + port selection (selection phase only)

**Question.** Of the ~48 Pine strategies in `Alorse/pinescript-strategies`, which contain a
mechanism the bot has never tested, and which of those have a *structurally* better cost
profile than Donchian breakout (54 trades/yr) or EMA/RSI/ADX momentum (~376 trades/yr)?

**Method.** Fetched all 74 `.pine` files (GitHub API tree → raw), read the source (the
README has no specs). Classified each against the two live families. Then ran a
**signal-rate probe only** (`scratchpad/rate_probe.py`, Yahoo 1h, 725d, published
defaults, no P&L, no tuning) on Gold / GBP/USD / EUR/USD / Crude / DXY to replace guessed
trade rates with measured ones.

**Evidence — measured signals/yr, 1h frame (Gold … DXY range):**

| mechanism | signals/yr | note |
|---|---|---|
| MACD regular-bullish divergence | 7.6–11.1 | lowest in the repo; ~n=19 per market over 2yr |
| Bollinger Breakout [kodify] (SMA350±2.5σ) | 20.7–35.9 | best cost profile with usable power |
| Supertrend flip (10, 3.7) | 47.5–66.7 | ≈ parity with the bot's breakout |
| BB Winner LITE (control) | 50.0–83.9 | low-rate control — cost cannot be its excuse |
| Double Supertrend entry | 64.7–94.5 | upper bound; real count lower (wide ST5 exit holds) |
| Pin Bar Magic setups | 67.2–197.0 | stop-entry + 3-bar cancel ⇒ fills are a fraction |
| Williams Vix Fix | 143.5–185.4 | a DAILY tool; on 1h it is not a rare event |
| QQE flips | 170.2–189.4 | |
| TTM Squeeze + EMA stack | 175.3–358.2 | EMA2500 filter is **not** selective (Gold 709 vs 720) |
| BB Divergence (band expansion) | 283.4–321.8 | |
| DMI Winner | 705.2–837.1 | calibration anchor for the momentum family |
| Flawless Victory v1 (control) | 968.9–1280.6 | BB at **1.0σ**; cost-hostile by construction |

**Findings.**
1. **HIGH — the repo contains no working volatility-squeeze strategy.** Both TTM scripts
   compute `diff = bband(20,2) − keltner(20,1)` (the actual squeeze) and then **never
   reference it**. The entry is a linreg momentum oscillator turning up below zero. The
   squeeze-regime idea therefore remains untested after this port, and any port claiming
   to test it would be mislabelled.
2. **HIGH — "mean-reversion" folder labels are unreliable.** `Bollinger Breakout [kodify]`
   is a slow volatility-normalised *breakout* (long on the upper band, exit on the SMA350
   re-cross), and `BB Divergence` is a band-*expansion* breakout (`close > upper` AND upper
   rising AND lower falling). Both were mis-filed; both are the interesting ones.
3. **MEDIUM — low trade rate and statistical power trade off directly.** MACD divergence at
   ~9 signals/yr is the best cost profile in the repo and also the least testable
   (n≈19/market/2yr; the `donttouchzero` gate I omitted from the probe pushes it lower
   still). It is a mechanism probe, not an edge test, and must be reported as such.
4. **MEDIUM — multi-week holds break the measured cost table.** The 0.012–0.372R per-market
   figures were measured on hours-to-days holds. `Bollinger Breakout [kodify]` exits on an
   SMA350 re-cross and can hold for weeks; overnight/weekend spread-bet funding is **not**
   in that table. Its attractive per-trade cost may be illusory.
5. **HIGH — a family-wise budget problem.** 12 strategies × 5 measurable markets = 60 cells.
   The briefing already establishes that best-of-12 under a global null medians PF 1.62.
   Best-of-60 is far worse. The pooled test must be primary; per-market is exploratory only.

**Selection (12 = 10 test + 2 negative controls).** Supertrend, Double Supertrend, Williams
Vix Fix, Pin Bar Magic v1, QQE, TTM Squeeze+EMA (labelled honestly), BB Divergence
(expansion), Bollinger Breakout [kodify], MACD + Divergences, DMI Winner (momentum
calibration anchor); controls = BB Winner LITE + Flawless Victory v1. The two controls were
picked to have *opposite* trade rates (~66/yr vs ~1150/yr) so a harness fault shows up
whether it flatters low-cost or high-cost arms.

**Open.** Fill convention (Pine fills next-bar-open), intrabar stop-vs-TP ordering, and the
R denominator for the four published strategies that carry **no stop at all** (QQE, Double
RSI, momentum/TTM, 2 EMA+RSI) are unresolved and must be pre-registered before any port —
each is a fork that can move a verdict.

---

# Pine Script Library vs the Bot — Verdict

**Question:** Is anything in the ported TradingView Pine Script library actually better than what the bot already runs?
**Date:** 2026-08-19 · **Scope:** 12 ported strategies × 5 measurable markets (Gold, GBP/USD, EUR/USD, Crude, DXY) × 1h, 730d (725/700 fallbacks where Yahoo rejected 730), 60/40 IS-OOS split.
**Sources:** 5 harness runs + 3 independent adversarial reviews (multiple-comparisons, porting-fidelity, novelty). Where the reviews corrected the harness runs, **the correction is what is recorded below.**

---

## 1. Direct answer

**No. Nothing in this library is better than what the bot already runs, and nothing in it has a structurally better cost profile.** Pooled across the five measurable markets out-of-sample — the pre-registered primary test — **all twelve ported strategies are net-negative**, the best being `bb_expansion_breakout` at −0.028R/trade (net PF 0.954) against the incumbent Donchian breakout's −0.049R (0.934). Of the 17 out-of-sample cells where an arm nominally beat the benchmark, the best raw one-sided p is 0.102; Holm at α=0.05 admits none, Benjamini–Hochberg at q=0.10 admits none, and Westfall–Young family-wise p for the single best cell is 0.9965 — while the null's median *maximum* beat is +1.119R against an observed maximum of +0.580R, i.e. the best "win" in the book is half of what pure noise routinely produces. Exactly one arm survived family-wise correction anywhere (`bb_expansion_breakout` on Gold, rotation p = 0.0004 full-period), and it then proved statistically indistinguishable from the bot's own Donchian breakout restricted to longs (difference +0.024R, SE 0.363, t = +0.07; the incumbent's long leg is *more* significant out-of-sample, p_rot 0.014 vs 0.046). The one thing this run did establish is that the failure is **not** a porting artefact and **not** a cost artefact: three independent re-transliterations found 12/12 arms bit-identical with no look-ahead, and 11 of 12 arms fail on **gross** in at least one half on GBP/USD before any charge is applied. The honest caveat that must travel with the "no": with a median cell of n=50 and per-trade sd of 1.64R, this design had **1.0% power at 60-cell correction against the +0.191R effect the study is about** — so "nothing survived correction" was arithmetically foreclosed before the first backtest ran. The load-bearing negatives are the pooled point estimates and the drift-matched rotation nulls, not the significance tests.

---

## 2. Pooled out-of-sample, sorted by cost share of gross

Trade-count-weighted pooling of the reported per-market OOS rows across Gold (0.012R), EUR/USD (0.084R), Crude (0.091R), GBP/USD (0.146R), DXY (0.372R). Reproduced independently — matches the multiple-comparisons reviewer's pooled figures exactly (bb_expansion −0.028, incumbent −0.049, macd_divergence −0.260 on n=50). Script: `scratchpad/synth/pool.py`.

**Baseline to beat: cost share 0.73.** Cost share is undefined (not favourable) where pooled gross ≤ 0.

| # | Strategy | n (OOS) | gross R | net R | cost R | **cost share** | mkts net+ | mean t/yr | annual cost drag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bb_expansion_breakout | 183 | +0.110 | **−0.028** | 0.137 | **1.25** | 2/5 | 46.5 | 6.71 R/yr |
| 2 | **BOT-BREAKOUT (benchmark)** | **230** | **+0.102** | **−0.049** | **0.151** | **1.48** | **3/5** | **56.2** | **8.29 R/yr** |
| 3 | dmi_winner *(calibration anchor)* | 224 | +0.091 | −0.048 | 0.139 | 1.53 | 2/5 | 55.6 | 7.80 R/yr |
| 4 | double_supertrend | 336 | +0.090 | −0.059 | 0.149 | 1.66 | 2/5 | 84.5 | 12.47 R/yr |
| 5 | control_bb_winner_lite *(NEG CONTROL)* | 234 | +0.059 | −0.090 | 0.148 | 2.53 | 1/5 | 58.0 | 8.83 R/yr |
| 6 | control_flawless_victory_v1 *(NEG CONTROL)* | 509 | +0.035 | −0.108 | 0.143 | 4.10 | 1/5 | 136.9 | 19.85 R/yr |
| 7 | pin_bar_magic | 341 | +0.033 | −0.109 | 0.143 | 4.28 | 2/5 | 83.1 | 11.87 R/yr |
| 8 | qqe_signals | 717 | +0.019 | −0.122 | 0.142 | 7.29 | 2/5 | 178.2 | 25.32 R/yr |
| — | *gross ≤ 0 — cost share meaningless* | | | | | | | | |
| 9 | williams_vix_fix | 366 | −0.025 | −0.182 | 0.157 | n/a | 2/5 | 92.4 | 13.51 R/yr |
| 10 | supertrend_flip | 233 | −0.029 | −0.173 | 0.144 | n/a | 1/5 | 56.5 | 8.18 R/yr |
| 11 | macd_divergence | 50 | −0.105 | −0.260 | 0.155 | n/a | 2/5 | 14.7 | 2.22 R/yr |
| 12 | ttm_squeeze_ema | 195 | −0.162 | −0.302 | 0.140 | n/a | 1/5 | 38.4 | 4.72 R/yr |
| 13 | kodify_bollinger_breakout | 39 | −0.484 | −0.642 | 0.158 | n/a | 0/5 | 10.0 | 1.49 R/yr |

**Reading the table:**

- **No arm reaches the 0.73 baseline.** The best cost share in the whole book is 1.25 — 1.7× worse than the number it has to beat — and it belongs to an arm that is still net-negative.
- **Only one arm beats the benchmark on cost share at all (bb_expansion, 1.25 vs 1.48), and it beats it by being net-negative slightly less.**
- **A negative control sits fifth.** `control_bb_winner_lite` has a better pooled OOS cost share (2.53) than seven candidate arms.
- **Trading less does not buy a cost profile.** The two arms with genuinely low cost drag — `kodify` at 1.49 R/yr (0.18× the incumbent) and `macd_divergence` at 2.22 R/yr (0.27×) — are the two with the *worst* pooled gross. A cheaper way to lose money is not an improvement.
- **Within a single market, cost share is a monotone restatement of gross R** (cost is a fixed per-trade charge), so it carries no independent information there and mechanically favours the lowest-n arms. It only becomes a real screen pooled across markets with different costs — which is what this table is, and it still finds nothing.

Pooled net PF where reported: bb_expansion 0.954, BOT-BREAKOUT 0.934 (multiple-comparisons review). Full-period pooled net R (novelty review, clustered bootstrap over markets then trades, 6000 draws) tells the same story with the sign of one arm flipped by tail trades: kodify +0.174 P(≤0)=0.500; macd_divergence +0.092 P(≤0)=0.374; dmi_winner +0.014 P(≤0)=0.505; **BOT-BREAKOUT −0.021 P(≤0)=0.565**; BOT-MOMENTUM (1h reference) −0.153 P(≤0)=0.882; all nine others −0.013 to −0.142. **Not one arm's pooled net is distinguishable from zero — including the incumbent's.**

---

## 3. Did the negative controls behave?

**Yes on four markets of five, and the one apparent failure was diagnosed and is not a harness fault. The run is not suspect.** — HIGH

| Market | control_bb_winner_lite (net PF, full) | control_flawless_victory_v1 (net PF, full) | Verdict |
|---|---|---|---|
| GBP/USD | 0.835 (IS 0.748 / OOS 0.984) | 0.831 (0.740 / 0.988) | pass |
| EUR/USD | 1.028 (mean net +0.018R, CI [−0.229,+0.273], t=+0.14) | 0.909 | pass |
| Crude | 0.941 (IS **1.072** / OOS 0.718) | 0.894 | pass |
| DXY | 0.399 | 0.612 | pass |
| **Gold** | **1.32 (IS 1.30 / OOS 1.35)** | **1.21 (IS 1.34 / OOS 1.02)** | **tripwire fired** |

On Gold **both refuted-family controls came out net-PF > 1 in both halves**, which by pre-registration means "treat the run as suspect". It was investigated three ways and the cause is the **sample, not the harness**:

- Gold rose **+73.7%** over the window (IS leg +57.0%, OOS +10.8%). 10 of 12 arms are long-only, and the mandated uniform exit (2×ATR stop, no take-profit, Donchian-27 trail) converts a random long entry into positive expectancy. A **Bernoulli(p) long signal on every bar**, at the benchmark's own trade rate, gives median gross PF **1.44–1.57** and P(gross R > 0) = 0.94–1.00 (400 seeds/rate). Random longs beat 8 of the 12 ported arms.
- Against the **correct** benchmark — a circular-rotation null preserving each arm's own trade count, clustering and exit rule while destroying price alignment — the controls behave exactly as pre-registered: `control_flawless_victory_v1` p_rot = **0.946** (worse than firing the same rule at random times), `control_bb_winner_lite` p_rot = **0.605**. The multiple-comparisons reviewer replicated this and sharpened it: under the rotation null a randomly-timed copy of each Gold arm is net-positive in *both* halves 43–79% of the time; observed 9 of 11 gives **P = 0.108**. "Positive in both halves on Gold" is worth nothing.
- The porting reviewer confirmed the same phenomenon from the opposite direction: restoring every arm's **published** exits makes both controls look *good* on Gold (ctrl_bb_lite +0.434R gross PF 1.82; ctrl_flawless +0.627R gross PF 1.71).

**Conclusion: absolute PF on this Gold window carries no information about entry skill and must not be quoted as evidence for anything.** On the four markets without the drift confound the controls sat at or below breakeven everywhere, which is the calibration that matters. The `dmi_winner` calibration anchor also failed net as pre-registered on GBP (0.830), EUR (0.826), Crude (0.872) and DXY (0.847) — but note it failed for the *wrong mechanism*: one-position-at-a-time cut it from ~390 published entries/yr to ~55 realised, so it was never actually tested in the cost-dominated regime it was chosen to anchor.

**One further control result worth recording:** with published exits restored on the two markets *without* the drift confound, the arm that clears "net-positive in both halves" on EUR/USD is bb_expansion at **t_net = +0.31** — and the arm that clears it on Crude is **`control_bb_winner_lite`, the refuted-family negative control, at t_net = +0.32**. The both-halves screen has no discriminating power at these sample sizes; that is demonstrated, not argued.

---

## 4. Anything that beat the benchmark — the three-gate test

A candidate must pass **all three**: (a) statistically real after multiplicity correction, (b) mechanically distinct, (c) faithfully ported.

### The complete list of OOS cells that beat the benchmark

17 of 60. Best raw one-sided p = **0.102** (qqe on DXY, where both arms lose money). Holm α=0.05: **0 survive**. BH q=0.10: **0 survive**. Westfall–Young FWER for the best cell: **0.9965**. Null median max-D +1.119R vs observed max-D +0.580R. **Four of the top ten "beats" are the two negative controls** — that is the tell.

### bb_expansion_breakout — the only arm that survived any correction

| Gate | Result |
|---|---|
| **(a) Statistically real** | **FAILS.** Gold full-period rotation p = 0.0002 (harness) / **0.00040** (independent re-implementation, 2500 draws) — survives BH over 12 arms and even Bonferroni over 60 (0.024). But that is the **full period, not out-of-sample**: OOS p_rot = **0.046 raw → 1.000 at 60 cells**, and Westfall–Young within Gold's own 11-arm family gives **p_WY = 0.496**. Beat-benchmark on Gold OOS: D = +0.061R, p_raw 0.460. **Pooled OOS it is net-negative (−0.028R).** It fails on GBP (net PF 0.927 IS → 0.588 OOS), EUR (1.105 → 0.685), Crude (0.621 → 1.388, full sign flip, randomisation p 0.355) and DXY (0.227 → 0.677). |
| **(b) Mechanically distinct** | **PASSES.** Bar-P&L correlation to BOT-BREAKOUT **0.36**, to BOT-BREAKOUT-LONGONLY 0.45, matched-trade 0.19; only **29%** of its entries fall within 6 bars of a bot long entry. For calibration, an actual reparameterisation (Donchian N=45 vs N=55) scores **0.96**, and the bot's own two live families (momentum vs breakout) score **0.658**. It is genuinely a different entry rule. |
| **(c) Faithfully ported** | **PASSES.** 148 signals, XOR = 0 against an independent bar-by-bar re-transliteration; no look-ahead by two independent methods. Deviations recorded: ported long-only (the published short leg is an author typo — `sellzone` mirrors the buy formula without swapping high for low, and removes **0 of 480** bars conditional on the setup); `pyramiding=1` not replicated. |

**Verdict: 2 of 3. Not adoptable.** And the reason it fails (a) is the finding, not a technicality: restricting BOT-BREAKOUT to longs — the like-for-like comparison, since bb_expansion is long-only and the bot trades both legs — gives 38.0 trades/yr, gross +0.725R, gross PF 2.55 on Gold, versus bb_expansion's 39.5/yr, +0.749R, 3.29. **Difference +0.024R, SE 0.363, t = +0.07; better OOS for the bot (+0.693 vs +0.476); rotation p bit-identical full-period (0.00040 both) and better for the bot out-of-sample (0.014 vs 0.046).** Two mechanically different entry rules delivering the same mean R is *stronger* evidence for the conclusion: on this sample the payoff came from the long-bias-plus-trailing-exit structure, not from either entry rule.

⚠️ **BOT-BREAKOUT-LONGONLY is a post-hoc leg selection made after seeing the short leg lose.** Use it for direction-matching only. Pooled across five markets it is +0.107R net with **P(≤0) = 0.353** and positive in 2/5 markets. Its Gold figures must never be quoted as "what the bot does".

### The other four nominal beats

| Arm | (a) Statistical | (b) Distinct | (c) Ported | Verdict |
|---|---|---|---|---|
| `macd_divergence` | **FAILS** — pooled OOS −0.260R on n=50, one-sided 95% UB +0.085; pooled full +0.048R t=0.24 p_WY 1.000. EUR result is **one trade**: top trade = 16.36R of 20.97R total (**78%**); drop it → net PF 1.158, drop two → +0.007R gross. Gold IS +0.746 → OOS −0.533. | PASSES — max |corr| to *anything* in the matrix = **0.17**, the only genuinely orthogonal arm | **FAILS as a strategy test** — the +16.36R trade was produced by the bot's Donchian-27 trail, not by the strategy. As published (7% stop + 1.5R limit off `lowest(low,10)`): **EUR net PF 0.44, Gold 1.02, Crude 0.54**. The port is faithful; what was measured is "divergence entry + the bot's exit". | **1/3. Do not carry forward.** The EUR run's recommendation to pool it as a mechanism probe is **refuted**: pooled it is negative and it does not deserve a budget line. |
| `dmi_winner` | FAILS — best gross anywhere (DXY +0.2254R) but 95% CI [−0.196, +0.701], P(net-positive) 26.2%, P(cost share < 0.73) **11.4%**; DXY needs 0.5096R gross to reach 0.73 | **FAILS the spirit** — corr 0.480 to BOT-BREAKOUT, the highest in the whole matrix; and it is closer to the bot's *breakout* (0.480) than to the bot's *momentum* rule it was named after (0.406) | PASSES | 1/3. It is the calibration anchor and it behaved. |
| `double_supertrend` | FAILS — pooled OOS −0.059R; Gold OOS beat D=+0.080 p_raw 0.433 | PASSES (0.40) | PASSES | 1/3 |
| `kodify_bollinger_breakout` | FAILS — pooled OOS −0.642R on **n=39**, 95% UB −0.183 (evidence of *harm*). Gold headline +1.873R is pure tail: top 3 trades = 135.7% of total R, sd 10.13R, bootstrap CI [−0.779, +6.474], P(mean≤0) = 0.274. Gold OOS: 7 trades, **all seven stopped at exactly −1.000R**. | PASSES (0.40) | PASSES, but **77% of its trades exit on the imposed 2×ATR stop, not its published SMA350 re-cross** — the multi-week-hold mechanism, and the funding question that was supposed to decide it, were never tested | 1/3 |

**No candidate passes all three gates. There is nothing to promote.**

---

## 5. Genuinely useful negative results — stop wondering about these

These are the payoff of the exercise. Each is now measured across 5 markets and 730d.

1. **Mean reversion is dead for a fourth time, on a third source and a fourth timeframe.** Both controls net-negative pooled OOS (−0.090 and −0.108). Consistent with the existing n=4737 / 49.50% barrier-hit / pooled gross PF 0.864 refutation. — HIGH
2. **A volatility-spike gate does NOT rescue mean reversion.** `williams_vix_fix` (synthetic-VIX capitulation + lower-BB) is inside its own rotation null on Gold (**p_rot 0.87**), pooled OOS net −0.182 with a **95% upper bound of −0.062** — positive evidence of harm, not absence of evidence. — HIGH
3. **ATR-band trailing (Supertrend) is not better than the Donchian channel as an entry.** At near-identical trade rate (56.5 vs 56.2/yr) it is pooled-OOS gross **−0.029R**. — HIGH
4. **Widening the exit band to hold through pullbacks (double_supertrend) buys nothing.** Pooled OOS −0.059R at 84.5 t/yr — 1.5× the cost drag for the same nothing. — HIGH
5. **Oscillator-space volatility ratchets (QQE) are cost-suicide.** 178 t/yr = **25.3 R/yr of cost drag** (3.1× the incumbent) against pooled OOS gross +0.019R. 95% UB on net −0.040. — HIGH
6. **TTM "squeeze" as published is not a squeeze strategy and does not work.** Pooled OOS net −0.302, 95% UB −0.079. Note: the *actual* squeeze condition is dead code in both Pine sources; the EMA2500 filter it was selected for removes only **1.4%** of signals (579 vs 587 on Gold, Pine-faithful; the catalogue's 709/720 came from a differently-seeded EMA and is not reproducible). — HIGH
7. **Slow volatility-normalised breakout (kodify) is degenerate, not cheap-and-good.** 10 trades/yr, n=39 pooled OOS, net −0.642R, 95% UB −0.183. — HIGH
8. **Single-candle rejection (pin bar) is not an edge.** Pooled OOS −0.109R; its verdict also moves ~0.10R/trade purely on fill convention (+0.220R honouring the published stop-entry vs +0.115R at next-bar open on Gold), both inside its rotation null (p 0.26 / 0.25). — HIGH
9. **"Just trade less to beat the cost" is refuted directly.** The two cheapest arms in the book by annual drag are the two with the worst pooled gross. Cost share is improved by a *larger per-trade edge*, and none of the twelve mechanisms produced one. — HIGH
10. **Mechanical novelty is not the binding constraint, and buys nothing.** All twelve arms are genuinely new (max corr to the incumbent 0.48 vs 0.91–0.96 for a true reparameterisation). Across the twelve, corr(similarity-to-incumbent, pooled gross R) = **Pearson 0.446 / Spearman 0.629, permutation p = 0.065** — arms are profitable roughly to the extent that they are *not* new — while corr(similarity-to-always-long, gross R) = −0.363, so market beta does not explain it. — MEDIUM
11. **Two "impressive-looking winners from noise" were caught in the act,** exactly as the brief predicted: `macd_divergence` on EUR/USD (78% of gross in one trade, net PF 2.40 IS / 1.99 OOS, cost share 0.101/0.176 — the *only* arm to clear the 0.73 screen in both halves anywhere) and `kodify` on Gold (net PF 5.75 in-sample, then seven consecutive −1.000R stop-outs). Both would have passed a naive screen. — HIGH

---

## 6. Findings by confidence

### HIGH
- **Nothing in the batch is adoptable.** Pooled OOS all 12 net-negative; zero survivors under Holm, BH or Westfall–Young; observed best-of-60 net PF (2.40) sits **below** the null median best-of-60 (2.65), replicating the prior study's 1.47-vs-1.62 result at 5× the family size.
- **The failure is not a cost artefact.** On GBP/USD, 11 of 12 arms fail on **gross** in at least one half. Re-tested at 0.031R (spread only), 0.052R and the mandated 0.146R — at every rung only BOT-BREAKOUT survives the both-halves screen. This conclusion is cost-model independent.
- **The failure is not a porting artefact.** 12/12 arms bit-identical to an independent transliteration; end-to-end truncation causality tests 8/8 identical at two truncation points; HTF merge audited on 310 sampled bars with zero look-ahead. Four catalogue errors were caught and corrected in the port itself (pin_bar enters at `high[1]` not the pin bar's high; macd stop is 7% not `lowest(low,10)`; the two Supertrend arms use opposite sign conventions; the pandas-3.0 `(s & ~s.shift(1))` episode counter silently returns `s.sum()`).
- **The uniform-exit override was conservative, not permissive.** Restoring published exits makes every Gold arm *better* — including both negative controls — and still leaves nothing standing on EUR/USD or Crude, where a refuted-family control clears the same both-halves bar (t=+0.32) as the best candidate (t=+0.31).
- **Absolute PF on the Gold 1h window is meaningless.** Random longs at the benchmark's trade rate give median gross PF 1.44–1.57 and P(gross R>0) = 0.94–1.00.
- **Cost share within one market is a monotone restatement of gross R.** It is only a structural screen pooled across markets with different costs, or against a hold-time-dependent cost like funding.
- **This design had ~1% power.** Median OOS cell n=50, per-trade gross-R sd 1.64R. Power against +0.191R: 20.5% uncorrected, **1.0% at 60-cell Bonferroni**; 80% power would need **~1174 trades in one cell ≈ 21 years** at the bot's 55 breakout trades/yr. "Nothing survived correction" was foreclosed in advance.

### MEDIUM
- **Overnight funding is missing from the cost table and is material for hours-to-days strategies.** Gold: ~0.065R/trade at 7% APR against 0.012R of execution — **~5×**, taking the incumbent's true all-in Gold cost share from 0.025 to ~0.16. EUR/USD: 0.069–0.421R/trade across arms (0.8× to 5× the execution charge), cross-checked against the project's own `funding.json` (12.7% of R per day held vs 11.8% computed). Charging it puts every EUR arm net-negative except macd_divergence, and the incumbent 0.934 → 0.733. **No verdict changes; the ranking of hold-length does.** Crude is exempt in kind — `EN.D.CL.Month1.IP` is a future, cost is in the forward price, not a daily debit. MEDIUM because 7% APR is an estimate; direction and order of magnitude are robust across 5–9%, the level is not. **IG's actual Gold DFB funding rate should be read from the account before this is used in a decision.**
- **The entry fill convention matters, but its sign is market-specific — do not apply book-wide.** Filling a Donchian break at `max(open, channel)` rather than next-bar open changes the incumbent's gross R by: Gold **+0.019 / +0.019**, GBP **+0.196 / +0.202**, EUR **−0.057 / −0.026**, Crude **−0.057**, DXY **+0.114 / +0.091** (two independent measurements where both exist; range −0.057 to +0.202, mean ≈ +0.046). The GBP run's inference that "the benchmark's honest cost share is ~0.42" does not generalise and should not be carried forward as fact. Under its *best* fill the incumbent is still gross-positive on only 2/5 markets.
- **The incumbent is not cleared either.** BOT-BREAKOUT pooled over the five measurable markets: OOS net **−0.049R** (net PF 0.934), full **−0.025R** (0.965), 95% upper bound +0.128R. The multiplicity machinery rejected twelve challengers; it did not endorse the defender.
- **The twelve arms are not twelve independent tests.** Mean pairwise bar-P&L correlation 0.234, largest pair 0.78; **effective independent arms = 5.78 of 12**. The 60-cell budget is really ~29 effective cells, so the corrections applied were *more* severe than necessary. Nothing survived anyway, which strengthens rather than weakens the conclusion.

### LOW
- The "profitable ∝ not-new" inversion (Spearman 0.629, permutation p 0.065) rests on n=12 arms on one pooled sample. Suggestive, not established.

### Two MEDIUM claims from the harness runs that are now REFUTED as actionable

Recording these explicitly so they are not carried into the v3 review as findings:

- **"Gold's HTF DAY filter is doing real work" — REFUTED as evidence.** Two-sample bootstrap HTF vs noHTF: IS +0.373 (p=0.111), **OOS +0.013 (p=0.482)**, full +0.239 (p=0.178) — none significant. The identical comparison on Crude has the **opposite sign** (IS −0.407 p=0.924, full −0.147). One market, one direction, post-hoc, nested, not significant, contradicted where replicated. It happens to support the status quo so it is harmless — **but it must not be quoted as validation of `52e97bd`.**
- **"The 2026-08-13 DXY HTF-DAY deploy does not replicate out-of-sample" — REFUTED as actionable.** IS−OOS gross difference +0.327R, 95% CI **[−0.276, +0.936]**, p = 0.146. The same split test applied to all five incumbents gives similar-sized differences with random signs (Gold +0.111 p=0.85, GBP −0.061 p=0.86, EUR +0.258 p=0.40, Crude −0.541 p=0.34, DXY +0.327 p=0.30). This is a five-market search for a split difference; one will always look big. **Do not revert config on it.** The separate, better-supported DXY concern stands on its own and belongs on the v3 agenda: **DXY's full-period breakout gross R is +0.0039 with P(≤0) = 0.51** — a literal coin flip on the book's most expensive market (0.372R/trade, requiring 0.5096R gross just to reach the 0.73 baseline).

---

## GAPS

1. **The seven index markets are unmeasurable here** (S&P, NASDAQ, Wall St, FTSE, Japan, HK, Russell): Yahoo cash proxies see only 43–89% of the bars inside the bot's trading window and 42.6% of real entries fire in hours Yahoo cannot see. Nothing in this study says anything about them.
2. **Power, restated as a gap:** this experiment could not have detected the effect it was testing for. Any future arm needs its power computed *before* it consumes budget.
3. **Budget overran ~4×, untallied.** Pre-registered 60 cells; actually reported ~204 primary strategy×market×period rows plus documented sensitivity re-runs = **≥260 looks**. Nothing passed, so no false positive was created — but **no figure in this study may be quoted as if it came from a 60-cell budget.**
4. **The trail-geometry diagnostic was never actually run.** `supertrend_flip` was selected to test ATR-band trail vs Donchian trail at equal trade count; the uniform harness discarded its published trail and gave it the bot's Donchian-27 trail. 76.3% of its trade pairs with the bot exit on the *same bar*, which is what produces its 0.806 matched-trade correlation (that correlation is **−0.107** on the 23.7% of pairs that exit on different bars). What was measured is ATR-band *entry* vs Donchian *entry*. The trail question is open.
5. **The TTM squeeze gate remains untested.** Both Pine sources compute `bband(20,2) − keltner(20,1)` and never reference it. Wiring it in would be authoring, not porting. The repository contains no working volatility-squeeze strategy; the priority-1 "squeeze regime" idea is still open and would need its own pre-registered hypothesis.
6. **Two arms are not the published strategy and their rows must be labelled so:** `pin_bar_magic`'s 1-pip published trailing stop was never ported, and `kodify`'s multi-week hold (and therefore its funding exposure, the issue that was supposed to decide it) was truncated by the imposed 2×ATR stop in 77% of trades.
7. **Two disclosed harness inhomogeneities.** DXY did not call `PATH_EXITS` for `ttm_squeeze_ema` (so the pooled ttm row is not one arm; the DXY sensitivity run gives n=30 gross −0.628 vs primary n=50). Exit-fill convention differs: Gold/GBP/EUR/Crude fill signal exits at the signal bar's close, DXY at the next bar's open (Pine-faithful). Measured size ≤ **+0.030R/trade**, and it flatters *candidates*, not the incumbent — so it makes the negative stronger.
8. **DXY's 0.372R cost is the single largest input to its column and rests on live fills, not a captured quote** — `dealing_rules.csv` for `CO.D.DX.Month1.IP` has bid == offer (status EDITS_ONLY, captured out of session). It would have to be wrong by >40% to change any verdict, but it is unverified. Separately, IG's DXY instrument is a Month1 **future** with roll effects Yahoo's spot index does not carry; the IG archive holds only 1588 bars / 25 days, far too short to test this.
9. **Funding rates are estimates throughout.** The actual IG DFB rates for Gold and the FX pairs have not been read from the account.

---

## 7. Recommendation

**Nothing here moves the go-live picture. The prior conclusion stands unchanged: a thin real edge, cost eats ~73% of it, the residual is statistically unmeasurable, and Gold is the only market cheap enough to be worth trading if anything is.**

Specifically:

- **Do not adopt any of the twelve strategies.** Not on Gold, not anywhere. The single arm that survived a family-wise correction did so on the full period on the one market with a 74% two-year trend, and the bot's own Donchian long leg matches it there and beats it out-of-sample.
- **Do not change any live config on the strength of this study.** In particular: do not revert the 2026-08-13 Gold/DXY HTF change (the "doesn't replicate" finding is a multiplicity artefact, p=0.146, same-sized differences with random signs in all five markets), and do not quote the Gold HTF filter result as validation of it either (p=0.482 OOS, opposite sign on Crude). Crude stays on breakout-shadow — this run independently reproduced the verdict that demoted it (gross PF 0.958, net 0.842, 0/14 arms net-PF>1 in both halves, family-wise randomisation p 0.993).
- **The go-live case gets no new support, and one small new piece of unfavourable evidence:** the incumbent breakout, pooled over the five measurable markets, is **net-negative out-of-sample (−0.049R, PF 0.934)** and net-negative full-period (−0.025R, PF 0.965), with a 95% upper bound of +0.128R. That is consistent with the prior +0.048R residual, not a contradiction of it — but it is the fourth independent look that has failed to find a positive net number, and DXY's full-period breakout gross R of +0.0039 with P(≤0)=0.51 on the book's most expensive market is worth an explicit v3 agenda line.
- **Do not manufacture a next test out of this.** The EUR run's proposal to pool `macd_divergence` across five markets is refuted (pooled it is −0.260R OOS on n=50; as *published* it is net PF 0.44 on EUR). At ~14 trades/yr per market it could never reach the ~1174 trades one cell needs. It does not deserve a budget line.
- **The one durable output is methodological, and it should be written into the review protocol.** Any future strategy candidate must, before it consumes family-wise budget: (1) have its **power computed first** — if the design cannot resolve +0.19R, do not run it; (2) be scored against a **rate-matched circular-rotation null**, not against absolute PF; (3) be compared against **BOT-BREAKOUT restricted to the same direction set**, not the both-legs version; and (4) carry a **funding charge derived from realised hold duration**, since on Gold that is ~5× the execution cost the current table charges. Every one of those four steps changed a verdict somewhere in this study.

**Next actions, in priority order:** (1) read IG's actual Gold DFB and FX overnight funding rates from the account and fold carry into the cost table — this is the only item here that changes numbers the bot already uses; (2) put "DXY breakout full-period gross R = +0.0039, P(≤0) = 0.51" on the v3 agenda alongside item 15(b), noting that its 0.372R cost requires 0.5096R gross to reach the 0.73 baseline; (3) close the open trail-geometry and TTM-squeeze diagnostics only if the owner specifically wants them — neither is on the go-live critical path, and both need their own pre-registration.
---

# Overnight funding: MEASURED from the IG account (2026-08-19)

**Why:** the cost-vs-edge synthesis named DFB overnight financing "the largest single
unmodelled cost", estimated at 2.7–12.7% of 1R per calendar day (assumed 7.5%/yr on
notional at 500–1300× implied leverage), i.e. 20–60% of 1R over a 5-day breakout hold,
and flagged that it "biases every long-hold result optimistically". That estimate was
never checked against the account. It has now been.

**Source:** `/history/transactions?type=ALL` over 180 days — free, does NOT touch the
10k/week historical allowance. 375 transactions: 345 DEAL, **30 financing rows**.
IG labels them `Daily Admin Fee - FX Interest` and `Daily Financing Adjustment - FX
Interest`, each naming the instrument and the number of nights.

## Measured rates, normalised per night

| Instrument | GBP total (180d) | nights | **GBP/night** | as R on the risk actually traded |
|---|---:|---:|---:|---|
| Spot Gold | −2.80 | 8 | **−0.350** | **−0.0100 R** (on £35 actual) / −0.0149 (on £23.50) |
| GBP/USD | −5.68 | 18 | **−0.316** | ≈ −0.0137 R (on £23.11 actual) |
| EUR/USD | +1.83 | 10 | **+0.183** | ≈ **+0.008 R — a CREDIT** |
| USD/JPY | +6.34 | 9 | **+0.704** | ≈ **+0.031 R — a CREDIT** |
| Dollar Index (DXY) | — | 0 | **none** | `CO.D.DX.Month1.IP` is a **future**, not a DFB — no daily financing at all |
| **All instruments** | **−0.31** | **45** | **−0.007** | ≈ **0** |

## Findings

**FF1. The "largest unmodelled cost" claim is REFUTED at portfolio level — HIGH.** Total
financing over 180 days is **−£0.31** against −£722.33 of trading P&L: **0.04%**. Two
reasons, both structural: (a) only **16 of 314 closed trades (5%) were held overnight**,
27 position-nights in total; (b) the FX legs **earn** carry as often as they pay it, and
the single largest financing row in the account is a **+£6.38 credit** (USD/JPY, 6 nights).

**FF2. But the per-night RATE assumption was roughly right — HIGH.** Gold at size 1.0
(notional ≈ £4,400) is charged £0.35/night ≈ 2.9%/yr on notional, against the 7.5%
assumed. So the synthesis's error was NOT the rate — it was multiplying that rate by a
hold pattern the bot does not have, and ignoring the credit side entirely.

**FF3. Gold's all-in cost roughly doubles-to-triples, and is still tiny — HIGH.**
Execution 0.012R + financing 0.010R/night:

    1-night hold -> 0.022R    2-night -> 0.032R    3-night -> 0.042R    (gross edge +0.191R)

Even at three nights Gold's all-in cost is **~22% of the pooled gross edge**, against the
book-average 0.139R that produced the headline "cost is 73% of the edge". **The 73% figure
is a book average dominated by expensive markets (DXY 0.372R); it materially overstates
the cost problem on the one market that is actually the go-live candidate.**

**FF4. Direction matters for FX, and no model in this repo captures it — MEDIUM.** EUR/USD
and USD/JPY carry positive; GBP/USD carries negative. A long-USD breakout can be
financing-POSITIVE. Treating financing as a uniform drag is wrong in sign for some trades.

**FF5. DXY escapes daily financing entirely — HIGH.** It is a Month1 future; the carry is
in the roll and the spread rather than a nightly charge. This slightly offsets its
worst-on-the-book 0.372R execution cost, but does not rescue it.

## The caveat that matters for the go-live decision

The 5%-overnight statistic is a property of the CURRENT book, which is mostly momentum
(holds minutes). **The go-live recommendation is breakout-only, and breakout holds days —
overnight exposure would go from 5% of trades to ~100%.** So financing becomes structural
rather than incidental. At the measured Gold rate that is still only 0.010R/night, but it
should be charged per realised hold duration in every future breakout backtest, not
ignored and not charged at the 7.5%-of-notional estimate.

## Gaps
- **n is small**: 45 instrument-nights across 4 instruments. Rates are indicative, not precise.
- Rates track central-bank policy and IG's spread over it; they will drift.
- **No index DFB financing was observed** because the bot has never held an index overnight.
  Indices are DFBs and WOULD be charged; that rate is still unmeasured.
- Gold's 8 financing nights mix momentum-era and breakout-era positions at size 1.0.

---

# STOP-PLACEMENT PRE-FLIGHT BRIEF — `min_stop_distance` vs IG's `minNormalStopOrLimitDistance`

*Investigation only. No repo file modified, nothing deployed. Scratch: `/private/tmp/claude-501/-Users-paulturner-IG/6f107130-5cce-4510-9c2a-c5bc07b3ecf6/scratchpad/`*

---

## 1. WHAT THE DEFECT ACTUALLY IS

**The original framing is wrong, and there is no defect in the six numbers.** `config.min_stop_distance` is not, and was never, a mirror of IG's broker minimum — `main.py:1684-1686` names the two concepts explicitly ("config.py min_stop_distance is the strategy floor, but IG's actual minimum varies by account tier / region / CFD-vs-spreadbet"), and reconciles them at order time with `max(strategy_floor, broker_floor)`. Git history contains **zero** commits setting a value to track a broker minimum; the only two deliberate edits (`84c67a68` S&P, `54b5ad86` Gold) moved the value to raise the *derived* `min_stop × 20` stop cap — the opposite use. Worse, the premise table is a sampling artefact: re-measured in-session, **NASDAQ (4.0) and Russell (1.0) match config exactly**, FTSE is under by 4× not 8×, and S&P's config sits *above* IG. What the investigation did surface is (i) a **semantic collision** — one field doing five jobs (×20/×25/×50 ceiling lever, spread multiple, observed-rejection floor, broker mirror once, 0.5%-of-price in the backtest map) which is what manufactured this false alarm; and (ii) **three genuine, unrelated defects**: breakout-path cap-skips are logged but never journaled, the clamp raises the stop without rescaling the take-profit, and `client.py:403` defaults a missing `dealingRules` to `0.0` which silently disables the clamp.

## 2. IS LIVE ORDER PLACEMENT AFFECTED?

**NO on both paths — clamp fires before sizing, verified in code today and against 217 post-clamp live trades (zero sub-minimum stops).**

| path | call chain | verdict |
|---|---|---|
| **Momentum** | `on_candle_complete` → `analyze_market_from_stream` (`main.py:1224`) → `strategy.analyze` floor `src/strategy.py:188` → **clamp `main.py:1688-1708`** (`ig_min = live_market_info.min_stop_distance + 0.5`) → **sizing `main.py:1710-1717`** → `client.open_position` `main.py:1817` | safe |
| **Breakout** | `analyze_forex_breakout` → `breakout.analyze_breakout` floor `src/breakout.py:128` → `_execute_breakout_entry` → **clamp `main.py:872-879`** (`signal.stop_distance = max(signal.stop_distance, ig_min)`) → **sizing `main.py:889-892`** → `client.open_position` `main.py:900` | safe |

These are the **only two** `client.open_position` call sites; `src/client.py:731-806` does no clamping of its own; `src/risk_manager.py` has **zero** references to `min_stop_distance`. Clamp observed firing live: `2026-08-10 09:10 | [Hong Kong HS50] Raising stop 30.79 -> 40.50 (IG min 40.00)` → journal #294 sized from 40.5.

**Three caveats that qualify "safe", not overturn it:**
- Both clamps sit inside `if info:` — `get_market_info` returns `None` with no retry/cache fallback, and `client.py:403` defaults `min_stop_distance` to `0.0` on a 200 without `dealingRules` (fails the `> 0` test). Fails **closed** (IG rejects, signal lost). Zero occurrences in 24 days of logs (2026-07-27 → 2026-08-19).
- The breakout clamp is **silent** (bare `max()`); momentum logs it. A breakout stop regression would leave no forensic trail.
- **The clamp costs live R:R.** `main.py:1707-1708` raises `limit_distance` only if it is *below* `ig_min` — it never rescales. HK #294 ran at realised 1.52 R:R against a configured 2.0; GBP/USD #126/#132 at 1.92/1.78. Forward-looking, 39.2% of HK momentum candidates clamp, median effective R:R **1.17 vs 2.0**. This makes "config only corrupts measurement" **false for Hong Kong** — but the fix is in the clamp, not in the config number.

## 3. MONEY TABLE

Budget: `risk_amount = balance × risk_per_trade / max_positions` = 9226.5 × 0.02 / 8 = **£23.07** (`src/risk_manager.py:73`). Cap-skip requires `floored_up AND min_size × stop > £45` (`risk_manager.py:131-157`) — since £23.07 < £45 this collapses to `stop > 45 / min_deal_size`.

| market | cfg | IG (21:50Z / ~11:10Z) | floor binds — momentum | binds — breakout 1h | risk/trade | cap threshold | breach £45? |
|---|---|---|---|---|---|---|---|
| FTSE 100 (shadow) | 1.0 | 8.0 / **4.0** | 0.00% vs IG+0.5 | 0.00% | £23.07 max | 1125 pt | no |
| Wall Street | 4.0 | 12.0 / 12.0 | 0.00% | 0.00% (min stop on 141 break bars = 93.75, 7.5× IG's 12.0) | £23.07 max | 1125 pt | no |
| NASDAQ 100 | 4.0 | 8.0 / **4.0** | 0.00% | 0.00% | £23.07 max | 4500 pt | no |
| Japan 225 | 20.0 | 40.0 / **20.0** | 0.00% | 0.00% | £23.05 max | 1125 pt | no |
| Hong Kong HS50 | 20.0 | 40.0 / 40.0 | **34–39%** clamped (stop +30.9% med, size −24%) | 0.09% | £23.06 max | 1125 pt | no |
| US Russell 2000 (shadow) | 1.0 | 2.0 / **1.0** | 0.00% (10% vs the 21:50Z value) | 0.00% | £23.07 max | 1125 pt | no |

**Raising the six changes 0 of ~9,583 candidate bars' cap outcome, 0.00% cap-skip everywhere, and does not move £ risk/trade — sizing is risk-based, so a wider stop buys a smaller size at constant £.** Nothing becomes untradeable from this change.

**Two markets ARE in the Copper failure mode — neither caused by, nor fixed by, `min_stop_distance`:**
- **Gold (LIVE breakout today** — `market_modes.json` read 11:47Z: Gold `breakout`, DXY/Crude `breakout-shadow`): `min_deal 1.0` × 2×ATR stop → **15.87% of Donchian break bars exceed £45**. Verified on the VPS today: **9 cap-skip log lines, Gold, 2026-08-06 04:20–05:00 container-local**, `Min size 1.0 × stop 46.0 risks £46.02 > £45.00` — one whole entry dropped.
- **Japan 225 breakout (latent)**: at in-window `min_deal 0.04`, 1h 2×ATR stop median 737 pt → 75–84% floored, median forced risk **£29.60** vs a £23.07 budget, **6.2–6.8% cap-skipped**. One `/mode nikkei breakout` walks into it.

## 4. MEASUREMENT-ONLY EFFECTS, AND SIGN OF BIAS

| consumer | affected? | sign |
|---|---|---|
| **Breakout shadow** (`main.py:2384` snapshot, `:2432/2443/2477` resolver) | **NO — 0 of 76 rows.** Tightest row 6.43 vs a 2.5 clamp; 0.00% of archive 1h bars would ever bind | none |
| **Momentum bench/shadow** (`main.py:1460`, `:1671` — pre-clamp *by documented design*, `main.py:1654-1655`) | **YES, 3 of 92 rows**, net −0.133R | **\|R\| overstated** (too-small denominator): row id 1 by 2.02×. Forward: HK 75%, Russell 48%, FTSE 41% of 5m bars would record an unplaceable stop |
| **`src/backtest.py`** (≈55 scripts) | **NO** — hard-coded `MIN_STOP_DISTANCE_MAP` (`:51-58`), config never read; FTSE/WallSt/Japan/HK/Russell get `0.0` | n/a (but S&P 30.0 / NASDAQ 100.0 in that map are 15×/12.5× IG — its own divergence) |
| **Breakout sweeps** | **NO** — synthetic `median_close × 0.0005` / `0.0003` | n/a — **the breakout PF/R figures in MEMORY.md are unaffected** |
| **Root `backtest.py`** (5–8 scripts) | **YES** — `:431` floor and `:433` cap `(min_stop × 20)` with **no price term**, unlike live | AI Index **54.78%** of 5m bars capped to 20 pt (live caps 0.00%); raising would *loosen* it and make pre/post runs **non-comparable** |

The **−15.86R breakout-shadow tally is not biased by `min_stop_distance`**; current value −16.034R (24 closed live rows). It *is* optimistic by cost: 0.845R of stored one-way spread → −16.879R net. **UNRECONCILED:** stored spread implies 0.035R/trade while the entry-fill work recorded `cost_pips ≈ 0.16R`; at 0.16R the tally is nearer −19.9R.

## 5. ARE IG'S VALUES SESSION-STABLE? — **NO. Proven, two ways.**

Four fresh `get_market_info` snapshots (10:56, 11:08, 11:10, 11:16 UTC on 2026-08-19), mutually identical, vs the 21:50Z capture: **5 of 13 minima halve** — FTSE 8→4, NASDAQ 8→4, Russell 2→1, S&P 2→1, Japan 40→20 — always by exactly 2×; **Japan `minDealSize` 0.5 → 0.04** (12.5×); spreads roughly quarter. Independent corroboration from pre-clamp orders (before `239afd6`, 2026-04-28), the only uncontaminated observations: IG **accepted** Russell stops of **1.77** (2026-04-17) and **1.70** (2026-04-27) — impossible if the 2.0 capture applied. `unit == "POINTS"` on all 13 epics (prior unknown, now closed). This **falsifies `src/client.py:1088`**, which asserts the value is "not a tick-by-tick quantity, so a day-long cache is ample" — the 24h cache can serve a stale out-of-hours value.

**Reviewer disagreement, adjudicated against `config.py` trading windows (all UTC):** the ~11:10Z samples fall **inside** the bot's window for FTSE (8–17), Wall Street / NASDAQ / Russell (4–20), AI (4–22) and forex/commodities (23–21) — the safety reviewer is right that those readings are the operative ones, since that is when orders are placed. But they fall **outside** Japan (0–8, `config.py:835-836`) and Hong Kong (1–9, `:847-848`) — the premise reviewer is right that **Japan and HK have never been sampled inside their own trading windows.** Partial rescue for HK only: the 2026-08-10 clamp log at 09:10 BST = **08:10 UTC** is inside HK's 1–9 window and confirms 40.00 with an accepted order. Japan's in-window minimum is **entirely unmeasured** (though 11 live Japan trades at 0.04-multiples prove `minDealSize = 0.04` in-window).

**Honest count: config is below IG's operative minimum on Wall Street (3×), FTSE (4×), and Hong Kong (2×, one in-window datapoint). NASDAQ, Russell match. Japan unknown.**

## 6. RANKED ASSUMPTIONS A PRE-FLIGHT MUST STATE (worst consequence first)

1. **"IG's minimum is a fixed instrument property, so a captured number can be hard-coded."** FALSIFIED (§5). Worst case: hard-coding the 21:50Z values pins config **above** IG's in-window minimum on FTSE/NASDAQ/Japan/Russell, permanently. Both clamps are monotone upward (`main.py:1701`, `:879`) — **nothing in the codebase ever lowers a stop.** A too-low config self-heals every order; a too-high one never does.
2. **"`min_stop_distance` is a broker mirror."** FALSIFIED (§1). Changing it silently retunes `max_sane_atr` (×50), `max_stop` (×20), `max_safe_stop` (×25) and the root-backtest cap (2–8× shift). Live ceilings are price-dominated today so nothing moves — but that is arithmetic on today's prices, not a property.
3. **"`rejected_signals` measures the cap-skip rate."** FALSIFIED, verified today: `main.py:891-893` logs and returns with no journal write. `rejected_signals` holds **12 cap-skip rows ever, all momentum, all 2026-06-10→06-29, none Gold** — while 9 real Gold breakout cap-skips sat in the log on 2026-08-06. The clearing query for this change returns a **false zero for every breakout market**.
4. **"The clamp makes live neutral, so the config value is measurement-only."** FALSIFIED for Hong Kong: the clamp raises the stop but not the limit (`main.py:1707-1708`), degrading R:R on ~39% of HK candidates to a median 1.17 vs 2.0. And raising HK's floor would move the take-profit (`src/strategy.py:204`) on **38.5%** of candidates — an unbacktested live P&L change.
5. **"IG accepts orders at IG's stated minimum."** Contradicted in-repo: `config.py:1134` EUR/USD "IG rejects stops at minimum"; `:1153` GBP/USD "IG rejects at 3.0 when pre-London spread widens"; `src/client.py:1047-1063` records a DXY amendment succeeding ~3.5 pt above a stated 10.0 pt minimum. The clamp buffer is +0.5.

## 7. OPTIONS AND RECOMMENDATION

**(a) Do nothing** — correct for the six numbers, wrong overall: it leaves the three collateral defects standing.
**(b) Edit the six numbers** — **reject**, because 2 of 6 already match in-window, 2 more are unmeasured in-window, and hard-coding a snapshot trades a self-healing error for an uncorrectable one.
**(c) Add a separate broker-minimum field refreshed from the API** — **reject as a live-path input**, because the live path already fetches fresh and uncached at order time, so a stored mirror only adds a staleness surface on a value now proven to swing 2× intraday. (A log-only/backtest-only variant is defensible, and would fix §4's root-`backtest.py` divergence.)

**RECOMMENDED — (d), three separable items, none of which touches the six numbers:**
1. **Map the schedule before anyone freezes a number.** `get_market_info` is free. Log `minNormalStopOrLimitDistance` + `minDealSize` for all 13 epics hourly for 48h; the 21:00–22:00 UTC transition and the Japan/HK windows (0–9 UTC) are unsampled. Reconsider `MIN_STOP_DISTANCE_TTL_HOURS = 24`.
2. **Journal the breakout path's rejections** (`main.py:891-893` → `_log_suppressed_signal`). Highest-value item here: it is blind to `/healthcheck` and to `rejected_signals`, it already cost one live Gold entry, and Gold is the only market on live breakout.
3. **Rename/document the field** (`strategy_min_stop`, or a rationale comment per value) so the two concepts stop sharing a name. Zero behavioural risk; this is the actual root cause of the false alarm.

Also worth a separate pre-flight, deliberately **not** bundled: the clamp's failure to rescale `limit_distance` (~40% haircut on HK's reward leg), and `client.py:403` defaulting to `0.0` instead of `None`.

## STILL UNKNOWN — HUMAN MUST DECIDE

1. **Japan's and Hong Kong's in-window minima** (0–8 / 1–9 UTC). Never sampled. HK has one 08:10 UTC datapoint (40.0); Japan has none. Any HK/Japan decision rests on this.
2. **Whether IG's stated minimum is IG's rejection threshold** — repo evidence says no on EUR/USD, GBP/USD and DXY. The +0.5 buffer may be too small at pre-London spreads.
3. **Whether momentum shadow *should* record pre-clamp stops.** `main.py:1654-1655` says yes by design (signal quality ≠ execution) — but that makes shadow R non-comparable to live R on FTSE/HK/Russell for 41–75% of future rows. Design call, not a bug.
4. **Cost charge for breakout-shadow R**: 0.035R (stored spread) vs 0.16R (`cost_pips`). Do not quote a cost-adjusted tally until reconciled; the gap is −16.9R vs −19.9R.
5. **The −15.86R figure's provenance** — irreproducible from the current table (−16.034R live-closed). Treat as a stale snapshot.
---

## 2026-08-20 — "Is it the strategy or the settings?" — the live book answers it in R

**Question.** After a full year of paper trading the bot is down in £ and no closer to
go-live. Two proposed routes: (a) go back to basics and source new strategies from the
TradingView library, or (b) keep fine-tuning what we run. Which?

**Hypotheses.**
- H1 — The entry signal is weak; better signals exist and we haven't found them yet.
- H2 — The settings are wrong; the right per-market parameters would make it profitable.
- H3 — The strategies are fine but execution cost eats the edge.
- H4 — There is no edge of *tradeable size* available from past-price rules on these
  instruments, and the paper year has now measured that rather than failed to.

### Evidence — live journal, decomposed in R (n=317 closed, size-invariant)

R = pnl / (size x initial stop_distance), so it is comparable across risk-% eras and
across the Gold min-size distortion that dominates the £ figures.

| bucket | n | avgR | totalR | t | 95% CI on avgR |
|---|---|---|---|---|---|
| ALL CLOSED | 317 | **-0.041** | -13.04 | -0.84 | [-0.137, +0.055] |
| pre-MACD-gate | 262 | -0.047 | -12.27 | -0.86 | — |
| post-MACD-gate | 55 | -0.014 | -0.76 | -0.13 | — |

Pooled per-trade sd of R = **0.87** (live momentum), materially lower than the breakout
study's 1.64R because momentum exits cut trades short. That matters for power:
n = (2*0.87/0.10)^2 = **302 trades to resolve a +0.10R edge at t=2 — and we have 317.**

**This is the key structural difference from the breakout power result.** The breakout
arm needs ~2,400 trades (~7 yr) because its R distribution is fat-tailed. The momentum
arm does not. **The paper-trading year was adequately powered for the momentum arm and
returned a negative.**

Best market in the book is NASDAQ 100 at +0.223R (n=31, t=+1.86, CI [-0.012, +0.457]) —
the only arm even approaching significance, and it does not clear it.

### Evidence — cost is NOT the binding constraint for momentum

Live spread snapshot 2026-08-20 08:27 BST, cost/R = spread / median realised stop:

| market | cost/R | | market | cost/R |
|---|---|---|---|---|
| NASDAQ 100 | 0.026 | | Wall Street | 0.064 |
| Japan 225 | 0.029 | | S&P 500 | 0.067 |
| GBP/USD | 0.030 | | Hong Kong HS50 | 0.094 |
| Gold / EUR/USD | 0.031 | | **Crude Oil** | **0.185 FAIL** |
| FTSE 100 | 0.041 | | **Dollar Index** | **0.400 FAIL** |

Ten of twelve markets sit **inside** the 0.10R go-live gate. This **corrects the framing
carried over from the breakout study**: the 0.139R cost figure there is dominated by
0.286xATR *entry slippage on the channel break*, not by spread. Momentum pays ~0.03-0.09R
and is still flat. So:

- **Momentum: cost is low, gross edge is ~zero.** Adding back cost puts gross at ~+0.01R.
- **Breakout: gross edge real (+0.191R, t=+3.85, n=676), 73% destroyed by entry slip.**

Two arms, two *different* failure modes. Conflating them has been a persistent error.

### Confidence

- **H4 — HIGH for the momentum arm specifically.** n=317, CI upper bound +0.055R.
  Caveat that lowers it from HIGH-plus: trades are **not independent** (correlated markets,
  clustered in time), so effective n < 317 and the true CI is wider — realistically the
  upper bound is +0.05 to +0.09R rather than +0.055R. Either way an edge worth trading
  (>= +0.10R) is excluded. Also note this rules out **momentum as operated across 21
  markets and many config eras**, not momentum in principle.
- **H3 — HIGH for breakout, REFUTED for momentum.** See cost table above.
- **H2 — REFUTED (prior work).** Per-EPIC walkforward: 21/21 profitable in-sample,
  12/20 out-of-sample. `min_confidence` is regime-gated and bimodal so sweeping it is a
  no-op. Settings do not port across timeframes.
- **H1 — LOW support, and the test was weak.** 12 Alorse Pine strategies ported: all 12
  net-negative pooled OOS; best -0.028R vs incumbent -0.049R. **But that design had ~1%
  power at 60-cell correction**, so "nothing survived" was foreclosed. The load-bearing
  evidence against H1 is not the significance tests — it is the pooled point estimates all
  leaning negative, plus the barrier test below.
- **Mean reversion — DEAD, HIGH, fourth independent confirmation.** Symmetric 1R-barrier
  hit rate 49.50%, 95% CI 48.08-50.93%, n=4737.

The barrier test is the deepest explanation available and it applies to *any* past-price
rule: at these horizons on these instruments, direction is a coin flip. TradingView hosts
~100k published strategies but every one of them is a function of the same OHLCV series.
Sampling more of them is sampling from a distribution already measured as centred on zero.

### Self-critique

- **Short holds look terrible (<=15m: n=141, -19.00R) and 1-4h looks great (n=29, +8.96R).
  This is CONFOUNDED and must not be acted on.** A trade that goes wrong immediately exits
  immediately; hold time is an *outcome*, not a treatment. Same trap for the RSI-exit
  bucket (+1.179R, WR 95.5%, n=22) — RSI-overbought only fires after price has already run,
  so it marks good trades rather than causing them. Neither is evidence for holding longer.
  Testing it needs a replay that forces longer holds, not a cross-section. — LOW
- The "LIMIT" classification (any `Stop/limit hit` finishing positive) is **contaminated by
  break-even-stop hits**, which is why it averages +0.376R rather than the ~+2R a real
  take-profit would pay. The BE-at-0.7 setting converting winners into scratches is a live
  hypothesis, NOT yet a finding — and it is adjacent to the already-REFUTED Gold
  profit-protection result, so it needs its own pre-registered test before any change.
- R uses the *initial* stop; break-even moves mean realised risk < initial risk on some
  trades, mildly inflating |R| on the loss side. Small, and it biases against the bot.
- Survivorship/snooping: the per-market table is 21 markets deep and the best one (NASDAQ)
  is exactly what you would expect the max of 21 noisy draws to look like. Do not promote
  on it.

### Conclusion

**Neither route the question offered is the right one.** Fine-tuning is refuted (H2) and
strategy-shopping is sampling from a measured-zero distribution (H1/H4). The one place a
*measured, statistically significant* quantity is being destroyed by something mechanically
fixable is **breakout entry slippage**: +0.191R gross, 0.143R lost to filling market-on-close
after the channel break. That is an execution problem, not a strategy problem, and it is the
only lever left that acts on a number we have actually established.

### Next steps (ranked)

1. **Breakout entry via resting stop-order at the channel level** instead of market-on-close.
   Directly attacks 0.143R of a +0.191R edge. Needs a pre-flight (order path).
2. **Retire the momentum arm** or cut it to the one or two markets with any positive lean.
   It has been measured and it is flat; continuing to run it 13-wide generates cost and
   noise without generating information.
3. **Pre-registered BE-stop test** (does BE-at-0.7 convert winners to scratches?).
4. Do NOT: source more TradingView strategies, sweep more parameters, or add markets.

**Open.** Non-price data (economic calendar, carry/rate differentials, cross-asset) is the
only genuinely unexplored axis. No evidence behind it, and it is a large build — flagged,
not recommended.

---

## 2026-08-20 — Per-market triage: enable, disable, or switch strategy?

**Question.** Does the accumulated data justify disabling any instrument, enabling any,
or switching any between momentum and breakout?

**Hypotheses.**
- H1 — Some markets are structurally untradeable (cost) and should go regardless of edge.
- H2 — Some markets show enough edge separation to promote/demote on performance.
- H3 — Some markets are on the wrong strategy and should switch momentum <-> breakout.
- H4 — Nothing is decidable at this sample size and the honest answer is "no change".

### Evidence 1 — cost, and a correction I nearly shipped

Cost is the ONE quantity measurable at this sample size, so it is the only defensible
basis for a per-market decision. First pass used the screener's `ATR/Spread` (logged
6x/day + every 30 min) pooled across all sessions:

| market | unweighted cost/R | verdict |
|---|---|---|
| Hong Kong HS50 | 0.173 | FAIL |
| S&P 500 | 0.121 | MARGINAL |
| Wall Street | 0.117 | MARGINAL |

**That was wrong, and the error was mine.** An unweighted median samples hours the bot
does not trade equally. Re-bucketing `ATR/Spread` by hour-of-day and weighting by the
hour-of-day distribution of each market's REAL entries:

| market | unweighted | **entry-weighted** | delta | gate |
|---|---|---|---|---|
| AI Index | 0.067 | **0.199** | +0.132 | FAIL (but n=2) |
| Hong Kong HS50 | 0.173 | **0.111** | −0.062 | MARGINAL |
| Crude Oil | 0.096 | 0.096 | 0.000 | pass |
| Wall Street | 0.117 | **0.092** | −0.025 | pass |
| S&P 500 | 0.121 | **0.083** | −0.038 | pass |
| Gold | 0.067 | 0.062 | −0.005 | pass |
| FTSE 100 | 0.079 | 0.060 | −0.019 | pass |
| EUR/USD | 0.044 | 0.043 | — | pass |
| GBP/USD | 0.045 | 0.042 | −0.004 | pass |
| NASDAQ 100 | 0.046 | 0.035 | −0.011 | pass |
| Japan 225 | 0.058 | 0.032 | −0.026 | pass |

**The bot times its entries into CHEAPER hours than the session average in 10 of 11
markets.** That is an unlooked-for positive and it reverses the S&P/Wall St verdict.
Only AI Index goes the other way (+0.132), and at n=2 entries the weighting is not
trustworthy — it is already `shadow_only` so nothing live depends on it.

DXY is absent from the screener series (scored `[OFF]` almost always); a direct snapshot
gives **0.400R** — spread 8.0 against a 20.0 stop. It is already `breakout-shadow`, so
this is a reason to keep it benched, not a new action. — HIGH

### Evidence 2 — edge does not separate anywhere

Per-market momentum edge in R, 95% CIs (n=317 pooled). Widths are ±0.27 to ±0.78R.
Best is NASDAQ +0.223R (n=32, CI [−0.012, +0.457]) — which is **exactly what the maximum
of 21 noisy draws looks like** under a global null. Largest non-Gold sample is Wall
Street, n=48, −0.051R, CI [−0.279, +0.177] — contains zero comfortably.

No market separates from zero. Promoting on the best or demoting on the worst is
selection, not inference. — HIGH

### Evidence 3 — no breakout flip is supported

Live-forward breakout shadow: **−0.53R/episode pooled, n=29, 10 of 11 markets negative**
(after repairing the corrupted NASDAQ row #134, see the cd0b4a8 entry). Per-market n is
1–6, so the per-market numbers decide nothing, but the *uniformity* is the signal: this
is the observer doing precisely the job it was installed for, and it says no.

Reverse direction (take something OFF breakout) is equally unsupported: Gold breakout
live −0.067R (n=7) vs Gold momentum −0.033R (n=66) — both ~zero, n=7 cannot decide.
GBP/USD breakout live −0.214R (n=6) but its 730d backtest is the book's best (PF 1.89);
six trades must not overturn two years. — HIGH for "no flips", LOW on any per-market number

### Confidence / self-critique

- **H4 is the supported answer.** H1 fails once cost is measured correctly; H2 and H3
  fail on power.
- **What would disprove this:** a market whose entry-weighted cost exceeds ~0.15R with a
  trustworthy n. AI Index is the only candidate and needs more entries to qualify.
- **Simpler explanation checked:** "no change" could be laziness dressed as rigour. It is
  not — the cost analysis was run to find disables, produced three candidates, and two of
  them dissolved under a *better* measurement rather than a weaker one.
- **Survivorship:** deliberately not acting on the per-market table is the whole point;
  it is 21 markets deep and its extremes are noise.
- **On demo, disabling destroys information.** While paper trading, a market that is not
  structurally untradeable is worth more ON than OFF. This asymmetry flips at go-live,
  where GO_LIVE_CRITERIA caps live markets at 3.

### Conclusion

**No instrument changes. No strategy switches.** Two watch items for v3:
AI Index entry-weighted cost (needs n), Hong Kong at 0.111R (marginal, over gate).

### Next steps

1. Let Phase 1 tick-entry logging accumulate — it is the only live experiment running.
2. Re-run this triage at v3 with the AI Index and HK entry counts grown.
3. Do NOT act on the per-market momentum table at any n reachable this year.

---

## 2026-08-21 — Wall Street: "signals wrong, or spread, or strategy?"

**Question.** Wall Street is the worst market in the book (−£149, ~5W/14L on the
headline). Is the cause (a) bad signals, (b) cost/spread, (c) the strategy, or
(d) none of these?

**Hypotheses.** H1 cost · H2 direction asymmetry · H3 exit mechanics ·
H4 noise at small n · H5 regime.

**Evidence.**

*H1 COST — REJECTED (MEDIUM).* Round-trip spread ÷ median stop actually used:
Wall St 0.151R, S&P 0.136R, NASDAQ 0.086R, Japan 0.061R, Hong Kong 1.084R.
Wall St is mid-pack and near-identical to S&P — yet S&P is break-even and Wall
St is not, so cost does not separate them. Hong Kong carries by far the worst
cost and is the second most profitable (+1.86R, PF 1.95). ⚠️ Measured ~22:00 BST
with US/HK cash closed, so all spreads are out-of-hours inflated; the relative
comparison is indicative, not exact. Re-measure in-session before quoting.

*H2 DIRECTION — CONFOUNDED, not supported (MEDIUM).* Live split looks strong:
BUY n=34 avgR −0.242, SELL n=16 avgR +0.252, diff +0.494R, t=2.22 (p≈0.03).
**But the sides are not interleaved.** Monthly: 2026-07 was 10 longs, −6.31R and
**zero shorts** — 77% of the entire long deficit sits in one month with no short
control. Sequence shows an 18-trade run of consecutive longs. So "longs are bad"
and "July was bad" are not separable in this sample. Prior Yahoo sweep across two
windows found Wall St **flat both ways in both windows — no asymmetry**
([[project-indices-direction-sweep-2026-07]]), which explicitly warns that
single-window direction verdicts are noise. Post-gate the asymmetry does persist
(BUY −3.04R/8, SELL +1.31R/5) but n is far too small to promote.

*H3 EXIT — entangled, not independent (LOW).* "MACD histogram negative for 3
candles" is the single biggest drag (n=15, −4.89R, avgR −0.326) — but that is the
LONG exit, so it re-states H2 rather than adding to it.

*H4 NOISE — STRONGLY SUPPORTED (HIGH).* All-time avgR −0.084 (n=50), and
post-MACD-gate avgR −0.133 with **95% CI [−0.507, +0.240] — comfortably includes
zero**. Wall Street is not statistically distinguishable from break-even.

*H5 REGIME — supported by prior work (MEDIUM).* The 2026-07-13 investigation of
this same question concluded "Wall St momentum broadly unprofitable across ALL
hours → a REGIME (chop) problem, not a session-structure problem"
([[project-wallst-session-filter-2026-07]]).

**KEY FINDING (HIGH).** Bucketed by config era, the damage is almost entirely
PRE-gate. Post-2026-07-24 the cumulative R runs:
`+0.34 −0.26 −1.00 +1.07 −0.46 +0.00 +0.12 +0.00 +0.82 +0.37 −1.00` = **exactly
0.00R over 11 trades**, then today's two losses take it to −1.73R over 13. Until
today, post-gate Wall Street was dead flat. The MACD coherence gate deployed at
the v2 review appears to have done its job, and today is a 2-trade drawdown on
top of a flat series — not evidence of a new problem.

⚠️ The headline **£** figure badly overstates recency: risk-per-trade changed on
2026-08-20, so recent trades carry far more £ per unit R. Use R, not £.

**Conclusion.** Most supported: **H4 + H5** — Wall Street is a marginal,
regime-sensitive market that is statistically indistinguishable from break-even,
and the alarming headline is dominated by pre-gate July. Ruled out: cost (H1).
Not supportable on this sample: direction asymmetry (H2), which is confounded
with time and contradicted by the prior two-window backtest.

**Open questions / next steps.**
1. Re-measure spreads IN-SESSION — the H1 rejection currently rests on
   out-of-hours quotes.
2. The prescribed test for H2 is the **IG candle archive walk-forward via live
   `analyze()`** (not Yahoo) — per the direction-sweep memory. Only that can
   separate "longs are bad" from "July was bad".
3. Decision question, separate from the statistics: per
   [[feedback-per-epic-profitability]], a reliably break-even market still
   consumes a position slot and adds variance without adding expectancy. Culling
   Wall Street is defensible on portfolio grounds even though "it is broken" is
   NOT supported. This belongs in the v3 review (agenda item 1), not a same-day
   reaction.

### 2026-08-21 (later) — IG archive walk-forward on the Wall Street direction question

**Method.** `scripts/backtest_wallst_direction_archive.py`. IG candle archive
(14,389 native 5m bars, 2026-06-12 → 2026-08-21), driven bar-by-bar through the
live `TradingStrategy.analyze()` and `should_close_position()`. HTF computed from
CLOSED hourly bars only (resample → shift(1) → asof-merge) to avoid the
`htf_series` look-ahead class. Run in `.venv-bt`, which is now byte-identical to
the container, so no CPU load on the live bot and no validity gap.

**⚠️ Two harness bugs, both caught by the POSITIVE CONTROL failing. Keep controls.**
1. One spread (4.8) applied to every market. S&P's median stop is 8.8 points, so
   that charged it 0.55R of pure cost per trade → S&P came out at PF 0.13. Fixed
   to per-market spreads; S&P moved to PF 0.79, near its live break-even.
2. Confidence gated on the STRATEGY PROFILE's `min_confidence` instead of
   `MarketConfig.min_confidence` (`main.py:1751`).

**STRUCTURAL FINDING — the direction question is largely moot (HIGH).**
`main.py:1685` applies a GLOBAL direction gate driven by the S&P 500 HTF trend:
regime BULLISH blocks every SELL, BEARISH blocks every BUY, NEUTRAL blocks all
trades. **A market's live long/short mix is therefore not a free choice — it is
dictated by what the S&P was doing at the time.** That mechanically explains why
2026-07 was 10 Wall Street longs and zero shorts (S&P regime bullish throughout),
which was the confound flagged earlier the same day. "Make Wall Street short-only"
is not a coherent request: the regime gate already decides direction book-wide.
Modelling the gate is also worth +5.89R over the window (n 87→70, −19.29R→−13.40R),
i.e. the gate is doing real work.

**RESULT — no direction asymmetry (HIGH).** Wall Street, gate on:
| side | n | WR | sumR | avgR | PF |
|---|---|---|---|---|---|
| BUY | 48 | 29.2% | −10.02 | −0.209 | 0.54 |
| SELL | 22 | 31.8% | −3.38 | −0.154 | 0.59 |

Gap is **0.055R/trade** — negligible, versus the +0.494R the live journal
suggested. Fold verdicts across 5 consecutive windows: `BUY, SELL, SELL, BUY,
SELL` — flips. Two independent methods now agree (Yahoo two-window 2026-07;
IG archive 2026-08): **Wall Street has no exploitable direction asymmetry.**
The live split was the July/regime confound, confirmed.

**⚠️ POWER CAVEAT (HIGH).** The S&P control did NOT reproduce its known long-only
edge (my run: BUY PF 0.64 vs SELL PF 0.92 — shorts better). This is the documented
behaviour, not a new bug: [[project-indices-direction-sweep-2026-07]] records that
S&P's asymmetry appears "ONLY on the big 1h sample (BUY PF 12.3 vs SELL 0.35), not
the thin 5m one". **So this harness can fail to find an asymmetry but cannot prove
one absent.** The Wall Street conclusion rests on the AGREEMENT of two methods,
not on this run alone.

**Secondary (MEDIUM).** Over the same window the archive says Wall Street momentum
is unprofitable in BOTH directions (PF 0.55, −13.40R over 70 signals) — a harsher
read than live's post-gate flat. The replay does not model the screener, position
caps or cooldowns, so live trades a filtered SUBSET of these signals; the gap
between −13.40R replayed and ~0.00R realised is the value those gates add.

**Next.** Re-run when the archive reaches ~180 days; at 70 days a 5m sample
provably cannot resolve a direction structure. Nothing here justifies a config
change.

### 2026-08-21 — "Is the screener any good? Can it be improved?"

**Premise check.** The screener does NOT decide whether a signal is any good — it
decides which 11 of 13 markets get slots. By volume it is only the 4th-largest
entry gate (all-time `rejected_signals`): Outside hours 195 · Confidence 178 ·
Direction-restricted 137 · **Screener-inactive 99** · Pullback-expired 85 ·
Regime NEUTRAL 84 · Regime BULLISH-blocks-SELL 63 · Regime BEARISH-blocks-BUY 35.
The three regime gates together (182) outweigh it.

**It passed Wall Street.** Today's screen had Wall Street ACTIVE at 62/100. So the
screener is neither saving nor failing us there — it is letting it through, which
is correct behaviour for its actual job (Wall Street's ADX/spread/vol were fine).

**The cap problem was already found AND fixed.** 2026-06 measurement
([[project-screener-veto-analysis]]): score 0–34 vetoes −0.99R (threshold correctly
blocked losers), score 45+ "Below top 8" vetoes **+3.27R** (the cap blocked
winners). Fix = cap 8→11 + 30-min re-screen (`cc84594`, `eaca8b2`).

**NEW (HIGH): the cap no longer binds.** Of 18 vetoes since that fix, 11 are CAP
vetoes and **all 11 are Hong Kong, all between 2026-06-12 and 2026-06-19**. There
have been **ZERO cap vetoes in the two months since 2026-06-19**; the remaining 7
are quality vetoes (score<40), the kind prior measurement showed correctly block
losers. This effectively answers the overdue "re-pull in ~2wk" review trigger: the
sample is thin *because the problem was fixed*, not because we stopped looking.

**Headroom (MEDIUM).** ~60 of the 100 score points duplicate what `analyze()`
already gates on (ADX 25, trend 20, HTF 15). The non-redundant contribution is the
ATR/spread tradeability filter (25) — the strategy is blind to spread — plus
portfolio allocation. So "improve the score" mostly means improving things the
strategy already does, on a component whose measured cost has already been removed.

**Conclusion.** Little measurable headroom left in the screener, and it is the
wrong lever for the Wall Street problem anyway: the screener ranks CURRENT
tradeability (a regime question), not "is this market ever worth trading" (a
config question). A market that should not trade should be disabled, not
down-scored. Conflating the two is what the question assumed.

**Next.** Nothing to change. If Wall Street is to go, that is a v3 cull decision.
