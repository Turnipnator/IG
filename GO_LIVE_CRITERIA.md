# Go-Live Criteria — pre-committed 2026-08-19

**Status: PRE-COMMITMENT. Written BEFORE the qualifying data exists, deliberately.**

Every criterion below was fixed on 2026-08-19, when no market qualified on the primary
gate. That is the point. Criteria written after seeing which market won are not criteria,
they are a rationalisation — and this book has already caught itself twice reading a
result and then reaching for the reason (the post-hoc long-only leg; the "cap-skips are
rare" claim resting on a query that returned a false zero).

Amendment rules are in §7. Read them before changing anything here.

---

## 1. What this is, and what it is not

Going live is **buying information, not expecting profit**. The measured position:

- Pooled gross breakout edge **+0.191R/trade** (t = 3.85, n = 676, 730d) — real.
- Measured cost **~0.139R/trade** book-average — about 73% of it. On Gold specifically,
  all-in cost is only ~0.022–0.042R, so the 73% is a book average, not Gold's number.
- Net residual **+0.048R/trade, t = 0.97** — positive on best estimate, statistically
  indistinguishable from zero.
- Resolving a +0.10R edge at 80% power needs **~2,400 trades ≈ 7 years** at the current
  rate. **No amount of further analysis closes this.** Only trades do.

So these criteria cannot ask "is the edge real?" — that question is unanswerable on any
useful timescale. They ask a weaker, answerable question: **is this market cheap enough,
structurally tradeable, and not contradicted by its own IG-native record?**

A market passing these gates is NOT proven profitable. It is proven *not obviously
broken*, on the axes that can actually be measured.

---

## 2. Hard gates — a market must pass ALL FOUR

### G1 — Cost ≤ 0.10R all-in
Execution cost plus financing at the strategy's expected hold. Measured per market, not
assumed. Rationale: the pooled gross edge is +0.191R; a cost above 0.10R consumes more
than half of it before anything else goes wrong. Cost is the ONLY quantity here measured
with near-certainty, so it carries the hardest gate.

    PASS  Wall St 0.009 · Gold 0.012 (+0.010/night) · S&P 0.054 · NASDAQ 0.077
          EUR/USD 0.084 (earns carry) · Crude 0.091
    FAIL  FTSE 0.145 · GBP/USD 0.146 · Hong Kong 0.157 · Russell 0.219
          Japan 0.239 · DXY 0.372

### G2 — Structurally tradeable
`IG min_deal_size × typical stop` must fit the risk budget, OR the excess must be an
explicit accepted decision with `max_risk_gbp` raised so setups are not SILENTLY skipped.
Copper was retired for failing this; Gold fails it and is accepted under §4 Tier 2.

### G3 — ≥30 IG-native trades on the exact strategy-market pair
**Live journal or breakout-shadow only. Yahoo backtests do NOT satisfy this gate for the
seven index markets** — Yahoo sees 43–89% of the bars inside the bot's own trading window
and 42.6% of real entries fire in hours it cannot see. 30 does not prove an edge; it
detects a catastrophically bad market (PF < 0.5). It is a floor, not a verdict.

### G4 — Total R positive over that sample, and no trustworthy backtest veto
Positive over the ≥30 sample. A 730d backtest may VETO but never QUALIFY, and only where
its data is trustworthy: **Gold, the FX pairs, Crude, DXY. Never the seven indices.**

---

## 3. Portfolio gates

- **Maximum 3 live markets at launch.** Not 8. Diversification without concentration.
- **Maximum 2 from any one `correlation_group`.** S&P, NASDAQ, Wall Street, FTSE, Japan,
  HK and Russell are all `equity_index` — three of them is one bet wearing three hats.
- **Existing correlation-cluster filter stays enforced.**
- Any market breaching a gate after going live is **demoted to shadow the same day**, no
  discussion, no grace period.

## 4. Which markets stand where — as of 2026-08-19

### Tier 1 — passes all four gates
| market | strategy | cost | IG-native n | record |
|---|---|---|---|---|
| **S&P 500** | momentum | 0.054R | **33** | PF 1.13, +£23.57 |
| **NASDAQ 100** | momentum | 0.077R | **31** | PF 2.06, +£74.31 |

Both are `equity_index`, so together they exhaust the correlation allowance. No third
Tier-1 market exists.

### Tier 2 — trustworthy backtest, insufficient live evidence (weaker basis, admitted explicitly)
| market | strategy | why admitted | why it fails G3 |
|---|---|---|---|
| **Gold** | breakout | 730d PF 1.42 on data verified faithful (n=109); cheapest all-in cost on the book | live breakout n=**7**, PF 0.84 |

**Gold is admitted on backtest evidence, not live evidence, and that is a weaker basis.**
Per §1's decision: accept ~£35/trade (1.5× the £23.50 target) and raise `max_risk_gbp`
above £45 so the highest-ATR 16% of setups stop being silently dropped — adverse
selection against exactly the trades the backtested edge lives on. Gold cannot be sized
down: IG's 1.0 minimum is the floor.

### Fails, and why — recorded so it is not relitigated
- **Wall Street** — cheapest market on the book (0.009R) and n=47, but all-time PF 0.95 /
  −£13.23 fails G4. Post-gate it is +£14.29 over 10 trades; **re-test at n≥30 post-gate.**
- **Japan 225** (+£17.65, PF 1.40) and **Hong Kong** (+£78.46, PF 2.89) — fail G1 on cost
  (0.239 / 0.157) and G3 on sample (11 / 10).
- **FTSE, Russell, GBP/USD, DXY, Crude, EUR/USD, AI Index** — fail two or more gates.
- **All breakout markets except Gold** — fail G3 badly (n = 2–9).

**Note the tension honestly: no market has both trustworthy backtest evidence AND
sufficient live evidence.** Gold has the first, S&P/NASDAQ have the second. That is the
actual state of knowledge, and no analysis available closes it.

---

## 5. Sizing and the risk budget

- **Cumulative loss limit: £500.** Hard stop. On reaching it the live experiment ENDS and
  everything returns to shadow — no averaging down, no "one more week".
- **Fund the live account at ~£9–10k**, comparable to the demo. The £500 is a drawdown
  stop, NOT the account size: position sizing is a percentage of balance and IG's
  minimums impose a floor, so a thinly funded account cannot size correctly at all.
- Per-trade risk stays at the current ~£23.50 (Gold forced to ~£35).

**£500 ≈ 21R. Understand what that means:** at the measured per-trade dispersion, a
process with a genuinely positive edge still hits a 21R drawdown roughly **40–50%** of
the time before it hits +21R. **The stop firing is NOT evidence the edge is absent.** It
is a budget limit. Do not read a verdict into it, in either direction.

## 6. Stop and review conditions

- **Stop immediately:** cumulative loss reaches £500 · any Tier-1 market's live record
  goes negative over its own next 20 trades · any correctness bug found in the order path.
- **Scheduled review:** at 100 live trades, or 3 months, whichever comes first.
- **At review:** re-apply §2 to every market on fresh data. Markets may be added ONLY by
  passing all four gates. Nothing is grandfathered.

## 7. Amendment rules — the part that matters

1. **Any change requires a written reason recorded here, dated, before the change.**
2. **Loosening a gate resets the evaluation window to zero trades.** Loosening after a bad
   run is exactly the failure this document exists to prevent.
3. **Never add a market mid-window.** Wait for review.
4. **A gate may not be weakened to admit a specific market.** If Gold cannot pass G3, the
   answer is to wait for its 30 trades, not to move G3 to 7.
5. **Re-running an analysis hoping for a friendlier answer is prohibited.** If a result is
   re-measured, the new number replaces the old regardless of which is more convenient —
   as happened on 2026-08-19 when a Gold breakout PF of 1.17 turned out to be a bad data
   fetch and the honest number, 1.44, was *better*; and again when the six
   `min_stop_distance` "defects" turned out to be a sampling artefact and no change was
   made.

---

## 8. Open items that would change this document

- **The seven index markets are UNMEASURABLE on Yahoo.** Only the shadow record can
  qualify them, and at ~0.17 trades/market/day that is ~6 months to n=30. This is the
  single highest-value passive process running.
- **DXY** — its live-faithful PF swings 0.63 to 1.13 on a spread that was derived, never
  measured. One mid-session `get_market_info` call settles it. It fails G1 either way.
- **EUR/USD** — shadow-only on a pre-look-ahead-fix figure (0.57); post-fix 730d is 1.083
  and it earns carry. Shadow record is the book's worst (−6.73R over 9). Unresolved.
