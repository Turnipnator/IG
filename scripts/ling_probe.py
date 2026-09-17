"""Knowledge probe: ask ling-3.0-flash-fin questions this repo has already measured (research_notes.md, 2026-09-15)."""
import concurrent.futures as cf, json, os, time
from pathlib import Path
import requests
HERE = Path(__file__).resolve().parent
KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "inclusionai/ling-3.0-flash-fin:free"
TAIL = ("\n\nAnswer in at most 150 words. Say plainly if you are unsure. End with exactly one line: "
        "ANSWER: <your single best number(s) or YES/NO> | CONFIDENCE: <low/medium/high>")
Q = [
 ("momentum_5m",
  "A bot trades UK spread bets on index CFDs (S&P 500, NASDAQ 100, Dow, FTSE 100, Nikkei 225, Hang Seng) on 5-minute bars. Long entry: EMA9 > EMA21 > EMA50, close > EMA50, RSI(7) < 70, ADX > 25, daily trend agreeing (mirror for shorts). After such signals, what is the average signed forward price move in the signal direction, in ATR(14) units, at horizons from 15 minutes to 8 hours? Does the entry signal carry a tradeable directional edge before costs?",
  "~0 ATR at every horizon (-0.10..+0.22, z -1.0..+0.4), n=1,078 signals; NO edge; book lost -0.07..-0.13R/trade"),
 ("spx_pullback",
  "S&P 500 cash index, daily bars, 1985-2026, long only. Rule: when close > 200-day SMA, buy at the close on a close below the lowest low of the prior 5 sessions; sell at the close on a close above the highest high of the prior 5 sessions or after 10 sessions; 3xATR(20) stop. Costs: 0.6 point spread round trip plus overnight financing at benchmark rate + 2.5%/yr. Define R = 2xATR(20) at entry. What is the mean net return per trade in R, and roughly what win rate?",
  "+0.203R/trade net, hit 69%, PF 1.57, n=363; +0.207R on never-fitted 1985-2003"),
 ("open_fade",
  "On major stock index CFDs (US, UK, Germany, Japan, Hong Kong), does fading the first 15 minutes of the cash session (sell if it rose, buy if it fell, holding about 1 hour) have a statistically reliable, tradeable edge after a typical retail spread?",
  "NO - primary test failed; cells +0.02..+0.57 ATR, z 0.0-1.3, never significant (volatility, not direction)"),
 ("overnight_drift",
  "US stock index returns are said to accrue mostly overnight. On a UK spread bet (IG Daily Funded Bet) on the S&P 500, would going long at the cash close and closing at the next cash open, every day, be profitable after the spread and overnight financing?",
  "NO (HIGH confidence)"),
 ("gold_daily_trend",
  "Gold daily bars 2004-2026, long only: buy on a close above the 55-day high, exit on a close below the 20-day low, initial stop 2xATR(20). Charge overnight financing of about 5.8%/yr of notional plus a normal spread. Define R = the initial stop distance. What is the mean net return per trade in R and the profit factor?",
  "+0.69R/trade, PF 2.06 at 5.8%/yr financing"),
 ("eurusd_breakout",
  "EUR/USD on 1-hour bars: enter on a break of the 55-bar Donchian channel only in the direction of the daily EMA9/EMA21 trend, 2xATR stop, trail with the opposite 20-bar channel, spread about 0.6-0.9 pip. Over roughly the last 1-2 years, what profit factor after spread would you expect? Is it profitable?",
  "PF ~0.57 after spread, NOT profitable (kept shadow); same rule on GBP/USD ~1.19 [truth confidence MEDIUM: recent-quarter figure]"),
 ("drawdown_math",
  "A strategy has a genuine positive edge: each trade independently wins +1.75R with probability 0.40 and loses -1R with probability 0.60 (expectancy +0.10R). Over 300 trades, what is the probability that the peak-to-trough drawdown of the cumulative R curve reaches 21R or more at some point? Also give the median maximum drawdown in R.",
  "P(maxDD>=21R)=0.293; median maxDD=17.0R; p5/p95 10/32.5R (local Monte Carlo n=100k)"),
]
def ask(item):
    key, q, truth = item
    t0 = time.time()
    for attempt in range(3):
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=600,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": q + TAIL}], "reasoning": {"enabled": True}})
        d = r.json()
        if r.status_code == 200 and d.get("choices"):
            m = d["choices"][0]["message"]
            return dict(key=key, truth=truth, content=m.get("content"), reasoning_tokens=d["usage"]["completion_tokens_details"]["reasoning_tokens"], elapsed=round(time.time()-t0,1))
        print(key, "attempt", attempt+1, str(d)[:300], flush=True); time.sleep(30)
    return dict(key=key, truth=truth, content=None)
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(ask, Q))
out = HERE.parent / "data" / "ling_forecast" / "ling_probe.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1))
for x in res:
    print(f"\n==== {x['key']}  ({x.get('elapsed')}s, {x.get('reasoning_tokens')} reasoning tok)\n{x['content']}\n>>> REPO TRUTH: {x['truth']}")
