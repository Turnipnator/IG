#!/usr/bin/env python3
"""
MACD exit-LENGTH A/B, per EPIC (v3 agenda item 14).

Question: the live exit closes on 3 consecutive opposing MACD-histogram bars
(strategy.py `range(1, 4)`). Is 3 right, per market?

Method — the backtest_crude_exit.py shape, but driven through the HARDENED
src/backtest.py engine rather than a parallel re-implementation, so the
2026-08-19 `117a4e8` fixes apply (intrabar stop/TP on low/high, gap-through
fills at the open, stop beats target within a bar, no HTF look-ahead):

  * ENTRY HELD FIXED at each market's live profile. Only `macd_exit_bars` varies.
  * Arms: N = 2, 3 (LIVE), 4, 5, and OFF (`use_macd_exit=False` equivalent) —
    OFF matters because "stop/limit + RSI only" is a real config, not a control.
  * SPREADS ARE TRADING-HOURS MEDIANS, not spot snapshots. A snapshot taken
    outside a market's session is worthless: Hong Kong's ATR/Spread runs 5-9x
    inside its 02:00-04:00 BST window and collapses to 0.1-0.6x outside it, so a
    midday reading gave 30pt when the real trading-hours figure is 7.11pt — a
    4.2x over-charge that made HK look uneconomic. Each spread here is
    median(archive ATR in hour h) / median(logged ATR-Spread ratio in hour h),
    taken over the market's OWN permitted trading hours (config trading_start..
    trading_end, converted UTC->BST to match archive stamps).
  * COST IS CHARGED. The engine charged nothing until 2026-08-31; an uncosted
    sweep flatters arms that trade more often, which is precisely the axis under
    test here. Costs are each market's REAL IG DFB spread, measured live, then
    swept to 1.5x and 2x for sensitivity.
  * `force_profile_stop=True` — the engine otherwise overrides the stop with the
    regime value, which live never applies.

Data: the IG-NATIVE candle archive (5m), ~80 days from 2026-06-12. Chosen over
Yahoo because it is the instrument actually traded (IG DFB prices and sessions),
it is longer than Yahoo's 60-day 5m cap, and ^GSPC/^NDX are cash indices whose
sessions do not match the DFB. Costs nothing against the 10k/week REST budget.

Report is STRICTLY PER MARKET. Pooling is what made the aggregate MACD-3 figure
misleading in the first place.
"""
import argparse, json, os, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETS as _ALL_MARKETS
from src.backtest import Backtester, DEFAULT_PARAMS, DISABLE_MACD_EXIT

# Archive timestamps are Europe/London; this window (Jun-Aug 2026) is entirely BST,
# so live's UTC trading hours are +1 in frame-local terms. Explicit, because a run
# spanning a DST boundary would need real tz conversion rather than this constant.
_BST_OFFSET = 1
_MKCFG = {m.name: m for m in _ALL_MARKETS}


def live_entry_hours(market):
    """(lo, hi) frame-local entry window mirroring main.py's live hours gate."""
    m = _MKCFG[market]
    return ((m.trading_start + _BST_OFFSET) % 24, (m.trading_end + _BST_OFFSET) % 24)

ARCHIVE = Path(os.environ.get("IG_ARCHIVE_DIR", "data/candle_archive"))

# name -> (epic, live profile overrides, measured IG DFB spread in points)
MARKETS = {
    "S&P 500": ("IX.D.SPTRD.DAILY.IP",
                dict(adx_threshold=25, stop_atr_multiplier=1.5, long_only=True), 0.6),
    "NASDAQ 100": ("IX.D.NASDAQ.CASH.IP",
                dict(adx_threshold=30, stop_atr_multiplier=2.0), 2.0),
    "Japan 225": ("IX.D.NIKKEI.DAILY.IP",
                dict(adx_threshold=30, stop_atr_multiplier=1.5), 9.63),
    "Hong Kong HS50": ("IX.D.HANGSENG.DAILY.IP",
                dict(adx_threshold=30, stop_atr_multiplier=1.5), 7.11),
}

# Shared live values for all four (indices* profiles, config.py)
BASE = dict(ema_fast=5, ema_medium=12, ema_slow=26, rsi_period=7,
            rsi_overbought=70, rsi_oversold=30, rsi_buy_max=55, rsi_sell_min=45,
            reward_risk_ratio=2.0, force_profile_stop=True)


class ArchiveBacktester(Backtester):
    """Backtester that reads the IG candle archive instead of Yahoo."""

    def __init__(self, epic, params=None):
        self.epic = epic
        self._cache = {}
        self.params = params or DEFAULT_PARAMS.copy()
        self.trades = []

    def fetch_data(self, market, days=30, interval="5m"):
        key = interval
        if key in self._cache:
            return self._cache[key].copy()
        path = ARCHIVE / f"{self.epic}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"])
        df = df[["date", "open", "high", "low", "close"]].dropna()
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        if interval in ("1h", "60m"):
            g = df.set_index("date").resample("1h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            df = g.reset_index()
        # NOTE: run() calls add_indicators() on the entry frame and adds
        # ema_9/ema_21 to the HTF frame itself — do not pre-apply either here.
        self._cache[key] = df
        return df.copy()


def arm_params(market, n, cost):
    p = DEFAULT_PARAMS.copy()
    p.update(BASE)
    p.update(MARKETS[market][1])
    p["cost_points"] = cost
    # Live REFUSES entries outside the market's hours (main.py:1780). Without this the
    # sweep trades thin overnight sessions the bot never touches -- worth 0.91->1.52 PF
    # on NASDAQ and 1.27->0.98 on Japan, i.e. it can flip a market's verdict outright.
    p["entry_hours"] = live_entry_hours(market)
    if n is not None:
        p["macd_exit_bars"] = n
    return p


def run_arm(market, n, cost, days):
    epic = MARKETS[market][0]
    DISABLE_MACD_EXIT.clear()
    if n is None:                      # OFF arm: no MACD exit at all
        DISABLE_MACD_EXIT.add(market)
    bt = ArchiveBacktester(epic, arm_params(market, n, cost))
    try:
        res = bt.run(market, days=days, interval="5m", htf_interval="1h",
                     require_htf_alignment=True, min_confidence=0.55,
                     min_hold_candles=1, reentry_cooldown_mins=30)
    finally:
        DISABLE_MACD_EXIT.clear()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--markets", default="")
    a = ap.parse_args()
    names = [m.strip() for m in a.markets.split(",") if m.strip()] or list(MARKETS)

    arms = [("N=2", 2), ("N=3 LIVE", 3), ("N=4", 4), ("N=5", 5), ("OFF", None)]
    for market in names:
        spread = MARKETS[market][2]
        print(f"\n{'='*78}\n{market}   (IG DFB spread = {spread} pts)\n{'='*78}")
        for mult, label in ((0.0, "frictionless"), (1.0, f"{spread}pt REAL"),
                            (2.0, f"{spread*2:g}pt 2x")):
            cost = spread * mult
            print(f"\n  -- cost {label} --")
            print(f"  {'arm':10s} {'n':>4s} {'WR':>6s} {'PF':>6s} {'totPnL%':>9s} {'avgW%':>7s} {'avgL%':>7s}")
            for label_a, n in arms:
                try:
                    r = run_arm(market, n, cost, a.days)
                except Exception as e:
                    print(f"  {label_a:10s}  ERROR {e}")
                    continue
                # NB BacktestResult.win_rate is a FRACTION (wins/total), not a
                # percentage — scale it here or 50% prints as "0.5%".
                print(f"  {label_a:10s} {r.total_trades:4d} {r.win_rate*100:5.1f}% "
                      f"{r.profit_factor:6.2f} {r.total_pnl:+9.2f} "
                      f"{r.avg_win:+7.2f} {r.avg_loss:+7.2f}")


if __name__ == "__main__":
    main()
