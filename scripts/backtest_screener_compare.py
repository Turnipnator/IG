"""
Backtest comparison: screener ON vs screener OFF.

Approach:
  1. Use the existing run_backtest to get the unfiltered trade list per market
     (this is the "screener OFF" world — strategy fires for every market).
  2. Build a screener timeline by running MarketScreener.score_market on each
     market's historical candles at fixed scoring intervals (every 4 hours).
     At each interval, mark the top N markets (score >= 40) as active.
  3. For the "screener ON" world, drop trades whose entry timestamp fell
     during a window when their market was NOT in the active set.
  4. Print aggregate comparison: trade count, win rate, total P&L, breakdown
     per market.

Limitations:
  - Only covers markets in TICKER_MAP (Yahoo Finance) — that's ~11 of 17.
    Six markets (NatGas, NY Cocoa, NY Cotton, Soybeans, AI Index, T-Notes,
    Silver) have no Yahoo data and are excluded. Result is directional, not
    exhaustive.
  - Spread is unknown historically — the screener's spread component gets a
    permissive default. We're explicitly testing whether the trend/vol/HTF
    components of the screener add value, not the spread filter (which
    everyone agrees is useful).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

from config import MARKETS, get_strategy_for_market
from backtest import run_backtest, fetch_data, fetch_htf_data, calculate_htf_trend, lookup_htf_trend, TICKER_MAP
from src.screener import MarketScreener


SCORING_INTERVAL_HOURS = 4
ACTIVE_TOP_N = 8
ACTIVE_MIN_SCORE = 40
DAYS = 30


def build_screener_timeline(market_dfs: dict, htf_dfs: dict) -> list[tuple]:
    """Return [(timestamp, set_of_active_market_names), ...] sorted by time.

    Score every market at each 4-hour boundary using its candles up to that
    point. Activate the top N with score >= threshold.
    """
    # Find global time range
    all_starts = [df["date"].min() for df in market_dfs.values() if not df.empty]
    all_ends = [df["date"].max() for df in market_dfs.values() if not df.empty]
    if not all_starts:
        return []
    start = min(all_starts)
    end = max(all_ends)

    # Need >= 50 candles before first scoring tick; round up to next 4h after start+50*5min
    first_score_at = start + timedelta(hours=12)  # plenty of warmup

    timeline = []
    screener = MarketScreener(client=None, max_active=ACTIVE_TOP_N)

    t = first_score_at
    while t <= end:
        scores = []
        for name, df in market_dfs.items():
            past = df[df["date"] <= t]
            if len(past) < 50:
                continue
            htf_trend = "NEUTRAL"
            htf_df = htf_dfs.get(name)
            if htf_df is not None and not htf_df.empty:
                htf_trends = calculate_htf_trend(htf_df)
                htf_trend = lookup_htf_trend(htf_trends, t)
            score = screener.score_market(
                epic=name, name=name, df=past.tail(100), spread=0, htf_trend=htf_trend
            )
            if score:
                scores.append(score)

        # Pick top N with score >= threshold
        scores.sort(key=lambda s: (s.score, s.atr_spread_ratio, s.adx), reverse=True)
        active = set()
        for i, s in enumerate(scores):
            if i < ACTIVE_TOP_N and s.score >= ACTIVE_MIN_SCORE:
                active.add(s.name)
        timeline.append((t, active))
        t += timedelta(hours=SCORING_INTERVAL_HOURS)

    return timeline


def active_at(timeline: list[tuple], ts: datetime) -> set:
    """Return the active set in effect at the given timestamp (most recent
    scoring tick at or before ts)."""
    if not timeline:
        return set()
    active = set()
    for tick_ts, tick_active in timeline:
        if tick_ts > ts:
            break
        active = tick_active
    return active


def main():
    backtestable = [m for m in MARKETS if m.name in TICKER_MAP]
    print(f"\nBackt­esting {len(backtestable)} markets ({DAYS} days)\n")

    # 1. Run no-screener backtests + collect candle dataframes
    market_dfs = {}
    htf_dfs = {}
    all_results = []
    for m in backtestable:
        print(f"  {m.name}...", end=" ", flush=True)
        df = fetch_data(m.name, days=DAYS, interval=f"{m.candle_interval}m")
        if df is None or df.empty:
            print("no data")
            continue
        market_dfs[m.name] = df
        htf_df = fetch_htf_data(m.name, days=DAYS)
        if htf_df is not None:
            htf_dfs[m.name] = htf_df
        result = run_backtest(m, days=DAYS)
        if result is None:
            print("no trades")
            continue
        all_results.append((m, result))
        print(f"{len(result.trades)} trades, P&L £{result.total_pnl:+.2f}")

    if not all_results:
        print("Nothing to compare.")
        return

    # 2. Build screener timeline
    print(f"\nBuilding screener timeline (every {SCORING_INTERVAL_HOURS}h)...")
    timeline = build_screener_timeline(market_dfs, htf_dfs)
    print(f"  {len(timeline)} scoring ticks")

    # Sample of active sets
    if timeline:
        sample_idx = len(timeline) // 2
        sample_ts, sample_active = timeline[sample_idx]
        print(f"  Sample (mid-window {sample_ts}): active={sorted(sample_active)}")

    # 3. Aggregate two ways: all trades, and screener-filtered
    def aggregate(label: str, trade_filter):
        total_pnl = 0.0
        wins = losses = bes = 0
        per_market = {}
        for m, result in all_results:
            kept = [t for t in result.trades if trade_filter(m, t)]
            for t in kept:
                total_pnl += t.pnl
                if t.pnl > 0.5:
                    wins += 1
                elif t.pnl < -0.5:
                    losses += 1
                else:
                    bes += 1
            per_market[m.name] = (
                len(kept),
                sum(1 for t in kept if t.pnl > 0.5),
                sum(1 for t in kept if t.pnl < -0.5),
                sum(t.pnl for t in kept),
            )
        total_trades = wins + bes + losses
        wr = (wins / (wins + losses) * 100) if (wins + losses) else 0
        print(f"\n=== {label} ===")
        print(f"Trades: {total_trades} ({wins}W / {bes}BE / {losses}L) | "
              f"WR: {wr:.1f}% | P&L: £{total_pnl:+.2f}")
        print("By market:")
        for name, (n, w, l, p) in sorted(per_market.items(), key=lambda x: x[1][3]):
            print(f"  {name:22s} | {n:3d}t {w:2d}W/{l:2d}L | £{p:+7.2f}")
        return total_pnl, total_trades, wins, losses

    pnl_off, n_off, w_off, l_off = aggregate("Screener OFF (all trades)", lambda m, t: True)

    def screener_keeps(m, t):
        active = active_at(timeline, t.entry_time)
        if not active:
            return True  # No timeline data yet → assume passes
        return m.name in active

    pnl_on, n_on, w_on, l_on = aggregate("Screener ON (filtered)", screener_keeps)

    print("\n=== Δ ===")
    print(f"Trades dropped by screener: {n_off - n_on}")
    print(f"P&L delta (ON − OFF): £{pnl_on - pnl_off:+.2f}")
    if (w_off + l_off) and (w_on + l_on):
        wr_off = w_off / (w_off + l_off) * 100
        wr_on = w_on / (w_on + l_on) * 100
        print(f"Win rate: OFF {wr_off:.1f}% → ON {wr_on:.1f}% (delta {wr_on - wr_off:+.1f}pp)")


if __name__ == "__main__":
    main()
