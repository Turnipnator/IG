"""
Backtest AI Index on 5m vs 15m candles using IG historical data directly.

AI Index is an IG-only synthetic — no Yahoo ticker exists, so we can't use
the standard yfinance-backed backtest. This script:
  1. Logs in to IG
  2. Fetches historical 5m AND 15m candles for IX.D.AIIDX.DAILY.IP
  3. Fetches 1h HTF data
  4. Patches backtest.fetch_data / fetch_htf_data to return our IG data
  5. Runs run_backtest on each timeframe with the indices strategy
  6. Compares trade count, win rate, P&L

API cost: ~2000 5m points + ~2000 15m points + ~200 1h points = ~4200
data points (vs 10k/week budget). Substantial but worth it for the
diagnostic.

Usage:
    docker exec ig-trading-bot python3 /app/scripts/backtest_ai_index.py
    (Run inside the container so it has IG creds via env)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import pandas as pd
from datetime import datetime, timedelta

from config import MarketConfig, load_ig_config
from src.client import IGClient
import backtest as bt
from backtest import run_backtest, print_result


EPIC = "IX.D.AIIDX.DAILY.IP"
NAME = "AI Index"

# On-disk cache so we don't refetch from IG on repeat runs (the user is
# rightly paranoid about burning API allowance).
DATA_CACHE = Path("/app/data") if Path("/app").exists() else Path("data")
CACHE_FILES = {
    "5m": DATA_CACHE / "ai_index_5m.json",
    "15m": DATA_CACHE / "ai_index_15m.json",
    "htf": DATA_CACHE / "ai_index_htf.json",
}
CACHE_MAX_AGE_HOURS = 24


def ig_to_df(prices: list) -> pd.DataFrame:
    """IGClient returns a DataFrame already, but normalize to backtest expects."""
    if prices is None or (hasattr(prices, "empty") and prices.empty):
        return None
    df = prices.copy()
    if "date" not in df.columns:
        # Ensure column name matches what fetch_data returns
        if "snapshotTime" in df.columns:
            df = df.rename(columns={"snapshotTime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_ig_history(client: IGClient, resolution: str, num_points: int, cache_key: str) -> pd.DataFrame:
    """Fetch via IGClient.get_historical_prices, with on-disk cache to avoid
    re-spending API allowance on re-runs."""
    cache_file = CACHE_FILES.get(cache_key)
    if cache_file and cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < CACHE_MAX_AGE_HOURS * 3600:
            df = pd.read_json(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            print(f"  [cache hit] {cache_key}: {len(df)} candles "
                  f"({age/3600:.1f}h old, range {df['date'].min()} -> {df['date'].max()})")
            return df

    print(f"  [API] Fetching {num_points} points at {resolution}...", flush=True)
    df = client.get_historical_prices(EPIC, resolution=resolution, num_points=num_points, use_cache=False)
    df = ig_to_df(df)
    if df is None or df.empty:
        print(f"  No data returned for {resolution}")
        return None
    print(f"  Got {len(df)} candles, range: {df['date'].min()} -> {df['date'].max()}")
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(cache_file, orient="records", date_format="iso")
        print(f"  Saved to {cache_file}")
    return df


def main():
    cfg = load_ig_config()
    client = IGClient(cfg)
    if not client.login():
        print("Login failed.")
        return
    spreadbet_id = client.get_spreadbet_account_id()
    if spreadbet_id and spreadbet_id != client.account_id:
        client.switch_account(spreadbet_id)

    print("Fetching IG historical data for AI Index...")
    print(f"--- 5-minute candles ---")
    df_5m = fetch_ig_history(client, "MINUTE_5", 2000, "5m")
    print(f"--- 15-minute candles ---")
    df_15m = fetch_ig_history(client, "MINUTE_15", 2000, "15m")
    print(f"--- 1-hour HTF candles ---")
    df_htf = fetch_ig_history(client, "HOUR", 200, "htf")

    if df_5m is None or df_15m is None:
        print("Missing data — aborting")
        return

    # Calculate days covered
    days_5m = (df_5m["date"].max() - df_5m["date"].min()).days
    days_15m = (df_15m["date"].max() - df_15m["date"].min()).days
    print(f"\n5m covers {days_5m} days, 15m covers {days_15m} days, HTF covers "
          f"{(df_htf['date'].max() - df_htf['date'].min()).days} days")

    # Patch backtest's data sources
    original_fetch_data = bt.fetch_data
    original_fetch_htf = bt.fetch_htf_data

    def patched_fetch_data(market_name, days=30, interval="5m", no_cache=False):
        if market_name != NAME:
            return original_fetch_data(market_name, days, interval, no_cache)
        if interval == "5m":
            return df_5m
        if interval == "15m":
            return df_15m
        return None

    def patched_fetch_htf(market_name, days=60, no_cache=False):
        if market_name != NAME:
            return original_fetch_htf(market_name, days, no_cache)
        return df_htf

    bt.fetch_data = patched_fetch_data
    bt.fetch_htf_data = patched_fetch_htf
    bt.TICKER_MAP[NAME] = "_IG_DATA_"  # any non-empty value to pass the ticker check

    # Run for both timeframes
    for interval_min in (5, 15):
        days_used = days_5m if interval_min == 5 else days_15m
        print(f"\n{'=' * 70}")
        print(f"Running backtest: {interval_min}m candles, {days_used} days")
        print('=' * 70)
        mc = MarketConfig(
            epic=EPIC,
            name=NAME,
            sector="Indices",
            min_stop_distance=1.0,  # current live config
            default_size=1.0,
            expiry="DFB",
            candle_interval=interval_min,
            htf_resolution="HOUR",
            min_confidence=0.55,
            strategy="indices",
        )
        result = run_backtest(mc, days=days_used)
        if result is None:
            print(f"  No result for {interval_min}m")
            continue
        print_result(result)

        # Per-trade timestamp dump + hours-window analysis
        print(f"\n  Individual trades (UTC times):")
        in_hours = 0  # 4-20 UTC (current AI Index trading window)
        out_hours = 0
        for t in result.trades:
            ts = t.entry_time
            hr = ts.hour if hasattr(ts, "hour") else pd.Timestamp(ts).hour
            inside = 4 <= hr < 20
            tag = "  inside" if inside else "OUTSIDE"
            if inside:
                in_hours += 1
            else:
                out_hours += 1
            print(f"    {tag} 4-20 UTC | {ts} | {t.direction} | "
                  f"P&L £{t.pnl:+.2f} | {t.exit_reason}")
        total = in_hours + out_hours
        if total > 0:
            print(f"\n  Hours summary: {in_hours}/{total} trades inside 4-20 UTC, "
                  f"{out_hours}/{total} blocked by current live filter")
            blocked_pnl = sum(t.pnl for t in result.trades
                              if not (4 <= (t.entry_time.hour if hasattr(t.entry_time, 'hour') else pd.Timestamp(t.entry_time).hour) < 20))
            kept_pnl = sum(t.pnl for t in result.trades
                           if 4 <= (t.entry_time.hour if hasattr(t.entry_time, 'hour') else pd.Timestamp(t.entry_time).hour) < 20)
            print(f"  P&L breakdown: kept £{kept_pnl:+.2f}, would-block £{blocked_pnl:+.2f}")

    # Restore
    bt.fetch_data = original_fetch_data
    bt.fetch_htf_data = original_fetch_htf


if __name__ == "__main__":
    main()
