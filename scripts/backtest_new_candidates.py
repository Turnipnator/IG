"""
Backtest candidate markets identified from spread bet discovery.

Candidates (Russell 2000 excluded — already known to be unprofitable):
  Indices:
    - France 40 (CAC)
    - Japan 225 (Nikkei)
    - Hong Kong HS50 (Hang Seng)
    - Netherlands 25 (AEX)
    - Switzerland Blue Chip (SMI)
    - Sweden 30 (OMX)
  Forex extensions:
    - AUD/USD
    - EUR/JPY
    - EUR/AUD
    - USD/CHF
  Crypto:
    - Cardano (ADA)
    - Litecoin (LTC)

Reuses backtest.py's run_backtest by patching TICKER_MAP and
IG_POINTS_MULT, and synthesising a MarketConfig per candidate.

Usage:
    python scripts/backtest_new_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

from config import MarketConfig, get_strategy_for_market
import backtest as bt
from backtest import run_backtest, print_result

# ---- Candidate definitions ----
# Each: (MarketConfig, yahoo_ticker, points_multiplier)

# For indices, point multiplier is 1 — Yahoo and IG quote in the same units.
# For forex, JPY-pairs use 100 (Yahoo 1.0850 -> IG 108.50), others 10000.
# For crypto, Yahoo and IG price equally; mult = 1 unless we hit issues.

CANDIDATES: list[tuple[MarketConfig, str, float]] = [
    # ---- Indices ----
    (MarketConfig(
        epic="IX.D.CAC.DAILY.IP", name="France 40", sector="Indices",
        min_stop_distance=4.0, default_size=0.5, expiry="DFB",
        candle_interval=5, htf_resolution="HOUR",
        min_confidence=0.55, strategy="indices",
        trading_start=7, trading_end=16,  # CAC 07-16 UTC roughly
    ), "^FCHI", 1.0),

    (MarketConfig(
        epic="IX.D.NIKKEI.DAILY.IP", name="Japan 225", sector="Indices",
        min_stop_distance=20.0, default_size=0.5, expiry="DFB",
        candle_interval=5, htf_resolution="HOUR",
        min_confidence=0.55, strategy="indices",
        trading_start=0, trading_end=8,  # Tokyo 00-08 UTC
    ), "^N225", 1.0),

    (MarketConfig(
        epic="IX.D.HANGSENG.DAILY.IP", name="Hong Kong HS50", sector="Indices",
        min_stop_distance=20.0, default_size=0.5, expiry="DFB",
        candle_interval=5, htf_resolution="HOUR",
        min_confidence=0.55, strategy="indices",
        trading_start=1, trading_end=9,  # HK 01-09 UTC
    ), "^HSI", 1.0),

    (MarketConfig(
        epic="IX.D.AEX.CASH.IP", name="Netherlands 25", sector="Indices",
        min_stop_distance=0.5, default_size=1.0, expiry="DFB",
        candle_interval=5, htf_resolution="HOUR",
        min_confidence=0.55, strategy="indices",
        trading_start=7, trading_end=16,
    ), "^AEX", 1.0),

    (MarketConfig(
        epic="IX.D.SMI.DFB.IP", name="Switzerland Blue Chip", sector="Indices",
        min_stop_distance=10.0, default_size=0.5, expiry="DFB",
        candle_interval=5, htf_resolution="HOUR",
        min_confidence=0.55, strategy="indices",
        trading_start=7, trading_end=16,
    ), "^SSMI", 1.0),

    (MarketConfig(
        epic="IX.D.OMX.CASH.IP", name="Sweden 30", sector="Indices",
        min_stop_distance=1.5, default_size=1.0, expiry="DFB",
        candle_interval=5, htf_resolution="HOUR",
        min_confidence=0.55, strategy="indices",
        trading_start=7, trading_end=16,
    ), "^OMX", 1.0),

    # ---- Forex Extensions ----
    (MarketConfig(
        epic="CS.D.AUDUSD.TODAY.IP", name="AUD/USD", sector="Forex",
        min_stop_distance=2.0, default_size=1.0, expiry="DFB",
        candle_interval=15, htf_resolution="HOUR",
        min_confidence=0.55, strategy="forex",
        trading_start=23, trading_end=21,  # 24/5
    ), "AUDUSD=X", 10000.0),

    (MarketConfig(
        epic="CS.D.EURJPY.TODAY.IP", name="EUR/JPY", sector="Forex",
        min_stop_distance=4.0, default_size=1.0, expiry="DFB",
        candle_interval=15, htf_resolution="HOUR",
        min_confidence=0.55, strategy="forex",
        trading_start=23, trading_end=21,
    ), "EURJPY=X", 100.0),

    (MarketConfig(
        epic="CS.D.EURAUD.TODAY.IP", name="EUR/AUD", sector="Forex",
        min_stop_distance=10.0, default_size=1.0, expiry="DFB",
        candle_interval=15, htf_resolution="HOUR",
        min_confidence=0.55, strategy="forex",
        trading_start=23, trading_end=21,
    ), "EURAUD=X", 10000.0),

    (MarketConfig(
        epic="CS.D.USDCHF.TODAY.IP", name="USD/CHF", sector="Forex",
        min_stop_distance=4.0, default_size=1.0, expiry="DFB",
        candle_interval=15, htf_resolution="HOUR",
        min_confidence=0.55, strategy="forex",
        trading_start=23, trading_end=21,
    ), "USDCHF=X", 10000.0),

    # ---- Crypto ----
    (MarketConfig(
        epic="CS.D.ADAUSD.TODAY.IP", name="Cardano", sector="Commodities",
        min_stop_distance=1.0, default_size=1.0, expiry="DFB",
        candle_interval=15, htf_resolution="HOUR",
        min_confidence=0.55, strategy="default",
        trading_start=0, trading_end=23,  # 24/7
    ), "ADA-USD", 1.0),

    (MarketConfig(
        epic="CS.D.LTCUSD.TODAY.IP", name="Litecoin", sector="Commodities",
        min_stop_distance=2.5, default_size=1.0, expiry="DFB",
        candle_interval=15, htf_resolution="HOUR",
        min_confidence=0.55, strategy="default",
        trading_start=0, trading_end=23,
    ), "LTC-USD", 1.0),
]

DAYS = 30


def main():
    # Patch backtest module's lookup tables so it knows our candidates
    for mc, ticker, mult in CANDIDATES:
        bt.TICKER_MAP[mc.name] = ticker
        if mult != 1.0:
            bt.IG_POINTS_MULT[mc.name] = mult

    results = []
    for mc, ticker, mult in CANDIDATES:
        print(f"\n=== {mc.name} ({ticker}, strategy={mc.strategy}) ===")
        try:
            r = run_backtest(mc, days=DAYS)
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        if r is None:
            print(f"  Skipped (no data or insufficient candles)")
            continue
        print_result(r)
        results.append((mc, r))

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY — sorted by total P&L")
    print("=" * 80)
    print(f"{'Market':<22} | {'Trades':>6} | {'WR':>5} | {'PF':>5} | "
          f"{'Total P&L':>10} | {'Avg Win':>8} | {'Avg Loss':>9} | {'MaxDD':>8}")
    print("-" * 80)
    results.sort(key=lambda x: x[1].total_pnl, reverse=True)
    for mc, r in results:
        n = r.win_count + r.loss_count + r.be_count
        wr = (r.win_count / (r.win_count + r.loss_count) * 100) if (r.win_count + r.loss_count) > 0 else 0
        print(f"{mc.name:<22} | {n:>6} | {wr:>4.1f}% | {r.profit_factor:>5.2f} | "
              f"£{r.total_pnl:>+9.2f} | £{r.avg_win:>+6.2f} | £{r.avg_loss:>+7.2f} | £{r.max_drawdown:>+6.2f}")

    # Verdict guidance
    print("\nVerdict guide:")
    print("  Profitable + PF > 1.3 + WR > 45%   = strong candidate, add to MARKETS")
    print("  Profitable but PF < 1.3            = marginal, watch live before committing")
    print("  Negative                            = skip")


if __name__ == "__main__":
    main()
