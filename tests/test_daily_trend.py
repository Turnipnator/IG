"""Daily-trend strategy (2026-09-09): long-only Donchian 55/20 on IG DAY bars.

What is pinned here and why:
  * PARITY — src.daily_trend.replay over the committed 22-year Gold file must
    reproduce, trade for trade, the list produced by the STUDY's independent
    engine (tests/fixtures/golden_daily_trend_gold.json). The live signal code
    and the validated rule are therefore the same object. evaluate() is then
    checked against replay() bar-by-bar, so the once-a-day decision path cannot
    drift from the engine either.
  * BAR HYGIENE — IG publishes a one-hour Sunday stub for spot Gold; it must be
    dropped, and a re-fetched date must replace its forming snapshot.
  * OWNERSHIP — the new 'daily-trend' bench type must be invisible to BOTH
    existing resolvers (momentum allowlist; breakout exact match). The 2026-08-10
    incident was exactly a resolver eating another strategy's rows.
  * ROUTING — daily-trend deals are excluded from every per-tick/per-candle exit
    manager and from the breakout's one-per-epic check (coexistence), and the
    re-adopt path restores the tag from the journal's strategy column.
  * MODES — main._daily_trend_mode and telegram._effective_daily_mode agree for
    every market and override; /daily persists and validates like /mode.
"""
import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from config import MARKETS, TelegramConfig, load_trading_config
from src import daily_bars, daily_trend
from src.daily_trend import DailyTrendConfig, VALID_DAILY_TREND_MODES, evaluate, prepare_daily, replay
from src.journal import TradeJournal
from tests.test_telegram_edited_message import edited_update

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
GOLD = "CS.D.USCGC.TODAY.IP"
BY_EPIC = {m.epic: m for m in MARKETS}


def gold_daily():
    return pd.read_csv(FIX / "gc_f_daily.csv", parse_dates=["date"])


class TestParityWithTheStudy(unittest.TestCase):
    def test_replay_reproduces_the_independent_engine(self):
        with open(FIX / "golden_daily_trend_gold.json") as fh:
            golden = json.load(fh)
        cfg = DailyTrendConfig(**golden["config"])
        trades = [t for t in replay(gold_daily(), cfg) if str(t["entry_date"].date()) >= "2005-01-01"]
        self.assertEqual(len(trades), len(golden["trades"]))
        for got, exp in zip(trades, golden["trades"]):
            self.assertEqual(str(got["entry_date"].date()), exp["entry_date"])
            self.assertEqual(str(got["exit_date"].date()), exp["exit_date"])
            self.assertAlmostEqual(got["pts"], exp["pts"], places=2)

    def test_study_numbers_still_hold(self):
        """The headline the decision rests on: positive mean gross R, both halves positive."""
        cfg = DailyTrendConfig()
        T = pd.DataFrame(replay(gold_daily(), cfg))
        T = T[T.entry_date >= "2005-01-01"]
        self.assertGreater(T.r_gross.mean(), 0.5)
        self.assertGreater(T[T.exit_date.dt.year <= 2015].r_gross.mean(), 0)
        self.assertGreater(T[T.exit_date.dt.year >= 2016].r_gross.mean(), 0)

    def test_evaluate_agrees_with_replay_on_every_bar(self):
        """Walk the fixture: at each bar, evaluate() on the prefix must decide what
        replay() did at that bar (entry arms where replay entered next bar; exit
        arms where replay left on a channel exit). Checked on 2018-2026 to keep
        the test fast; the rule is stateless beyond in_position."""
        cfg = DailyTrendConfig()
        df = prepare_daily(gold_daily())
        trades = replay(df, cfg)
        entry_signal_dates = set()
        channel_exit_signal_dates = set()
        dates = list(df["date"])
        idx = {d: i for i, d in enumerate(dates)}
        for t in trades:
            entry_signal_dates.add(dates[idx[t["entry_date"]] - 1])
            if t["reason"] == "channel":
                channel_exit_signal_dates.add(dates[idx[t["exit_date"]] - 1])
        in_pos = False
        pos_exit_bar = None
        checked = 0
        for i in range(len(df)):
            d = dates[i]
            if d < pd.Timestamp("2018-01-01"):
                continue
            # reconstruct replay's position state at bar i close
            for t in trades:
                if t["entry_date"] == d: in_pos = True; pos_exit_bar = t["exit_date"]
            if pos_exit_bar is not None and d == pos_exit_bar and not any(t["entry_date"] == d for t in trades):
                in_pos = False
            sig = evaluate(df.iloc[: i + 1], cfg, in_position=in_pos)
            if sig.action == "WAIT":
                continue
            checked += 1
            if not in_pos:
                self.assertEqual(sig.action == "ENTER_LONG", d in entry_signal_dates, f"{d.date()} {sig.reason}")
            else:
                self.assertEqual(sig.action == "EXIT", d in channel_exit_signal_dates, f"{d.date()} {sig.reason}")
            if sig.action == "EXIT":
                in_pos = False
        self.assertGreater(checked, 1500)


class TestBarHygiene(unittest.TestCase):
    def test_sunday_stub_dropped_and_last_duplicate_wins(self):
        rows = [
            {"date": "2026-09-04", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0},  # Fri
            {"date": "2026-09-06", "open": 1.5, "high": 1.6, "low": 1.4, "close": 1.55, "volume": 0},  # Sun stub
            {"date": "2026-09-07", "open": 1.5, "high": 2.5, "low": 1.4, "close": 2.0, "volume": 0},  # Mon forming
            {"date": "2026-09-07", "open": 1.5, "high": 2.7, "low": 1.3, "close": 2.2, "volume": 0},  # Mon final
        ]
        d = prepare_daily(pd.DataFrame(rows))
        self.assertEqual([str(x.date()) for x in d["date"]], ["2026-09-04", "2026-09-07"])
        self.assertEqual(float(d.iloc[-1]["high"]), 2.7)

    def test_store_merge_new_rows_win_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old = pd.DataFrame([{"date": "2026-09-05", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0},
                                {"date": "2026-09-08", "open": 1, "high": 2, "low": 0.5, "close": 1.0, "volume": 0}])
            old["date"] = pd.to_datetime(old["date"])
            daily_bars.store_daily_bars(GOLD, old, base)
            new = pd.DataFrame([{"date": "2026-09-08", "open": 1, "high": 3, "low": 0.5, "close": 2.5, "volume": 0},
                                {"date": "2026-09-09", "open": 2.5, "high": 3, "low": 2, "close": 2.8, "volume": 0}])
            new["date"] = pd.to_datetime(new["date"])
            merged = daily_bars.merge_daily_bars(daily_bars.load_daily_bars(GOLD, base), new)
            daily_bars.store_daily_bars(GOLD, merged, base)
            back = daily_bars.load_daily_bars(GOLD, base)
            self.assertEqual(len(back), 3)
            self.assertEqual(float(back[back.date == "2026-09-08"].close.iloc[0]), 2.5)

    def test_refresh_bypasses_shared_cache_and_sizes_the_fetch(self):
        calls = []
        class FakeClient:
            def get_historical_prices(self, epic, resolution, num_points, use_cache):
                calls.append((resolution, num_points, use_cache))
                d = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=num_points, freq="D"),
                                  "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 0})
                return d
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            daily_bars.refresh_daily_bars(FakeClient(), GOLD, min_bars=56, base=base)   # thin -> seed
            daily_bars.refresh_daily_bars(FakeClient(), GOLD, min_bars=56, base=base)   # full -> top-up
        self.assertEqual(calls[0], ("DAY", daily_bars.SEED_POINTS, False))
        self.assertEqual(calls[1], ("DAY", daily_bars.REFRESH_POINTS, False))

    def test_evaluate_waits_on_a_short_frame(self):
        d = prepare_daily(gold_daily()).head(30)
        self.assertEqual(evaluate(d, DailyTrendConfig(), in_position=False).action, "WAIT")


class TestShadowResolver(unittest.TestCase):
    def test_resolve_open_episode_matches_replay(self):
        cfg = DailyTrendConfig()
        df = prepare_daily(gold_daily())
        trades = [t for t in replay(df, cfg) if t["entry_date"] >= pd.Timestamp("2015-01-01")][:15]
        dates = list(df["date"]); idx = {d: i for i, d in enumerate(dates)}
        for t in trades:
            signal_date = dates[idx[t["entry_date"]] - 1]
            res = daily_trend.resolve_open_episode(df, cfg, signal_date, t["entry"], cfg.stop_mult * t["atr_at_entry"])
            self.assertIsNotNone(res, t)
            self.assertEqual(res["exit_date"], t["exit_date"])
            self.assertAlmostEqual(res["exit"], t["exit"], places=6)
            self.assertEqual(res["reason"], t["reason"])

    def test_resolve_returns_none_while_open(self):
        cfg = DailyTrendConfig()
        df = prepare_daily(gold_daily())
        last = df.iloc[-1]
        # an episode entered on the last bar with a far stop is still open
        self.assertIsNone(daily_trend.resolve_open_episode(df, cfg, last["date"], float(last["close"]), 1e9))


class TestLedgerOwnership(unittest.TestCase):
    def test_daily_trend_rows_are_invisible_to_the_other_resolvers(self):
        self.assertNotIn("daily-trend", TradeJournal.MOMENTUM_BENCH_TYPES)
        src = (REPO / "src" / "journal.py").read_text()
        # breakout resolver selects by exact bench_type, never by prefix
        self.assertNotIn("bench_type LIKE 'breakout", src)
        self.assertIn("bench_type='breakout-shadow' AND status='OPEN'", src)
        self.assertIn("bench_type='daily-trend' AND status='OPEN'", src)

    def test_shadow_row_roundtrip(self):
        import src.journal as journal_mod
        with tempfile.TemporaryDirectory() as tmp:
            saved = (journal_mod.DB_DIR, journal_mod.DB_FILE)
            journal_mod.DB_DIR = Path(tmp); journal_mod.DB_FILE = Path(tmp) / "j.db"
            try:
                j = TradeJournal()
                self._roundtrip(j)
                j.db.close()
            finally:
                journal_mod.DB_DIR, journal_mod.DB_FILE = saved

    def _roundtrip(self, j):
        if True:
            j.log_daily_trend_shadow(GOLD, "Gold", 4400.0, 158.0, "2026-09-09T00:00:00", 0.4)
            rows = j.get_open_daily_trend_shadow(GOLD)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bench_type"], "daily-trend")
            self.assertEqual(j.get_open_benched(GOLD), [])            # momentum resolver: nothing
            self.assertEqual(j.get_open_breakout_shadow(GOLD), [])    # breakout resolver: nothing
            j.resolve_daily_trend_shadow(rows[0]["id"], "WIN", "channel", 12, 1.7, 4670.0)
            self.assertEqual(j.get_open_daily_trend_shadow(GOLD), [])


class TestRoutingAndCoexistence(unittest.TestCase):
    """Static guards on main.py: the order-path wiring this strategy depends on."""
    def setUp(self):
        self.src = (REPO / "main.py").read_text()

    def test_exit_managers_skip_daily_trend_deals(self):
        self.assertIn("if deal_id in breakout_deals or _daily_managed(deal_id):", self.src)  # BE/ATR tick manager
        self.assertIn("if _daily_managed(position.deal_id):\n            continue\n        if position.deal_id in breakout_deals:",
                      self.src)  # candle exit routing, before the trail/momentum exits
        self.assertIn("return deal_id in daily_trend_deals or deal_id in pullback_deals", self.src)

    def test_breakout_one_per_epic_ignores_daily_trend(self):
        self.assertIn("p.epic == epic and not _daily_managed(p.deal_id)", self.src)

    def test_tag_cleanup_on_every_close_path_and_readopt(self):
        self.assertEqual(self.src.count("daily_trend_deals.discard("), 4)   # 2 external-close sites, momentum close, daily close
        self.assertIn('if row.get("strategy") == "daily-trend":', self.src)

    def test_jobs_scheduled_on_the_london_clock(self):
        self.assertIn('schedule.every().day.at("22:30").do(refresh_daily_trend_bars)', self.src)
        self.assertIn('schedule.every().day.at("23:15").do(run_daily_trend)', self.src)
        self.assertIn('schedule.every().day.at("07:15").do(run_daily_trend)', self.src)

    def test_risk_helper_and_cap(self):
        import main
        self.assertAlmostEqual(main._daily_trend_risk_gbp(1.0, 158.0), 158.0)
        self.assertEqual(load_trading_config().daily_trend_max_risk_gbp, 250.0)


class TestModes(unittest.TestCase):
    def test_config_defaults_match_the_evidence_tier(self):
        JAPAN = "IX.D.NIKKEI.DAILY.IP"
        self.assertEqual(BY_EPIC[GOLD].daily_trend, "live")
        self.assertEqual(BY_EPIC[JAPAN].daily_trend, "shadow")     # Tier 2: plateau positive, lumpy years
        self.assertEqual(set(daily_trend.DAILY_TREND_CONFIGS), {GOLD, JAPAN})
        for m in MARKETS:
            if m.epic not in (GOLD, JAPAN):
                self.assertIsNone(m.daily_trend, m.name)

    def test_main_and_telegram_resolve_identically(self):
        import main
        import src.telegram_bot as tb
        class Fake:
            def __init__(self, modes): self.daily_trend_modes = modes
            _effective_daily_mode = tb.TelegramBot._effective_daily_mode
        original = getattr(main, "telegram", None)
        try:
            for override in (None,) + VALID_DAILY_TREND_MODES + ("bogus",):
                for m in MARKETS:
                    fake = Fake({} if override is None else {m.epic: override})
                    main.telegram = fake
                    self.assertEqual(main._daily_trend_mode(m), fake._effective_daily_mode(m), (m.epic, override))
        finally:
            main.telegram = original


class TestDailyCommand(unittest.TestCase):
    def setUp(self):
        import src.telegram_bot as tb
        self.tb = tb
        self.tmp = tempfile.TemporaryDirectory(); tmp = Path(self.tmp.name)
        self.modes_file = tmp / "daily_trend_modes.json"
        self.patches = [mock.patch.object(tb, "STATS_DIR", tmp),
                        mock.patch.object(tb, "MARKET_MODES_FILE", tmp / "market_modes.json"),
                        mock.patch.object(tb, "LEGACY_FOREX_MODE_FILE", tmp / "forex_mode.json"),
                        mock.patch.object(tb, "DAILY_TREND_MODES_FILE", self.modes_file)]
        for p in self.patches: p.start()
        self.bot = tb.TelegramBot(TelegramConfig(bot_token="x", chat_id="1"))

    def tearDown(self):
        for p in self.patches: p.stop()
        self.tmp.cleanup()

    def run_daily(self, *args):
        update, msg = edited_update()
        asyncio.run(self.bot.daily_command(update, SimpleNamespace(args=list(args))))
        return msg

    def test_board_and_toggle_persist(self):
        board = self.run_daily().replies[0]
        self.assertIn("Gold: `live`", board)
        msg = self.run_daily("gold", "shadow")
        self.assertEqual(self.bot.daily_trend_modes[GOLD], "shadow")
        self.assertEqual(json.loads(self.modes_file.read_text())["daily_trend_modes"][GOLD], "shadow")
        self.assertIn("shadow", msg.replies[0])
        self.run_daily("gold", "default")
        self.assertNotIn(GOLD, self.bot.daily_trend_modes)

    def test_rejects_unknown_mode_and_unconfigured_market(self):
        self.run_daily("gold", "bogus")
        self.assertEqual(self.bot.daily_trend_modes, {})
        msg = self.run_daily("s&p", "live")
        self.assertEqual(self.bot.daily_trend_modes, {})
        self.assertIn("no daily-trend config", msg.replies[0])

    def test_mode_board_shows_the_daily_line(self):
        update, msg = edited_update()
        asyncio.run(self.bot.mode_command(update, SimpleNamespace(args=[])))
        self.assertIn("daily-trend: `live`", msg.replies[0])


if __name__ == "__main__":
    unittest.main()
