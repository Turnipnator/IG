"""Pullback-in-uptrend strategy (2026-09-09, sweep 2): long-only mean reversion on
cash-session bars. Pins: parity with the sweep engine's goldens (S&P + NASDAQ),
evaluate==replay per bar, session-bar construction (window, holidays, partial
days), resolver ownership, routing/coexistence guards, mode parity, /pullback."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from config import MARKETS, TelegramConfig, load_trading_config
from src import pullback, session_bars
from src.journal import TradeJournal
from src.pullback import PullbackConfig, VALID_PULLBACK_MODES, evaluate, prepare_daily, replay
from tests.test_telegram_edited_message import edited_update

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
SPX, NDX = "IX.D.SPTRD.DAILY.IP", "IX.D.NASDAQ.CASH.IP"
BY_EPIC = {m.epic: m for m in MARKETS}


def daily(name):
    return pd.read_csv(FIX / name, parse_dates=["date"])


class TestParityWithTheSweep(unittest.TestCase):
    def test_replay_reproduces_the_sweep_engine(self):
        with open(FIX / "golden_pullback.json") as fh:
            G = json.load(fh)
        cfg = PullbackConfig(**G["config"])
        for key, fn in (("sp500", "gspc_daily.csv"), ("nasdaq100", "ndx_daily.csv")):
            T = replay(daily(fn), cfg, start="2005-01-01")
            self.assertEqual(len(T), len(G[key]), key)
            for got, exp in zip(T, G[key]):
                self.assertEqual(str(got["entry_date"].date()), exp["entry_date"])
                self.assertEqual(str(got["exit_date"].date()), exp["exit_date"])
                self.assertEqual(got["reason"], exp["reason"])
                self.assertAlmostEqual(got["pts"], exp["pts"], places=2)

    def test_study_numbers_still_hold(self):
        cfg = PullbackConfig()
        for fn in ("gspc_daily.csv", "ndx_daily.csv"):
            T = pd.DataFrame(replay(daily(fn), cfg, start="2005-01-01"))
            self.assertGreater(T.r_gross.mean(), 0.15, fn)
            self.assertGreater(T[T.exit_date.dt.year <= 2015].r_gross.mean(), 0, fn)
            self.assertGreater(T[T.exit_date.dt.year >= 2016].r_gross.mean(), 0, fn)
            self.assertGreater((T.r_gross > 0).mean(), 0.6, fn)

    def test_evaluate_agrees_with_replay(self):
        """At every bar 2019-2026, evaluate() on the prefix must decide what replay did."""
        cfg = PullbackConfig()
        df = prepare_daily(daily("gspc_daily.csv"))
        trades = replay(df, cfg)
        entries = {t["entry_date"] for t in trades}
        exits_at_close = {t["exit_date"] for t in trades if t["reason"] in ("channel", "time")}
        by_entry = {t["entry_date"]: t for t in trades}
        dates = list(df["date"]); idx = {d: i for i, d in enumerate(dates)}
        in_pos = False; e_date = None; checked = 0
        last_exit = max(t["exit_date"] for t in trades)   # a still-open position after this is invisible to replay()
        for i, d in enumerate(dates):
            if d > last_exit:
                break
            if d < pd.Timestamp("2019-01-01"):
                if d in entries: in_pos = True; e_date = d
                for t in trades:
                    if t["exit_date"] == d: in_pos = False
                continue
            # stop exits happen intraday (before the close decision): reflect them first
            stopped_today = False
            for t in trades:
                if t["exit_date"] == d and t["reason"] == "stop": in_pos = False; stopped_today = True
            held = (idx[d] - idx[e_date]) if in_pos else 0
            sig = evaluate(df.iloc[: i + 1], cfg, in_position=in_pos, bars_held=held, exited_today=stopped_today)
            if sig.action == "WAIT":
                continue
            checked += 1
            if in_pos:
                self.assertEqual(sig.action == "EXIT", d in exits_at_close, f"{d.date()} {sig.reason}")
                if d in exits_at_close: in_pos = False
            else:
                self.assertEqual(sig.action == "ENTER_LONG", d in entries, f"{d.date()} {sig.reason}")
                if d in entries: in_pos = True; e_date = d
        self.assertGreater(checked, 1500)

    def test_sma_from_a_longer_close_series(self):
        """Live: the session frame is short (months) and the SMA200 comes from IG DAY
        closes. Feeding the same closes as sma_closes must give the same decision."""
        cfg = PullbackConfig()
        df = prepare_daily(daily("ndx_daily.csv"))
        full = evaluate(df, cfg, in_position=False)
        short = df.tail(40)
        closes = pd.Series(df["close"].values, index=df["date"].values)
        via_series = evaluate(short, cfg, in_position=False, sma_closes=closes)
        self.assertEqual(full.action, via_series.action)
        self.assertAlmostEqual(full.sma, via_series.sma, places=6)
        self.assertEqual(evaluate(short, cfg, in_position=False).action, "WAIT")   # no SMA -> WAIT
        # Sunday stub rows in an IG DAY series must not enter the SMA
        sundays = pd.date_range("2015-01-04", df["date"].iloc[-1], freq="W-SUN")
        polluted = pd.concat([closes, pd.Series(1e-6, index=sundays)]).sort_index()
        self.assertAlmostEqual(evaluate(short, cfg, in_position=False, sma_closes=polluted).sma, full.sma, places=6)

    def test_store_rows_needed_covers_sunday_stubs(self):
        import main
        self.assertGreaterEqual(int(PullbackConfig().sma_n * 7 / 5) + 15, 295)
        self.assertIn("rows_needed = int(pullback.get_pullback_config(m.epic).sma_n * 7 / 5) + 15", (REPO / "main.py").read_text())


class TestSessionBars(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        (Path(self.tmp.name) / f"{SPX}.jsonl").write_bytes((FIX / "spx_5m.jsonl").read_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def test_window_and_aggregation(self):
        g = session_bars.build_session_bars(SPX, archive_dir=Path(self.tmp.name))
        self.assertGreater(len(g), 3)
        self.assertTrue((g["n"] == 78).all(), g["n"].tolist())        # full sessions only in the fixture head
        raw = pd.read_json(FIX / "spx_5m.jsonl", lines=True); raw["date"] = pd.to_datetime(raw["timestamp"])
        d0 = g["date"].iloc[0]
        day = raw[(raw.date.dt.normalize() == d0) & (raw.date.dt.hour * 60 + raw.date.dt.minute >= 14 * 60 + 30)
                  & (raw.date.dt.hour * 60 + raw.date.dt.minute < 21 * 60)]
        self.assertAlmostEqual(float(g["open"].iloc[0]), float(day.iloc[0]["open"]))
        self.assertAlmostEqual(float(g["close"].iloc[0]), float(day.iloc[-1]["close"]))
        self.assertAlmostEqual(float(g["low"].iloc[0]), float(day["low"].min()))

    def test_partial_sessions_and_holidays_dropped(self):
        raw = pd.read_json(FIX / "spx_5m.jsonl", lines=True)
        raw["date"] = pd.to_datetime(raw["timestamp"])
        day = raw.date.dt.normalize().iloc[0]
        partial = raw[(raw.date.dt.normalize() == day)].head(200)          # only part of a day
        (Path(self.tmp.name) / f"{SPX}.jsonl").write_text("\n".join(json.dumps({"timestamp": str(r.date), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": 0}) for r in partial.itertuples()))
        g = session_bars.build_session_bars(SPX, archive_dir=Path(self.tmp.name))
        self.assertTrue(g.empty or (g["n"] >= session_bars.MIN_BARS).all())
        self.assertTrue(session_bars.is_nyse_holiday("2026-09-07"))       # Labor Day 2026
        self.assertTrue(session_bars.is_nyse_holiday("2026-07-03"))       # Independence Day observed
        self.assertFalse(session_bars.is_nyse_holiday("2026-09-08"))

    def test_stream_frame_merges_with_archive(self):
        g0 = session_bars.build_session_bars(SPX, archive_dir=Path(self.tmp.name))
        last = g0["date"].iloc[-1]
        # a synthetic next session from a "stream" frame
        ts = pd.date_range(last + pd.Timedelta(days=1, hours=14, minutes=30), periods=78, freq="5min")
        stream = pd.DataFrame({"date": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 0})
        g1 = session_bars.build_session_bars(SPX, stream_df=stream, archive_dir=Path(self.tmp.name))
        if (ts[0].weekday() < 5) and not session_bars.is_nyse_holiday(ts[0]):
            self.assertEqual(len(g1), len(g0) + 1)
            self.assertEqual(float(g1["high"].iloc[-1]), 2.0)


class TestShadowResolver(unittest.TestCase):
    def test_resolver_matches_replay(self):
        cfg = PullbackConfig()
        df = prepare_daily(daily("gspc_daily.csv"))
        for t in [t for t in replay(df, cfg) if t["entry_date"] >= pd.Timestamp("2020-01-01")][:20]:
            res = pullback.resolve_open_episode(df, cfg, t["entry_date"], t["entry"], cfg.stop_mult * t["atr_at_entry"])
            self.assertIsNotNone(res, t)
            self.assertEqual(res["exit_date"], t["exit_date"]); self.assertEqual(res["reason"], t["reason"])
            self.assertAlmostEqual(res["exit"], t["exit"], places=6)


class TestLedgerOwnership(unittest.TestCase):
    def test_pullback_rows_invisible_to_other_resolvers(self):
        self.assertNotIn("pullback", TradeJournal.MOMENTUM_BENCH_TYPES)
        src = (REPO / "src" / "journal.py").read_text()
        self.assertIn("bench_type='pullback' AND status='OPEN'", src)

    def test_roundtrip(self):
        import src.journal as journal_mod
        with tempfile.TemporaryDirectory() as tmp:
            saved = (journal_mod.DB_DIR, journal_mod.DB_FILE)
            journal_mod.DB_DIR = Path(tmp); journal_mod.DB_FILE = Path(tmp) / "j.db"
            try:
                j = TradeJournal()
                j.log_pullback_shadow(SPX, "S&P 500", 7600.0, 200.0, "2026-09-09T00:00:00", 0.6)
                rows = j.get_open_pullback_shadow(SPX)
                self.assertEqual(len(rows), 1)
                self.assertEqual(j.get_open_benched(SPX), []); self.assertEqual(j.get_open_breakout_shadow(SPX), [])
                self.assertEqual(j.get_open_daily_trend_shadow(SPX), [])
                j.resolve_pullback_shadow(rows[0]["id"], "WIN", "channel", 3, 0.4, 7680.0)
                self.assertEqual(j.get_open_pullback_shadow(SPX), [])
                j.db.close()
            finally:
                journal_mod.DB_DIR, journal_mod.DB_FILE = saved


class TestRoutingAndConfig(unittest.TestCase):
    def setUp(self):
        self.src = (REPO / "main.py").read_text()

    def test_daily_managed_helper_guards_every_site(self):
        self.assertIn("return deal_id in daily_trend_deals or deal_id in pullback_deals", self.src)
        self.assertIn("if deal_id in breakout_deals or _daily_managed(deal_id):", self.src)
        self.assertIn("if _daily_managed(position.deal_id):\n            continue", self.src)
        self.assertIn("p.epic == epic and not _daily_managed(p.deal_id)", self.src)
        self.assertEqual(self.src.count("pullback_deals.discard("), 4)
        self.assertIn('if row.get("strategy") == "pullback":', self.src)

    def test_jobs_and_config(self):
        self.assertIn('schedule.every().day.at("21:05").do(run_pullback)', self.src)
        self.assertIn('schedule.every().day.at("21:35").do(run_pullback)', self.src)
        self.assertEqual(BY_EPIC[SPX].pullback, "live"); self.assertEqual(BY_EPIC[NDX].pullback, "live")
        self.assertEqual(set(pullback.PULLBACK_CONFIGS), {SPX, NDX})
        for m in MARKETS:
            if m.epic not in (SPX, NDX):
                self.assertIsNone(m.pullback, m.name)
        self.assertEqual(load_trading_config().pullback_risk_gbp, 100.0)

    def test_main_and_telegram_resolve_identically(self):
        import main
        import src.telegram_bot as tb
        class Fake:
            def __init__(self, modes): self.pullback_modes = modes
            _effective_pullback_mode = tb.TelegramBot._effective_pullback_mode
        original = getattr(main, "telegram", None)
        try:
            for override in (None,) + VALID_PULLBACK_MODES + ("bogus",):
                for m in MARKETS:
                    fake = Fake({} if override is None else {m.epic: override}); main.telegram = fake
                    self.assertEqual(main._pullback_mode(m), fake._effective_pullback_mode(m), (m.epic, override))
        finally:
            main.telegram = original


class TestPullbackCommand(unittest.TestCase):
    def setUp(self):
        import src.telegram_bot as tb
        self.tb = tb; self.tmp = tempfile.TemporaryDirectory(); tmp = Path(self.tmp.name)
        self.modes_file = tmp / "pullback_modes.json"
        self.patches = [mock.patch.object(tb, "STATS_DIR", tmp),
                        mock.patch.object(tb, "MARKET_MODES_FILE", tmp / "market_modes.json"),
                        mock.patch.object(tb, "LEGACY_FOREX_MODE_FILE", tmp / "forex_mode.json"),
                        mock.patch.object(tb, "DAILY_TREND_MODES_FILE", tmp / "daily_trend_modes.json"),
                        mock.patch.object(tb, "PULLBACK_MODES_FILE", self.modes_file)]
        for p in self.patches: p.start()
        self.bot = tb.TelegramBot(TelegramConfig(bot_token="x", chat_id="1"))

    def tearDown(self):
        for p in self.patches: p.stop()
        self.tmp.cleanup()

    def run_cmd(self, *args):
        update, msg = edited_update()
        asyncio.run(self.bot.pullback_command(update, SimpleNamespace(args=list(args))))
        return msg

    def test_board_toggle_persist_and_reject(self):
        board = self.run_cmd().replies[0]
        self.assertIn("S&P 500: `live`", board); self.assertIn("NASDAQ 100: `live`", board)
        self.run_cmd("nasdaq", "shadow")
        self.assertEqual(json.loads(self.modes_file.read_text())["pullback_modes"][NDX], "shadow")
        self.run_cmd("nasdaq", "default"); self.assertNotIn(NDX, self.bot.pullback_modes)
        msg = self.run_cmd("gold", "live"); self.assertEqual(self.bot.pullback_modes, {}); self.assertIn("no pullback config", msg.replies[0])
        update, msg = edited_update()
        asyncio.run(self.bot.mode_command(update, SimpleNamespace(args=[])))
        self.assertIn("pullback: `live`", msg.replies[0])


if __name__ == "__main__":
    unittest.main()
