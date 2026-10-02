"""NY-open opening-range breakout SHADOW (2026-10-02, src/orb.py). Pins: parity with the
Oanda backtest engine (Oanda_Gold/research_orb_friend.js, exported goldens), the NY clock
across the US/UK DST-mismatch weeks, the virtual position's exits, one trade per day,
holiday skip, journal ownership, the tick path end to end incl. a restart mid-trade,
mode parity, and that no "live" mode exists anywhere."""
import asyncio
import gzip
import json
import tempfile
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from config import MARKETS, TelegramConfig
from src import orb
from src.journal import TradeJournal
from src.orb import Bar, OrbConfig, OrbDay, ShadowTrade, replay_day, swing_trend
from src.session_bars import is_nyse_holiday
from src.streaming import Candle, MarketStream
from tests.test_telegram_edited_message import edited_update

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
DOW = "IX.D.DOW.DAILY.IP"
BY_EPIC = {m.epic: m for m in MARKETS}


def golden():
    with gzip.open(FIX / "orb_us30_golden.json.gz", "rt") as fh:
        return json.load(fh)


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


class TestParityWithTheBacktest(unittest.TestCase):
    """The JS engine is the study; the Python port must make the same decision on every day."""

    def test_replay_matches_engine_day_by_day(self):
        cfg, G = OrbConfig(), golden()
        self.assertGreaterEqual(sum(1 for d in G if d["pop"]), 15)
        checked = 0
        for d in G:
            got = replay_day(cfg, d["rows"], d["m5h"], d["m5l"])
            if is_nyse_holiday(d["date"]):      # engine traded thin holiday sessions; the port skips them
                self.assertIsNone(got["pop"], d["date"]); self.assertIsNone(got["plain"], d["date"])
                continue
            for v in ("pop", "plain"):
                exp, g = d[v], got[v]
                if exp is None:
                    self.assertIsNone(g, (d["date"], v)); continue
                self.assertIsNotNone(g, (d["date"], v))
                self.assertEqual(g["side"], exp["side"], (d["date"], v))
                self.assertAlmostEqual(g["entry"], exp["entry"], places=6, msg=(d["date"], v))
                self.assertEqual(g["reason"], exp["reason"], (d["date"], v))
                self.assertAlmostEqual(g["r"], exp["R"], places=3, msg=(d["date"], v))
                checked += 1
        self.assertGreater(checked, 40)

    def test_swing_trend_matches_engine(self):
        for d in golden():
            self.assertEqual(swing_trend(d["m5h"], d["m5l"]), d["trend"], d["date"])

    def test_swing_trend_basics(self):
        up_h = [1, 2, 5, 2, 1, 3, 7, 3, 2, 4]; up_l = [x - 1 for x in up_h]
        lows_up = [3, 2, 0, 2, 3, 2, 1, 2, 3, 4]
        self.assertEqual(swing_trend(up_h, lows_up), 1)                  # HH (5->7) and HL (0->1)
        self.assertEqual(swing_trend([1, 2, 3], [0, 1, 2]), 0)           # too few pivots -> no trade


class TestClock(unittest.TestCase):
    def test_ny_open_in_dst_mismatch_weeks(self):
        # 2026-10-26: UK already on GMT, US still on EDT -> 09:30 NY = 13:30 UTC (not 14:30)
        self.assertEqual(orb.ny_minute(utc(2026, 10, 26, 13, 30)), ("2026-10-26", 570))
        # 2026-03-09: US on EDT, UK still GMT -> 09:30 NY = 13:30 UTC
        self.assertEqual(orb.ny_minute(utc(2026, 3, 9, 13, 30)), ("2026-03-09", 570))
        # normal week: 14:30 UTC in winter
        self.assertEqual(orb.ny_minute(utc(2026, 12, 7, 14, 30)), ("2026-12-07", 570))

    def test_m5_before_reads_london_naive_candles(self):
        # 2026-10-26 13:25 London (=UTC) candle ends 13:30 UTC, inside a 13:35 UTC cutoff; 13:35 is not
        cs = [Candle(datetime(2026, 10, 26, 13, 25), 1, 10, 0, 5), Candle(datetime(2026, 10, 26, 13, 30), 1, 11, 0, 5),
              Candle(datetime(2026, 10, 26, 13, 35), 1, 99, 0, 5)]
        hs, _ = orb.m5_before(cs, utc(2026, 10, 26, 13, 35))
        self.assertEqual(hs, [10.0, 11.0])
        # summer: naive 14:25 London (BST) = 13:25 UTC
        hs, _ = orb.m5_before([Candle(datetime(2026, 7, 6, 14, 30), 1, 7, 0, 5)], utc(2026, 7, 6, 13, 35))
        self.assertEqual(hs, [7.0])

    def test_holidays_and_weekends_skip(self):
        for d in ("2026-07-03", "2026-11-26", "2026-10-03"):     # observed July 4, Thanksgiving, Saturday
            self.assertTrue(all(s == "skipped" for s in orb.new_day(d).status.values()), d)
        self.assertEqual(orb.new_day("2026-10-05").status, {"pop": "waiting", "plain": "waiting"})


class TestShadowTrade(unittest.TestCase):
    def trade(self, side=1):
        e = 100.0
        return ShadowTrade(side=side, entry=e, stop=e - side * 50, target=e + side * 95, be_trigger=50,
                           stop_dist=50, entry_ts=utc(2026, 10, 5, 13, 40).isoformat())

    def test_target(self):
        t = self.trade(); ts = utc(2026, 10, 5, 13, 41)
        self.assertIsNone(t.on_tick(100, 101, ts, False))
        ex = t.on_tick(196, 197, ts, False)
        self.assertEqual(ex["reason"], "tp"); self.assertAlmostEqual(ex["r"], 1.9)

    def test_breakeven_then_scratch(self):
        t = self.trade(); ts = utc(2026, 10, 5, 13, 41)
        t.on_tick(100, 101, ts, False); t.on_tick(151, 152, ts, False)
        self.assertTrue(t.be_done); self.assertEqual(t.stop, 100.0)
        ex = t.on_tick(99.5, 100.5, ts, False)
        self.assertEqual(ex["reason"], "be"); self.assertLessEqual(ex["r"], 0)

    def test_stop_short_and_clock_exit(self):
        t = self.trade(side=-1); ts = utc(2026, 10, 5, 13, 41)
        t.on_tick(99, 100, ts, False)
        ex = t.on_tick(150, 151, ts, False)
        self.assertEqual(ex["reason"], "sl"); self.assertAlmostEqual(ex["r"], -1.02)   # gap: filled at the tick's offer
        t2 = self.trade(); t2.on_tick(100, 101, ts, False)
        ex = t2.on_tick(120, 121, utc(2026, 10, 5, 19, 30), True)
        self.assertEqual(ex["reason"], "time"); self.assertAlmostEqual(ex["r"], 0.4)

    def test_roundtrip(self):
        t = self.trade(); t.on_tick(151, 152, utc(2026, 10, 5, 13, 41), False)
        self.assertEqual(ShadowTrade.from_dict(json.loads(json.dumps(t.to_dict()))), t)


class TestOrbDay(unittest.TestCase):
    def bars(self, minute0, closes, rng=1.0, ticks=10, start=utc(2026, 10, 5, 13, 0)):
        return [Bar(start + timedelta(minutes=minute0 + i), c, c + rng / 2, c - rng / 2, c, ticks) for i, c in enumerate(closes)]

    def test_one_signal_per_variant_per_day_and_creep_skip(self):
        cfg = OrbConfig(); day = OrbDay(date="2026-10-05")
        hist = self.bars(0, [100.0] * 30)                       # 09:00-09:29 NY (13:00 UTC = 09:00 EDT)
        for b in self.bars(30, [100.0] * 5):                    # range 09:30-09:34: 99.5-100.5
            day.on_bar(cfg, b, hist, lambda: 1); hist.append(b)
        creep = Bar(utc(2026, 10, 5, 13, 35), 100, 101, 100, 100.8, 10)   # breaks, but no pop
        ev = day.on_bar(cfg, creep, hist, lambda: 1)
        self.assertEqual(sorted(ev), [("plain", "signal"), ("pop", "skip:creep")])
        later = Bar(utc(2026, 10, 5, 13, 40), 100, 130, 100, 129, 99)
        self.assertEqual(day.on_bar(cfg, later, hist, lambda: 1), [])   # plain already pending, pop done for the day
        self.assertEqual(day.trend, 1)

    def test_pop_needs_history_after_restart(self):
        cfg = OrbConfig(); day = OrbDay(date="2026-10-05")
        for b in self.bars(30, [100.0] * 5):
            day.on_bar(cfg, b, [], lambda: -1)
        pop = Bar(utc(2026, 10, 5, 13, 36), 100, 100, 80, 81, 99)
        ev = day.on_bar(cfg, pop, [], lambda: -1)
        self.assertIn(("pop", "skip:no-history"), ev); self.assertIn(("plain", "signal"), ev)

    def test_missing_range_minute_skips(self):
        cfg = OrbConfig(); day = OrbDay(date="2026-10-05")
        for b in self.bars(31, [100.0] * 4):                    # 09:30 bar missing (e.g. restart at 09:30:30)
            day.on_bar(cfg, b, [], lambda: 1)
        ev = day.on_bar(cfg, Bar(utc(2026, 10, 5, 13, 35), 100, 120, 100, 119, 50), [], lambda: 1)
        self.assertEqual(sorted(ev), [("plain", "skip:no-range"), ("pop", "skip:no-range")])


class TestLedgerOwnership(unittest.TestCase):
    def test_rows_invisible_to_other_resolvers_and_tally(self):
        import src.journal as journal_mod
        for bt in TradeJournal.ORB_BENCH_TYPES:
            self.assertNotIn(bt, TradeJournal.MOMENTUM_BENCH_TYPES)
        self.assertEqual(set(TradeJournal.ORB_BENCH_TYPES), set(orb.VARIANTS.values()))
        with tempfile.TemporaryDirectory() as tmp:
            saved = (journal_mod.DB_DIR, journal_mod.DB_FILE)
            journal_mod.DB_DIR = Path(tmp); journal_mod.DB_FILE = Path(tmp) / "j.db"
            try:
                j = TradeJournal()
                rid = j.log_orb_shadow(DOW, "Wall Street", "SELL", 46000.0, 50, 95, "2026-10-05T13:40:00+00:00", 2.4, "orb-shadow")
                j.log_orb_shadow(DOW, "Wall Street", "SELL", 46000.0, 50, 95, "2026-10-05T13:40:00+00:00", 2.4, "orb-shadow-plain")
                self.assertIsNone(j.log_orb_shadow(DOW, "Wall Street", "BUY", 1, 50, 95, "x", 0, "shadow"))
                self.assertEqual(len(j.get_open_orb_shadow(DOW)), 2)
                self.assertEqual(j.get_open_benched(DOW), []); self.assertEqual(j.get_open_breakout_shadow(DOW), [])
                self.assertEqual(j.get_open_pullback_shadow(DOW), [])
                j.resolve_breakout_shadow(rid, "WIN", "tp", 30, 1.9, 45905.0)
                t = j.get_orb_tally(DOW)
                self.assertEqual(t["orb-shadow"], {"n": 1, "wins": 1, "sum_r": 1.9, "open": 0, "expired": 0})
                self.assertEqual(t["orb-shadow-plain"]["open"], 1)
                j.db.close()
            finally:
                journal_mod.DB_DIR, journal_mod.DB_FILE = saved


class TestTickPathAndRestart(unittest.TestCase):
    """Drive main._orb_step with ticks synthesised from a real golden day; kill the
    in-memory state mid-trade and check the restart resumes the SAME journal row."""

    def setUp(self):
        import main
        import src.journal as journal_mod
        self.main = main; self.tmp = tempfile.TemporaryDirectory(); tmp = Path(self.tmp.name)
        self.saved_db = (journal_mod.DB_DIR, journal_mod.DB_FILE)
        journal_mod.DB_DIR = tmp; journal_mod.DB_FILE = tmp / "j.db"
        self.journal = TradeJournal()
        self.patches = [mock.patch.object(main, "ORB_STATE_FILE", tmp / "orb_state.json"),
                        mock.patch.object(main, "journal", self.journal),
                        mock.patch.object(main, "telegram", None)]
        for p in self.patches: p.start()
        self._reset()

    def tearDown(self):
        import src.journal as journal_mod
        for p in self.patches: p.stop()
        self._reset(); self.journal.db.close()
        journal_mod.DB_DIR, journal_mod.DB_FILE = self.saved_db
        self.tmp.cleanup()

    def _reset(self):
        for d in (self.main._orb_days, self.main._orb_aggs, self.main._orb_trades):
            d.clear()
        self.main._orb_restored.clear()

    def _market(self, d):
        ms = MarketStream(epic=DOW, name="Wall Street")
        range_end = datetime.fromisoformat(d["date"]).replace(tzinfo=orb.NY) + timedelta(minutes=575)
        end = range_end.astimezone(orb.LONDON).replace(tzinfo=None)
        n = len(d["m5h"])
        ms.candles = deque([Candle(end - timedelta(minutes=5 * (n - i)), 0, h, l, 0)
                            for i, (h, l) in enumerate(zip(d["m5h"], d["m5l"]))], maxlen=100)
        return ms

    def _ticks(self, rows):
        """O, H, L, then C repeated so each minute carries as many ticks as the golden
        bar's volume (Oanda volume = price updates) — the pop test needs that activity."""
        for r in rows:
            t0 = datetime.fromtimestamp(r[0], timezone.utc)
            n = max(4, int(r[9]))
            seq = [(1, 5), (2, 6), (3, 7)] + [(4, 8)] * (n - 3)
            for k, (bi, ai) in enumerate(seq):
                yield t0 + timedelta(seconds=59 * k / n), r[bi], r[ai]

    def test_signal_journal_and_restart_resume(self):
        main = self.main
        day = next(d for d in golden() if d["pop"] and d["pop"]["reason"] in ("tp", "sl", "time")
                   and not is_nyse_holiday(d["date"]))
        ms = self._market(day); mc = BY_EPIC[DOW]; cfg = orb.ORB_CONFIGS[DOW]
        ticks = list(self._ticks(day["rows"]))
        entry_at = next(i for i, (ts, _, _) in enumerate(ticks)
                        if ts >= datetime.fromtimestamp(day["pop"]["ms"] / 1000, timezone.utc))
        cut = entry_at + 6
        for ts, b, a in ticks[:cut]:
            ms.bid, ms.offer = b, a
            main._orb_step(DOW, mc, cfg, ms, b, a, ts)
        open_rows = {r["bench_type"]: r for r in self.journal.get_open_orb_shadow(DOW)}
        self.assertIn("orb-shadow", open_rows)
        self.assertEqual(open_rows["orb-shadow"]["direction"], "BUY" if day["pop"]["side"] == 1 else "SELL")
        self.assertAlmostEqual(open_rows["orb-shadow"]["entry_price"], day["pop"]["entry"], places=6)
        row_id = open_rows["orb-shadow"]["id"]

        self._reset()                                         # "restart": memory gone, state file remains
        for ts, b, a in ticks[cut:]:
            ms.bid, ms.offer = b, a
            main._orb_step(DOW, mc, cfg, ms, b, a, ts)
        row = self.journal.db.execute("SELECT * FROM benched_outcomes WHERE id=?", (row_id,)).fetchone()
        self.assertIn(row["status"], ("WIN", "LOSS"))         # resolved, not expired as an orphan
        self.assertEqual(self.journal.get_open_orb_shadow(DOW), [])
        self.assertTrue(json.loads(main.ORB_STATE_FILE.read_text())[DOW]["day"]["status"]["pop"] == "done")

    def test_orphan_row_without_state_is_expired(self):
        rid = self.journal.log_orb_shadow(DOW, "Wall Street", "BUY", 1.0, 50, 95, "2026-10-01T13:40:00+00:00", 2.4, "orb-shadow")
        ms = MarketStream(epic=DOW, name="Wall Street"); ms.bid, ms.offer = 100.0, 102.4
        self.main._orb_step(DOW, BY_EPIC[DOW], orb.ORB_CONFIGS[DOW], ms, 100.0, 102.4, utc(2026, 10, 5, 12, 0))
        row = self.journal.db.execute("SELECT status, outcome FROM benched_outcomes WHERE id=?", (rid,)).fetchone()
        self.assertEqual((row["status"], row["outcome"]), ("EXPIRED", "state-lost"))


class TestModesAndConfig(unittest.TestCase):
    def test_no_live_mode_anywhere(self):
        self.assertEqual(orb.VALID_ORB_MODES, ("off", "shadow"))
        self.assertEqual(set(orb.ORB_CONFIGS), {DOW})
        self.assertEqual(BY_EPIC[DOW].orb, "shadow")
        for m in MARKETS:
            if m.epic != DOW:
                self.assertIsNone(m.orb, m.name)
        src = (REPO / "main.py").read_text()
        section = src[src.index("NY-OPEN 5-MIN OPENING-RANGE BREAKOUT"):src.index("# PULLBACK-IN-UPTREND strategy")]
        for forbidden in ("open_position", "place_order", "create_position", "risk_manager", "client."):
            self.assertNotIn(forbidden, section, forbidden)

    def test_main_and_telegram_resolve_identically(self):
        import main
        import src.telegram_bot as tb

        class Fake:
            def __init__(self, modes): self.orb_modes = modes
            _effective_orb_mode = tb.TelegramBot._effective_orb_mode
        original = getattr(main, "telegram", None)
        try:
            for override in (None,) + orb.VALID_ORB_MODES + ("live", "bogus"):
                for m in MARKETS:
                    fake = Fake({} if override is None else {m.epic: override}); main.telegram = fake
                    self.assertEqual(main._orb_mode(m), fake._effective_orb_mode(m), (m.epic, override))
        finally:
            main.telegram = original

    def test_tick_hook_is_wired_before_the_position_early_return(self):
        src = (REPO / "main.py").read_text()
        hook = src.index("    _orb_on_tick(epic, market)")
        self.assertLess(hook, src.index("    if not known_positions:\n        return", src.index("def on_price_update")))


class TestOrbCommand(unittest.TestCase):
    def setUp(self):
        import src.telegram_bot as tb
        self.tb = tb; self.tmp = tempfile.TemporaryDirectory(); tmp = Path(self.tmp.name)
        self.modes_file = tmp / "orb_modes.json"
        self.patches = [mock.patch.object(tb, "STATS_DIR", tmp),
                        mock.patch.object(tb, "MARKET_MODES_FILE", tmp / "market_modes.json"),
                        mock.patch.object(tb, "LEGACY_FOREX_MODE_FILE", tmp / "forex_mode.json"),
                        mock.patch.object(tb, "DAILY_TREND_MODES_FILE", tmp / "daily_trend_modes.json"),
                        mock.patch.object(tb, "PULLBACK_MODES_FILE", tmp / "pullback_modes.json"),
                        mock.patch.object(tb, "ORB_MODES_FILE", self.modes_file)]
        for p in self.patches: p.start()
        self.bot = tb.TelegramBot(TelegramConfig(bot_token="x", chat_id="1"))

    def tearDown(self):
        for p in self.patches: p.stop()
        self.tmp.cleanup()

    def run_cmd(self, *args):
        update, msg = edited_update()
        asyncio.run(self.bot.orb_command(update, SimpleNamespace(args=list(args))))
        return msg

    def test_board_toggle_persist_and_reject_live(self):
        self.assertIn("Wall Street: `shadow`", self.run_cmd().replies[0])
        self.run_cmd("wall", "off")
        self.assertEqual(json.loads(self.modes_file.read_text())["orb_modes"][DOW], "off")
        self.run_cmd("wall", "default"); self.assertNotIn(DOW, self.bot.orb_modes)
        msg = self.run_cmd("wall", "live")
        self.assertIn("no live ORB mode", msg.replies[0]); self.assertEqual(self.bot.orb_modes, {})
        msg = self.run_cmd("gold", "shadow")
        self.assertIn("no ORB config", msg.replies[0]); self.assertEqual(self.bot.orb_modes, {})


if __name__ == "__main__":
    unittest.main()
