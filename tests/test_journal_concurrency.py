"""TradeJournal must be safe under the thread-per-market fan-out.

main.on_candle_complete spawns one thread per market (main.py:900) and every one
of them reads and writes the same TradeJournal, whose sqlite3 connection is opened
check_same_thread=False and shared. Python's sqlite3 caches prepared statements by
SQL TEXT, so two threads running the same query with different bound parameters can
be handed the same underlying statement — one thread's bind/reset then clobbers the
other's in-flight iteration.

That is not theoretical. On 2026-08-20 the breakout-shadow resolver was handed
another market's rows and stamped NASDAQ episode #134 (entry 29837.3) with a Gold
high as its exit, booking +211.39R and flipping the pooled readout from -15.27R to
+196.12R (see the frame-ownership guard at main.py:2676). Reproduced locally on
Python 3.12.14 / SQLite 3.53.4 in 8/8 runs within 20s.

Both observed failure modes are asserted here, because get_open_benched swallows
exceptions and returns [], so a corrupted read is silent either way:
  - cross-epic leakage: rows returned for an epic the caller did not ask for
  - truncated/oversized result sets: wrong row count for a known-size fixture
"""

import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.journal as journal_mod


EPICS = ("IX.D.NASDAQ.CASH.IP", "CS.D.USCGC.TODAY.IP", "IX.D.SPTRD.DAILY.IP", "CO.D.DX.Month1.IP")
ROWS_PER_EPIC = 40
DURATION_S = 12


class TestJournalThreadSafety(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        # __init__ reads both module globals at call time, so patching here is enough
        # and no production signature has to change to make this testable.
        self._saved = (journal_mod.DB_DIR, journal_mod.DB_FILE)
        journal_mod.DB_DIR = tmp
        journal_mod.DB_FILE = tmp / "trade_journal.db"
        self.journal = journal_mod.TradeJournal()

        for epic in EPICS:
            for i in range(ROWS_PER_EPIC):
                self.journal.log_benched(
                    epic=epic, market_name=epic.split(".")[2], direction="BUY",
                    entry_price=100.0 + i, stop_distance=10.0, limit_distance=20.0,
                    score=50, bench_type="cap", benched_at=f"2026-08-20T{i % 24:02d}:00:00",
                )

    def tearDown(self):
        self.journal.db.close()
        journal_mod.DB_DIR, journal_mod.DB_FILE = self._saved
        self._tmp.cleanup()

    def test_concurrent_reads_do_not_cross_contaminate(self):
        """Every thread gets exactly its own epic's rows, under sustained contention."""
        failures = []
        stop = threading.Event()

        def worker(epic):
            try:
                while not stop.is_set():
                    rows = self.journal.get_open_benched(epic)
                    wrong = sorted({r["epic"] for r in rows if r["epic"] != epic})
                    if wrong:
                        failures.append(f"{epic} was handed rows for {wrong}")
                        stop.set()
                        return
                    if len(rows) != ROWS_PER_EPIC:
                        failures.append(
                            f"{epic} got {len(rows)} rows, expected {ROWS_PER_EPIC} "
                            f"(result set clobbered by a peer thread)"
                        )
                        stop.set()
                        return
            except Exception as e:  # a shared statement can also raise outright
                failures.append(f"{epic} raised {type(e).__name__}: {e}")
                stop.set()

        threads = [threading.Thread(target=worker, args=(e,), daemon=True) for e in EPICS]
        for t in threads:
            t.start()
        stop.wait(timeout=DURATION_S)
        stop.set()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual([], failures[:5], "journal reads cross-contaminated under concurrency")

    def test_concurrent_writes_all_land(self):
        """Interleaved writes from many threads must not lose or duplicate rows."""
        per_thread = 25
        errors = []

        def writer(n):
            try:
                for i in range(per_thread):
                    self.journal.log_benched(
                        epic="WRITE.TEST.EPIC", market_name="WriteTest", direction="SELL",
                        entry_price=float(n * 1000 + i), stop_distance=5.0,
                        limit_distance=10.0, score=1, bench_type="quality",
                        benched_at=f"2026-08-21T00:00:{i % 60:02d}",
                    )
            except Exception as e:
                errors.append(f"writer {n}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual([], errors)
        landed = self.journal.get_open_benched("WRITE.TEST.EPIC")
        self.assertEqual(4 * per_thread, len(landed), "writes were lost or duplicated")
        self.assertEqual(
            4 * per_thread, len({r["entry_price"] for r in landed}),
            "row contents were corrupted across threads",
        )


if __name__ == "__main__":
    unittest.main()
