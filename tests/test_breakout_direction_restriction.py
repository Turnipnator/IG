"""MarketConfig.allowed_direction must be honoured by the BREAKOUT path, live only.

Until 2026-09-15 `allowed_direction` was read in exactly one place —
analyze_market_from_stream, the MOMENTUM path — and src/breakout.py never referenced
it. Setting it on a breakout market was a SILENT NO-OP: the flag looked applied,
logged nothing, and blocked nothing. Gold and GBP/USD went long-only on 2026-09-15
(user decision) and both trade breakout LIVE, so that no-op would have been the whole
change.

Two live routes reach an order — the hour-close path (analyze_forex_breakout) and the
tick-cross path (_check_breakout_tick_trigger). Both funnel through
_execute_breakout_entry, which is where the hard gate lives so a future third caller
inherits it. analyze_forex_breakout additionally routes a blocked LIVE signal into the
shadow branch, so the blocked side is still snapshotted and resolved in R and the
pre-registered 2026-09-14 short-leg test still settles.

Live only: shadow/observer arms must keep recording shorts, and must keep their exact
`Breakout-shadow:` reject_reason prefix — scripts/backfill_breakout_shadow.py:111
filters on it with a LIKE.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import main

GOLD = "CS.D.USCGC.TODAY.IP"      # live breakout, allowed_direction="BUY"
CRUDE = "CC.D.CL.USS.IP"          # breakout-shadow, no direction restriction
SPX = "IX.D.SPTRD.DAILY.IP"       # allowed_direction="BUY" but momentum-shadow only


def _frame(n=80):
    idx = pd.date_range("2026-09-01", periods=n, freq="h")
    return pd.DataFrame({"date": idx, "open": 4300.0, "high": 4310.0, "low": 4290.0,
                         "close": 4305.0, "volume": 0})


class _Market:
    market_state = "TRADEABLE"
    name = "Gold"
    mid_price = 4282.0
    bid = 4281.0
    offer = 4283.0

    def to_dataframe(self):
        return _frame()


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (main.journal, dict(main.htf_trends), dict(main._breakout_armed),
                       main.BREAKOUT_TICK_LATCH_FILE, dict(main._breakout_entry_refusal_last))
        main.journal = MagicMock()
        main._breakout_armed.clear()
        main._breakout_entry_refusal_last.clear()   # the refusal log is throttled per epic/kind
        main.BREAKOUT_TICK_LATCH_FILE = Path(self._tmp.name) / "breakout_tick_latch.json"

    def tearDown(self):
        journal, htf, armed, latch, refusal = self._saved
        main.journal = journal
        main.htf_trends.clear(); main.htf_trends.update(htf)
        main._breakout_armed.clear(); main._breakout_armed.update(armed)
        main.BREAKOUT_TICK_LATCH_FILE = latch
        main._breakout_entry_refusal_last.clear()
        main._breakout_entry_refusal_last.update(refusal)
        self._tmp.cleanup()

    def _cfg(self, epic):
        return next(m for m in main.MARKETS if m.epic == epic)

    def _sig(self, epic, direction, name="Gold"):
        return main.TradeSignal(
            signal=main.Signal.BUY if direction == "BUY" else main.Signal.SELL,
            epic=epic, market_name=name, confidence=0.7, entry_price=4282.0,
            stop_distance=36.0, limit_distance=0.0,
            reason=f"Breakout: {direction} break of 55-bar channel @ 4280.0",
            atr=18.0, break_level=4280.0)


class Helper(_Base):
    def test_wrong_way_on_restricted_market_is_blocked(self):
        self.assertTrue(main._breakout_direction_blocked(self._cfg(GOLD), self._sig(GOLD, "SELL")))

    def test_allowed_way_passes(self):
        self.assertFalse(main._breakout_direction_blocked(self._cfg(GOLD), self._sig(GOLD, "BUY")))

    def test_unrestricted_market_never_blocks(self):
        cfg = self._cfg(CRUDE)
        self.assertEqual(cfg.allowed_direction, "")
        for d in ("BUY", "SELL"):
            self.assertFalse(main._breakout_direction_blocked(cfg, self._sig(CRUDE, d, "Crude Oil")))

    def test_missing_signal_is_not_blocked(self):
        self.assertFalse(main._breakout_direction_blocked(self._cfg(GOLD), None))


class HourClosePath(_Base):
    """analyze_forex_breakout: blocked LIVE shorts divert into the shadow branch."""

    def _run(self, epic, direction, mode, name="Gold"):
        sig = self._sig(epic, direction, name)
        with patch.object(main.breakout, "analyze_breakout", return_value=sig), \
             patch.object(main, "_arm_breakout_levels"), \
             patch.object(main, "_resolve_breakout_shadow"), \
             patch.object(main, "_snapshot_breakout_shadow") as snap, \
             patch.object(main, "_execute_breakout_entry") as execute, \
             patch.object(main, "_live_breakout_position", return_value=False):
            main.analyze_forex_breakout(epic, _Market(), self._cfg(epic), mode)
        return snap, execute

    def test_live_short_is_not_executed(self):
        snap, execute = self._run(GOLD, "SELL", "breakout")
        execute.assert_not_called()

    def test_live_short_is_still_snapshotted_for_the_counterfactual(self):
        snap, _ = self._run(GOLD, "SELL", "breakout")
        snap.assert_called_once()
        reason = main.journal.log_rejected_signal.call_args.kwargs["reject_reason"]
        self.assertIn("direction-restricted", reason)

    def test_live_long_still_executes(self):
        snap, execute = self._run(GOLD, "BUY", "breakout")
        execute.assert_called_once()
        snap.assert_not_called()

    def test_unrestricted_market_short_still_executes(self):
        _, execute = self._run(CRUDE, "SELL", "breakout", name="Crude Oil")
        execute.assert_called_once()

    def test_shadow_mode_short_keeps_the_original_reject_prefix(self):
        """scripts/backfill_breakout_shadow.py filters `Breakout-shadow:%` with a LIKE.
        S&P/FTSE/AI Index carry allowed_direction for momentum while running the
        breakout observer, so relabelling their rows would drop them from that query."""
        snap, execute = self._run(SPX, "SELL", "breakout-shadow", name="S&P 500")
        execute.assert_not_called()
        snap.assert_called_once()
        reason = main.journal.log_rejected_signal.call_args.kwargs["reject_reason"]
        self.assertTrue(reason.startswith("Breakout-shadow:"), reason)
        self.assertNotIn("direction-restricted", reason)


class OrderFunnelBackstop(_Base):
    """_execute_breakout_entry is the single funnel both live routes pass through."""

    def test_wrong_way_refused_before_any_broker_call(self):
        with patch.object(main, "client") as client, \
             patch.object(main, "_log_entry_refusal") as refusal:
            main._execute_breakout_entry(GOLD, _Market(), self._cfg(GOLD),
                                         self._sig(GOLD, "SELL"), None)
        client.get_market_info.assert_not_called()
        client.open_position.assert_not_called()
        self.assertEqual(refusal.call_args.args[3], "direction-restricted")

    def test_allowed_way_passes_the_direction_gate(self):
        """Reaches the NEXT gate rather than being refused on direction."""
        with patch.object(main, "client") as client, \
             patch.object(main, "_log_entry_refusal") as refusal:
            client.get_market_info.return_value = None
            client.get_balance.return_value = 9200.0
            main._execute_breakout_entry(GOLD, _Market(), self._cfg(GOLD),
                                         self._sig(GOLD, "BUY"), None)
        kinds = [c.args[3] for c in refusal.call_args_list]
        self.assertNotIn("direction-restricted", kinds)


class ConfigCoherence(_Base):
    def test_live_breakout_long_only_markets_are_actually_restricted(self):
        """The whole point of 2026-09-15: the flag must be SET on the live arms."""
        for epic in (GOLD, "CS.D.GBPUSD.TODAY.IP"):
            cfg = self._cfg(epic)
            self.assertEqual(cfg.default_mode, "breakout", f"{cfg.name} is not live breakout")
            self.assertEqual(cfg.allowed_direction, "BUY", f"{cfg.name} is not long-only")


if __name__ == "__main__":
    unittest.main()
