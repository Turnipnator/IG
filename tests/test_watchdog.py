"""Tests for watchdog.sh.

Driven through the REAL script with `docker` and `systemctl` stubbed onto PATH,
rather than by reimplementing its logic here — a test that reimplements the
thing it tests only ever proves the test agrees with itself.

The property that matters most is not "does it detect a dead bot" (easy) but
"does it stay quiet when nothing is wrong" (hard, and the reason watchdogs get
muted and then ignored). Most of what follows is about silence.
"""

import os
import pathlib
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "watchdog.sh"


@unittest.skipUnless(SCRIPT.exists(), "watchdog.sh not present (excluded from the image)")
class WatchdogTestCase(unittest.TestCase):
    # Defaults describe a completely healthy system; each test overrides one thing.
    RUNNING = "true"
    HEALTH = "healthy"
    RESTARTS = "0"
    STARTED_MINS_AGO = 120
    WATCHER_ACTIVE = True

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "bin").mkdir()
        (self.tmp / "data").mkdir()
        (self.tmp / "logs").mkdir()
        self.log = self.tmp / "logs" / "ig_bot.log"
        self.log.write_text("2026-08-21 15:00:00 | INFO | [STREAM] Gold: HOLD\n")
        self.notify_file = self.tmp / "notifications.txt"
        self.state = self.tmp / "data" / ".watchdog_state"
        self.heartbeat = self.tmp / "data" / ".watchdog_heartbeat"

    def _stub(self, name, body):
        p = self.tmp / "bin" / name
        p.write_text("#!/bin/bash\n" + textwrap.dedent(body))
        p.chmod(0o755)

    def run_watchdog(self, **overrides):
        started = time.strftime(
            "%Y-%m-%dT%H:%M:%S.000000000Z",
            time.gmtime(time.time() - overrides.pop("started_mins_ago", self.STARTED_MINS_AGO) * 60),
        )
        running = overrides.pop("running", self.RUNNING)
        health = overrides.pop("health", self.HEALTH)
        restarts = overrides.pop("restarts", self.RESTARTS)
        watcher = overrides.pop("watcher_active", self.WATCHER_ACTIVE)

        # `docker inspect -f <fmt> <name>` — note the format string is $3, not
        # $2, because `inspect` is itself an argument. Match on "$*" so the
        # stub cannot be broken by argument order again.
        # STRICT: only the format strings docker actually understands are
        # answered; anything else exits non-zero with no output. A permissive
        # stub is worse than none — the first version matched
        # *State.RestartCount*, which is a docker TEMPLATE ERROR rather than a
        # real field, so it happily confirmed a probe that was dead in
        # production. The stub must not be more forgiving than the real tool.
        self._stub("docker", f"""
            case "$*" in
              *"{{{{.State.Running}}}}"*)       [ -n "{running}" ] && echo "{running}" ;;
              *"{{{{.State.StartedAt}}}}"*)     echo "{started}" ;;
              *"{{{{.State.Health.Status}}}}"*) echo "{health}" ;;
              *"{{{{.RestartCount}}}}"*)        echo "{restarts}" ;;
              *) echo "template parsing error" >&2; exit 1 ;;
            esac
            exit 0
        """)
        self._stub("systemctl", f"exit {0 if watcher else 3}\n")

        env = dict(os.environ)
        env["PATH"] = f"{self.tmp / 'bin'}:{env['PATH']}"
        env.update({
            "WATCHDOG_BOT_DIR": str(self.tmp),
            "WATCHDOG_LOG_FILE": str(self.log),
            "WATCHDOG_STATE_FILE": str(self.state),
            "WATCHDOG_NOTIFY_FILE": str(self.notify_file),
            "WATCHDOG_HEARTBEAT_FILE": str(self.heartbeat),
        })
        for k, v in overrides.items():
            env[f"WATCHDOG_{k.upper()}"] = str(v)

        before = self.notify_file.read_text() if self.notify_file.exists() else ""
        subprocess.run(["/bin/bash", str(SCRIPT)], env=env, capture_output=True, check=False)
        after = self.notify_file.read_text() if self.notify_file.exists() else ""
        return [ln for ln in after[len(before):].splitlines() if ln.strip()]


class TestStaysQuietWhenHealthy(WatchdogTestCase):
    def test_healthy_system_sends_nothing(self):
        self.assertEqual(self.run_watchdog(), [])

    def test_repeated_healthy_runs_send_nothing(self):
        for _ in range(3):
            self.assertEqual(self.run_watchdog(), [])

    def test_quiet_weekend_log_is_not_stale(self):
        """Sat/Sun are ~600-850 lines vs ~5500, but 11 days of history showed
        no gap over 20 min. A 19-minute-old log must NOT alert."""
        os.utime(self.log, (time.time() - 19 * 60, time.time() - 19 * 60))
        self.assertEqual(self.run_watchdog(), [])


class TestHeartbeat(WatchdogTestCase):
    """Proof the watchdog RAN.

    A healthy run is otherwise completely invisible: it writes nothing to its
    cron log, and `>>` with no output does not touch that file's mtime. So a
    watchdog whose cron entry had been deleted would look identical to one
    reporting all-clear. The thing that watches everything else could not
    demonstrate it was alive.
    """

    def test_written_on_a_healthy_run(self):
        self.run_watchdog()
        self.assertTrue(self.heartbeat.exists(), "no heartbeat written on a healthy run")
        self.assertIn("ok", self.heartbeat.read_text())

    def test_written_when_problems_exist(self):
        """The run that matters most must still leave proof it happened —
        otherwise a broken system and a dead watchdog look the same."""
        self.run_watchdog(running="false")
        text = self.heartbeat.read_text()
        self.assertIn("problem", text, text)
        self.assertIn("container_stopped", text, text)

    def test_written_even_though_the_script_exits_nonzero(self):
        """It exits 1 when problems exist; the heartbeat must be written
        BEFORE that exit, not skipped by it."""
        self.run_watchdog(watcher_active=False)
        self.assertTrue(self.heartbeat.exists())

    def test_timestamp_advances_between_runs(self):
        """mtime, not just presence — a file written once and never again
        would satisfy an existence check forever."""
        self.run_watchdog()
        first = self.heartbeat.stat().st_mtime
        time.sleep(1.1)
        self.run_watchdog()
        self.assertGreater(self.heartbeat.stat().st_mtime, first,
                           "heartbeat did not advance on the second run")

    def test_records_which_problems_were_seen(self):
        """Content, not just a timestamp: reading the file should tell you what
        the last run concluded without re-running anything."""
        self.run_watchdog(running="false", watcher_active=False)
        text = self.heartbeat.read_text()
        self.assertIn("container_stopped", text)
        self.assertIn("watcher_down", text)


class TestProbeContract(WatchdogTestCase):
    """The probes must ask docker for fields docker actually has.

    A wrong format string is a template parsing ERROR, not an empty result, and
    the script's `[ -n "$restarts" ]` guard turns that into silence rather than
    a failure. Live verification caught it; these keep it caught.
    """

    def test_healthy_run_records_the_restart_count(self):
        """If the field name is wrong, `restarts` is empty and this line is
        never written — which is exactly how crash-loop detection died."""
        self.run_watchdog(restarts="3")
        self.assertIn("_restarts=3", self.state.read_text(),
                      "state file has no _restarts line — the RestartCount probe returned nothing")

    def test_script_queries_top_level_RestartCount(self):
        """Comment lines are stripped first. The script documents the bad field
        name in prose to explain why it is wrong, and a naive substring search
        over the raw text reports a violation that exists only in a comment —
        found by exactly that false positive."""
        code = "\n".join(ln for ln in SCRIPT.read_text().splitlines()
                         if not ln.lstrip().startswith("#"))
        self.assertIn("{{.RestartCount}}", code)
        self.assertNotIn("{{.State.RestartCount}}", code,
                         "RestartCount is top-level in docker inspect; .State.RestartCount "
                         "is a template error and silently disables crash-loop detection")


class TestDeployGrace(WatchdogTestCase):
    def test_fresh_container_does_not_alert_on_starting_health(self):
        """Immediately after a deploy the container is legitimately 'starting'
        and its log thin. Alerting here would fire on every single deploy,
        which is how a channel gets muted."""
        msgs = self.run_watchdog(health="starting", started_mins_ago=1)
        self.assertEqual(msgs, [], f"alerted during the deploy grace window: {msgs}")

    def test_fresh_container_does_not_alert_on_thin_log(self):
        os.utime(self.log, (time.time() - 90 * 60, time.time() - 90 * 60))
        self.assertEqual(self.run_watchdog(started_mins_ago=1), [])

    def test_grace_does_NOT_suppress_a_stopped_container(self):
        """A young container is exactly when 'not running' matters most —
        the grace window must never hide it."""
        msgs = self.run_watchdog(running="false", started_mins_ago=1)
        self.assertTrue(any("STOPPED" in m for m in msgs), msgs)


class TestDetection(WatchdogTestCase):
    def test_stopped_container(self):
        self.assertTrue(any("STOPPED" in m for m in self.run_watchdog(running="false")))

    def test_missing_container(self):
        self.assertTrue(any("does not exist" in m for m in self.run_watchdog(running="")))

    def test_unhealthy_container(self):
        self.assertTrue(any("unhealthy" in m.lower() for m in self.run_watchdog(health="unhealthy")))

    def test_stale_log(self):
        os.utime(self.log, (time.time() - 45 * 60, time.time() - 45 * 60))
        self.assertTrue(any("no log output" in m for m in self.run_watchdog()))

    def test_polling_mode_degradation(self):
        """The 2026-07 class: the bot keeps running, quieter and worse, and
        raises no alarm of its own."""
        self.log.write_text(self.log.read_text() + "2026-08-21 15:05:00 | INFO | Polling cycle started\n")
        self.assertTrue(any("POLLING MODE" in m for m in self.run_watchdog()))

    def test_rebuild_watcher_down(self):
        """If the watcher dies, writing the trigger file does nothing at all."""
        msgs = self.run_watchdog(watcher_active=False)
        self.assertTrue(any("deploys will silently do nothing" in m for m in msgs), msgs)

    def test_wedged_trigger_file(self):
        trig = self.tmp / "data" / "rebuild_trigger"
        trig.write_text("2026-08-21T00:00:00Z")
        os.utime(trig, (time.time() - 30 * 60, time.time() - 30 * 60))
        self.assertTrue(any("wedged" in m for m in self.run_watchdog()))

    def test_crash_loop_detected_from_a_rising_restart_count(self):
        self.run_watchdog(restarts="0")                       # seed the state
        msgs = self.run_watchdog(restarts="4")
        self.assertTrue(any("crash-looping" in m for m in msgs), msgs)

    def test_restart_count_steady_is_not_a_crash_loop(self):
        self.run_watchdog(restarts="7")
        self.assertEqual(self.run_watchdog(restarts="7"), [])


class TestEdgeTriggering(WatchdogTestCase):
    def test_a_persistent_problem_alerts_only_once(self):
        first = self.run_watchdog(running="false")
        self.assertTrue(first)
        for _ in range(4):
            self.assertEqual(self.run_watchdog(running="false"), [],
                             "re-alerted on an unchanged problem — this is how alerts get muted")

    def test_renag_fires_after_the_configured_interval(self):
        self.run_watchdog(running="false")
        msgs = self.run_watchdog(running="false", renag_sec=0)
        self.assertTrue(any("still unresolved" in m for m in msgs), msgs)

    def test_recovery_is_announced_once(self):
        self.run_watchdog(running="false")
        recovered = self.run_watchdog(running="true")
        self.assertTrue(any("recovered" in m for m in recovered), recovered)
        self.assertEqual(self.run_watchdog(running="true"), [],
                         "repeated the recovery message")

    def test_independent_problems_alert_independently(self):
        first = self.run_watchdog(watcher_active=False)
        self.assertEqual(len(first), 1, first)
        second = self.run_watchdog(watcher_active=False, health="unhealthy")
        self.assertEqual(len(second), 1, f"expected only the NEW problem, got {second}")
        self.assertTrue(any("unhealthy" in m.lower() for m in second))


if __name__ == "__main__":
    unittest.main()
