"""Guards for the two hygiene rules that fail SILENTLY when broken.

Neither of these produces an error when violated. A gitleaks config without
[extend] reports "no leaks found" forever, which is indistinguishable from a
clean repo. A .dockerignore that excludes scripts/ breaks a weekly cron into a
log file nobody reads. Both are one-line mistakes with no feedback, which is
exactly what a test is for.
"""

import pathlib
import tomllib
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent

# These assert facts about the REPOSITORY, not about the running system. The
# suite is deliberately shipped inside the image so the golden tests can be run
# against the container (`docker exec ig-trading-bot python3 -m unittest
# discover -s tests -t .`), but .dockerignore excludes itself from that image,
# so in there these have nothing to check. Skip rather than fail — and key the
# skip on being at /app, the same discriminator main.py uses, so that a
# .dockerignore actually DELETED from the repo still fails loudly in CI instead
# of quietly skipping.
IN_CONTAINER = REPO == pathlib.Path("/app")


@unittest.skipIf(IN_CONTAINER, "repository-hygiene test; not meaningful inside the image")
class TestGitleaksConfig(unittest.TestCase):
    def test_config_extends_the_default_ruleset(self):
        """A custom gitleaks config REPLACES the defaults unless it extends
        them. Drop this block and every scan passes while checking nothing —
        a guard that fails open."""
        path = REPO / ".gitleaks.toml"
        self.assertTrue(path.exists(), ".gitleaks.toml is missing")
        cfg = tomllib.loads(path.read_text())
        self.assertTrue(
            cfg.get("extend", {}).get("useDefault") is True,
            ".gitleaks.toml must set [extend] useDefault = true, or it silently "
            "scans with an empty ruleset and always reports clean",
        )


@unittest.skipIf(IN_CONTAINER, "repository-hygiene test; .dockerignore excludes itself from the image")
class TestDockerignore(unittest.TestCase):
    def setUp(self):
        path = REPO / ".dockerignore"
        self.assertTrue(path.exists(), ".dockerignore is missing")
        self.rules = [ln.strip() for ln in path.read_text().splitlines()
                      if ln.strip() and not ln.strip().startswith("#")]

    def test_credentials_and_runtime_state_are_excluded(self):
        """The reason this file exists: .env was baked into an image layer."""
        for required in (".env", "data/", "logs/"):
            self.assertIn(required, self.rules, f".dockerignore must exclude {required}")

    def test_scripts_are_NOT_excluded(self):
        """A weekly VPS cron runs scripts/ from INSIDE the container:
            0 9 * * 1 docker exec ig-trading-bot python3 /app/scripts/tick_entry_readout.py
        Excluding scripts/ would break that readout with no visible symptom."""
        for forbidden in ("scripts", "scripts/", "/scripts", "*.py"):
            self.assertNotIn(
                forbidden, self.rules,
                f".dockerignore excludes {forbidden!r} — the weekly tick-entry readout "
                f"runs scripts/ from inside the image and would break silently",
            )

    def test_source_the_bot_needs_is_not_excluded(self):
        for forbidden in ("src", "src/", "main.py", "config.py",
                          "requirements.txt", "requirements.lock.txt"):
            self.assertNotIn(forbidden, self.rules,
                             f".dockerignore excludes {forbidden!r}, which the bot needs at runtime")


@unittest.skipIf(IN_CONTAINER, "repository-hygiene test; .github/ is excluded from the image")
class TestDeployGateWiring(unittest.TestCase):
    """The name of the CI job is load-bearing in THREE files.

    rebuild-watcher.sh will not deploy a commit unless a check run with that
    exact name reports success. Rename the job in ci.yml alone and the gate
    stops finding it — and because the gate fails CLOSED, every deploy silently
    refuses. You would discover it on the day you needed to ship a fix.

    health.yml's `gate` job queries the same name weekly for the same reason;
    it is the watchdog on the watchdog, and it too must agree.
    """

    JOB = "verify"

    def test_ci_declares_the_job_the_watcher_looks_for(self):
        ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
        self.assertIn(f'name: {self.JOB}', ci,
                      f"ci.yml no longer declares a job named {self.JOB!r}")

    def test_rebuild_watcher_looks_for_that_job(self):
        watcher = (REPO / "rebuild-watcher.sh").read_text()
        self.assertIn(f'CI_JOB_NAME="{self.JOB}"', watcher,
                      f"rebuild-watcher.sh CI_JOB_NAME is not {self.JOB!r} — the gate will "
                      f"never match and every deploy will be refused")

    def test_health_workflow_watches_the_same_job(self):
        health = (REPO / ".github" / "workflows" / "health.yml").read_text()
        self.assertIn(f'select(.name == "{self.JOB}")', health,
                      "health.yml's gate job queries a different job name than the watcher uses, "
                      "so it is no longer testing the real deploy gate")

    def test_watcher_uses_check_runs_not_the_legacy_status_endpoint(self):
        """/commits/{sha}/status returns state=pending with zero statuses
        forever on an Actions-only repo. A gate built on it either blocks every
        deploy or never blocks anything."""
        watcher = (REPO / "rebuild-watcher.sh").read_text()
        self.assertIn("/check-runs", watcher)
        self.assertNotIn("/status?", watcher)

    def test_watcher_still_has_the_force_escape_hatch(self):
        """The gate fails closed. Without the override, a GitHub outage or a
        rollback to a pre-CI commit means no deploy is possible at all."""
        watcher = (REPO / "rebuild-watcher.sh").read_text()
        self.assertIn("rebuild_trigger_force", watcher)


if __name__ == "__main__":
    unittest.main()
