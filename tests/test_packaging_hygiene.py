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


if __name__ == "__main__":
    unittest.main()
