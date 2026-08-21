"""derive_github_repo() in rebuild-watcher.sh.

The gate queries GitHub for the commit it is about to deploy, so this function
decides WHICH repository's CI is consulted. Get it wrong in either direction
and the consequences are asymmetric but both bad: a wrong owner blocks every
deploy (404, fails closed), and a permissive parser that accepted a non-GitHub
remote would query a URL that cannot answer.

Exercised through the real function against real `git remote` state rather than
by asserting on the source text.
"""

import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
WATCHER = REPO / "rebuild-watcher.sh"

# These drive the real function against real `git remote` state, so they need a
# git binary. The bot image deliberately has none — it runs the bot, it does not
# build it — and the suite is shipped in that image so the golden tests can be
# run against the container. Skip rather than error there; CI and any dev
# machine have git and run them for real.
HAVE_GIT = shutil.which("git") is not None


@unittest.skipUnless(WATCHER.exists(), "rebuild-watcher.sh not present")
@unittest.skipUnless(HAVE_GIT, "git not installed (bot image) — parser tests need real remotes")
class TestDeriveGithubRepo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        body = WATCHER.read_text().splitlines()
        start = next(i for i, l in enumerate(body) if l.startswith("derive_github_repo() {"))
        end = next(i for i, l in enumerate(body[start:], start) if l == "}")
        cls.fn = "\n".join(body[start:end + 1])

    def derive(self, remote_url):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        if remote_url:
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=tmp, check=True)
        script = f'{self.fn}\nBOT_DIR="{tmp}"\nUPSTREAM_REMOTE=origin\nderive_github_repo\n'
        out = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True)
        return out.stdout.strip()

    def test_ssh_scp_form(self):
        self.assertEqual(self.derive("git@github.com:Turnipnator/IG.git"), "Turnipnator/IG")

    def test_ssh_scp_form_without_dot_git(self):
        self.assertEqual(self.derive("git@github.com:Turnipnator/IG"), "Turnipnator/IG")

    def test_https_form(self):
        self.assertEqual(self.derive("https://github.com/Turnipnator/IG.git"), "Turnipnator/IG")

    def test_ssh_url_form(self):
        self.assertEqual(self.derive("ssh://git@github.com/owner/repo.git"), "owner/repo")

    def test_a_fork_resolves_to_the_fork_not_upstream(self):
        """The whole point: a fork must gate against its OWN CI. Resolving to
        upstream here would query a repo that has never seen the fork's SHA."""
        self.assertEqual(self.derive("https://github.com/somemate/IG-fork.git"), "somemate/IG-fork")

    def test_non_github_remote_yields_nothing(self):
        for url in ("git@gitlab.com:owner/repo.git", "https://example.com/owner/repo.git"):
            self.assertEqual(self.derive(url), "", f"{url} should not resolve")

    def test_missing_remote_yields_nothing(self):
        self.assertEqual(self.derive(None), "")


if __name__ == "__main__":
    unittest.main()
