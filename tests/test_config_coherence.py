"""Config-coherence guards.

The failure these exist to prevent is silent. This bot is held together by
strings that nothing checks: an epic in data/market_modes.json, a
correlation_group name, a mode literal duplicated in two modules. Rename one
side and everything still imports, still runs, and simply stops doing what you
think it does — a market quietly reverts to its config default, a cluster stops
being a cluster, a /mode value stops being accepted.

Nothing in Python's import machinery can see any of it, because as far as the
interpreter is concerned they are all just str.
"""

import contextlib
import json
import os
import pathlib
import unittest
import warnings

warnings.filterwarnings("ignore")

import config
from config import MARKETS
from src.breakout import BREAKOUT_CONFIGS

REPO = pathlib.Path(__file__).resolve().parent.parent
EPICS = {m.epic for m in MARKETS}


class TestModeConstants(unittest.TestCase):
    """main.VALID_MARKET_MODES and telegram_bot.MARKET_MODES are declared
    SEPARATELY in two modules and must stay equal — main._market_mode and
    telegram._effective_mode both use their own copy to validate a /mode
    override. If one gains a mode the other doesn't, the two disagree about
    what is valid: telegram accepts and persists a mode that main then
    ignores, so the board shows one thing and the bot does another."""

    def test_mode_tuples_are_identical(self):
        import main
        import src.telegram_bot as telegram_bot

        self.assertEqual(
            tuple(main.VALID_MARKET_MODES),
            tuple(telegram_bot.MARKET_MODES),
            "main.VALID_MARKET_MODES and telegram_bot.MARKET_MODES have diverged",
        )

    def test_mode_resolution_agrees_across_both_modules(self):
        """main._market_mode and telegram._effective_mode carry docstrings
        telling each other to 'keep in sync'. That is a comment; this is the
        test. Exercised across every market AND every possible override."""
        import main
        import src.telegram_bot as telegram_bot

        class FakeTelegram:
            def __init__(self, modes):
                self.market_modes = modes

            _effective_mode = telegram_bot.TelegramBot._effective_mode

        original = getattr(main, "telegram", None)
        try:
            for override in (None,) + tuple(main.VALID_MARKET_MODES) + ("bogus",):
                for market in MARKETS:
                    modes = {} if override is None else {market.epic: override}
                    fake = FakeTelegram(modes)
                    main.telegram = fake
                    self.assertEqual(
                        main._market_mode(market),
                        fake._effective_mode(market),
                        f"mode resolution diverged for {market.epic} with override={override!r}",
                    )
        finally:
            main.telegram = original


class TestMarketModesFile(unittest.TestCase):
    """data/market_modes.json holds runtime /mode overrides.

    Parsed exactly as telegram_bot.load_market_modes does — `.get("market_modes",
    {})`, not the top level, which also carries a "saved_at" stamp. Reimplementing
    the format here was wrong once already; reading it the same way the loader
    does is the only version that cannot drift from it.

    Note what the loader does and does not check: it drops entries whose MODE is
    invalid, but never validates the EPIC. An override naming an epic that has
    since been renamed or disabled therefore survives into self.market_modes and
    simply never matches anything — dead config that looks live on the /mode
    board. That gap is what test_every_override_names_a_real_epic covers.
    """

    def setUp(self):
        self.path = REPO / "data" / "market_modes.json"
        if not self.path.exists():
            self.skipTest("no data/market_modes.json (VPS runtime state, absent in a fresh checkout)")
        raw = json.loads(self.path.read_text())
        self.overrides = raw.get("market_modes", {})

    def test_every_override_names_a_real_epic(self):
        for epic in self.overrides:
            self.assertIn(epic, EPICS,
                          f"market_modes.json overrides {epic!r}, which is not in MARKETS — "
                          f"the loader will keep it and it will never match")

    def test_every_override_has_a_mode_the_loader_will_keep(self):
        """An invalid mode is dropped silently by load_market_modes, so the
        market reverts to its config default while the file still claims
        otherwise."""
        import main
        for epic, mode in self.overrides.items():
            self.assertIn(mode, main.VALID_MARKET_MODES,
                          f"{epic} has mode {mode!r}, which load_market_modes will silently drop")


class TestBreakoutCoverage(unittest.TestCase):
    def test_breakout_configs_name_real_epics(self):
        """A BREAKOUT_CONFIGS entry for an epic not in MARKETS is unreachable."""
        for epic in BREAKOUT_CONFIGS:
            self.assertIn(epic, EPICS, f"BREAKOUT_CONFIGS has orphan epic {epic!r}")

    def test_every_breakout_market_has_a_config(self):
        """A market whose default_mode is a breakout flavour but which has no
        BREAKOUT_CONFIGS entry can never fire: analyze_breakout returns early
        on the missing config, silently, forever."""
        for market in MARKETS:
            mode = getattr(market, "default_mode", None)
            if mode in ("breakout", "breakout-shadow"):
                self.assertIn(
                    market.epic, BREAKOUT_CONFIGS,
                    f"{market.name} defaults to {mode!r} but has no BREAKOUT_CONFIGS entry",
                )

    def test_exit_channel_lookback_is_sane(self):
        """m = n//2 is the Turtle exit. m < 2 would make exit_channel's
        prior-window slice degenerate."""
        for epic, cfg in BREAKOUT_CONFIGS.items():
            self.assertGreaterEqual(cfg.n, 4, f"{epic}: n={cfg.n} too small for a half-length exit")
            self.assertGreaterEqual(cfg.m, 2, f"{epic}: m={cfg.m} degenerate")


class TestMarketFields(unittest.TestCase):
    def test_epics_are_unique(self):
        epics = [m.epic for m in MARKETS]
        self.assertEqual(len(epics), len(set(epics)), "duplicate epic in MARKETS")

    def test_htf_resolution_is_a_real_ig_resolution(self):
        """htf_resolution is passed straight to IG's /prices endpoint. A typo
        does not raise at import — it fails at runtime, per market, as an HTF
        fetch that never succeeds."""
        valid = {"MINUTE", "MINUTE_2", "MINUTE_3", "MINUTE_5", "MINUTE_10",
                 "MINUTE_15", "MINUTE_30", "HOUR", "HOUR_2", "HOUR_3",
                 "HOUR_4", "DAY", "WEEK", "MONTH"}
        for m in MARKETS:
            self.assertIn(m.htf_resolution, valid, f"{m.name}: bad htf_resolution {m.htf_resolution!r}")

    def test_trading_window_is_in_utc_range_and_non_empty(self):
        """start > end is DELIBERATE and means a wrap-around window: Crude Oil
        runs 23->21 UTC, i.e. nearly 24h. main.py:925 branches on exactly that
        (`(h < ts or h >= te) if ts < te else (te <= h < ts)`), so the only
        degenerate case is start == end, which makes every hour outside."""
        for m in MARKETS:
            self.assertTrue(0 <= m.trading_start <= 23, f"{m.name}: trading_start out of range")
            self.assertTrue(0 <= m.trading_end <= 24, f"{m.name}: trading_end out of range")
            self.assertNotEqual(m.trading_start, m.trading_end,
                                f"{m.name}: start == end is an always-closed window")

    def test_wrap_around_windows_match_the_gate_in_main(self):
        """Guards the wrap branch itself: for every market, at least one UTC
        hour must be INSIDE its window under main.py's own comparison."""
        for m in MARKETS:
            ts, te = m.trading_start, m.trading_end
            open_hours = [h for h in range(24)
                          if not ((h < ts or h >= te) if ts < te else (te <= h < ts))]
            self.assertTrue(open_hours, f"{m.name}: window {ts}-{te} is never open")

    def test_candle_interval_is_supported(self):
        """_trail_frame branches on candle_interval >= 60. An interval that is
        neither a real IG resolution nor a clean resample target silently
        produces a frame nobody validated."""
        for m in MARKETS:
            self.assertIn(m.candle_interval, (1, 2, 3, 5, 10, 15, 30, 60),
                          f"{m.name}: unsupported candle_interval {m.candle_interval}")

    def test_strategy_profile_exists(self):
        for m in MARKETS:
            self.assertIn(m.strategy, config.STRATEGY_PROFILES,
                          f"{m.name}: strategy {m.strategy!r} is not in STRATEGY_PROFILES")


class TestCorrelationGroups(unittest.TestCase):
    """The cluster filter blocks a second same-direction entry within a
    correlation_group. A typo'd group name does not error — it just makes that
    market its own singleton cluster, silently disabling the filter for it.
    CLUSTER_FILTER_ENFORCE has been True since 2026-07-06, so this is live."""

    def test_no_singleton_groups(self):
        groups = {}
        for m in MARKETS:
            g = getattr(m, "correlation_group", None)
            if g:
                groups.setdefault(g, []).append(m.name)
        for name, members in groups.items():
            self.assertGreater(
                len(members), 1,
                f"correlation_group {name!r} has only {members} — a one-member cluster "
                f"cannot block anything; likely a typo against another group name",
            )


@contextlib.contextmanager
def env(**overrides):
    """Temporarily set/clear env vars. None clears, so defaults can be exercised."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class TestTradingConfigBounds(unittest.TestCase):
    """Every value in TradingConfig feeds position sizing, and .env is hand-edited
    on the VPS. Unvalidated, RISK_PER_TRADE=2 is 200% risk rather than 2%, and
    MAX_POSITIONS=0 is a ZeroDivisionError at risk_manager.py:73 raised inside a
    daemon market thread the moment a signal fires — not at startup, where anyone
    would see it. These are fat-finger rails, not a view on risk."""

    def test_live_vps_values_load(self):
        """The bounds must not lock the running deployment out of booting."""
        with env(RISK_PER_TRADE="0.02", MAX_POSITIONS="8", CHECK_INTERVAL="5",
                 MAX_RISK_GBP=None, PRICE_DATA_POINTS=None, CACHE_TTL_MINUTES=None):
            cfg = config.load_trading_config()
        self.assertEqual(0.02, cfg.risk_per_trade)
        self.assertEqual(8, cfg.max_positions)
        self.assertEqual(45.0, cfg.max_risk_gbp)

    def test_defaults_load(self):
        with env(RISK_PER_TRADE=None, MAX_POSITIONS=None, MAX_RISK_GBP=None,
                 CHECK_INTERVAL=None, PRICE_DATA_POINTS=None, CACHE_TTL_MINUTES=None):
            self.assertIsNotNone(config.load_trading_config())

    def test_out_of_range_values_are_rejected(self):
        cases = {
            "RISK_PER_TRADE": ["2", "0.5", "0", "-0.01", "abc", ""],
            "MAX_POSITIONS": ["0", "-1", "21", "eight", ""],
            "MAX_RISK_GBP": ["0", "-45", "100000", ""],
            "CHECK_INTERVAL": ["0", "-5"],
            "PRICE_DATA_POINTS": ["0", "-1"],
            "CACHE_TTL_MINUTES": ["-1"],
        }
        for var, bad_values in cases.items():
            for bad in bad_values:
                with self.subTest(var=var, value=bad):
                    with env(**{var: bad}):
                        with self.assertRaises(config.ConfigError) as ctx:
                            config.load_trading_config()
                    self.assertIn(var, str(ctx.exception),
                                  "the error must name the offending variable")


class TestAccountTypeCannotFailOpen(unittest.TestCase):
    """base_url used to be `if DEMO: demo else: live`, so ANY unrecognised
    IG_ACC_TYPE — a trailing space, a typo, an empty value — silently pointed a
    demo-tested bot at real money. Unknown values must raise, never default."""

    def test_demo_and_live_resolve_to_their_own_endpoints(self):
        for raw, host in (("DEMO", "demo-api.ig.com"), ("LIVE", "api.ig.com"),
                          ("demo", "demo-api.ig.com"), (" live ", "api.ig.com")):
            with self.subTest(raw=raw), env(IG_ACC_TYPE=raw):
                url = config.load_ig_config().base_url
            self.assertIn(host, url)
            if host == "api.ig.com":
                self.assertNotIn("demo-api", url)

    def test_unknown_account_type_raises_rather_than_defaulting_to_live(self):
        for bad in ("DEMP", "", "PROD", "demo1", "TEST"):
            with self.subTest(value=bad), env(IG_ACC_TYPE=bad):
                with self.assertRaises(config.ConfigError):
                    config.load_ig_config()

    def test_hand_built_config_with_bad_acc_type_raises_at_use(self):
        """Defence in depth: the property must not guess even if load_ig_config
        is bypassed."""
        cfg = config.IGConfig(api_key="k", username="u", password="p", acc_type="DEMP")
        with self.assertRaises(config.ConfigError):
            _ = cfg.base_url


if __name__ == "__main__":
    unittest.main()
