"""PFO cycle-timing windows from the Market Timing Report monthly sheets.

INSTRUMENTATION ONLY. Nothing in the live trading path reads this module. It
exists so cycle-window performance can be MEASURED before anyone decides whether
to act on it. See research_notes.md and v3 agenda item 19.

Cycle state is a PURE FUNCTION of (epic, date) -- unlike ADX/RSI it does not
depend on the frame live happened to hold, so it never needs capturing at
runtime and ANY historical trade can be tagged retroactively. That is why this
is a lookup and not a logger, and why no journal schema change was needed.

Two kinds of marking, per the report legend:
  * cross ("PFO date point")  -- a candidate daily trend-change date
  * red box                   -- a strong WEEKLY cycle, a multi-day band
A cross inside a red band is the strongest signal ("look for an important trend
change when a daily PFO date aligns with a weekly cycle").

TOLERANCE. The report's author states a cross "can happen day before or after",
so a cross at D marks [D-1, D+1]. Bands are already multi-day and are NOT
widened -- widening both would make ~all of some months in-window and destroy
the contrast the test depends on. `cross_tolerance_days` exposes this so the
sensitivity can be swept rather than assumed.

Sheets live in cycles/*.json, committed rather than in the gitignored data/,
because they are source evidence for a measurement and must stay diffable.
"""
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

CYCLES_DIR = Path(__file__).resolve().parent.parent / "cycles"

# Report instrument -> EPIC. "US Equities" is S&P 500 ONLY (confirmed by the
# report's reader 2026-09-01) -- deliberately NOT the other index EPICs, which
# would silently multiply the sample with markets the sheet does not cover.
INSTRUMENT_EPICS: dict[str, tuple[str, ...]] = {
    "US Equities":  ("IX.D.SPTRD.DAILY.IP",),
    "Crude Oil":    ("CC.D.CL.USS.IP",),
    "Gold":         ("CS.D.USCGC.TODAY.IP",),
    "Dollar Index": ("CC.D.DX.USS.IP",),
    "EUR/USD":      ("CS.D.EURUSD.TODAY.IP",),
    "Bitcoin":      ("CS.D.BITCOIN.TODAY.IP",),
}


@dataclass(frozen=True)
class CycleState:
    """What the sheet says about one (epic, day)."""
    cross: bool = False       # a PFO date point, tolerance already applied
    week: bool = False        # inside a strong weekly (red) band
    volatility: bool = False  # inside a yellow volatility band

    @property
    def strong(self) -> bool:
        """Cross aligned with a weekly cycle -- the report's own 'strongest'."""
        return self.cross and self.week

    @property
    def any(self) -> bool:
        return self.cross or self.week or self.volatility

    @property
    def label(self) -> str:
        if self.strong:
            return "strong"
        if self.cross:
            return "cross"
        if self.week:
            return "week"
        if self.volatility:
            return "volatility"
        return "none"


def _as_date(when) -> date:
    if isinstance(when, datetime):
        return when.date()
    if isinstance(when, date):
        return when
    return datetime.fromisoformat(str(when)[:19]).date()


def _expand(lo: str, hi: str):
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    while a <= b:
        yield a
        a += timedelta(days=1)


def load_cycles(cycles_dir: Path = CYCLES_DIR, cross_tolerance_days: int = 1
                ) -> dict[str, dict[date, CycleState]]:
    """Build {epic: {date: CycleState}} from every sheet in `cycles_dir`.

    Sheets are merged, so overlapping months compose rather than collide. An
    unreadable sheet is skipped with a warning rather than killing the run --
    these are transcriptions of photographs and a torn one should not be fatal.
    """
    out: dict[str, dict[date, CycleState]] = {}

    def mark(epic: str, d: date, **kw) -> None:
        cur = out.setdefault(epic, {}).get(d, CycleState())
        out[epic][d] = CycleState(
            cross=cur.cross or kw.get("cross", False),
            week=cur.week or kw.get("week", False),
            volatility=cur.volatility or kw.get("volatility", False),
        )

    for path in sorted(Path(cycles_dir).glob("*.json")):
        try:
            sheet = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"cycle sheet {path.name} unreadable, skipped: {e}")
            continue
        for instrument, spec in (sheet.get("instruments") or {}).items():
            epics = INSTRUMENT_EPICS.get(instrument)
            if not epics:
                logger.debug(f"{path.name}: no EPIC mapped for {instrument!r}")
                continue
            for epic in epics:
                for iso in spec.get("crosses", []):
                    d0 = date.fromisoformat(iso)
                    for off in range(-cross_tolerance_days, cross_tolerance_days + 1):
                        mark(epic, d0 + timedelta(days=off), cross=True)
                for lo, hi in spec.get("weeks", []):
                    for d in _expand(lo, hi):
                        mark(epic, d, week=True)
                for lo, hi in spec.get("volatility", []):
                    for d in _expand(lo, hi):
                        mark(epic, d, volatility=True)
    return out


_CACHE: dict[int, dict[str, dict[date, CycleState]]] = {}
NO_CYCLE = CycleState()


def cycle_state(epic: str, when, cross_tolerance_days: int = 1) -> CycleState:
    """Cycle state for one EPIC on one day. Unknown epic/date -> all-False.

    An all-False result is NOT the same as "no sheet covers this date" -- callers
    that need to exclude uncovered periods must check `covered_range()`, or a
    month with no sheet silently reads as a quiet out-of-window period and
    biases any in/out comparison.
    """
    if cross_tolerance_days not in _CACHE:
        _CACHE[cross_tolerance_days] = load_cycles(
            cross_tolerance_days=cross_tolerance_days)
    return _CACHE[cross_tolerance_days].get(epic, {}).get(_as_date(when), NO_CYCLE)


def covered_range(cycles_dir: Path = CYCLES_DIR) -> tuple[date, date] | None:
    """First and last calendar day any sheet covers, from the month fields."""
    months = []
    for path in sorted(Path(cycles_dir).glob("*.json")):
        try:
            m = json.loads(path.read_text()).get("month")
        except (json.JSONDecodeError, OSError):
            continue
        if m:
            months.append(m)
    if not months:
        return None
    lo = date.fromisoformat(f"{min(months)}-01")
    y, mm = map(int, max(months).split("-"))
    hi = (date(y + (mm == 12), (mm % 12) + 1, 1) - timedelta(days=1))
    return lo, hi
