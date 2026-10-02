"""NY-open 5-minute opening-range breakout (ORB) — SHADOW ONLY (2026-10-02).

Origin: a friend's discretionary US30 checklist, mechanised and backtested in
Oanda_Gold/research_orb_friend.js (research_notes.md Part 13/13c). Only one reading
survived: range = the 09:30-09:35 New York 1-min bars, trend = 5-min swing structure
(last two fractal highs AND lows rising / falling), entry on the FIRST 1-min close
beyond the range in the trend direction, and only if that bar is a "pop" (range and
activity >= 1.5x the previous 20 bars), else stand aside for the day. SL 50 pts,
TP 95 pts (his "close at 90-95% of 100"), stop to entry at +50, flatten 15:30 NY,
one trade a day. Oanda 25 months: +0.37R/trade, n=53, PF 2.28 (found post hoc).

This module is pure logic (no broker, no journal) so it can be pinned against the
backtest. main.py feeds it IG ticks and records what it WOULD have done in
benched_outcomes. It has no order path and VALID_ORB_MODES has no "live" by design:
promotion needs GO_LIVE_CRITERIA.md's IG-native record first, and then a reviewed
change, not a mode flip.

Two variants run side by side on the same range and trend:
  orb-shadow        the candidate above (pop filter)
  orb-shadow-plain  any first close beyond the range (backtest ~0R): a fast parity
                    check on IG data, ~14 signals/month vs ~2.

IG ticks carry no volume. Oanda's M1 "volume" is a count of price updates, so the
activity half of the pop test counts ticks per minute here: the same quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")

VALID_ORB_MODES = ("off", "shadow")
VARIANTS = {"pop": "orb-shadow", "plain": "orb-shadow-plain"}


@dataclass(frozen=True)
class OrbConfig:
    range_start_min: int = 9 * 60 + 30   # NY minutes after midnight
    range_bars: int = 5                  # 1-min bars in the range (09:30-09:34)
    cutoff_min: int = 15 * 60 + 30       # no entries at/after, flatten at 15:30 NY
    stop_pts: float = 50.0
    target_pts: float = 95.0
    be_trigger_pts: float = 50.0         # stop -> entry once price has gone this far
    pop_lookback: int = 20
    pop_range_mult: float = 1.5
    pop_ticks_mult: float = 1.5

    @property
    def first_entry_min(self) -> int:
        return self.range_start_min + self.range_bars


ORB_CONFIGS: dict[str, OrbConfig] = {
    "IX.D.DOW.DAILY.IP": OrbConfig(),    # Wall Street — the only market the study covered
}


def has_orb_config(epic: str) -> bool:
    return epic in ORB_CONFIGS


def ny_minute(ts: datetime) -> tuple[str, int]:
    """(NY date ISO, NY minutes after midnight) for an aware timestamp."""
    t = ts.astimezone(NY)
    return t.date().isoformat(), t.hour * 60 + t.minute


# ---------------------------------------------------------------------------
# 1-minute bars from mid ticks
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    start: datetime          # aware UTC, minute start
    open: float
    high: float
    low: float
    close: float
    ticks: int = 1


class MinuteAggregator:
    """Mid ticks -> closed 1-min bars. A bar closes on the first tick of a later
    minute (same convention as streaming._update_candle), so a silent minute simply
    produces no bar — exactly like a missing Oanda M1 row."""

    def __init__(self, keep: int = 40):
        self.keep = keep
        self.closed: list[Bar] = []
        self.current: Optional[Bar] = None

    def on_tick(self, mid: float, ts: datetime) -> Optional[Bar]:
        start = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
        cur = self.current
        if cur is not None and start == cur.start:
            cur.high = max(cur.high, mid); cur.low = min(cur.low, mid); cur.close = mid; cur.ticks += 1
            return None
        if cur is not None and start < cur.start:
            return None                     # out-of-order tick: ignore rather than rewrite a bar
        self.current = Bar(start, mid, mid, mid, mid, 1)
        if cur is None:
            return None
        self.closed.append(cur)
        if len(self.closed) > self.keep:
            del self.closed[: len(self.closed) - self.keep]
        return cur


# ---------------------------------------------------------------------------
# 5-minute swing structure
# ---------------------------------------------------------------------------

def swing_trend(highs: list[float], lows: list[float]) -> int:
    """+1 if the last two confirmed fractal highs AND lows are rising, -1 if neither is,
    else 0. A 2-left/2-right fractal at p is only known once bar p+2 has closed, so
    callers pass COMPLETED bars only. Mirrors research_orb_friend.js prep()."""
    sh: list[float] = []
    sl: list[float] = []
    n = len(highs)
    for p in range(2, n - 2):
        h, l = highs[p], lows[p]
        if h > highs[p - 1] and h > highs[p - 2] and h >= highs[p + 1] and h >= highs[p + 2]:
            sh.append(h)
        if l < lows[p - 1] and l < lows[p - 2] and l <= lows[p + 1] and l <= lows[p + 2]:
            sl.append(l)
    if len(sh) < 2 or len(sl) < 2:
        return 0
    hh, hl = sh[-1] > sh[-2], sl[-1] > sl[-2]
    if hh and hl:
        return 1
    if not hh and not hl:
        return -1
    return 0


def m5_before(candles, cutoff_utc: datetime, keep: int = 100) -> tuple[list[float], list[float]]:
    """Highs/lows of 5-min candles that had CLOSED by cutoff_utc. Stream candle
    timestamps are naive bar starts on the container's London clock."""
    hs, ls = [], []
    for c in candles:
        ts = c.timestamp
        start = ts.replace(tzinfo=LONDON) if ts.tzinfo is None else ts
        if start + timedelta(minutes=5) <= cutoff_utc:
            hs.append(float(c.high)); ls.append(float(c.low))
    return hs[-keep:], ls[-keep:]


# ---------------------------------------------------------------------------
# The virtual position
# ---------------------------------------------------------------------------

@dataclass
class ShadowTrade:
    side: int                 # +1 long, -1 short
    entry: float
    stop: float
    target: float
    be_trigger: float         # price distance that arms breakeven
    stop_dist: float
    entry_ts: str             # ISO UTC
    row_id: Optional[int] = None
    be_done: bool = False
    first_bar: bool = True    # gap fills only apply after the entry bar (JS j > entryIdx)
    last_ts: Optional[str] = None

    def on_bar(self, bid_o, bid_h, bid_l, ask_o, ask_h, ask_l, ts: datetime, cutoff: bool) -> Optional[dict]:
        """Advance one bar (a tick is a bar with O=H=L). Pessimistic ordering, as in
        the backtest: clock exit at the open, then stop before target, then BE (which
        only takes effect from the next bar). Returns the exit dict or None."""
        long = self.side == 1
        xo, xh, xl = (bid_o, bid_h, bid_l) if long else (ask_o, ask_h, ask_l)
        self.last_ts = ts.astimezone(timezone.utc).isoformat()
        if not self.first_bar and cutoff:
            return self._exit(xo, "time")
        sl_hit = xl <= self.stop if long else xh >= self.stop
        tp_hit = xh >= self.target if long else xl <= self.target
        if sl_hit:
            gap = (not self.first_bar) and (xo < self.stop if long else xo > self.stop)
            self.first_bar = False
            return self._exit(xo if gap else self.stop, "be" if self.be_done else "sl")
        if tp_hit:
            self.first_bar = False
            return self._exit(self.target, "tp")
        if not self.be_done and (xh >= self.entry + self.be_trigger if long else xl <= self.entry - self.be_trigger):
            self.stop = self.entry; self.be_done = True
        self.first_bar = False
        return None

    def on_tick(self, bid: float, offer: float, ts: datetime, cutoff: bool) -> Optional[dict]:
        return self.on_bar(bid, bid, bid, offer, offer, offer, ts, cutoff)

    def _exit(self, price: float, reason: str) -> dict:
        r = (price - self.entry) / self.stop_dist * self.side
        return {"exit": float(price), "reason": reason, "r": float(r)}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowTrade":
        return cls(**d)


# ---------------------------------------------------------------------------
# One NY session
# ---------------------------------------------------------------------------

@dataclass
class OrbDay:
    """State for one NY date. Status per variant: waiting -> armed -> entered -> done,
    or skipped(reason). JSON-serialisable via to_dict/from_dict so a restart resumes."""
    date: str
    range_hi: Optional[float] = None
    range_lo: Optional[float] = None
    range_minutes: list = field(default_factory=list)
    trend: Optional[int] = None
    status: dict = field(default_factory=lambda: {v: "waiting" for v in VARIANTS})
    reason: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)      # variant -> side, set on the signal bar close
    trades: dict = field(default_factory=dict)       # variant -> ShadowTrade dict

    def skip(self, variant: str, reason: str) -> None:
        self.status[variant] = "skipped"; self.reason[variant] = reason

    def on_bar(self, cfg: OrbConfig, bar: Bar, history: list[Bar], trend_fn) -> list[tuple[str, str]]:
        """Feed a CLOSED 1-min bar. history = closed bars before it (oldest first).
        trend_fn() is called once, when the range completes. Returns events
        [(variant, 'signal'|'skip:<reason>')] for the caller to log."""
        events = []
        _, m = ny_minute(bar.start)
        rs, re = cfg.range_start_min, cfg.range_start_min + cfg.range_bars
        if rs <= m < re:
            self.range_minutes.append(m)
            self.range_hi = bar.high if self.range_hi is None else max(self.range_hi, bar.high)
            self.range_lo = bar.low if self.range_lo is None else min(self.range_lo, bar.low)
            return events
        if m < cfg.first_entry_min:
            return events
        if self.trend is None:
            # first bar after the range: decide the day's setup once
            if sorted(set(self.range_minutes)) != list(range(rs, re)):
                for v in VARIANTS:
                    if self.status[v] == "waiting":
                        self.skip(v, "no-range"); events.append((v, "skip:no-range"))
                self.trend = 0
                return events
            self.trend = int(trend_fn())
            for v in VARIANTS:
                if self.status[v] != "waiting":
                    continue
                if self.trend == 0:
                    self.skip(v, "no-trend"); events.append((v, "skip:no-trend"))
                else:
                    self.status[v] = "armed"
        if m >= cfg.cutoff_min:
            for v in VARIANTS:
                if self.status[v] == "armed":
                    self.skip(v, "no-break"); events.append((v, "skip:no-break"))
            return events
        broke = (bar.close > self.range_hi) if self.trend == 1 else (bar.close < self.range_lo)
        if not broke:
            return events
        for v in VARIANTS:
            if self.status[v] != "armed":
                continue
            if v == "pop" and len(history) < cfg.pop_lookback:
                self.skip(v, "no-history"); events.append((v, "skip:no-history"))   # restart: can't judge, don't guess
                continue
            if v == "pop" and not is_pop(cfg, bar, history):
                self.skip(v, "creep"); events.append((v, "skip:creep"))
                continue
            self.status[v] = "pending"; self.pending[v] = self.trend
            events.append((v, "signal"))
        return events

    def open_pending(self, cfg: OrbConfig, bid: float, offer: float, ts: datetime) -> list[tuple[str, ShadowTrade]]:
        """Fill every pending signal on this tick (the next bar's open: offer for a long,
        bid for a short). Returns the new trades for the caller to journal."""
        out = []
        for v, side in list(self.pending.items()):
            entry = offer if side == 1 else bid
            t = ShadowTrade(side=side, entry=float(entry),
                            stop=float(entry - side * cfg.stop_pts), target=float(entry + side * cfg.target_pts),
                            be_trigger=cfg.be_trigger_pts, stop_dist=cfg.stop_pts,
                            entry_ts=ts.astimezone(timezone.utc).isoformat())
            self.trades[v] = t.to_dict(); self.status[v] = "entered"
            out.append((v, t))
        self.pending.clear()
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OrbDay":
        return cls(**d)


def new_day(date: str) -> OrbDay:
    """A fresh session. Weekends and NYSE holidays are skipped up front: IG's DFB
    still quotes on a holiday, but there is no cash open behind it (the backtest's
    only two parity breaks were exactly such days, Presidents' Day and July 4)."""
    from src.session_bars import is_nyse_holiday
    day = OrbDay(date=date)
    d = datetime.fromisoformat(date)
    if d.weekday() >= 5 or is_nyse_holiday(d):
        for v in VARIANTS:
            day.skip(v, "holiday")
    return day


def is_pop(cfg: OrbConfig, bar: Bar, history: list[Bar]) -> bool:
    """Break bar's range AND tick count >= mult x the mean of the previous N bars.
    Fewer than N prior bars (e.g. a restart at 09:20) -> not a pop, never a guess."""
    prev = history[-cfg.pop_lookback:]
    if len(prev) < cfg.pop_lookback:
        return False
    avg_rng = sum(b.high - b.low for b in prev) / len(prev)
    avg_ticks = sum(b.ticks for b in prev) / len(prev)
    return (bar.high - bar.low) >= cfg.pop_range_mult * avg_rng and bar.ticks >= cfg.pop_ticks_mult * avg_ticks


# ---------------------------------------------------------------------------
# Bar replay — the same state machine driven by 1-min bid/ask bars instead of ticks.
# Used by the parity test against the backtest, and by any later report script.
# ---------------------------------------------------------------------------

def replay_day(cfg: OrbConfig, rows: list, m5_highs: list[float], m5_lows: list[float]) -> dict:
    """rows: [unix_s, bidO, bidH, bidL, bidC, askO, askH, askL, askC, ticks] for one NY
    session (plus some bars before the open for the pop lookback). Returns
    {variant: {side, entry, exit, R, reason, entry_ts} | None}."""
    if not rows:
        return {v: None for v in VARIANTS}
    first = datetime.fromtimestamp(rows[0][0], timezone.utc)
    date = ny_minute(datetime.fromtimestamp(rows[-1][0], timezone.utc))[0]
    day = new_day(date)
    history: list[Bar] = []
    open_trades: dict[str, ShadowTrade] = {}
    result: dict = {v: None for v in VARIANTS}
    for r in rows:
        ts = datetime.fromtimestamp(r[0], timezone.utc)
        d, m = ny_minute(ts)
        cutoff = m >= cfg.cutoff_min or d != date
        if day.pending and d == date:
            for v, t in day.open_pending(cfg, r[1], r[5], ts):
                open_trades[v] = t
        for v, t in list(open_trades.items()):
            ex = t.on_bar(r[1], r[2], r[3], r[5], r[6], r[7], ts, cutoff)
            if ex:
                result[v] = {"side": t.side, "entry": t.entry, **ex, "entry_ts": t.entry_ts}
                del open_trades[v]
        bar = Bar(ts, (r[1] + r[5]) / 2, (r[2] + r[6]) / 2, (r[3] + r[7]) / 2, (r[4] + r[8]) / 2, int(r[9]))
        if d == date and ts >= first and not all(st == "skipped" for st in day.status.values()):
            day.on_bar(cfg, bar, history, lambda: swing_trend(m5_highs, m5_lows))
        history.append(bar)
    return result
