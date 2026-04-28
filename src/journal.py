"""
Trade Journal — SQLite-based trade logging for pattern analysis.

Logs entry/exit details, indicator snapshots, and exit reasons.
Provides query methods for win rate, P&L by instrument, and more.
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_DIR = Path("/app/data") if os.path.exists("/app") else Path("data")
DB_FILE = DB_DIR / "trade_journal.db"


class TradeJournal:
    """SQLite trade journal for post-trade analysis."""

    def __init__(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(DB_FILE), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Trade journal initialized: {DB_FILE}")

    def _create_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deal_id TEXT UNIQUE,
                epic TEXT NOT NULL,
                market_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                size REAL NOT NULL,
                strategy TEXT,

                -- Entry snapshot
                entry_price REAL,
                entry_time TEXT,
                stop_distance REAL,
                limit_distance REAL,
                confidence REAL,
                reason TEXT,
                adx REAL,
                rsi REAL,
                atr REAL,
                ema_fast REAL,
                ema_medium REAL,
                ema_slow REAL,
                htf_trend TEXT,

                -- Exit snapshot (filled on close)
                exit_price REAL,
                exit_time TEXT,
                exit_reason TEXT,
                pnl REAL,
                duration_mins REAL,
                adx_at_exit REAL,

                -- Status
                status TEXT DEFAULT 'OPEN'
            );

            CREATE TABLE IF NOT EXISTS rejected_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epic TEXT NOT NULL,
                market_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                adx REAL,
                rsi REAL,
                reject_reason TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trades_epic ON trades(epic);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_rejected_epic ON rejected_signals(epic);
        """)
        self.db.commit()

    def log_entry(
        self,
        deal_id: str,
        epic: str,
        market_name: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_distance: float,
        limit_distance: float,
        confidence: float,
        reason: str,
        strategy: str = "",
        adx: float = 0.0,
        rsi: float = 0.0,
        atr: float = 0.0,
        ema_fast: float = 0.0,
        ema_medium: float = 0.0,
        ema_slow: float = 0.0,
        htf_trend: str = "",
    ) -> None:
        """Log a trade entry with indicator snapshot."""
        try:
            self.db.execute(
                """INSERT OR REPLACE INTO trades
                   (deal_id, epic, market_name, direction, size, strategy,
                    entry_price, entry_time, stop_distance, limit_distance,
                    confidence, reason, adx, rsi, atr,
                    ema_fast, ema_medium, ema_slow, htf_trend, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (deal_id, epic, market_name, direction, size, strategy,
                 entry_price, datetime.now().isoformat(), stop_distance, limit_distance,
                 confidence, reason, adx, rsi, atr,
                 ema_fast, ema_medium, ema_slow, htf_trend),
            )
            self.db.commit()
            logger.info(f"Journal: logged entry for {market_name} ({deal_id})")
        except Exception as e:
            logger.warning(f"Journal: failed to log entry: {e}")

    def log_exit(
        self,
        deal_id: str,
        pnl: float,
        exit_reason: str,
        exit_price: float = 0.0,
        adx_at_exit: float = 0.0,
        status: str = "CLOSED",
    ) -> None:
        """Log a trade exit with P&L and reason.

        status: 'CLOSED' when pnl is broker-confirmed (deal confirmation or
        matched IG transaction); 'PROVISIONAL' when pnl is the cached stream
        value and awaits later reconciliation against transaction history.
        """
        try:
            # Calculate duration
            row = self.db.execute(
                "SELECT entry_time FROM trades WHERE deal_id = ?", (deal_id,)
            ).fetchone()

            duration_mins = 0.0
            if row and row["entry_time"]:
                entry_dt = datetime.fromisoformat(row["entry_time"])
                duration_mins = (datetime.now() - entry_dt).total_seconds() / 60

            self.db.execute(
                """UPDATE trades SET
                   exit_price = ?, exit_time = ?, exit_reason = ?,
                   pnl = ?, duration_mins = ?, adx_at_exit = ?, status = ?
                   WHERE deal_id = ?""",
                (exit_price, datetime.now().isoformat(), exit_reason,
                 pnl, duration_mins, adx_at_exit, status, deal_id),
            )
            self.db.commit()
            marker = " [PROVISIONAL]" if status == "PROVISIONAL" else ""
            logger.info(f"Journal: logged exit for {deal_id} — £{pnl:.2f} ({exit_reason}){marker}")
        except Exception as e:
            logger.warning(f"Journal: failed to log exit: {e}")

    # --- Reconciliation (PROVISIONAL → CLOSED via IG transaction history) ---

    def has_provisional(self) -> bool:
        """Cheap check — true if any PROVISIONAL rows exist (gates expensive API calls)."""
        try:
            row = self.db.execute(
                "SELECT 1 FROM trades WHERE status = 'PROVISIONAL' LIMIT 1"
            ).fetchone()
            return row is not None
        except Exception:
            return False

    def get_provisional_rows(self, max_age_hours: int = 24) -> list[dict]:
        """All PROVISIONAL trades from the last N hours (IG txn history window)."""
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        rows = self.db.execute(
            """SELECT deal_id, market_name, direction, size, entry_price,
                      entry_time, exit_time, pnl
               FROM trades
               WHERE status = 'PROVISIONAL' AND entry_time > ?""",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def confirm_provisional(self, deal_id: str, pnl: float, exit_price: float) -> None:
        """Promote a PROVISIONAL row to CLOSED with broker-confirmed values."""
        try:
            self.db.execute(
                """UPDATE trades SET pnl = ?, exit_price = ?, status = 'CLOSED'
                   WHERE deal_id = ? AND status = 'PROVISIONAL'""",
                (pnl, exit_price, deal_id),
            )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Journal: confirm_provisional failed for {deal_id}: {e}")

    def mark_unmatched(self, deal_id: str) -> None:
        """Stop trying to reconcile — IG transaction never appeared (rare)."""
        try:
            self.db.execute(
                """UPDATE trades SET status = 'UNMATCHED'
                   WHERE deal_id = ? AND status = 'PROVISIONAL'""",
                (deal_id,),
            )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Journal: mark_unmatched failed for {deal_id}: {e}")

    def log_rejected_signal(
        self,
        epic: str,
        market_name: str,
        direction: str,
        confidence: float,
        adx: float,
        rsi: float,
        reject_reason: str,
    ) -> None:
        """Log a signal that was generated but rejected (filters, cooldowns, etc.)."""
        try:
            self.db.execute(
                """INSERT INTO rejected_signals
                   (epic, market_name, direction, timestamp, confidence, adx, rsi, reject_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (epic, market_name, direction, datetime.now().isoformat(),
                 confidence, adx, rsi, reject_reason),
            )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Journal: failed to log rejection: {e}")

    # --- Query methods ---

    def get_stats_by_market(self, days: int = 30) -> list[dict]:
        """Win rate and avg P&L per market over the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.db.execute(
            """SELECT market_name,
                      COUNT(*) as total,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                      SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                      ROUND(SUM(pnl), 2) as total_pnl,
                      ROUND(AVG(pnl), 2) as avg_pnl,
                      ROUND(AVG(duration_mins), 0) as avg_duration
               FROM trades
               WHERE status = 'CLOSED' AND entry_time > ?
               GROUP BY market_name
               ORDER BY total_pnl DESC""",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats_by_exit_reason(self, days: int = 30) -> list[dict]:
        """P&L breakdown by exit type."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.db.execute(
            """SELECT exit_reason,
                      COUNT(*) as total,
                      ROUND(SUM(pnl), 2) as total_pnl,
                      ROUND(AVG(pnl), 2) as avg_pnl
               FROM trades
               WHERE status = 'CLOSED' AND entry_time > ?
               GROUP BY exit_reason
               ORDER BY total_pnl DESC""",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_overall_stats(self, days: int = 30) -> dict:
        """Overall performance summary."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        row = self.db.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                      SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                      ROUND(SUM(pnl), 2) as total_pnl,
                      ROUND(AVG(pnl), 2) as avg_pnl,
                      ROUND(AVG(CASE WHEN pnl > 0 THEN pnl END), 2) as avg_win,
                      ROUND(AVG(CASE WHEN pnl <= 0 THEN pnl END), 2) as avg_loss,
                      ROUND(AVG(duration_mins), 0) as avg_duration,
                      ROUND(AVG(adx), 1) as avg_adx_entry
               FROM trades
               WHERE status = 'CLOSED' AND entry_time > ?""",
            (cutoff,),
        ).fetchone()
        return dict(row) if row else {}

    def get_rejected_count_by_market(self, days: int = 7) -> list[dict]:
        """Count rejected signals per market (helps assess filter strictness)."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.db.execute(
            """SELECT market_name,
                      COUNT(*) as rejected,
                      ROUND(AVG(adx), 1) as avg_adx,
                      ROUND(AVG(confidence), 2) as avg_confidence
               FROM rejected_signals
               WHERE timestamp > ?
               GROUP BY market_name
               ORDER BY rejected DESC""",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 10) -> list[dict]:
        """Get most recent closed trades."""
        rows = self.db.execute(
            """SELECT market_name, direction, entry_price, exit_price,
                      pnl, exit_reason, duration_mins, adx, confidence,
                      entry_time, htf_trend
               FROM trades
               WHERE status = 'CLOSED'
               ORDER BY exit_time DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
