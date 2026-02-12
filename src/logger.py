"""Trade logger — SQLite persistence for trades and bot state."""

import sqlite3
import logging
from datetime import datetime

log = logging.getLogger(__name__)


class TradeLogger:
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        log.info("Trade logger initialized: %s", db_path)

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL,
                side        TEXT    NOT NULL,
                module      TEXT    NOT NULL,
                regime      TEXT    NOT NULL,
                entry_price REAL    NOT NULL,
                exit_price  REAL,
                size        REAL    NOT NULL,
                entry_time  TEXT    NOT NULL,
                exit_time   TEXT,
                pnl         REAL,
                pnl_pct     REAL,
                exit_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                date       TEXT PRIMARY KEY,
                pnl        REAL    NOT NULL DEFAULT 0,
                num_trades INTEGER NOT NULL DEFAULT 0,
                wins       INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS bot_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ── trade lifecycle ──────────────────────────────────────────

    def log_entry(
        self,
        symbol: str,
        side: str,
        module: str,
        regime: str,
        entry_price: float,
        size: float,
        entry_time: datetime,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO trades
               (symbol, side, module, regime, entry_price, size, entry_time)
               VALUES (?,?,?,?,?,?,?)""",
            (symbol, side, module, regime, entry_price, size,
             entry_time.isoformat()),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: datetime,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
    ):
        self._conn.execute(
            """UPDATE trades
               SET exit_price=?, exit_time=?, pnl=?, pnl_pct=?, exit_reason=?
               WHERE id=?""",
            (exit_price, exit_time.isoformat(), pnl, pnl_pct,
             exit_reason, trade_id),
        )
        date_str = exit_time.strftime("%Y-%m-%d")
        win = 1 if pnl > 0 else 0
        self._conn.execute(
            """INSERT INTO daily_pnl (date, pnl, num_trades, wins)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(date) DO UPDATE SET
                   pnl = pnl + excluded.pnl,
                   num_trades = num_trades + 1,
                   wins = wins + excluded.wins""",
            (date_str, pnl, win),
        )
        self._conn.commit()

    # ── queries ──────────────────────────────────────────────────

    def get_daily_pnl(self, date_str: str) -> float:
        row = self._conn.execute(
            "SELECT pnl FROM daily_pnl WHERE date=?", (date_str,),
        ).fetchone()
        return row["pnl"] if row else 0.0

    def get_open_trades(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE exit_price IS NULL ORDER BY id",
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_trades(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY id",
        ).fetchall()
        return [dict(r) for r in rows]

    # ── key/value state ──────────────────────────────────────────

    def get_state(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM bot_state WHERE key=?", (key,),
        ).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str):
        self._conn.execute(
            """INSERT INTO bot_state (key, value) VALUES (?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (key, value),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
