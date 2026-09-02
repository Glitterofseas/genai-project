"""Recruiter schedule access.

Two backends, same schema as dbo.Schedule: the committed SQLite fixture
(default) and the live SQL Server built by db_Tech.sql.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .rules import DEFAULT_POSITION

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS Schedule (
    ScheduleID INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT    NOT NULL,
    time       TEXT    NOT NULL,
    position   TEXT    NOT NULL,
    available  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_schedule_lookup ON Schedule (position, date, time);
"""


@dataclass(frozen=True)
class Slot:
    schedule_id: int
    date: dt.date
    time: dt.time
    position: str
    available: bool

    @property
    def start(self) -> dt.datetime:
        return dt.datetime.combine(self.date, self.time)

    def label(self) -> str:
        """e.g. 'Tuesday 07 May 2024 at 2:00 PM'. Hand-built: %-I isn't portable."""
        hour = self.time.hour
        suffix = "AM" if hour < 12 else "PM"
        display = hour % 12 or 12
        return f"{self.start:%A} {self.start:%d %b %Y} at {display}:{self.time.minute:02d} {suffix}"


def _row_to_slot(row: Sequence) -> Slot:
    raw_date, raw_time = row[1], row[2]
    date = raw_date if isinstance(raw_date, dt.date) else dt.date.fromisoformat(str(raw_date)[:10])
    if isinstance(raw_time, dt.time):
        time = raw_time
    else:
        parts = str(raw_time).split(":")
        time = dt.time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    return Slot(int(row[0]), date, time, str(row[3]), bool(row[4]))


class ScheduleStore:
    """Read/write access to the schedule.

    read_only keeps the evaluation from booking slots as it replays; otherwise
    each run would degrade the calendar for the next.
    """

    def __init__(self, read_only: bool = False):
        self.read_only = read_only
        self.attempted_bookings: list[int] = []

    def _query(self, sql: str, params: Sequence = ()) -> list[Slot]:
        raise NotImplementedError

    def _execute(self, sql: str, params: Sequence = ()) -> None:
        raise NotImplementedError

    # ---------- reads ----------

    def count(self) -> int:
        raise NotImplementedError

    def check_slot(self, date: dt.date, time: dt.time,
                   position: str = DEFAULT_POSITION) -> Slot | None:
        """The row for an exact slot, or None if there is no such slot.

        None isn't the same as unavailable - there are no Mondays or Saturdays
        in the calendar at all.
        """
        rows = self._query(
            "SELECT ScheduleID, date, time, position, available FROM Schedule "
            "WHERE position = ? AND date = ? AND time = ?",
            (position, date.isoformat(), time.strftime("%H:%M:%S")),
        )
        return rows[0] if rows else None

    def nearest_available(self, target: dt.datetime, position: str = DEFAULT_POSITION,
                          n: int = 3, not_before: dt.datetime | None = None,
                          exclude: set[dt.datetime] | None = None) -> list[Slot]:
        """The n available slots closest to `target`.

        not_before keeps suggestions in the future; exclude drops times already
        offered, so a declined slot isn't handed straight back.
        """
        floor = not_before or target
        window_lo = (floor - dt.timedelta(days=1)).date().isoformat()
        window_hi = (target + dt.timedelta(days=60)).date().isoformat()
        candidates = self._query(
            "SELECT ScheduleID, date, time, position, available FROM Schedule "
            "WHERE position = ? AND available = 1 AND date BETWEEN ? AND ? "
            "ORDER BY date, time",
            (position, window_lo, window_hi),
        )
        blocked = exclude or set()
        future = [s for s in candidates if s.start >= floor and s.start not in blocked]
        future.sort(key=lambda s: (abs((s.start - target).total_seconds()), s.start))
        return future[:n]

    # ---------- writes ----------

    def book(self, schedule_id: int) -> bool:
        """Mark a slot taken. Records intent but doesn't write when read_only."""
        self.attempted_bookings.append(schedule_id)
        if self.read_only:
            return True
        self._execute("UPDATE Schedule SET available = 0 WHERE ScheduleID = ?", (schedule_id,))
        return True


class SqliteScheduleStore(ScheduleStore):
    """SQLite-backed schedule.

    Streamlit caches this and reruns on a different thread, so the connection
    needs check_same_thread=False and a lock.
    """

    def __init__(self, path: Path, read_only: bool = False):
        super().__init__(read_only=read_only)
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()

    def _query(self, sql: str, params: Sequence = ()) -> list[Slot]:
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_slot(r) for r in rows]

    def _execute(self, sql: str, params: Sequence = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM Schedule").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    @classmethod
    def create(cls, path: Path, rows: Iterable[tuple]) -> "SqliteScheduleStore":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(path)
        conn.executescript(CREATE_SQL)
        conn.executemany(
            "INSERT INTO Schedule (date, time, position, available) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        return cls(path)


class MssqlScheduleStore(ScheduleStore):
    """Live SQL Server backend. Requires the MSSQLSERVER service to be running."""

    def __init__(self, conn_str: str, read_only: bool = False):
        super().__init__(read_only=read_only)
        import pyodbc  # imported lazily so sqlite-only users need no ODBC driver

        self._conn = pyodbc.connect(conn_str, timeout=5)

    def _query(self, sql: str, params: Sequence = ()) -> list[Slot]:
        cur = self._conn.cursor()
        # pyodbc uses the same '?' placeholder as sqlite3.
        cur.execute(sql, tuple(params)) if params else cur.execute(sql)
        return [_row_to_slot(r) for r in cur.fetchall()]

    def _execute(self, sql: str, params: Sequence = ()) -> None:
        cur = self._conn.cursor()
        cur.execute(sql, tuple(params))
        self._conn.commit()

    def count(self) -> int:
        return int(self._conn.cursor().execute("SELECT COUNT(*) FROM dbo.Schedule").fetchone()[0])


def working_copy_path(seed: Path) -> Path:
    """The app's writable copy, seeded from the committed fixture on first use.

    Bookings have to write somewhere, and the committed fixture is what the
    evaluation reads - a demo must not touch it.
    """
    working = seed.with_name(seed.stem + ".local" + seed.suffix)
    if not working.exists() and seed.exists():
        working.write_bytes(seed.read_bytes())
    return working


def get_store(settings=None, read_only: bool = False) -> ScheduleStore:
    """Build the backend named by SCHEDULE_BACKEND, defaulting to sqlite.

    Read-only callers get the committed fixture, writable ones the working copy.
    """
    if settings is None:
        from ..config.settings import get_settings

        settings = get_settings()
    if settings.schedule_backend == "mssql":
        return MssqlScheduleStore(settings.mssql_conn, read_only=read_only)
    path = settings.sqlite_path if read_only else working_copy_path(settings.sqlite_path)
    return SqliteScheduleStore(path, read_only=read_only)
