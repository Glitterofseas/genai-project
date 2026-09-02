"""Access to the recruiter schedule.

Two backends behind one interface (option C):

  * sqlite  - a committed, deterministic fixture in data/schedule.sqlite.
              The default, so the project is reproducible on any machine and
              the evaluation numbers in the notebook are stable.
  * mssql   - the live SQL Server built by data/db_Tech.sql, for demonstrating
              the real T-SQL artefact.

Both speak the same schema as dbo.Schedule:
    ScheduleID, date, time, position, available
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
        """Human-readable slot, e.g. 'Tuesday 07 May 2024 at 2:00 PM'.

        Built by hand rather than with strftime because %-I (hour without a
        leading zero) is not portable - it raises ValueError on Windows.
        """
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

    `read_only` exists because the evaluation harness replays 15 conversations
    through the real agent: without it, every replay would book slots and the
    next run would see a different - permanently degraded - calendar.
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
        """The row for an exact slot, or None when the calendar has no such slot.

        None is meaningful: db_Tech.sql generates no Mondays or Saturdays, so a
        Monday proposal is not 'unavailable', it simply does not exist.
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
        """The n available slots closest in time to `target`.

        Spec: "it then suggests the three nearest available time slots".
        `not_before` keeps suggestions in the conversation's future, and
        `exclude` drops times already offered - a candidate who says "I can't
        at that time" must not be handed the same slot back.
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
        """Mark a slot taken. A no-op that still records intent in read_only mode."""
        self.attempted_bookings.append(schedule_id)
        if self.read_only:
            return True
        self._execute("UPDATE Schedule SET available = 0 WHERE ScheduleID = ?", (schedule_id,))
        return True


class SqliteScheduleStore(ScheduleStore):
    """SQLite-backed schedule.

    `check_same_thread=False` plus an explicit lock is required because
    Streamlit caches this object with @st.cache_resource and then reruns the
    script on a different thread than the one that opened the connection.
    Without it, the first message in the UI raises ProgrammingError.
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
        # pyodbc uses the same '?' placeholder as sqlite3, so the SQL in the
        # base class is portable as written.
        cur.execute(sql, tuple(params)) if params else cur.execute(sql)
        return [_row_to_slot(r) for r in cur.fetchall()]

    def _execute(self, sql: str, params: Sequence = ()) -> None:
        cur = self._conn.cursor()
        cur.execute(sql, tuple(params))
        self._conn.commit()

    def count(self) -> int:
        return int(self._conn.cursor().execute("SELECT COUNT(*) FROM dbo.Schedule").fetchone()[0])


def working_copy_path(seed: Path) -> Path:
    """Path to the app's writable copy of the schedule, created on first use.

    data/schedule.sqlite is committed and is the reproducibility anchor: the
    evaluation reads it and the notebook quotes numbers derived from it. But a
    real booking has to write somewhere, and a demo that mutated the committed
    file would leave the repository dirty and the published numbers unrepeatable.
    So the app books against a gitignored copy seeded from it.
    """
    working = seed.with_name(seed.stem + ".local" + seed.suffix)
    if not working.exists() and seed.exists():
        working.write_bytes(seed.read_bytes())
    return working


def get_store(settings=None, read_only: bool = False) -> ScheduleStore:
    """Build the store named by SCHEDULE_BACKEND, falling back to sqlite.

    Read-only callers (the evaluation harness) get the committed seed itself;
    writable callers (the CLI and Streamlit app) get the working copy.
    """
    if settings is None:
        from ..config.settings import get_settings

        settings = get_settings()
    if settings.schedule_backend == "mssql":
        return MssqlScheduleStore(settings.mssql_conn, read_only=read_only)
    path = settings.sqlite_path if read_only else working_copy_path(settings.sqlite_path)
    return SqliteScheduleStore(path, read_only=read_only)
