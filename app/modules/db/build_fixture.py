"""Build data/schedule.sqlite, the committed schedule fixture.

db_Tech.sql seeds availability with NEWID(), so every run of it produces a
different calendar and no quoted number would survive a re-run. The schedule is
therefore built once and frozen.

    --from-mssql   export 2024 from the live SQL Server (service must be up)
    (default)      reproduce the same rules in Python, seeded and byte-stable

Either way the current-year rows are generated too - db_Tech.sql only covers
2024, and a live demo needs bookable dates now.
"""
from __future__ import annotations

import argparse
import datetime as dt

from ..config.settings import get_settings
from . import rules
from .store import SqliteScheduleStore

SEED_YEAR = rules.SEED_YEAR              # 2024, what db_Tech.sql populates
DEMO_YEARS = (2026, 2027)                # so "next Friday" resolves today too


def rows_from_mssql(conn_str: str, year: int) -> list[tuple]:
    import pyodbc

    conn = pyodbc.connect(conn_str, timeout=5)
    cur = conn.cursor()
    cur.execute(
        "SELECT [date], [time], position, available FROM dbo.Schedule "
        "WHERE YEAR([date]) = ? ORDER BY ScheduleID",
        year,
    )
    out = []
    for date, time, position, available in cur.fetchall():
        date_s = date.isoformat() if isinstance(date, dt.date) else str(date)[:10]
        time_s = time.strftime("%H:%M:%S") if isinstance(time, dt.time) else str(time)[:8]
        out.append((date_s, time_s, str(position), int(bool(available))))
    conn.close()
    return out


def build(from_mssql: bool = False) -> None:
    settings = get_settings()

    if from_mssql:
        print(f"Exporting {SEED_YEAR} from SQL Server (dbo.Schedule)...")
        seed_rows = rows_from_mssql(settings.mssql_conn, SEED_YEAR)
        provenance = "live SQL Server, seeded by data/db_Tech.sql"
    else:
        print(f"Generating {SEED_YEAR} deterministically from db_Tech.sql rules...")
        seed_rows = list(rules.generate_rows(SEED_YEAR))
        provenance = "deterministic reproduction of data/db_Tech.sql rules"

    all_rows = list(seed_rows)
    for year in DEMO_YEARS:
        generated = list(rules.generate_rows(year))
        all_rows.extend(generated)
        print(f"Generating {year} for live demos... {len(generated):,} rows")

    store = SqliteScheduleStore.create(settings.sqlite_path, all_rows)
    total = store.count()

    print(f"\nWrote {settings.sqlite_path.relative_to(settings.sqlite_path.parents[1])}")
    print(f"  provenance : {provenance}")
    print(f"  {SEED_YEAR} rows  : {len(seed_rows):,}")
    print(f"  total rows : {total:,}")

    available = store._query(
        "SELECT ScheduleID, date, time, position, available FROM Schedule "
        "WHERE available = 1 AND position = 'Python Dev' LIMIT 1"
    )
    print(f"  sanity     : {'first bookable Python Dev slot = ' + available[0].label() if available else 'NO AVAILABLE SLOTS'}")
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-mssql",
        action="store_true",
        help="export 2024 from the live SQL Server instead of regenerating it",
    )
    build(**vars(parser.parse_args()))
