"""Schedule rules, transcribed from data/db_Tech.sql.

The T-SQL is the source of truth; this mirrors it so any year can be
reproduced deterministically. Keep the two in sync.
"""
from __future__ import annotations

import datetime as dt
import random

# db_Tech.sql: WHERE DATENAME(WEEKDAY,d) NOT IN ('Saturday','Monday')
VALID_WEEKDAYS = {1, 2, 3, 4, 6}    # Python weekdays: Tue, Wed, Thu, Fri, Sun

# db_Tech.sql: SELECT CAST('09:00' AS TIME) ... WHERE t < '17:00'  -> 09:00..17:00 inclusive
FIRST_HOUR, LAST_HOUR = 9, 17
SLOT_HOURS = tuple(range(FIRST_HOUR, LAST_HOUR + 1))   # 9 slots per valid day

POSITIONS = ("Python Dev", "Sql Dev", "Analyst", "ML")
DEFAULT_POSITION = "Python Dev"

SEED_YEAR = 2024  # the year db_Tech.sql itself populates


def is_valid_day(d: dt.date) -> bool:
    """True when db_Tech.sql would have generated rows for this date."""
    return d.weekday() in VALID_WEEKDAYS


def valid_dates(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if is_valid_day(d):
            yield d
        d += dt.timedelta(days=1)


def _availability(rng: random.Random) -> int:
    """Mirror db_Tech.sql's availability expression.

    Two uniforms summed and thresholded at their mean: ~50% available, but
    triangular rather than uniform, as the original intends.
    """
    return 1 if (rng.randint(0, 99) + rng.randint(0, 99)) / 200.0 >= 0.5 else 0


def generate_rows(year: int, seed: int | None = None):
    """Yield (date, time, position, available) for a whole year.

    Seeded, unlike NEWID() - which is what keeps the committed fixture and the
    evaluation numbers stable.
    """
    rng = random.Random(seed if seed is not None else year)
    for d in valid_dates(dt.date(year, 1, 1), dt.date(year, 12, 31)):
        for hour in SLOT_HOURS:
            for position in POSITIONS:
                yield (d.isoformat(), f"{hour:02d}:00:00", position, _availability(rng))
