"""Resolve the natural-language date/time phrases candidates actually use.

Spec (Additional Implementation Steps): "if the user mentions 'next Friday', it
infers the current date from the time the conversation took place and combines
it with the user's input."  So every resolution is relative to an ANCHOR - the
conversation's own timestamp - never to the wall clock.

Deliberately rule-based, not LLM-based: it must be deterministic, free, and
unit-testable.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3,
    "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}

# The dataset uses U+202F NARROW NO-BREAK SPACE inside times ("10 AM").
# Normalising every unicode space class to a plain space is what makes the
# time regex below actually match real transcript text.
_SPACE_RE = re.compile(r"[\s  -​  　]+")

_TIME_RE = re.compile(
    r"\b(?P<hour>1[0-2]|0?[1-9])(?::(?P<minute>[0-5][0-9]))?\s*(?P<mer>am|pm)\b"
    r"|\b(?P<h24>[01]?[0-9]|2[0-3]):(?P<m24>[0-5][0-9])\b",
    re.IGNORECASE,
)

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Longest-first so "thursday" wins over "thu" and "tues" over "tue".
_DAY_ALT = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_MONTH_ALT = "|".join(MONTHS)

# Order is significant. An explicit "30 Apr 2024" must be matched before the
# bare-weekday branch, and must swallow any weekday that precedes it: the bot's
# own offers read "Tuesday 30 Apr 2024 at 11:00 AM", and resolving that
# "Tuesday" as the *next* Tuesday would confirm an interview a week late.
_DATE_RE = re.compile(
    r"\b(?:(?:" + _DAY_ALT + r")[a-z]*[,]?\s+)?"
    r"(?P<dmy_day>\d{1,2})\s+(?P<dmy_mon>" + _MONTH_ALT + r")[a-z]*\.?\s+(?P<dmy_year>\d{4})\b"
    r"|\b(?:(?P<rel>next|this|coming)\s+)?(?P<day>" + _DAY_ALT + r")\b"
    r"|\b(?P<today>today)\b"
    r"|\b(?P<tomorrow>tomorrow)\b"
    r"|\b(?P<iso>\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _SPACE_RE.sub(" ", text).strip()


def _to_time(m: re.Match | None) -> dt.time | None:
    if m is None:
        return None
    if m.group("h24") is not None:
        return dt.time(int(m.group("h24")), int(m.group("m24")))
    hour = int(m.group("hour"))
    minute = int(m.group("minute") or 0)
    mer = m.group("mer").lower()
    if mer == "pm" and hour != 12:
        hour += 12
    elif mer == "am" and hour == 12:
        hour = 0
    return dt.time(hour, minute)


def parse_time(text: str) -> dt.time | None:
    """Extract the first clock time. Handles '10 AM', '2:30 PM', '14:00'."""
    return _to_time(_TIME_RE.search(normalise(text)))


def _upcoming(anchor: dt.date, weekday: int) -> dt.date:
    """First occurrence of `weekday` strictly after `anchor`."""
    delta = (weekday - anchor.weekday() - 1) % 7 + 1
    return anchor + dt.timedelta(days=delta)


def _resolve_date_match(m: re.Match, anchor: dt.date) -> dt.date:
    if m.group("dmy_day"):
        return dt.date(
            int(m.group("dmy_year")),
            MONTHS[m.group("dmy_mon").lower()[:3]],
            int(m.group("dmy_day")),
        )
    if m.group("today"):
        return anchor
    if m.group("tomorrow"):
        return anchor + dt.timedelta(days=1)
    if m.group("iso"):
        year, month, day = m.group("iso").split("-")
        return dt.date(int(year), int(month), int(day))
    candidate = _upcoming(anchor, WEEKDAYS[m.group("day").lower()])
    if (m.group("rel") or "").lower() == "next":
        # "next X" means the following week, not merely the next occurrence.
        if candidate.isocalendar()[:2] == anchor.isocalendar()[:2]:
            candidate += dt.timedelta(days=7)
    return candidate


def parse_offers(text: str, anchor: dt.datetime) -> list[tuple[dt.date, dt.time | None]]:
    """Every date/time offer in the text, in order of appearance.

    Recruiter turns routinely offer two slots in one message - "this Friday at
    11 AM or next Monday at 9 AM" - so each date is paired with the time that
    follows it and precedes the next date, never with a time from another offer.
    """
    text = normalise(text)
    date_matches = list(_DATE_RE.finditer(text))
    if not date_matches:
        return []
    time_matches = list(_TIME_RE.finditer(text))

    offers: list[tuple[dt.date, dt.time | None]] = []
    for i, dm in enumerate(date_matches):
        boundary = date_matches[i + 1].start() if i + 1 < len(date_matches) else len(text)
        paired = next((tm for tm in time_matches if dm.end() <= tm.start() < boundary), None)
        offers.append((_resolve_date_match(dm, anchor.date()), _to_time(paired)))
    return offers


def parse_date(text: str, anchor: dt.date) -> dt.date | None:
    """First date mentioned, resolved against the anchor."""
    m = _DATE_RE.search(normalise(text))
    return _resolve_date_match(m, anchor) if m else None


def parse_datetime(text: str, anchor: dt.datetime) -> tuple[dt.date | None, dt.time | None]:
    """First offer only. Either half may be None when the text does not say."""
    offers = parse_offers(text, anchor)
    if offers:
        date, time = offers[0]
        return date, time
    return None, parse_time(text)


def target_datetime(text: str, anchor: dt.datetime, default_hour: int = 9) -> dt.datetime | None:
    """Best-effort single datetime, used as the centre for 'nearest slots'."""
    date, time = parse_datetime(text, anchor)
    if date is None and time is None:
        return None
    if date is None:
        date = anchor.date()
    return dt.datetime.combine(date, time or dt.time(default_hour))
