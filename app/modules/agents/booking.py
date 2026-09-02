"""Turning an acceptance into an actual booking.

Without this the bot confirms interviews it never writes, and hands the same
slot to everyone. It also verifies first: candidates accept times that were
never offered, including Mondays, which the calendar has none of.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum

from ..db import dates
from ..db.store import ScheduleStore, Slot
from .types import ConversationContext


class BookingStatus(str, Enum):
    BOOKED = "booked"
    NOT_ON_CALENDAR = "not_on_calendar"
    ALREADY_TAKEN = "already_taken"
    UNCLEAR = "unclear"


@dataclass
class BookingResult:
    status: BookingStatus
    slot: Slot | None = None
    requested: dt.datetime | None = None
    alternatives: list[Slot] | None = None

    @property
    def confirmed(self) -> bool:
        return self.status is BookingStatus.BOOKED

    def summary(self) -> str:
        if self.status is BookingStatus.BOOKED and self.slot:
            return f"booked {self.slot.label()}"
        if self.status is BookingStatus.NOT_ON_CALENDAR:
            return f"{self.requested:%A %d %b %H:%M} is not on the calendar" if self.requested else "requested time is not on the calendar"
        if self.status is BookingStatus.ALREADY_TAKEN:
            return f"{self.requested:%A %d %b %H:%M} is already taken" if self.requested else "requested time is already taken"
        return "no specific time was named"


class SlotBooker:
    """Resolves the time a candidate accepted, verifies it, and books it."""

    def __init__(self, store: ScheduleStore):
        self.store = store

    def book_from(self, context: ConversationContext) -> BookingResult:
        requested = self._requested_datetime(context)
        if requested is None:
            return BookingResult(BookingStatus.UNCLEAR)

        slot = self.store.check_slot(
            requested.date(), requested.time(), context.position
        )
        if slot is None:
            return BookingResult(
                BookingStatus.NOT_ON_CALENDAR,
                requested=requested,
                alternatives=self._alternatives(requested, context),
            )
        if not slot.available:
            return BookingResult(
                BookingStatus.ALREADY_TAKEN,
                requested=requested,
                alternatives=self._alternatives(requested, context),
            )

        self.store.book(slot.schedule_id)
        return BookingResult(BookingStatus.BOOKED, slot=slot, requested=requested)

    def _alternatives(self, requested: dt.datetime, context: ConversationContext) -> list[Slot]:
        return self.store.nearest_available(
            requested, position=context.position, n=3, not_before=context.anchor
        )

    @staticmethod
    def _requested_datetime(context: ConversationContext) -> dt.datetime | None:
        """The time the candidate accepted.

        Their own words win, even if they differ from what was offered. A bare
        "that works" falls back to the last slot on the table.
        """
        last = context.last_candidate_message
        offers = dates.parse_offers(last, context.anchor)
        for date, time in offers:
            if time is not None:
                return dt.datetime.combine(date, time)

        recruiter_offers = SlotBooker._last_recruiter_offers(context)
        if not recruiter_offers:
            return None

        # "11 AM works" gives a time but no day - match it to an offer.
        bare_time = dates.parse_time(last)
        if bare_time is not None:
            matching = [(d, t) for d, t in recruiter_offers if t == bare_time]
            if len(matching) == 1:
                date, time = matching[0]
                return dt.datetime.combine(date, time)
            return None

        # "that works" is only unambiguous with one slot on the table.
        if len(recruiter_offers) == 1:
            date, time = recruiter_offers[0]
            return dt.datetime.combine(date, time)
        return None

    @staticmethod
    def _last_recruiter_offers(context: ConversationContext):
        for turn in reversed(context.history):
            if turn.speaker != "recruiter":
                continue
            offers = [
                (d, t) for d, t in dates.parse_offers(turn.text, context.anchor) if t
            ]
            if offers:
                return offers
        return []
