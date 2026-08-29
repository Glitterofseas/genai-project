"""Tests for turning an acceptance into a real, verified booking.

Two failure modes matter here:
  * confirming an interview without ever writing it to the calendar, and
  * confirming a time the calendar does not contain - which the transcripts
    do repeatedly, since candidates accept Mondays and Mondays do not exist.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.modules.agents.booking import BookingStatus, SlotBooker
from app.modules.agents.types import ConversationContext
from app.modules.config.settings import get_settings
from app.modules.db.store import SqliteScheduleStore, get_store
from app.modules.evaluation.dataset import Turn

ANCHOR = dt.datetime(2024, 4, 30, 11, 19)   # a Tuesday
NB = " "                                     # narrow no-break space, as in the data


def ctx(*messages: tuple[str, str], anchor: dt.datetime = ANCHOR) -> ConversationContext:
    turns = tuple(
        Turn(i + 1, speaker, anchor, text, None)
        for i, (speaker, text) in enumerate(messages)
    )
    return ConversationContext(history=turns, anchor=anchor)


@pytest.fixture
def writable_store(tmp_path):
    """A throwaway copy, so booking tests never touch the committed fixture."""
    import shutil

    source = get_settings().sqlite_path
    target = tmp_path / "schedule.sqlite"
    shutil.copy(source, target)
    return SqliteScheduleStore(target)


class TestRequestedTimeResolution:
    def test_candidate_stated_time_wins(self, writable_store):
        booker = SlotBooker(writable_store)
        context = ctx(
            ("recruiter", f"Wednesday at 10{NB}AM or Thursday at 2{NB}PM?"),
            ("candidate", f"Thursday at 2{NB}PM works."),
        )
        assert booker._requested_datetime(context) == dt.datetime(2024, 5, 2, 14, 0)

    def test_bare_acceptance_falls_back_to_a_single_offer(self, writable_store):
        booker = SlotBooker(writable_store)
        context = ctx(
            ("recruiter", f"Would Thursday at 4{NB}PM work?"),
            ("candidate", "That works, thanks."),
        )
        assert booker._requested_datetime(context) == dt.datetime(2024, 5, 2, 16, 0)

    def test_bare_acceptance_of_two_offers_is_ambiguous(self, writable_store):
        booker = SlotBooker(writable_store)
        context = ctx(
            ("recruiter", f"Wednesday at 10{NB}AM or Thursday at 2{NB}PM?"),
            ("candidate", "Sounds great."),
        )
        assert booker._requested_datetime(context) is None


class TestBooking:
    def test_available_slot_is_actually_written(self, writable_store):
        slot = writable_store.nearest_available(ANCHOR, n=1, not_before=ANCHOR)[0]
        hour = slot.time.hour
        display = hour % 12 or 12
        suffix = "AM" if hour < 12 else "PM"
        context = ctx(
            ("recruiter", f"Would {slot.date.isoformat()} at {display} {suffix} work?"),
            ("candidate", "That works."),
            anchor=ANCHOR,
        )
        result = SlotBooker(writable_store).book_from(context)
        assert result.status is BookingStatus.BOOKED
        assert result.slot.schedule_id == slot.schedule_id

        after = writable_store.check_slot(slot.date, slot.time, slot.position)
        assert not after.available, "the slot must be marked taken"

    def test_monday_acceptance_is_refused_with_alternatives(self, writable_store):
        # Straight from conversation 1: offered Wed/Thu, candidate says Monday.
        context = ctx(
            ("recruiter", f"How about Thursday at 4{NB}PM instead?"),
            ("candidate", f"Monday at 3{NB}PM is good."),
        )
        result = SlotBooker(writable_store).book_from(context)
        assert result.status is BookingStatus.NOT_ON_CALENDAR
        assert result.alternatives, "must offer a way forward, not just refuse"
        assert all(s.start >= ANCHOR for s in result.alternatives)

    def test_taken_slot_is_refused(self, writable_store):
        slot = writable_store.nearest_available(ANCHOR, n=1, not_before=ANCHOR)[0]
        writable_store.book(slot.schedule_id)

        hour = slot.time.hour
        display = hour % 12 or 12
        suffix = "AM" if hour < 12 else "PM"
        context = ctx(
            ("recruiter", f"Would {slot.date.isoformat()} at {display} {suffix} work?"),
            ("candidate", "That works."),
        )
        result = SlotBooker(writable_store).book_from(context)
        assert result.status is BookingStatus.ALREADY_TAKEN

    def test_no_time_named_is_unclear(self, writable_store):
        context = ctx(("recruiter", "Tell me about your experience."),
                      ("candidate", "Sure, five years of Python."))
        assert SlotBooker(writable_store).book_from(context).status is BookingStatus.UNCLEAR

    def test_double_booking_is_impossible(self, writable_store):
        slot = writable_store.nearest_available(ANCHOR, n=1, not_before=ANCHOR)[0]
        hour = slot.time.hour
        display = hour % 12 or 12
        suffix = "AM" if hour < 12 else "PM"
        context = ctx(
            ("recruiter", f"Would {slot.date.isoformat()} at {display} {suffix} work?"),
            ("candidate", "That works."),
        )
        booker = SlotBooker(writable_store)
        assert booker.book_from(context).status is BookingStatus.BOOKED
        assert booker.book_from(context).status is BookingStatus.ALREADY_TAKEN


class TestEvaluationStillReadOnly:
    def test_booking_never_writes_during_evaluation(self):
        """The harness must stay reproducible now that booking exists."""
        from app.modules.agents.rule_based import build_rule_based_agent
        from app.modules.evaluation.dataset import load_conversations
        from app.modules.evaluation.harness import evaluate

        store = get_store(read_only=True)
        before = store.count()
        available_before = len(
            store._query(
                "SELECT ScheduleID, date, time, position, available FROM Schedule "
                "WHERE available = 1 AND date LIKE '2024-05%'"
            )
        )
        evaluate(build_rule_based_agent(store), load_conversations())
        available_after = len(
            store._query(
                "SELECT ScheduleID, date, time, position, available FROM Schedule "
                "WHERE available = 1 AND date LIKE '2024-05%'"
            )
        )
        assert store.count() == before
        assert available_before == available_after, "evaluation must not book slots"


class TestWorkingCopyIsolation:
    """A demo booking must never dirty the committed fixture."""

    def test_writable_store_does_not_use_the_committed_seed(self):
        from app.modules.db.store import get_store, working_copy_path

        seed = get_settings().sqlite_path
        writable = get_store(read_only=False)
        readonly = get_store(read_only=True)
        assert writable.path == working_copy_path(seed)
        assert readonly.path == seed
        assert writable.path != readonly.path

    def test_booking_leaves_the_seed_byte_identical(self):
        import hashlib

        from app.modules.db.store import get_store

        seed = get_settings().sqlite_path
        before = hashlib.md5(seed.read_bytes()).hexdigest()

        store = get_store(read_only=False)
        slot = store.nearest_available(ANCHOR, n=1, not_before=ANCHOR)[0]
        store.book(slot.schedule_id)

        assert hashlib.md5(seed.read_bytes()).hexdigest() == before


class TestBareTimeAcceptance:
    """'11 AM works' - a time with no day, which candidates say constantly."""

    def test_bare_time_matches_the_offer_it_refers_to(self, writable_store):
        booker = SlotBooker(writable_store)
        context = ctx(
            ("recruiter", f"Wednesday at 10{NB}AM or Thursday at 2{NB}PM?"),
            ("candidate", f"2{NB}PM works."),
        )
        assert booker._requested_datetime(context) == dt.datetime(2024, 5, 2, 14, 0)

    def test_bare_time_matching_no_offer_is_unclear(self, writable_store):
        booker = SlotBooker(writable_store)
        context = ctx(
            ("recruiter", f"Wednesday at 10{NB}AM or Thursday at 2{NB}PM?"),
            ("candidate", f"4{NB}PM works."),
        )
        assert booker._requested_datetime(context) is None

    def test_ambiguous_bare_time_is_unclear(self, writable_store):
        booker = SlotBooker(writable_store)
        context = ctx(
            ("recruiter", f"Wednesday at 10{NB}AM or Thursday at 10{NB}AM?"),
            ("candidate", f"10{NB}AM works."),
        )
        assert booker._requested_datetime(context) is None


class TestNoRepeatedOffers:
    """A candidate who declines a time must not be handed it straight back."""

    def test_declined_slots_are_excluded_from_the_next_offer(self, writable_store):
        from app.modules.agents.rule_based import (
            RuleBasedSchedulingAdvisor,
            previously_offered,
        )

        first = ctx(
            ("recruiter", "Tell me about your experience."),
            ("candidate", "Five years of Python."),
        )
        advisor = RuleBasedSchedulingAdvisor(writable_store)
        offered = advisor.evaluate(first).slots
        assert offered

        rendered = " or ".join(s.label() for s in offered[:2])
        second = ctx(
            ("recruiter", "Tell me about your experience."),
            ("candidate", "Five years of Python."),
            ("recruiter", f"Could we schedule for {rendered}?"),
            ("candidate", "I can't at that time."),
        )
        assert previously_offered(second) == {s.start for s in offered[:2]}

        again = advisor.evaluate(second).slots
        assert again, "must still offer something"
        assert not ({s.start for s in again} & {s.start for s in offered[:2]}), (
            "declined slots were offered again"
        )

    def test_store_honours_the_exclude_set(self, writable_store):
        anchor = ANCHOR
        first = writable_store.nearest_available(anchor, n=3, not_before=anchor)
        excluded = {first[0].start}
        second = writable_store.nearest_available(
            anchor, n=3, not_before=anchor, exclude=excluded
        )
        assert first[0].start not in {s.start for s in second}
