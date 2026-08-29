"""Unit tests for the relative-date resolver.

This is the highest-risk pure function in the project: every scheduling
decision depends on resolving a phrase like "next Tuesday" against the
conversation's own timestamp rather than the wall clock.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.modules.db import dates
from app.modules.db.rules import is_valid_day

NB = " "  # narrow no-break space, as used throughout sms_conversations.json

# 2024-04-30 is a Tuesday; it is the start time of conversation 15.
TUE = dt.datetime(2024, 4, 30, 11, 19)
SUN = dt.datetime(2024, 4, 28, 11, 13)   # conversation 14 starts on a Sunday
WED = dt.datetime(2024, 4, 3, 15, 12)    # conversation 1 starts on a Wednesday


class TestParseTime:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("10 AM", dt.time(10, 0)),
            (f"10{NB}AM", dt.time(10, 0)),          # the real transcript spelling
            ("2 PM", dt.time(14, 0)),
            ("2:30 PM", dt.time(14, 30)),
            ("12 AM", dt.time(0, 0)),
            ("12 PM", dt.time(12, 0)),
            ("14:00", dt.time(14, 0)),
            ("no time here", None),
        ],
    )
    def test_times(self, text, expected):
        assert dates.parse_time(text) == expected


class TestParseDate:
    def test_bare_weekday_is_next_occurrence(self):
        assert dates.parse_date("Thursday at 4 PM", TUE.date()) == dt.date(2024, 5, 2)

    def test_this_weekday_stays_in_week(self):
        assert dates.parse_date("this Friday", TUE.date()) == dt.date(2024, 5, 3)

    def test_next_weekday_pushes_a_week_out(self):
        # Plain "next Thursday" would be 05-02, inside the anchor's own week,
        # so it must resolve to the following Thursday instead.
        assert dates.parse_date("next Thursday", TUE.date()) == dt.date(2024, 5, 9)

    def test_next_same_weekday_as_anchor(self):
        assert dates.parse_date("next Tuesday", TUE.date()) == dt.date(2024, 5, 7)

    def test_today_and_tomorrow(self):
        assert dates.parse_date("today", TUE.date()) == dt.date(2024, 4, 30)
        assert dates.parse_date("tomorrow", TUE.date()) == dt.date(2024, 5, 1)

    def test_month_boundary(self):
        assert dates.parse_date("Wednesday", TUE.date()) == dt.date(2024, 5, 1)

    def test_iso_date_passes_through(self):
        assert dates.parse_date("on 2024-06-11", TUE.date()) == dt.date(2024, 6, 11)

    def test_no_date_returns_none(self):
        assert dates.parse_date("I'm busy then", TUE.date()) is None

    @pytest.mark.parametrize("anchor", [TUE, SUN, WED])
    def test_resolution_is_always_in_the_future(self, anchor):
        for phrase in ("Monday", "Friday", "next Sunday", "this Wednesday"):
            resolved = dates.parse_date(phrase, anchor.date())
            assert resolved > anchor.date(), f"{phrase} from {anchor.date()}"


class TestParseOffers:
    def test_two_slot_offer_pairs_each_time_with_its_own_day(self):
        text = f"Could we schedule a chat this Friday at 11{NB}AM or next Monday at 9{NB}AM?"
        assert dates.parse_offers(text, TUE) == [
            (dt.date(2024, 5, 3), dt.time(11, 0)),
            (dt.date(2024, 5, 6), dt.time(9, 0)),
        ]

    def test_second_offer_time_never_leaks_to_the_first(self):
        text = f"Wednesday at 10{NB}AM or Thursday at 2{NB}PM"
        offers = dates.parse_offers(text, TUE)
        assert offers[0] == (dt.date(2024, 5, 1), dt.time(10, 0))
        assert offers[1] == (dt.date(2024, 5, 2), dt.time(14, 0))

    def test_day_without_a_time_yields_none_time(self):
        assert dates.parse_offers("How about next Thursday?", TUE) == [
            (dt.date(2024, 5, 9), None)
        ]

    def test_no_offers_in_plain_refusal(self):
        assert dates.parse_offers("I can't at that time - I'm busy.", TUE) == []


class TestScheduleRules:
    def test_mondays_and_saturdays_are_not_bookable(self):
        # db_Tech.sql excludes them, yet the transcripts propose Mondays -
        # the advisor must be able to detect that.
        assert not is_valid_day(dt.date(2024, 5, 6))   # Monday
        assert not is_valid_day(dt.date(2024, 5, 4))   # Saturday

    def test_tue_to_fri_and_sunday_are_bookable(self):
        for day in (dt.date(2024, 4, 30), dt.date(2024, 5, 1), dt.date(2024, 5, 2),
                    dt.date(2024, 5, 3), dt.date(2024, 5, 5)):
            assert is_valid_day(day)

    def test_monday_offer_in_real_transcript_is_flagged(self):
        offers = dates.parse_offers(f"Monday at 3{NB}PM is good.", TUE)
        assert offers and not is_valid_day(offers[0][0])


class TestExplicitDates:
    """The bot renders its own offers as 'Tuesday 30 Apr 2024 at 11:00 AM'.

    Re-parsing that must yield 30 April, not the next Tuesday - otherwise the
    bot confirms an interview a week after the one it offered.
    """

    def test_explicit_date_beats_the_weekday_in_front_of_it(self):
        assert dates.parse_date("Tuesday 30 Apr 2024", TUE.date()) == dt.date(2024, 4, 30)

    def test_explicit_date_without_a_weekday(self):
        assert dates.parse_date("on 7 May 2024", TUE.date()) == dt.date(2024, 5, 7)

    def test_round_trip_through_the_slot_label(self):
        from app.modules.db.store import Slot

        slot = Slot(1, dt.date(2024, 4, 30), dt.time(11, 0), "Python Dev", True)
        offers = dates.parse_offers(f"Could we schedule for {slot.label()}?", TUE)
        assert offers == [(dt.date(2024, 4, 30), dt.time(11, 0))]

    def test_two_rendered_offers_round_trip(self):
        from app.modules.db.store import Slot

        a = Slot(1, dt.date(2024, 4, 30), dt.time(11, 0), "Python Dev", True)
        b = Slot(2, dt.date(2024, 4, 30), dt.time(12, 0), "Python Dev", True)
        offers = dates.parse_offers(f"{a.label()} or {b.label()}?", TUE)
        assert offers == [
            (dt.date(2024, 4, 30), dt.time(11, 0)),
            (dt.date(2024, 4, 30), dt.time(12, 0)),
        ]

    def test_weekday_alone_still_resolves_forward(self):
        assert dates.parse_date("Thursday at 4 PM", TUE.date()) == dt.date(2024, 5, 2)
