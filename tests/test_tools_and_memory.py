"""Tests for the LangChain tools over the SQL calendar, and conversation memory.

These cover the two spec requirements that were previously unimplemented:
"Function Calling to interact with the SQL database" and the Memories part of
"LangChain - Agents, Memories, Tool".

The tools are exercised directly rather than through the model, so the suite
stays offline and free; a live tool-calling round trip is verified separately.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.modules.agents.memory import ConversationMemory
from app.modules.agents.tools import ScheduleToolkit, build_schedule_toolkit
from app.modules.agents.types import ConversationContext
from app.modules.config.settings import get_settings
from app.modules.db.store import SqliteScheduleStore, get_store
from app.modules.evaluation.dataset import Turn

ANCHOR = dt.datetime(2024, 4, 30, 11, 19)   # a Tuesday
NB = " "


def ctx(*messages: tuple[str, str], anchor: dt.datetime = ANCHOR) -> ConversationContext:
    turns = tuple(
        Turn(i + 1, speaker, anchor, text, None)
        for i, (speaker, text) in enumerate(messages)
    )
    return ConversationContext(history=turns, anchor=anchor)


@pytest.fixture
def toolkit():
    return build_schedule_toolkit(get_store(read_only=True), ctx(("candidate", "hi")))


class TestToolDefinitions:
    def test_both_sql_tools_are_exposed(self, toolkit):
        names = {t.name for t in toolkit.as_tools()}
        assert names == {"check_interview_slot", "find_nearest_interview_slots"}

    def test_tools_declare_argument_schemas(self, toolkit):
        by_name = {t.name: t for t in toolkit.as_tools()}
        assert set(by_name["check_interview_slot"].args_schema.model_fields) == {"date", "time"}
        assert set(by_name["find_nearest_interview_slots"].args_schema.model_fields) == {
            "around_date", "around_time", "count",
        }

    def test_tools_have_descriptions_for_the_model(self, toolkit):
        for tool in toolkit.as_tools():
            assert len(tool.description) > 40, f"{tool.name} needs a usable description"

    def test_no_write_tool_is_exposed(self, toolkit):
        """A model that can book would book interviews nobody agreed to."""
        names = " ".join(t.name for t in toolkit.as_tools())
        assert "book" not in names


class TestCheckSlotTool:
    def test_monday_reports_not_on_the_calendar(self, toolkit):
        result = toolkit._check_slot("2024-05-06", "15:00")   # a Monday
        assert "NOT on the calendar" in result
        assert toolkit.returned_slots == []

    def test_available_slot_is_reported_and_recorded(self, toolkit):
        slot = toolkit.store.nearest_available(ANCHOR, n=1, not_before=ANCHOR)[0]
        result = toolkit._check_slot(slot.date.isoformat(), slot.time.strftime("%H:%M"))
        assert "AVAILABLE" in result
        assert toolkit.returned_slots[0].schedule_id == slot.schedule_id

    def test_malformed_arguments_do_not_raise(self, toolkit):
        assert "not a valid" in toolkit._check_slot("next tuesday", "3pm")

    def test_calls_are_logged(self, toolkit):
        toolkit._check_slot("2024-05-06", "15:00")
        assert toolkit.call_log and "check_interview_slot" in toolkit.call_log[0]


class TestNearestSlotsTool:
    def test_returns_three_real_slots(self, toolkit):
        result = toolkit._nearest_slots(around_date="2024-05-07", count=3)
        assert "Nearest available slots" in result
        assert len(toolkit.unique_slots(3)) == 3

    def test_never_returns_a_slot_before_the_anchor(self, toolkit):
        toolkit._nearest_slots(around_date="2024-01-05")
        assert all(s.start >= ANCHOR for s in toolkit.returned_slots)

    def test_count_is_clamped(self, toolkit):
        toolkit._nearest_slots(around_date="2024-05-07", count=99)
        assert len(toolkit.returned_slots) <= 5

    def test_bad_date_is_reported_not_raised(self, toolkit):
        assert "not a valid date" in toolkit._nearest_slots(around_date="whenever")

    def test_unique_slots_deduplicates_and_orders(self, toolkit):
        toolkit._nearest_slots(around_date="2024-05-07", count=3)
        toolkit._nearest_slots(around_date="2024-05-07", count=3)
        slots = toolkit.unique_slots(3)
        ids = [s.schedule_id for s in slots]
        assert len(ids) == len(set(ids))
        assert slots == sorted(slots, key=lambda s: s.start)


class TestFallback:
    def test_fallback_returns_slots_when_tools_are_unused(self, toolkit):
        """The advisor must never degrade if tool calling fails."""
        assert len(toolkit.fallback_slots(3)) == 3


class TestConversationMemory:
    def test_speakers_map_to_langchain_message_types(self):
        from langchain_core.messages import AIMessage, HumanMessage

        memory = ConversationMemory(ANCHOR)
        memory.add_recruiter("Tell me about your experience.")
        memory.add_candidate("Five years of Python.")
        assert isinstance(memory.messages[0], AIMessage)
        assert isinstance(memory.messages[1], HumanMessage)

    def test_turns_round_trip_with_metadata(self):
        memory = ConversationMemory(ANCHOR)
        memory.add_recruiter("a")
        memory.add_candidate("b")
        turns = memory.turns
        assert [t.turn_id for t in turns] == [1, 2]
        assert [t.speaker for t in turns] == ["recruiter", "candidate"]

    def test_clock_advances_between_turns(self):
        memory = ConversationMemory(ANCHOR)
        memory.add_recruiter("a")
        memory.add_candidate("b")
        assert memory.turns[1].timestamp > memory.turns[0].timestamp

    def test_context_is_what_the_advisors_expect(self):
        memory = ConversationMemory(ANCHOR, position="Python Dev", candidate_phone="+1-555")
        memory.add_recruiter("Tell me about your experience.")
        memory.add_candidate("Five years of Django.")
        context = memory.context()
        assert isinstance(context, ConversationContext)
        assert context.last_candidate_message == "Five years of Django."
        assert context.position == "Python Dev"
        assert len(context.history) == 2

    def test_clear_resets(self):
        memory = ConversationMemory(ANCHOR)
        memory.add_recruiter("a")
        memory.clear()
        assert len(memory) == 0 and memory.turns == ()


class TestMemoryDoesNotLeakIntoEvaluation:
    def test_harness_builds_context_from_transcripts_not_memory(self):
        """Evaluation is teacher-forced; memory must play no part in it."""
        import inspect

        from app.modules.evaluation import harness

        source = inspect.getsource(harness)
        assert "ConversationMemory" not in source
        assert "history_before" in source
