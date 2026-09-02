"""Tests for the orchestration loop, the advisors, and the schedule store.

All of these run offline: the loop is exercised with stub advisors, and the
schedule comes from the committed SQLite fixture. No API key required.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.modules.agents.main_agent import MainAgent
from app.modules.agents.rule_based import build_rule_based_agent
from app.modules.agents.types import (
    Action,
    AdvisorName,
    AdvisorVerdict,
    ConversationContext,
    action_from_verdicts,
)
from app.modules.db.store import get_store
from app.modules.evaluation.dataset import Turn, load_conversations, split
from app.modules.evaluation.harness import baseline_reports, context_for, evaluate

ANCHOR = dt.datetime(2024, 4, 30, 11, 19)


def ctx(*messages: tuple[str, str], anchor: dt.datetime = ANCHOR) -> ConversationContext:
    turns = tuple(
        Turn(i + 1, speaker, anchor, text, None)
        for i, (speaker, text) in enumerate(messages)
    )
    return ConversationContext(history=turns, anchor=anchor)


class StubAdvisor:
    def __init__(self, name: AdvisorName, decision: bool):
        self.name = name
        self.decision = decision
        self.calls = 0

    def evaluate(self, context):
        self.calls += 1
        return AdvisorVerdict(self.name, self.decision, "stub")


class SequenceRouter:
    """Routes through a fixed list, so the loop itself is what gets tested."""

    def __init__(self, order):
        self.order = list(order)

    def choose(self, context, consulted):
        return self.order[len(consulted)] if len(consulted) < len(self.order) else None

    def consult_again(self, context, consulted):
        return not consulted[-1].decision


class NullComposer:
    def compose(self, context, consulted, action, booking=None):
        return f"<{action.value}>"


class TestActionDerivation:
    def test_exit_outranks_sched(self):
        verdicts = [
            AdvisorVerdict(AdvisorName.SCHED, True),
            AdvisorVerdict(AdvisorName.EXIT, True),
        ]
        assert action_from_verdicts(verdicts) is Action.END

    def test_sched_when_exit_declines(self):
        verdicts = [
            AdvisorVerdict(AdvisorName.EXIT, False),
            AdvisorVerdict(AdvisorName.SCHED, True),
        ]
        assert action_from_verdicts(verdicts) is Action.SCHEDULE

    def test_continue_when_all_decline(self):
        verdicts = [
            AdvisorVerdict(AdvisorName.EXIT, False),
            AdvisorVerdict(AdvisorName.SCHED, False),
            AdvisorVerdict(AdvisorName.INFO, False),
        ]
        assert action_from_verdicts(verdicts) is Action.CONTINUE

    def test_no_advisors_means_continue(self):
        assert action_from_verdicts([]) is Action.CONTINUE


class TestOrchestrationLoop:
    def test_positive_verdict_stops_the_loop(self):
        exit_advisor = StubAdvisor(AdvisorName.EXIT, True)
        sched = StubAdvisor(AdvisorName.SCHED, True)
        agent = MainAgent(
            router=SequenceRouter([AdvisorName.EXIT, AdvisorName.SCHED]),
            advisors={AdvisorName.EXIT: exit_advisor, AdvisorName.SCHED: sched},
            composer=NullComposer(),
        )
        decision = agent.decide(ctx(("recruiter", "hi"), ("candidate", "not interested")))
        assert decision.action is Action.END
        assert exit_advisor.calls == 1
        assert sched.calls == 0, "a positive verdict must end the turn"

    def test_negative_verdict_consults_the_next_advisor(self):
        exit_advisor = StubAdvisor(AdvisorName.EXIT, False)
        sched = StubAdvisor(AdvisorName.SCHED, True)
        agent = MainAgent(
            router=SequenceRouter([AdvisorName.EXIT, AdvisorName.SCHED]),
            advisors={AdvisorName.EXIT: exit_advisor, AdvisorName.SCHED: sched},
            composer=NullComposer(),
        )
        decision = agent.decide(ctx(("recruiter", "hi"), ("candidate", "3 years Django")))
        assert decision.action is Action.SCHEDULE
        assert exit_advisor.calls == 1 and sched.calls == 1

    def test_loop_is_capped(self):
        advisors = {
            name: StubAdvisor(name, False)
            for name in (AdvisorName.EXIT, AdvisorName.SCHED, AdvisorName.INFO)
        }
        agent = MainAgent(
            router=SequenceRouter(list(advisors) * 5),
            advisors=advisors,
            composer=NullComposer(),
            max_advisor_calls=2,
        )
        decision = agent.decide(ctx(("candidate", "hello")))
        assert len(decision.consulted) == 2


@pytest.fixture(scope="module")
def store():
    return get_store(read_only=True)


class TestScheduleStore:
    def test_fixture_is_populated(self, store):
        assert store.count() > 20_000

    def test_monday_slot_does_not_exist(self, store):
        assert store.check_slot(dt.date(2024, 5, 6), dt.time(15, 0)) is None

    def test_nearest_slots_are_after_the_anchor(self, store):
        anchor = dt.datetime(2024, 4, 30, 11, 19)
        slots = store.nearest_available(anchor, n=3, not_before=anchor)
        assert len(slots) == 3
        assert all(s.start >= anchor for s in slots)
        assert all(s.available for s in slots)

    def test_saturday_anchor_rolls_forward(self, store):
        saturday = dt.datetime(2026, 8, 29, 10, 0)
        assert saturday.weekday() == 5
        slots = store.nearest_available(saturday, n=3, not_before=saturday)
        assert slots and all(s.start > saturday for s in slots)

    def test_read_only_store_records_but_does_not_write(self, store):
        before = store.count()
        slot = store.nearest_available(dt.datetime(2024, 4, 30), n=1)[0]
        store.book(slot.schedule_id)
        assert slot.schedule_id in store.attempted_bookings
        again = store.check_slot(slot.date, slot.time, slot.position)
        assert again.available, "read_only must not mutate the calendar"
        assert store.count() == before


class TestSlotFormatting:
    """label() is what the candidate actually reads, so its edges matter."""

    @pytest.mark.parametrize(
        "hour,expected",
        [
            (0, "12:00 AM"), (9, "9:00 AM"), (11, "11:00 AM"),
            (12, "12:00 PM"), (13, "1:00 PM"), (17, "5:00 PM"), (23, "11:00 PM"),
        ],
    )
    def test_twelve_hour_clock(self, hour, expected):
        from app.modules.db.store import Slot

        slot = Slot(1, dt.date(2024, 5, 7), dt.time(hour, 0), "Python Dev", True)
        assert slot.label().endswith(expected)

    def test_label_is_portable(self):
        """Guards against strftime directives that raise on Windows."""
        from app.modules.db.store import Slot

        for hour in range(24):
            Slot(1, dt.date(2024, 5, 7), dt.time(hour, 30), "Python Dev", True).label()


class TestEvaluationHarness:
    def test_teacher_forced_context_excludes_the_turn_itself(self):
        conversation = load_conversations()[0]
        turn = conversation.labelled_turns[-1]
        context = context_for(conversation, turn)
        assert all(t.turn_id < turn.turn_id for t in context.history)
        assert context.anchor == turn.timestamp

    def test_split_is_disjoint_and_complete(self):
        train, test = split()
        train_ids = {c.conversation_id for c in train}
        test_ids = {c.conversation_id for c in test}
        assert not train_ids & test_ids
        assert len(train_ids) == 10 and len(test_ids) == 5

    def test_positional_baseline_is_reported(self):
        reports = baseline_reports(load_conversations())
        names = [r.name for r in reports]
        assert any("last turn = end" in n for n in names)

    def test_rule_agent_beats_every_trivial_baseline(self):
        conversations = load_conversations()
        best_baseline = max(r.accuracy for r in baseline_reports(conversations))
        report = evaluate(build_rule_based_agent(get_store(read_only=True)), conversations)
        assert report.accuracy > best_baseline

    def test_evaluation_does_not_book_slots(self):
        store = get_store(read_only=True)
        before = store.count()
        evaluate(build_rule_based_agent(store), load_conversations())
        assert store.count() == before


class TestRejectionBeatsAcceptance:
    """'Those slots don't work for me' contains 'work for me'.

    Both phrasings appear verbatim in the transcripts (conversations 8 and 12),
    where the correct action is `schedule`, not `end`.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "Those slots don't work for me.",
            "I can't at that time - I'm busy.",
            "I'm unavailable then; do you have any other times?",
            "That doesn't work for me.",
        ],
    )
    def test_rejections_do_not_end_the_conversation(self, message):
        from app.modules.agents.rule_based import RuleBasedExitAdvisor

        context = ctx(
            ("recruiter", "Would 2024-05-07 at 11 AM work?"),
            ("candidate", message),
        )
        assert RuleBasedExitAdvisor().evaluate(context).decision is False

    @pytest.mark.parametrize(
        "message", ["Tuesday at 10 AM works.", "That works, thanks.", "Sounds great."]
    )
    def test_genuine_acceptances_still_end(self, message):
        from app.modules.agents.rule_based import RuleBasedExitAdvisor

        context = ctx(
            ("recruiter", "Would 2024-05-07 at 11 AM work?"),
            ("candidate", message),
        )
        assert RuleBasedExitAdvisor().evaluate(context).decision is True
