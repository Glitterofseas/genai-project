"""LangChain tools over the SQL schedule.

Bound to the model, which decides when to call them and with what arguments.

Read-only on purpose: a model that could write would book interviews nobody
agreed to, so writes stay in SlotBooker. Every slot a tool returns is recorded
on the toolkit, so offers come from real rows rather than the model's prose.
"""
from __future__ import annotations

import datetime as dt

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..db.rules import DEFAULT_POSITION
from ..db.store import ScheduleStore, Slot
from .types import ConversationContext


class CheckSlotArgs(BaseModel):
    date: str = Field(description="Calendar date to check, as YYYY-MM-DD")
    time: str = Field(description="Start time on that date, 24-hour, as HH:MM")


class NearestSlotsArgs(BaseModel):
    around_date: str = Field(
        default="",
        description="Preferred date as YYYY-MM-DD. Leave empty to search from today.",
    )
    around_time: str = Field(
        default="",
        description="Preferred time, 24-hour, as HH:MM. Leave empty for any time.",
    )
    count: int = Field(default=3, description="How many slots to return (default 3)")


class ScheduleToolkit:
    """The schedule tools for one turn.

    Holds the context, so the model is never told - or trusted with - the anchor
    date or the position.
    """

    def __init__(self, store: ScheduleStore, context: ConversationContext):
        self.store = store
        self.context = context
        self.returned_slots: list[Slot] = []
        self.call_log: list[str] = []

    # ---------- tool implementations ----------

    def _check_slot(self, date: str, time: str) -> str:
        self.call_log.append(f"check_interview_slot(date={date!r}, time={time!r})")
        try:
            parsed_date = dt.date.fromisoformat(date.strip())
            hour, _, minute = time.strip().partition(":")
            parsed_time = dt.time(int(hour), int(minute or 0))
        except (ValueError, TypeError):
            return f"'{date} {time}' is not a valid date and time. Use YYYY-MM-DD and HH:MM."

        slot = self.store.check_slot(parsed_date, parsed_time, self.context.position)
        if slot is None:
            return (
                f"{parsed_date:%A %d %B %Y} at {time} is NOT on the calendar at all. "
                "Interviews run Tuesday to Friday and Sunday, 09:00-17:00 only."
            )
        if not slot.available:
            return f"{slot.label()} exists but is already booked."
        self.returned_slots.append(slot)
        return f"{slot.label()} is AVAILABLE and can be offered."

    def _nearest_slots(self, around_date: str = "", around_time: str = "", count: int = 3) -> str:
        self.call_log.append(
            f"find_nearest_interview_slots(around_date={around_date!r}, "
            f"around_time={around_time!r}, count={count})"
        )
        target = self.context.anchor
        if around_date.strip():
            try:
                date = dt.date.fromisoformat(around_date.strip())
                time = dt.time(9)
                if around_time.strip():
                    hour, _, minute = around_time.strip().partition(":")
                    time = dt.time(int(hour), int(minute or 0))
                target = dt.datetime.combine(date, time)
            except (ValueError, TypeError):
                return f"'{around_date}' is not a valid date. Use YYYY-MM-DD."

        from .rule_based import previously_offered

        slots = self.store.nearest_available(
            target,
            position=self.context.position,
            n=max(1, min(int(count or 3), 5)),
            not_before=self.context.anchor,
            exclude=previously_offered(self.context),
        )
        if not slots:
            return "No available slots were found near that date."
        self.returned_slots.extend(slots)
        listed = "; ".join(s.label() for s in slots)
        return f"Nearest available slots: {listed}"

    # ---------- binding ----------

    def as_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._check_slot,
                name="check_interview_slot",
                description=(
                    "Check whether one exact interview slot is on the calendar and "
                    "still free. Use this when the candidate names a specific day "
                    "and time. Returns whether it exists, and whether it is available."
                ),
                args_schema=CheckSlotArgs,
            ),
            StructuredTool.from_function(
                func=self._nearest_slots,
                name="find_nearest_interview_slots",
                description=(
                    "Find the interview slots closest to a preferred date and time, "
                    "or to today when none is given. Use this to obtain times to "
                    "offer the candidate. Returns up to five real, bookable slots."
                ),
                args_schema=NearestSlotsArgs,
            ),
        ]

    def unique_slots(self, limit: int = 3) -> list[Slot]:
        """Distinct slots the tools actually returned, earliest first."""
        seen: dict[int, Slot] = {}
        for slot in self.returned_slots:
            seen.setdefault(slot.schedule_id, slot)
        return sorted(seen.values(), key=lambda s: s.start)[:limit]

    def fallback_slots(self, limit: int = 3) -> list[Slot]:
        """Deterministic slots, for when tool calling returns nothing usable."""
        from .rule_based import previously_offered

        return self.store.nearest_available(
            self.context.anchor,
            position=self.context.position,
            n=limit,
            not_before=self.context.anchor,
            exclude=previously_offered(self.context),
        )


def build_schedule_toolkit(
    store: ScheduleStore, context: ConversationContext
) -> ScheduleToolkit:
    return ScheduleToolkit(store, context)


__all__ = [
    "ScheduleToolkit",
    "build_schedule_toolkit",
    "CheckSlotArgs",
    "NearestSlotsArgs",
    "DEFAULT_POSITION",
]
