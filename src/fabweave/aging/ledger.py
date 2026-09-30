from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fabweave.aging.clock import AgingEvaluation, WipLot, evaluate_lot, rank_aging_risks
from fabweave.core.timebox import TimeBox
from fabweave.core.versioning import RevisionHistory


_EVENT_KINDS = frozenset(
    {
        "step_entered",
        "step_completed",
        "due_declared",
        "risk_declared",
        "hold_started",
        "hold_released",
    }
)


@dataclass(frozen=True, slots=True)
class AgingEvent:
    event_id: str
    lot_id: str
    occurred_at: int
    kind: str
    data: dict[str, Any]
    arrived_seq: int


class AgingLedger:
    """Append-only lot events that tolerate late and out-of-order arrival.

    Events are accepted in any arrival order; projection always replays them in
    event time, so a ``hold_started`` record showing up after its release cannot
    corrupt the pause accounting. Re-posting an event id is idempotent, while
    reusing an id with different content fails closed.
    """

    def __init__(self) -> None:
        self._events: dict[str, AgingEvent] = {}

    def append(
        self,
        event_id: str,
        lot_id: str,
        occurred_at: int,
        kind: str,
        data: dict[str, Any] | None = None,
    ) -> AgingEvent:
        if kind not in _EVENT_KINDS:
            raise ValueError(f"unknown aging event kind {kind!r}")
        candidate = AgingEvent(event_id, lot_id, occurred_at, kind, dict(data or {}), len(self._events))
        existing = self._events.get(event_id)
        if existing is not None:
            if (
                existing.lot_id,
                existing.occurred_at,
                existing.kind,
                existing.data,
            ) != (candidate.lot_id, candidate.occurred_at, candidate.kind, candidate.data):
                raise ValueError("event id was reused with different content")
            return existing
        self._events[event_id] = candidate
        return candidate

    def lot_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.lot_id for event in self._events.values()}))

    def events_for(self, lot_id: str) -> tuple[AgingEvent, ...]:
        ordered = sorted(
            (event for event in self._events.values() if event.lot_id == lot_id),
            key=lambda event: (event.occurred_at, event.arrived_seq),
        )
        return tuple(ordered)


@dataclass(frozen=True, slots=True)
class _LotState:
    route_step: str
    entered_at: int
    completed_at: int | None
    due_at: int
    risk_rank: int
    hold_intervals: tuple[TimeBox, ...]


def _project(events: tuple[AgingEvent, ...], as_of: int) -> _LotState | None:
    route_step: str | None = None
    entered_at: int | None = None
    completed_at: int | None = None
    due_at: int | None = None
    risk_rank = 0
    open_holds: dict[str, int] = {}
    closed_holds: list[TimeBox] = []
    # Last writer at a given instant wins for declaration events; a different
    # value at the same event time is an ambiguity we refuse to guess through.
    latest_declaration: dict[str, tuple[int, object]] = {}

    for event in events:
        if event.occurred_at > as_of:
            continue
        data = event.data
        if event.kind == "step_entered":
            step = str(data["route_step"])
            if route_step is not None and route_step != step:
                raise ValueError("conflicting step entries for the same lot")
            if entered_at is not None and entered_at != event.occurred_at:
                raise ValueError("conflicting step entries for the same lot")
            route_step, entered_at = step, event.occurred_at
        elif event.kind == "step_completed":
            if completed_at is not None and completed_at != event.occurred_at:
                raise ValueError("conflicting step completions for the same lot")
            completed_at = event.occurred_at
        elif event.kind in {"due_declared", "risk_declared"}:
            value: object = data["due_at"] if event.kind == "due_declared" else data["risk_rank"]
            prior = latest_declaration.get(event.kind)
            if prior is not None and prior[0] == event.occurred_at and prior[1] != value:
                raise ValueError(f"conflicting {event.kind} events at the same time")
            latest_declaration[event.kind] = (event.occurred_at, value)
            if event.kind == "due_declared":
                due_at = int(value)  # type: ignore[arg-type]
            else:
                risk_rank = int(value)  # type: ignore[arg-type]
        elif event.kind == "hold_started":
            hold_id = str(data["hold_id"])
            if hold_id in open_holds:
                raise ValueError(f"hold {hold_id!r} started again before release")
            open_holds[hold_id] = event.occurred_at
        else:  # hold_released
            hold_id = str(data["hold_id"])
            start = open_holds.pop(hold_id, None)
            if start is None:
                raise ValueError(f"hold {hold_id!r} released without a start")
            if event.occurred_at > start:
                closed_holds.append(TimeBox(start, event.occurred_at))

    if entered_at is None:
        return None
    if completed_at is not None and completed_at < entered_at:
        raise ValueError("step completion precedes entry")
    if due_at is None:
        raise ValueError("lot entered a step without a declared due date")

    hold_intervals = tuple(closed_holds) + tuple(
        TimeBox(start, as_of) for start in open_holds.values() if start < as_of
    )
    return _LotState(
        route_step or "",
        entered_at,
        completed_at,
        due_at,
        risk_rank,
        hold_intervals,
    )


def evaluate_active_lots(
    ledger: AgingLedger,
    thresholds: dict[str, RevisionHistory[int]],
    as_of: int,
) -> list[AgingEvaluation]:
    """Project every lot that is WIP at ``as_of``, compliant ones included."""

    evaluations: list[AgingEvaluation] = []
    for lot_id in ledger.lot_ids():
        state = _project(ledger.events_for(lot_id), as_of)
        if state is None or state.entered_at > as_of:
            continue
        if state.completed_at is not None and state.completed_at <= as_of:
            continue
        lot = WipLot(
            lot_id=lot_id,
            route_step=state.route_step,
            entered_at=state.entered_at,
            due_at=state.due_at,
            risk_rank=state.risk_rank,
        )
        evaluations.append(evaluate_lot(lot, thresholds, list(state.hold_intervals), as_of))
    return evaluations


def aging_risk_report(
    ledger: AgingLedger,
    thresholds: dict[str, RevisionHistory[int]],
    as_of: int,
) -> list[AgingEvaluation]:
    """Return the exceeded lots among active WIP at ``as_of``.

    Lots whose step was already completed by ``as_of`` are no longer WIP and
    drop out; events occurring after ``as_of`` (including late arrivals) are
    ignored so the report describes that instant exactly. The result is stable:
    risk rank first, then most-overdue commitment, then lot id.
    """

    return rank_aging_risks(evaluate_active_lots(ledger, thresholds, as_of))
