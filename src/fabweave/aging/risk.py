"""WIP aging risk screening.

A lot's aging clock runs from the moment it enters a route step.  The clock
stops while an engineering/quality hold is in place (planned detention does
not count), then resumes when the hold is released.  The limits that decide a
breach are versioned per route step, and the revision in force *when the lot
entered the step* is the one that applies for the whole sojourn.

Events may arrive out of order ("late" telemetry): every calculation is a
pure replay over event time, not wall-clock arrival time.  Overlapping or
nested holds are merged, and every pause interval is clipped to the step's
half-open ``[entry, now)`` sojourn window, which keeps elapsed, paused and
effective durations internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from fabweave.core.timebox import TimeBox, covered_duration, merge_timeboxes
from fabweave.core.versioning import Revision, RevisionHistory


ENTERED = "step_entered"
EXITED = "step_exited"
HOLD_PLACED = "hold_placed"
HOLD_RELEASED = "hold_released"

EventKind = Literal["step_entered", "step_exited", "hold_placed", "hold_released"]


@dataclass(frozen=True, slots=True)
class AgingLimit:
    """Maximum *effective* minutes a lot may age at one route step."""

    route_step: str
    max_minutes: int
    revision: int

    def __post_init__(self) -> None:
        if self.max_minutes <= 0 or self.revision <= 0:
            raise ValueError("aging limit and revision must be positive")


@dataclass(frozen=True, slots=True)
class AgingEvent:
    lot_id: str
    occurred_at: int
    kind: EventKind
    route_step: str | None = None
    hold_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind in (ENTERED, EXITED) and not self.route_step:
            raise ValueError("step events require a route step")
        if self.kind in (HOLD_PLACED, HOLD_RELEASED) and not self.hold_id:
            raise ValueError("hold events require a hold id")


class AgingLimitBook:
    """Versioned aging limits keyed by route step.

    The engine resolves the value effective at step entry, so a revision
    published mid-sojourn never retroactively changes a lot's threshold.
    """

    def __init__(self) -> None:
        self._histories: dict[str, RevisionHistory[AgingLimit]] = {}

    def publish(self, limit: AgingLimit, effective_at: int) -> None:
        history = self._histories.setdefault(limit.route_step, RevisionHistory())
        history.publish(limit.revision, effective_at, limit)

    def at(self, route_step: str, instant: int) -> Revision[AgingLimit]:
        try:
            history = self._histories[route_step]
        except KeyError:
            raise LookupError(f"no aging limit published for route step {route_step}") from None
        return history.at(instant)


@dataclass(frozen=True, slots=True)
class AgingRuleHit:
    """A rule that fired for a lot at evaluation time.

    Limit fields are present for aging-threshold rules and absent for rules
    unrelated to a threshold (e.g. a due-date miss).
    """

    rule: str
    over_minutes: int
    limit_minutes: int | None = None
    limit_revision: int | None = None


RiskLevel = Literal["critical", "high", "medium"]


@dataclass(frozen=True, slots=True)
class AgingEvaluation:
    lot_id: str
    route_step: str
    entered_at: int
    evaluated_at: int
    elapsed_minutes: int
    paused_minutes: int
    effective_minutes: int
    limit_minutes: int
    limit_revision: int
    remaining_minutes: int
    over_minutes: int
    risk_level: RiskLevel | None
    minutes_to_due: int | None
    pause_windows: tuple[TimeBox, ...]
    hits: tuple[AgingRuleHit, ...]

    @property
    def breached(self) -> bool:
        return any(hit.rule == "aging_limit_exceeded" for hit in self.hits)


class _OpenStep(NamedTuple):
    route_step: str
    entered_at: int
    raw_pauses: tuple[TimeBox, ...]


# Ordering of risk bands, from most to least urgent.
_RISK_ORDER: dict[str, int] = {"critical": 0, "high": 1, "medium": 2}

_EVENT_ORDER = {ENTERED: 0, HOLD_PLACED: 1, HOLD_RELEASED: 1, EXITED: 2}


def risk_level(over_minutes: int, limit_minutes: int) -> RiskLevel:
    """Band the breach magnitude relative to the limit itself.

    At least one full limit beyond the threshold is critical, at least half a
    limit beyond is high, and anything else over the line is medium.
    """

    if over_minutes <= 0:
        raise ValueError("risk only applies to breached lots")
    if over_minutes >= limit_minutes:
        return "critical"
    if 2 * over_minutes >= limit_minutes:
        return "high"
    return "medium"


def replay_step(lot_id: str, events: list[AgingEvent], now: int) -> _OpenStep | None:
    """Replay one lot's events in event time.

    Returns the lot's currently open step with its raw pause intervals, or
    ``None`` when the lot has completed its last step (no WIP) or has not yet
    entered.  Telemetry stamped after ``now`` belongs to a future evaluation
    and is ignored, which makes replay deterministic.

    Holds are paired independently of step boundaries: a detention placed
    upstream and released inside this step (or a release telemetry packet
    arriving late) is handled by clipping the resulting interval to the
    sojourn window, never by discarding the event.
    """

    ordered = sorted(
        (event for event in events if event.occurred_at <= now),
        # Deterministic order at the same timestamp: entry, then hold
        # placement/release, then exit, matching the half-open convention
        # where an instant at a boundary belongs to the starting interval.
        key=lambda event: (event.occurred_at, _EVENT_ORDER[event.kind]),
    )

    # First pass: pair every detention across the whole history.  Pairing is
    # independent of step boundaries so a hold carried in from upstream (or a
    # release packet arriving late) still resolves; clipping happens later.
    open_holds: dict[str, int] = {}
    all_pauses: list[TimeBox] = []
    for event in ordered:
        if event.kind == HOLD_PLACED:
            if event.hold_id in open_holds:
                raise ValueError(f"lot {lot_id} hold {event.hold_id} placed twice")
            open_holds[event.hold_id] = event.occurred_at
        elif event.kind == HOLD_RELEASED:
            placed_at = open_holds.pop(event.hold_id, None)
            if placed_at is None:
                raise ValueError(f"lot {lot_id} hold {event.hold_id} released without placement")
            if event.occurred_at < placed_at:
                raise ValueError(f"lot {lot_id} hold {event.hold_id} released before placement")
            if event.occurred_at > placed_at:
                all_pauses.append(TimeBox(placed_at, event.occurred_at))

    # Second pass: locate the currently open step, validating step transitions.
    route_step: str | None = None
    entered_at: int | None = None
    exited = True
    for event in ordered:
        if event.kind == ENTERED:
            if not exited:
                raise ValueError(f"lot {lot_id} entered a step without leaving {route_step}")
            route_step = event.route_step
            entered_at = event.occurred_at
            exited = False
        elif event.kind == EXITED:
            if exited:
                raise ValueError(f"lot {lot_id} exited a step it never entered")
            if event.route_step != route_step:
                raise ValueError(f"lot {lot_id} exit for {event.route_step} does not match open step {route_step}")
            exited = True

    if entered_at is None or exited:
        return None

    # Detentions still open at evaluation time stop the clock up to `now`.
    raw_pauses = [
        *all_pauses,
        *(TimeBox(placed_at, now) for placed_at in open_holds.values() if placed_at < now),
    ]
    return _OpenStep(route_step, entered_at, tuple(raw_pauses))


def evaluate_lot(
    lot_id: str,
    events: list[AgingEvent],
    limits: AgingLimitBook,
    now: int,
    due_at: int | None = None,
) -> AgingEvaluation | None:
    """Evaluate one lot at ``now``; ``None`` when the lot is not WIP there."""

    open_step = replay_step(lot_id, events, now)
    if open_step is None:
        return None
    route_step, entered_at, raw_pauses = open_step

    # Resolve the threshold by route step and the revision effective at entry.
    revision = limits.at(route_step, entered_at)
    limit = revision.value

    window = TimeBox(entered_at, now)
    elapsed = window.duration
    # Clip to [entry, now) and merge first, so nested/overlapping holds and
    # holds spanning the boundary are counted once; then derive effective age.
    pause_windows = tuple(merge_timeboxes([part for pause in raw_pauses if (part := pause.intersection(window))]))
    paused = covered_duration(list(pause_windows), window)
    effective = elapsed - paused
    remaining = limit.max_minutes - effective
    over = max(0, -remaining)

    hits: list[AgingRuleHit] = []
    if over > 0:
        hits.append(AgingRuleHit("aging_limit_exceeded", over, limit.max_minutes, limit.revision))
    if due_at is not None and due_at < now:
        hits.append(AgingRuleHit("due_date_overdue", now - due_at))

    minutes_to_due = None if due_at is None else due_at - now
    return AgingEvaluation(
        lot_id=lot_id,
        route_step=route_step,
        entered_at=entered_at,
        evaluated_at=now,
        elapsed_minutes=elapsed,
        paused_minutes=paused,
        effective_minutes=effective,
        limit_minutes=limit.max_minutes,
        limit_revision=limit.revision,
        remaining_minutes=remaining,
        over_minutes=over,
        risk_level=risk_level(over, limit.max_minutes) if over else None,
        minutes_to_due=minutes_to_due,
        pause_windows=pause_windows,
        hits=tuple(hits),
    )


def _sort_key(item: AgingEvaluation) -> tuple[int, float, int, str]:
    # Risk band first, then least slack to the commit date (lots with no due
    # date sort last), then deepest breach, then lot id so the ranking never
    # depends on input order or dict hashing.
    due_slack = item.minutes_to_due if item.minutes_to_due is not None else float("inf")
    return (_RISK_ORDER[item.risk_level], due_slack, -item.over_minutes, item.lot_id)  # type: ignore[arg-type]


def screen_wip_aging(
    events_by_lot: dict[str, list[AgingEvent]],
    limits: AgingLimitBook,
    now: int,
    due_at_by_lot: dict[str, int] | None = None,
) -> list[AgingEvaluation]:
    """Return lots with a real aging risk, stably ranked.

    Only lots past their *effective* aging limit are returned: planned
    detention is excluded from age.  Thresholds resolve by route step and the
    revision effective at step entry, and ranking is by risk band, remaining
    time to the due date, breach magnitude and lot id.
    """

    due_at_by_lot = due_at_by_lot or {}
    findings: list[AgingEvaluation] = []
    for lot_id, events in events_by_lot.items():
        evaluation = evaluate_lot(lot_id, events, limits, now, due_at_by_lot.get(lot_id))
        if evaluation is not None and evaluation.breached:
            findings.append(evaluation)
    return sorted(findings, key=_sort_key)


class AgingEventLog:
    """Append-only lot events; late arrivals are simply replayed by event time."""

    def __init__(self) -> None:
        self._events: dict[str, list[AgingEvent]] = {}

    def record(self, event: AgingEvent) -> AgingEvent:
        """Append an event.

        Idempotent on ``(lot, time, kind, step/hold)`` so retried telemetry
        does not create phantom holds or duplicate step entries.
        """

        events = self._events.setdefault(event.lot_id, [])
        for existing in events:
            if (
                existing.occurred_at == event.occurred_at
                and existing.kind == event.kind
                and existing.route_step == event.route_step
                and existing.hold_id == event.hold_id
            ):
                return existing
        events.append(event)
        return event

    def events_for(self, lot_id: str) -> list[AgingEvent]:
        return list(self._events.get(lot_id, ()))

    def screen(
        self,
        limits: AgingLimitBook,
        now: int,
        due_at_by_lot: dict[str, int] | None = None,
    ) -> list[AgingEvaluation]:
        return screen_wip_aging(dict(self._events), limits, now, due_at_by_lot)
