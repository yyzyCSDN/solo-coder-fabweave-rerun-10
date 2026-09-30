from __future__ import annotations

from dataclasses import dataclass

from fabweave.core.timebox import TimeBox, covered_duration
from fabweave.core.versioning import Revision, RevisionHistory


@dataclass(frozen=True, slots=True)
class HitRule:
    """The queue-aging limit that a lot was measured against."""

    route_step: str
    max_minutes: int
    revision: int
    effective_at: int


@dataclass(frozen=True, slots=True)
class WipLot:
    lot_id: str
    route_step: str
    entered_at: int
    due_at: int
    risk_rank: int = 0
    completed_at: int | None = None


@dataclass(frozen=True, slots=True)
class AgingEvaluation:
    lot_id: str
    route_step: str
    elapsed_minutes: int
    paused_minutes: int
    effective_minutes: int
    remaining_minutes: int
    exceeded_minutes: int
    risk_rank: int
    remaining_to_due: int
    hit_rule: HitRule

    @property
    def exceeded(self) -> bool:
        return self.exceeded_minutes > 0


def _resolve_rule(
    route_step: str,
    thresholds: dict[str, RevisionHistory[int]],
    at: int,
) -> HitRule:
    history = thresholds.get(route_step)
    if history is None:
        raise LookupError(f"no aging threshold published for route step {route_step!r}")
    revision: Revision[int] = history.at(at)
    return HitRule(route_step, revision.value, revision.version, revision.effective_at)


def evaluate_lot(
    lot: WipLot,
    thresholds: dict[str, RevisionHistory[int]],
    planned_holds: list[TimeBox] | None = None,
    as_of: int | None = None,
) -> AgingEvaluation:
    """Measure one lot's effective wait against the limit in force at ``as_of``.

    The limit revision is the one effective at the evaluation instant, so a lot
    waiting across a reissue of the rule is judged by the then-current version.
    Planned-hold intervals are clipped to [entry, evaluation] and merged before
    they are subtracted, which keeps overlapping and touching holds from being
    double counted.
    """

    end = lot.completed_at if as_of is None else as_of
    if end < lot.entered_at:
        raise ValueError("evaluation precedes step entry")
    rule = _resolve_rule(lot.route_step, thresholds, end)
    elapsed = end - lot.entered_at
    paused = (
        0
        if elapsed == 0
        else covered_duration(planned_holds or [], TimeBox(lot.entered_at, end))
    )
    effective = elapsed - paused
    remaining = rule.max_minutes - effective
    return AgingEvaluation(
        lot_id=lot.lot_id,
        route_step=lot.route_step,
        elapsed_minutes=elapsed,
        paused_minutes=paused,
        effective_minutes=effective,
        remaining_minutes=remaining,
        exceeded_minutes=max(0, -remaining),
        risk_rank=lot.risk_rank,
        remaining_to_due=lot.due_at - end,
        hit_rule=rule,
    )


def rank_aging_risks(evaluations: list[AgingEvaluation]) -> list[AgingEvaluation]:
    """Return only exceeded lots, ordered by risk then by how late the commit is.

    Higher ``risk_rank`` comes first; within a risk band the lot whose customer
    commitment is most overdue (smallest ``remaining_to_due``) leads. Ties are
    broken on ``lot_id`` so the order never depends on input ordering.
    """

    return sorted(
        (item for item in evaluations if item.exceeded),
        key=lambda item: (-item.risk_rank, item.remaining_to_due, item.lot_id),
    )
