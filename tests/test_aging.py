from __future__ import annotations

import pytest

from fabweave.aging.clock import WipLot, evaluate_lot, rank_aging_risks
from fabweave.aging.ledger import AgingLedger, aging_risk_report, evaluate_active_lots
from fabweave.core.timebox import TimeBox
from fabweave.core.versioning import RevisionHistory


def _history(step: str, minutes: int, revision: int = 1, effective_at: int = 0) -> RevisionHistory[int]:
    history: RevisionHistory[int] = RevisionHistory()
    history.publish(revision, effective_at, minutes)
    return history


def etch_history() -> RevisionHistory[int]:
    """Revision 1 (60m) until t=150, revision 2 (80m) from t=150 on."""

    history: RevisionHistory[int] = RevisionHistory()
    history.publish(1, 0, 60)
    history.publish(2, 150, 80)
    return history


def report_history() -> RevisionHistory[int]:
    """Revision 1 (60m) until t=200, revision 2 (80m) from t=200 on."""

    history: RevisionHistory[int] = RevisionHistory()
    history.publish(1, 0, 60)
    history.publish(2, 200, 80)
    return history


def test_effective_clock_splits_elapsed_pause_and_limit_remainder():
    result = evaluate_lot(
        WipLot("LOT-1", "ETCH", entered_at=100, due_at=220),
        {"ETCH": _history("ETCH", 60)},
        [TimeBox(110, 130), TimeBox(125, 140)],  # union 110-140 = 30
        as_of=200,
    )
    assert result.elapsed_minutes == 100
    assert result.paused_minutes == 30
    assert result.effective_minutes == 70
    assert result.remaining_minutes == -10
    assert result.exceeded_minutes == 10
    assert result.exceeded
    assert result.hit_rule.revision == 1
    assert result.hit_rule.max_minutes == 60
    assert result.remaining_to_due == 20


def test_planned_holds_are_clipped_to_the_step_window():
    result = evaluate_lot(
        WipLot("LOT-1", "ETCH", entered_at=100, due_at=260),
        {"ETCH": _history("ETCH", 60)},
        [TimeBox(80, 110), TimeBox(190, 230)],  # clipped to 100-110 and 190-200
        as_of=200,
    )
    assert result.paused_minutes == 20
    assert result.effective_minutes == 80
    assert result.remaining_minutes == -20


def test_half_open_touching_intervals_neither_overlap_nor_gap():
    result = evaluate_lot(
        WipLot("LOT-1", "ETCH", entered_at=0, due_at=100),
        {"ETCH": _history("ETCH", 60)},
        [TimeBox(10, 20), TimeBox(20, 30)],
        as_of=60,
    )
    assert result.paused_minutes == 20
    assert result.effective_minutes == 40
    assert not result.exceeded


def test_threshold_uses_revision_in_force_at_evaluation_time():
    thresholds = {"ETCH": etch_history()}
    before = evaluate_lot(WipLot("LOT-1", "ETCH", 0, 300), thresholds, [], as_of=149)
    after = evaluate_lot(WipLot("LOT-1", "ETCH", 0, 300), thresholds, [], as_of=150)
    # 149 effective minutes breach the 60-minute revision 1 rule...
    assert before.hit_rule.revision == 1 and before.exceeded_minutes == 89
    # ...but at the revision-2 boundary the same wait is judged against 80.
    assert after.hit_rule.revision == 2 and after.exceeded_minutes == 70


def test_missing_threshold_fails_closed():
    with pytest.raises(LookupError):
        evaluate_lot(WipLot("LOT-1", "UNKNOWN", 0, 10), {"ETCH": _history("ETCH", 60)}, [], as_of=50)


def test_ranking_is_risk_then_due_then_lot_id_and_excludes_compliant_lots():
    thresholds = {"ETCH": _history("ETCH", 60), "CLEAN": _history("CLEAN", 100)}

    def evaluation(lot_id: str, risk: int, due: int, step: str = "ETCH"):
        return evaluate_lot(WipLot(lot_id, step, 0, due, risk_rank=risk), thresholds, [], as_of=100)

    ok = evaluation("LOT-OK", 9, 500, step="CLEAN")  # exactly at limit, never appears
    alpha = evaluation("LOT-A", 1, 120)  # risk 1, due in 20
    beta = evaluation("LOT-B", 1, 110)  # risk 1, due in 10 -> ahead of A
    high = evaluation("LOT-H", 2, 400)  # risk 2, far from due -> still first
    ranked = rank_aging_risks([ok, beta, high, alpha])
    assert [item.lot_id for item in ranked] == ["LOT-H", "LOT-B", "LOT-A"]
    # equal risk and due date -> stable lot-id tie break, input order irrelevant
    twin_one = evaluation("LOT-2", 1, 110)
    twin_two = evaluation("LOT-1", 1, 110)
    assert [item.lot_id for item in rank_aging_risks([twin_one, twin_two])] == ["LOT-1", "LOT-2"]
    assert [item.lot_id for item in rank_aging_risks([twin_two, twin_one])] == ["LOT-1", "LOT-2"]


def _ledger_with_late_hold() -> AgingLedger:
    ledger = AgingLedger()
    # Deliberately shuffled arrival order: release and completion land before
    # the late-arriving hold start and before the entry record.
    ledger.append("E-CMP", "LOT-1", 260, "step_completed")
    ledger.append("E-HOLD-REL", "LOT-1", 200, "hold_released", {"hold_id": "PLAN-1"})
    ledger.append("E-DUE", "LOT-1", 0, "due_declared", {"due_at": 260})
    ledger.append("E-ENTER", "LOT-1", 100, "step_entered", {"route_step": "ETCH"})
    ledger.append("E-HOLD-START", "LOT-1", 140, "hold_started", {"hold_id": "PLAN-1"})
    return ledger


def test_late_arriving_events_replay_in_event_time_and_pause_is_not_counted():
    report = aging_risk_report(_ledger_with_late_hold(), {"ETCH": report_history()}, as_of=250)
    assert len(report) == 1
    item = report[0]
    assert item.elapsed_minutes == 150
    assert item.paused_minutes == 60  # 140-200 hold, recovered despite late start
    assert item.effective_minutes == 90
    assert item.exceeded_minutes == 10  # revision-2 limit of 80 in force at t=250
    assert item.hit_rule.revision == 2
    assert item.remaining_to_due == 10


def test_report_ignores_events_after_the_reporting_instant():
    ledger = _ledger_with_late_hold()
    # At t=199 the hold is still open and revision 1 (60m) is in force. The
    # release (t=200) and completion (t=260) both lie in the future.
    assert aging_risk_report(ledger, {"ETCH": report_history()}, as_of=199) == []
    item = evaluate_active_lots(ledger, {"ETCH": report_history()}, as_of=199)[0]
    assert item.paused_minutes == 59  # open hold 140-199
    assert item.effective_minutes == 40
    assert not item.exceeded
    assert item.hit_rule.revision == 1


def test_completed_lots_leave_the_wip_report():
    ledger = _ledger_with_late_hold()  # completes at t=260
    assert aging_risk_report(ledger, {"ETCH": report_history()}, as_of=260) == []
    assert aging_risk_report(ledger, {"ETCH": report_history()}, as_of=300) == []


def test_overlapping_holds_union_without_double_counting_in_replay():
    ledger = AgingLedger()
    ledger.append("E-ENTER", "L", 0, "step_entered", {"route_step": "ETCH"})
    ledger.append("E-DUE", "L", 0, "due_declared", {"due_at": 100})
    ledger.append("H1-S", "L", 10, "hold_started", {"hold_id": "H1"})
    ledger.append("H2-S", "L", 20, "hold_started", {"hold_id": "H2"})
    ledger.append("H1-E", "L", 40, "hold_released", {"hold_id": "H1"})
    ledger.append("H2-E", "L", 30, "hold_released", {"hold_id": "H2"})
    item = aging_risk_report(ledger, {"ETCH": _history("ETCH", 60)}, as_of=100)[0]
    assert item.paused_minutes == 30  # union of [10,40) and [20,30)
    assert item.effective_minutes == 70


def test_event_repost_is_idempotent_but_conflicting_reuse_fails():
    ledger = AgingLedger()
    first = ledger.append("E1", "L", 0, "due_declared", {"due_at": 100})
    again = ledger.append("E1", "L", 0, "due_declared", {"due_at": 100})
    assert first == again
    with pytest.raises(ValueError):
        ledger.append("E1", "L", 0, "due_declared", {"due_at": 101})


def test_release_without_start_and_double_start_fail_closed():
    ledger = AgingLedger()
    ledger.append("E-ENTER", "L", 0, "step_entered", {"route_step": "ETCH"})
    ledger.append("E-DUE", "L", 0, "due_declared", {"due_at": 100})
    ledger.append("REL", "L", 10, "hold_released", {"hold_id": "H"})
    with pytest.raises(ValueError):
        aging_risk_report(ledger, {"ETCH": _history("ETCH", 60)}, as_of=50)

    ledger = AgingLedger()
    ledger.append("E-ENTER", "L", 0, "step_entered", {"route_step": "ETCH"})
    ledger.append("E-DUE", "L", 0, "due_declared", {"due_at": 100})
    ledger.append("S1", "L", 5, "hold_started", {"hold_id": "H"})
    ledger.append("S2", "L", 6, "hold_started", {"hold_id": "H"})
    with pytest.raises(ValueError):
        aging_risk_report(ledger, {"ETCH": _history("ETCH", 60)}, as_of=50)


def test_arrival_order_does_not_change_the_report():
    appends = [
        ("E-ENTER", "L", 0, "step_entered", {"route_step": "ETCH"}),
        ("E-DUE", "L", 0, "due_declared", {"due_at": 120}),
        ("H-S", "L", 20, "hold_started", {"hold_id": "H"}),
        ("H-E", "L", 60, "hold_released", {"hold_id": "H"}),
    ]

    def build(reverse: bool) -> AgingLedger:
        ledger = AgingLedger()
        for args in (reversed(appends) if reverse else appends):
            ledger.append(*args)
        return ledger

    forward = aging_risk_report(build(False), {"ETCH": _history("ETCH", 60)}, as_of=101)
    backward = aging_risk_report(build(True), {"ETCH": _history("ETCH", 60)}, as_of=101)
    assert forward == backward
    assert forward[0].paused_minutes == 40
    assert forward[0].effective_minutes == 61
