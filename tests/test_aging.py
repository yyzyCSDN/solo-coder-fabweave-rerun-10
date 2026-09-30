from __future__ import annotations

import pytest

from fabweave.aging.risk import (
    ENTERED,
    EXITED,
    HOLD_PLACED,
    HOLD_RELEASED,
    AgingEvent,
    AgingEventLog,
    AgingLimit,
    AgingLimitBook,
    evaluate_lot,
    replay_step,
    risk_level,
    screen_wip_aging,
)
from fabweave.core.timebox import TimeBox


def _book() -> AgingLimitBook:
    book = AgingLimitBook()
    book.publish(AgingLimit("ETCH", 100, 1), 0)
    book.publish(AgingLimit("ETCH", 60, 2), 150)
    book.publish(AgingLimit("CLEAN", 30, 1), 0)
    return book


def _ev(lot_id: str, at: int, kind: str, route_step: str | None = None, hold_id: str | None = None) -> AgingEvent:
    return AgingEvent(lot_id, at, kind, route_step, hold_id)  # type: ignore[arg-type]


def test_threshold_uses_revision_effective_at_step_entry_not_now():
    # Limit tightens from 100 (rev1) to 60 (rev2) at t=150.
    old = [_ev("OLD", 100, ENTERED, "ETCH")]
    new = [_ev("NEW", 200, ENTERED, "ETCH")]
    old_eval = evaluate_lot("OLD", old, _book(), 250)
    new_eval = evaluate_lot("NEW", new, _book(), 250)
    # OLD entered under rev1: 150 effective vs a 100 limit -> breached.
    assert old_eval.limit_revision == 1 and old_eval.over_minutes == 50 and old_eval.breached
    # NEW entered under rev2: 50 effective vs a 60 limit -> still healthy.
    assert new_eval.limit_revision == 2 and new_eval.remaining_minutes == 10 and not new_eval.breached


def test_planned_holds_pause_clock_and_durations_reconcile():
    events = [
        _ev("L", 0, ENTERED, "ETCH"),
        _ev("L", 20, HOLD_PLACED, hold_id="H1"),
        _ev("L", 40, HOLD_RELEASED, hold_id="H1"),
    ]
    result = evaluate_lot("L", events, _book(), 150)
    assert (result.elapsed_minutes, result.paused_minutes, result.effective_minutes) == (150, 20, 130)
    assert result.over_minutes == 30 and result.breached
    hit = result.hits[0]
    assert hit.rule == "aging_limit_exceeded" and hit.over_minutes == 30
    assert (hit.limit_minutes, hit.limit_revision) == (100, 1)
    assert result.pause_windows == (TimeBox(20, 40),)


def test_overlapping_nested_and_boundary_holds_merge_and_clip_without_double_counting():
    events = [
        _ev("L", 0, ENTERED, "ETCH"),
        _ev("L", -10, HOLD_PLACED, hold_id="BEFORE"),
        _ev("L", 20, HOLD_RELEASED, hold_id="BEFORE"),
        _ev("L", 10, HOLD_PLACED, hold_id="OUTER"),
        _ev("L", 20, HOLD_PLACED, hold_id="INNER"),
        _ev("L", 30, HOLD_RELEASED, hold_id="INNER"),
        _ev("L", 40, HOLD_RELEASED, hold_id="OUTER"),
        _ev("L", 120, HOLD_PLACED, hold_id="AFTER"),
        _ev("L", 200, HOLD_RELEASED, hold_id="AFTER"),
    ]
    result = evaluate_lot("L", events, _book(), 100)
    # (-10,20] clipped to (0,20) merges with (10,40) into (0,40): 40 minutes.
    assert result.paused_minutes == 40
    assert result.pause_windows == (TimeBox(0, 40),)
    assert result.effective_minutes == 60 and not result.breached


def test_open_ended_hold_stops_clock_only_until_evaluation_time():
    events = [
        _ev("L", 0, ENTERED, "ETCH"),
        _ev("L", 30, HOLD_PLACED, hold_id="OPEN"),
    ]
    result = evaluate_lot("L", events, _book(), 100)
    assert result.paused_minutes == 70 and result.effective_minutes == 30 and not result.breached
    assert result.pause_windows == (TimeBox(30, 100),)


def test_future_dated_events_do_not_leak_into_evaluation():
    events = [
        _ev("L", 0, ENTERED, "ETCH"),
        _ev("L", 300, HOLD_PLACED, hold_id="FUTURE"),
    ]
    result = evaluate_lot("L", events, _book(), 100)
    assert result.paused_minutes == 0 and result.effective_minutes == 100


def test_late_arriving_events_are_replayed_by_event_time():
    chronological = AgingEventLog()
    chronological.record(_ev("L", 0, ENTERED, "ETCH"))
    chronological.record(_ev("L", 20, HOLD_PLACED, hold_id="H1"))
    chronological.record(_ev("L", 50, HOLD_RELEASED, hold_id="H1"))

    scrambled = AgingEventLog()
    for event in reversed(chronological.events_for("L")):
        scrambled.record(event)
    # Late telemetry appended after later events: a second hold placed at 60
    # whose release packet (stamped 90) arrives first, plus a retried entry.
    scrambled.record(_ev("L", 90, HOLD_RELEASED, hold_id="H2"))
    scrambled.record(_ev("L", 60, HOLD_PLACED, hold_id="H2"))
    scrambled.record(_ev("L", 0, ENTERED, "ETCH"))  # exact duplicate, ignored

    expected = evaluate_lot(
        "L",
        chronological.events_for("L")
        + [_ev("L", 60, HOLD_PLACED, hold_id="H2"), _ev("L", 90, HOLD_RELEASED, hold_id="H2")],
        _book(),
        100,
    )
    actual = evaluate_lot("L", scrambled.events_for("L"), _book(), 100)
    assert (actual.paused_minutes, actual.effective_minutes) == (expected.paused_minutes, expected.effective_minutes)
    # (20,50) and (60,90) stay disjoint regardless of arrival order.
    assert actual.pause_windows == (TimeBox(20, 50), TimeBox(60, 90))
    assert actual.paused_minutes == 60 and actual.effective_minutes == 40


def test_screen_only_returns_breached_wip_and_ranks_stably():
    def lot(lot_id: str, held: int = 0, due_at: int | None = None, now: int = 200) -> list[AgingEvent]:
        events = [_ev(lot_id, 0, ENTERED, "ETCH")]
        if held:
            events += [
                _ev(lot_id, 0, HOLD_PLACED, hold_id="H"),
                _ev(lot_id, held, HOLD_RELEASED, hold_id="H"),
            ]
        return events

    lots = {
        "B": lot("B", 50, due_at=150),   # high, over 50, overdue (slack -50)
        "E": lot("E", 150),              # effective 50 -> healthy, excluded
        "A": lot("A"),                   # critical, over 100
        "ZZ": lot("ZZ", 70),             # medium, over 30
        "D": lot("D", 70),               # medium, over 30
        "C": lot("C", 50, due_at=250),   # high, over 50, slack 50
        "AA": lot("AA", 70),             # medium, over 30
        "DONE": [
            _ev("DONE", 0, ENTERED, "ETCH"),
            _ev("DONE", 10, EXITED, "ETCH"),
        ],
    }
    duedates = {"B": 150, "C": 250}

    forward = screen_wip_aging(lots, _book(), 200, duedates)
    reverse = screen_wip_aging(dict(reversed(list(lots.items()))), _book(), 200, duedates)
    assert [item.lot_id for item in forward] == ["A", "B", "C", "AA", "D", "ZZ"]
    assert [item.lot_id for item in reverse] == ["A", "B", "C", "AA", "D", "ZZ"]

    by_id = {item.lot_id: item for item in forward}
    assert by_id["A"].risk_level == "critical"
    assert by_id["B"].risk_level == by_id["C"].risk_level == "high"
    assert by_id["AA"].risk_level == "medium"
    # Overdue breached lot carries both hit rules; aging breach drives inclusion.
    assert {hit.rule for hit in by_id["B"].hits} == {"aging_limit_exceeded", "due_date_overdue"}
    assert {hit.rule for hit in by_id["A"].hits} == {"aging_limit_exceeded"}


def test_risk_bands_scale_with_limit():
    assert risk_level(100, 100) == "critical"
    assert risk_level(50, 100) == "high"
    assert risk_level(49, 100) == "medium"
    with pytest.raises(ValueError):
        risk_level(0, 100)


def test_replay_tracks_segments_and_rejects_invalid_histories():
    lot_id = "L"

    with pytest.raises(ValueError, match="released without placement"):
        replay_step(lot_id, [_ev(lot_id, 0, ENTERED, "ETCH"), _ev(lot_id, 10, HOLD_RELEASED, hold_id="X")], 100)

    with pytest.raises(ValueError, match="placed twice"):
        replay_step(
            lot_id,
            [
                _ev(lot_id, 0, ENTERED, "ETCH"),
                _ev(lot_id, 5, HOLD_PLACED, hold_id="X"),
                _ev(lot_id, 6, HOLD_PLACED, hold_id="X"),
            ],
            100,
        )

    with pytest.raises(ValueError, match="does not match open step"):
        replay_step(
            lot_id,
            [_ev(lot_id, 0, ENTERED, "ETCH"), _ev(lot_id, 5, EXITED, "CLEAN")],
            100,
        )

    # Release ordered before its placement in event time has no placement to
    # match, which fails closed as a dangling release.
    with pytest.raises(ValueError, match="released without placement"):
        replay_step(
            lot_id,
            [
                _ev(lot_id, 0, ENTERED, "ETCH"),
                _ev(lot_id, 10, HOLD_PLACED, hold_id="X"),
                _ev(lot_id, 5, HOLD_RELEASED, hold_id="X"),
            ],
            100,
        )

    # A detention carried in from upstream and released inside this step is
    # clipped to the entry boundary rather than rejected.
    carried = replay_step(
        lot_id,
        [
            _ev(lot_id, -10, HOLD_PLACED, hold_id="X"),
            _ev(lot_id, 0, ENTERED, "ETCH"),
            _ev(lot_id, 20, HOLD_RELEASED, hold_id="X"),
        ],
        100,
    )
    assert carried.raw_pauses == (TimeBox(-10, 20),)
    # Completed steps are not WIP, then a later open segment is tracked.
    replayed = replay_step(
        lot_id,
        [
            _ev(lot_id, 0, ENTERED, "ETCH"),
            _ev(lot_id, 10, EXITED, "ETCH"),
            _ev(lot_id, 20, ENTERED, "CLEAN"),
        ],
        40,
    )
    assert replayed.route_step == "CLEAN" and replayed.entered_at == 20


def test_pauses_from_completed_segments_do_not_leak_into_current_step():
    # Hold entirely inside the prior (closed) segment must vanish after
    # clipping to the current segment; one carried across does not.
    events = [
        _ev("L", 0, ENTERED, "ETCH"),
        _ev("L", 5, HOLD_PLACED, hold_id="OLD"),
        _ev("L", 8, HOLD_RELEASED, hold_id="OLD"),
        _ev("L", 10, EXITED, "ETCH"),
        _ev("L", 20, ENTERED, "CLEAN"),
        _ev("L", 25, HOLD_PLACED, hold_id="CARRIED"),
    ]
    result = evaluate_lot("L", events, _book(), 45)
    assert result.route_step == "CLEAN"
    assert result.pause_windows == (TimeBox(25, 45),)
    assert result.paused_minutes == 20 and result.effective_minutes == 5


def test_evaluation_is_fail_closed_on_unknown_step():
    with pytest.raises(LookupError):
        evaluate_lot("L", [_ev("L", 0, ENTERED, "UNKNOWN")], _book(), 100)


def test_event_and_limit_validation():
    with pytest.raises(ValueError):
        AgingEvent("L", 0, ENTERED)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AgingEvent("L", 0, HOLD_PLACED)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AgingLimit("ETCH", 0, 1)
    with pytest.raises(ValueError):
        AgingLimit("ETCH", 10, 0)


def test_event_log_screen_is_idempotent_on_retried_telemetry():
    log = AgingEventLog()
    event = _ev("L", 0, ENTERED, "ETCH")
    assert log.record(event) is event
    assert log.record(AgingEvent("L", 0, ENTERED, "ETCH")) == event
    findings = log.screen(_book(), 150)
    assert [item.lot_id for item in findings] == ["L"]
    assert findings[0].elapsed_minutes == 150 and findings[0].paused_minutes == 0
