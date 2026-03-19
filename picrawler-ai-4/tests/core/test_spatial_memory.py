"""
Tests for SpatialMemory — action recording, stuck detection, turn preferences,
cooldown expiry, and memory pruning.
"""
from __future__ import annotations

import time

import pytest

from core.spatial_memory import SpatialMemory, ActionRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def record(memory: SpatialMemory, action: str, success: bool,
           distance_before: float = None, distance_after: float = None) -> None:
    memory.record_action(action, success, distance_before, distance_after)


def fill_history(memory: SpatialMemory, action: str, success: bool, n: int,
                 distance: float = 50.0) -> None:
    for _ in range(n):
        record(memory, action, success, distance_before=distance, distance_after=distance)


# ---------------------------------------------------------------------------
# record_action
# ---------------------------------------------------------------------------

class TestRecordAction:
    def test_action_added_to_history(self, spatial_memory):
        record(spatial_memory, "forward", True)
        assert len(spatial_memory.action_history) == 1

    def test_success_increments_success_count(self, spatial_memory):
        record(spatial_memory, "forward", True)
        assert spatial_memory.success_counts["forward"] == 1
        assert spatial_memory.failure_counts["forward"] == 0

    def test_failure_increments_failure_count(self, spatial_memory):
        record(spatial_memory, "forward", False)
        assert spatial_memory.failure_counts["forward"] == 1
        assert spatial_memory.success_counts["forward"] == 0

    def test_turn_outcomes_tracked(self, spatial_memory):
        record(spatial_memory, "turn_left", True)
        record(spatial_memory, "turn_right", False)
        assert True in spatial_memory.turn_outcomes["turn_left"]
        assert False in spatial_memory.turn_outcomes["turn_right"]

    def test_history_pruned_at_capacity(self):
        mem = SpatialMemory(history_size=5)
        for i in range(10):
            record(mem, "forward", True)
        assert len(mem.action_history) == 5

    def test_turn_outcomes_pruned_at_20(self, spatial_memory):
        for _ in range(25):
            record(spatial_memory, "turn_left", True)
        assert len(spatial_memory.turn_outcomes["turn_left"]) == 20


# ---------------------------------------------------------------------------
# get_action_scores
# ---------------------------------------------------------------------------

class TestGetActionScores:
    def test_fresh_memory_forward_highest(self, spatial_memory):
        scores = spatial_memory.get_action_scores()
        assert scores["forward"] >= scores["backward"]

    def test_failing_action_gets_penalised(self, spatial_memory):
        scores_before = spatial_memory.get_action_scores()
        fill_history(spatial_memory, "forward", False, 10)
        scores_after = spatial_memory.get_action_scores()
        assert scores_after["forward"] < scores_before["forward"]

    def test_succeeding_action_maintains_score(self, spatial_memory):
        fill_history(spatial_memory, "forward", True, 10)
        scores = spatial_memory.get_action_scores()
        assert scores["forward"] > 0.5

    def test_heavy_penalty_for_repeated_failures(self, spatial_memory):
        # 3 of the last 5 are same action with 2+ failures → heavy penalty
        for i in range(5):
            record(spatial_memory, "forward", i < 2)  # 2 successes, 3 failures total
        # Actually we need count>=3 AND failures>=2
        for _ in range(3):
            record(spatial_memory, "forward", False)
        scores = spatial_memory.get_action_scores()
        assert scores["forward"] < 0.5

    def test_all_scores_between_zero_and_one(self, spatial_memory):
        fill_history(spatial_memory, "turn_left", False, 20)
        fill_history(spatial_memory, "forward", True, 10)
        scores = spatial_memory.get_action_scores()
        for action, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {action} out of range: {score}"


# ---------------------------------------------------------------------------
# get_best_turn_direction
# ---------------------------------------------------------------------------

class TestGetBestTurnDirection:
    def test_prefers_left_when_left_succeeds_more(self, spatial_memory):
        fill_history(spatial_memory, "turn_left", True, 5)
        fill_history(spatial_memory, "turn_right", False, 5)
        assert spatial_memory.get_best_turn_direction() == "turn_left"

    def test_prefers_right_when_right_succeeds_more(self, spatial_memory):
        fill_history(spatial_memory, "turn_right", True, 5)
        fill_history(spatial_memory, "turn_left", False, 5)
        assert spatial_memory.get_best_turn_direction() == "turn_right"

    def test_returns_valid_direction_with_no_history(self, spatial_memory):
        result = spatial_memory.get_best_turn_direction()
        assert result in ("turn_left", "turn_right")

    def test_alternates_when_equal_success_rates(self, spatial_memory):
        # Both have same number of successes: should return a valid direction
        fill_history(spatial_memory, "turn_left", True, 3)
        fill_history(spatial_memory, "turn_right", True, 3)
        result = spatial_memory.get_best_turn_direction()
        assert result in ("turn_left", "turn_right")


# ---------------------------------------------------------------------------
# is_stuck
# ---------------------------------------------------------------------------

class TestIsStuck:
    def test_not_stuck_with_empty_history(self, spatial_memory):
        assert not spatial_memory.is_stuck()

    def test_not_stuck_with_varied_non_oscillating_actions(self, spatial_memory):
        # forward + backward pattern — not an oscillation (no alternating turns)
        for a in ["forward", "backward", "forward", "backward"] * 3:
            record(spatial_memory, a, True)
        assert not spatial_memory.is_stuck()

    def test_stuck_when_same_action_mostly_fails(self, spatial_memory):
        # 10 records, all same action, 7 failures → stuck
        for i in range(10):
            record(spatial_memory, "forward", i >= 7)  # 7 failures, 3 successes
        assert spatial_memory.is_stuck()

    def test_not_stuck_if_failure_count_below_threshold(self, spatial_memory):
        # 10 records, same action, only 5 failures → NOT stuck (threshold is 7)
        for i in range(10):
            record(spatial_memory, "forward", i >= 5)  # 5 failures
        assert not spatial_memory.is_stuck()

    def test_stuck_on_oscillation(self, spatial_memory):
        # Alternating left/right pattern fills history
        for _ in range(5):
            record(spatial_memory, "turn_left", True)
            record(spatial_memory, "turn_right", True)
        assert spatial_memory.is_stuck()

    def test_not_stuck_with_fewer_than_10_records(self, spatial_memory):
        for _ in range(9):
            record(spatial_memory, "forward", False)
        assert not spatial_memory.is_stuck()


# ---------------------------------------------------------------------------
# _is_oscillating
# ---------------------------------------------------------------------------

class TestIsOscillating:
    def test_not_oscillating_with_few_turns(self, spatial_memory):
        actions = ["turn_left", "turn_right"]
        assert not spatial_memory._is_oscillating(actions)

    def test_not_oscillating_with_short_list(self, spatial_memory):
        actions = ["forward", "turn_left", "forward"]
        assert not spatial_memory._is_oscillating(actions)

    def test_oscillating_with_alternating_turns(self, spatial_memory):
        actions = ["turn_left", "turn_right"] * 5
        assert spatial_memory._is_oscillating(actions)


# ---------------------------------------------------------------------------
# _no_spatial_progress
# ---------------------------------------------------------------------------

class TestNoSpatialProgress:
    def test_progress_detected_with_changing_distances(self, spatial_memory):
        records = [
            ActionRecord("forward", 0.0, True, 50.0, 60.0, "ok"),
            ActionRecord("forward", 0.0, True, 60.0, 70.0, "ok"),
            ActionRecord("forward", 0.0, True, 70.0, 80.0, "ok"),
            ActionRecord("forward", 0.0, True, 80.0, 90.0, "ok"),
            ActionRecord("forward", 0.0, True, 90.0, 100.0, "ok"),
        ]
        assert not spatial_memory._no_spatial_progress(records)

    def test_no_progress_with_static_distances(self, spatial_memory):
        records = [
            ActionRecord("forward", 0.0, False, 15.0, 15.0, "blocked")
            for _ in range(10)
        ]
        assert spatial_memory._no_spatial_progress(records)

    def test_not_triggered_with_few_distance_readings(self, spatial_memory):
        records = [
            ActionRecord("forward", 0.0, True, None, None, "no_sensor")
            for _ in range(10)
        ]
        assert not spatial_memory._no_spatial_progress(records)


# ---------------------------------------------------------------------------
# record_escape / recovery escalation
# ---------------------------------------------------------------------------

class TestRecordEscape:
    def test_escape_count_increments(self, spatial_memory):
        spatial_memory.record_escape()
        assert spatial_memory.escape_count == 1

    def test_recovery_level_one_after_first_escape(self, spatial_memory):
        spatial_memory.record_escape()
        assert spatial_memory.recovery_level == 1

    def test_recovery_escalates_when_stuck_again_quickly(self, spatial_memory):
        spatial_memory.record_escape()
        # Immediately escape again (within 30s)
        spatial_memory.record_escape()
        assert spatial_memory.recovery_level >= 2

    def test_recovery_resets_when_stuck_after_delay(self, spatial_memory):
        spatial_memory.record_escape()
        # Simulate a long pause by backdating last_stuck_time
        spatial_memory.last_stuck_time = time.time() - 60.0
        spatial_memory.record_escape()
        # Should not be at max escalation
        assert spatial_memory.recovery_level == 1

    def test_recovery_level_capped_at_3(self, spatial_memory):
        for _ in range(10):
            spatial_memory.record_escape()
        assert spatial_memory.get_recovery_level() <= 3


# ---------------------------------------------------------------------------
# reset helpers
# ---------------------------------------------------------------------------

class TestResets:
    def test_reset_failure_counts(self, spatial_memory):
        fill_history(spatial_memory, "forward", False, 5)
        spatial_memory.reset_failure_counts()
        assert dict(spatial_memory.failure_counts) == {}

    def test_reset_recovery_escalation(self, spatial_memory):
        spatial_memory.record_escape()
        spatial_memory.record_escape()
        spatial_memory.reset_recovery_escalation()
        assert spatial_memory.recovery_level == 0
        assert spatial_memory.consecutive_stuck_count == 0


# ---------------------------------------------------------------------------
# get_recent_actions / get_recent_failures
# ---------------------------------------------------------------------------

class TestAccessors:
    def test_get_recent_actions(self, spatial_memory):
        for a in ["forward", "turn_left", "backward"]:
            record(spatial_memory, a, True)
        recent = spatial_memory.get_recent_actions(3)
        assert recent == ["forward", "turn_left", "backward"]

    def test_get_recent_failures(self, spatial_memory):
        record(spatial_memory, "forward", True)
        record(spatial_memory, "turn_left", False)
        failures = spatial_memory.get_recent_failures(10)
        assert len(failures) == 1
        assert failures[0].action == "turn_left"

    def test_to_dict_has_required_keys(self, spatial_memory):
        d = spatial_memory.to_dict()
        for key in ("recent_actions", "failure_counts", "success_counts",
                    "action_scores", "is_stuck", "escape_count", "best_turn"):
            assert key in d
