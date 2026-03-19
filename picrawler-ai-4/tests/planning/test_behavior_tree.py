"""
Tests for behavior tree nodes: SequenceNode, FallbackNode, condition nodes,
action nodes, and smart recovery sequences.
"""
from __future__ import annotations

import pytest

from core.world_model import WorldModel, ObstacleInfo
from core.spatial_memory import SpatialMemory
from planning.behavior_tree import (
    Status,
    BehaviorContext,
    BehaviorNode,
    SequenceNode,
    FallbackNode,
    CheckStuckCondition,
    CheckPathClear,
    CheckFreeSpace,
    MoveForwardAction,
    TurnAction,
    BackupAction,
    SmartTurnAction,
    StopAction,
    RecoverySequence,
    build_exploration_tree,
    build_cautious_exploration_tree,
)


# ---------------------------------------------------------------------------
# Stub nodes
# ---------------------------------------------------------------------------

class _Always(BehaviorNode):
    def __init__(self, status: Status):
        super().__init__(f"Always{status.name}")
        self._status = status

    def execute(self, context: BehaviorContext) -> Status:
        return self._status


SUCCESS = lambda: _Always(Status.SUCCESS)
FAILURE = lambda: _Always(Status.FAILURE)
RUNNING = lambda: _Always(Status.RUNNING)


class _Counting(BehaviorNode):
    """Records how many times execute() is called."""
    def __init__(self, status: Status):
        super().__init__("Counting")
        self._status = status
        self.calls = 0

    def execute(self, context: BehaviorContext) -> Status:
        self.calls += 1
        return self._status


# ---------------------------------------------------------------------------
# SequenceNode
# ---------------------------------------------------------------------------

class TestSequenceNode:
    def test_all_success_returns_success(self, behavior_context):
        seq = SequenceNode("s", [SUCCESS(), SUCCESS(), SUCCESS()])
        assert seq.execute(behavior_context) == Status.SUCCESS

    def test_first_failure_returns_failure(self, behavior_context):
        seq = SequenceNode("s", [FAILURE(), SUCCESS(), SUCCESS()])
        assert seq.execute(behavior_context) == Status.FAILURE

    def test_running_child_returns_running(self, behavior_context):
        seq = SequenceNode("s", [SUCCESS(), RUNNING(), SUCCESS()])
        assert seq.execute(behavior_context) == Status.RUNNING

    def test_aborts_after_first_failure(self, behavior_context):
        counter = _Counting(Status.SUCCESS)
        seq = SequenceNode("s", [FAILURE(), counter])
        seq.execute(behavior_context)
        assert counter.calls == 0  # Never reached

    def test_all_children_executed_on_all_success(self, behavior_context):
        nodes = [_Counting(Status.SUCCESS) for _ in range(3)]
        seq = SequenceNode("s", nodes)
        seq.execute(behavior_context)
        assert all(n.calls == 1 for n in nodes)

    def test_empty_sequence_returns_success(self, behavior_context):
        assert SequenceNode("s", []).execute(behavior_context) == Status.SUCCESS


# ---------------------------------------------------------------------------
# FallbackNode
# ---------------------------------------------------------------------------

class TestFallbackNode:
    def test_all_failure_returns_failure(self, behavior_context):
        fb = FallbackNode("f", [FAILURE(), FAILURE(), FAILURE()])
        assert fb.execute(behavior_context) == Status.FAILURE

    def test_first_success_returns_success(self, behavior_context):
        fb = FallbackNode("f", [SUCCESS(), FAILURE(), FAILURE()])
        assert fb.execute(behavior_context) == Status.SUCCESS

    def test_short_circuits_after_first_success(self, behavior_context):
        counter = _Counting(Status.FAILURE)
        fb = FallbackNode("f", [SUCCESS(), counter])
        fb.execute(behavior_context)
        assert counter.calls == 0

    def test_running_stops_iteration(self, behavior_context):
        counter = _Counting(Status.SUCCESS)
        fb = FallbackNode("f", [FAILURE(), RUNNING(), counter])
        result = fb.execute(behavior_context)
        assert result == Status.RUNNING
        assert counter.calls == 0

    def test_empty_fallback_returns_failure(self, behavior_context):
        assert FallbackNode("f", []).execute(behavior_context) == Status.FAILURE


# ---------------------------------------------------------------------------
# CheckStuckCondition
# ---------------------------------------------------------------------------

class TestCheckStuckCondition:
    def test_success_when_stuck(self, behavior_context):
        # Fill memory with stuck pattern
        for _ in range(10):
            behavior_context.memory.record_action("forward", False, 15.0, 15.0)
        node = CheckStuckCondition()
        assert node.execute(behavior_context) == Status.SUCCESS

    def test_failure_when_not_stuck(self, behavior_context):
        node = CheckStuckCondition()
        assert node.execute(behavior_context) == Status.FAILURE


# ---------------------------------------------------------------------------
# CheckPathClear
# ---------------------------------------------------------------------------

class TestCheckPathClear:
    def test_success_when_path_clear(self, behavior_context):
        behavior_context.world_model.update_ultrasonic(50.0)
        node = CheckPathClear()
        assert node.execute(behavior_context) == Status.SUCCESS

    def test_failure_when_obstacle(self, behavior_context):
        behavior_context.world_model.update_ultrasonic(10.0)
        node = CheckPathClear()
        assert node.execute(behavior_context) == Status.FAILURE


# ---------------------------------------------------------------------------
# CheckFreeSpace
# ---------------------------------------------------------------------------

class TestCheckFreeSpace:
    def test_success_when_enough_free_space(self, behavior_context):
        # All directions clear → free_space = 1.0
        for d in ("front", "left", "right", "rear"):
            behavior_context.world_model.obstacles[d] = ObstacleInfo(None, 0.0, "unknown", False)
        node = CheckFreeSpace(min_score=0.3)
        assert node.execute(behavior_context) == Status.SUCCESS

    def test_failure_when_insufficient_free_space(self, behavior_context):
        # All blocked → free_space = 0.0
        for d in ("front", "left", "right", "rear"):
            behavior_context.world_model.obstacles[d] = ObstacleInfo(5.0, 0.9, "ultrasonic", True)
        node = CheckFreeSpace(min_score=0.5)
        assert node.execute(behavior_context) == Status.FAILURE

    def test_custom_threshold_respected(self, behavior_context):
        for d in ("front", "left", "right", "rear"):
            behavior_context.world_model.obstacles[d] = ObstacleInfo(None, 0.0, "unknown", False)
        # With very high threshold, should fail
        node = CheckFreeSpace(min_score=1.5)
        assert node.execute(behavior_context) == Status.FAILURE


# ---------------------------------------------------------------------------
# StopAction
# ---------------------------------------------------------------------------

class TestStopAction:
    def test_stop_calls_robot_execute(self, behavior_context):
        node = StopAction()
        result = node.execute(behavior_context)
        assert result == Status.SUCCESS
        assert ("stop", 0.1) in behavior_context.robot.calls


# ---------------------------------------------------------------------------
# TurnAction
# ---------------------------------------------------------------------------

class TestTurnAction:
    def test_turn_left_calls_robot(self, behavior_context):
        node = TurnAction("left", duration=0.5)
        result = node.execute(behavior_context)
        assert result == Status.SUCCESS
        assert any(action == "turn_left" for action, _ in behavior_context.robot.calls)

    def test_turn_right_calls_robot(self, behavior_context):
        node = TurnAction("right", duration=0.5)
        result = node.execute(behavior_context)
        assert result == Status.SUCCESS
        assert any(action == "turn_right" for action, _ in behavior_context.robot.calls)

    def test_turn_records_action_in_memory(self, behavior_context):
        node = TurnAction("left", duration=0.5)
        node.execute(behavior_context)
        recent = behavior_context.memory.get_recent_actions(5)
        assert "turn_left" in recent


# ---------------------------------------------------------------------------
# BackupAction
# ---------------------------------------------------------------------------

class TestBackupAction:
    def test_backup_calls_robot(self, behavior_context):
        node = BackupAction(duration=1.0)
        result = node.execute(behavior_context)
        assert result == Status.SUCCESS
        assert any(action == "backward" for action, _ in behavior_context.robot.calls)

    def test_backup_records_in_memory(self, behavior_context):
        node = BackupAction(duration=1.0)
        node.execute(behavior_context)
        assert "backward" in behavior_context.memory.get_recent_actions(5)


# ---------------------------------------------------------------------------
# MoveForwardAction
# ---------------------------------------------------------------------------

class TestMoveForwardAction:
    def test_forward_succeeds_when_path_clear(self, behavior_context):
        behavior_context.world_model.update_ultrasonic(50.0)
        node = MoveForwardAction(duration=0.5)
        result = node.execute(behavior_context)
        assert result == Status.SUCCESS
        assert any(action == "forward" for action, _ in behavior_context.robot.calls)

    def test_forward_fails_when_blocked(self, behavior_context):
        behavior_context.world_model.update_ultrasonic(5.0)
        node = MoveForwardAction(duration=0.5)
        result = node.execute(behavior_context)
        assert result == Status.FAILURE

    def test_forward_records_in_memory(self, behavior_context):
        behavior_context.world_model.update_ultrasonic(50.0)
        node = MoveForwardAction(duration=0.5)
        node.execute(behavior_context)
        assert "forward" in behavior_context.memory.get_recent_actions(5)


# ---------------------------------------------------------------------------
# SmartTurnAction
# ---------------------------------------------------------------------------

class TestSmartTurnAction:
    def test_smart_turn_executes_a_turn(self, behavior_context):
        node = SmartTurnAction(duration=0.5)
        result = node.execute(behavior_context)
        assert result == Status.SUCCESS
        assert any("turn" in action for action, _ in behavior_context.robot.calls)

    def test_smart_turn_uses_memory_preference(self, behavior_context):
        # Seed memory so left has higher success rate
        for _ in range(5):
            behavior_context.memory.record_action("turn_left", True)
            behavior_context.memory.record_action("turn_right", False)
        node = SmartTurnAction(duration=0.5)
        node.execute(behavior_context)
        assert any(action == "turn_left" for action, _ in behavior_context.robot.calls)


# ---------------------------------------------------------------------------
# RecoverySequence
# ---------------------------------------------------------------------------

class TestRecoverySequence:
    def test_light_recovery_at_level_0(self, behavior_context):
        assert behavior_context.memory.get_recovery_level() == 0
        node = RecoverySequence()
        result = node.execute(behavior_context)
        assert result in (Status.SUCCESS, Status.FAILURE)  # Runs without error

    def test_escape_recorded_in_memory(self, behavior_context):
        node = RecoverySequence()
        behavior_context.world_model.update_ultrasonic(50.0)
        node.execute(behavior_context)
        assert behavior_context.memory.escape_count == 1

    def test_recovery_level_escalates_on_repeat(self, behavior_context):
        node = RecoverySequence()
        behavior_context.world_model.update_ultrasonic(50.0)
        node.execute(behavior_context)
        node.execute(behavior_context)
        assert behavior_context.memory.get_recovery_level() >= 1


# ---------------------------------------------------------------------------
# Pre-built trees
# ---------------------------------------------------------------------------

class TestPrebuiltTrees:
    def test_exploration_tree_builds(self):
        tree = build_exploration_tree()
        assert tree is not None
        assert tree.name == "ExplorationRoot"

    def test_cautious_tree_builds(self):
        tree = build_cautious_exploration_tree()
        assert tree is not None
        assert tree.name == "CautiousExplorationRoot"

    def test_exploration_tree_executes_without_error(self, behavior_context):
        behavior_context.world_model.update_ultrasonic(50.0)
        tree = build_exploration_tree()
        result = tree.execute(behavior_context)
        assert result in Status

    def test_cautious_tree_executes_without_error(self, behavior_context):
        behavior_context.world_model.update_ultrasonic(50.0)
        tree = build_cautious_exploration_tree()
        result = tree.execute(behavior_context)
        assert result in Status

    def test_exploration_tree_config_overrides_applied(self):
        config = {
            "behavior_tree_settings": {
                "move_forward_duration_s": 2.5,
                "turn_duration_s": 0.6,
            }
        }
        tree = build_exploration_tree(config)
        assert tree is not None  # Does not crash with config
