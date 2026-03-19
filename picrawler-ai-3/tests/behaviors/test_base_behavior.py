"""
Tests for BaseBehavior — anti-loop detection, stuck detection, escape logic,
and ban management. All hardware/AI deps are mocked via conftest.py.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# conftest.py registers stubs for pyttsx3 / openai / hardware before imports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fill_actions(behavior, action: str, count: int) -> None:
    """Push `count` identical actions into the behavior's recent history."""
    for _ in range(count):
        behavior._recent_actions.append(action)


# ---------------------------------------------------------------------------
# _is_repeating
# ---------------------------------------------------------------------------

class TestIsRepeating:
    def test_not_repeating_with_no_history(self, base_behavior):
        assert not base_behavior._is_repeating("forward")

    def test_not_repeating_with_mixed_actions(self, base_behavior):
        for a in ["forward", "turn_left", "forward"]:
            base_behavior._recent_actions.append(a)
        assert not base_behavior._is_repeating("forward")

    def test_repeating_when_threshold_consecutive_same(self, base_behavior):
        fill_actions(base_behavior, "forward", base_behavior._repeat_threshold)
        assert base_behavior._is_repeating("forward")

    def test_not_repeating_one_short_of_threshold(self, base_behavior):
        fill_actions(base_behavior, "forward", base_behavior._repeat_threshold - 1)
        assert not base_behavior._is_repeating("forward")


# ---------------------------------------------------------------------------
# _is_oscillating
# ---------------------------------------------------------------------------

class TestIsOscillating:
    def test_not_oscillating_with_few_actions(self, base_behavior):
        for a in ["turn_left", "turn_right"]:
            base_behavior._recent_actions.append(a)
        assert not base_behavior._is_oscillating()

    def test_oscillating_with_abab_pattern(self, base_behavior):
        for a in ["turn_left", "turn_right", "turn_left", "turn_right"]:
            base_behavior._recent_actions.append(a)
        assert base_behavior._is_oscillating()

    def test_not_oscillating_with_uniform_actions(self, base_behavior):
        fill_actions(base_behavior, "forward", 4)
        assert not base_behavior._is_oscillating()


# ---------------------------------------------------------------------------
# _ban / _is_banned
# ---------------------------------------------------------------------------

class TestBanMechanism:
    def test_action_not_banned_initially(self, base_behavior):
        assert not base_behavior._is_banned("forward")

    def test_action_banned_after_ban_call(self, base_behavior):
        base_behavior._ban("forward", seconds=10.0)
        assert base_behavior._is_banned("forward")

    def test_action_not_banned_after_expiry(self, base_behavior):
        base_behavior._ban("forward", seconds=0.001)
        time.sleep(0.01)
        assert not base_behavior._is_banned("forward")

    def test_banning_one_action_does_not_ban_others(self, base_behavior):
        base_behavior._ban("forward", seconds=10.0)
        assert not base_behavior._is_banned("turn_left")


# ---------------------------------------------------------------------------
# _pick_non_banned_fallback
# ---------------------------------------------------------------------------

class TestPickNonBannedFallback:
    def test_returns_different_action_when_preferred_banned(self, base_behavior):
        base_behavior._ban("forward", seconds=10.0)
        fallback = base_behavior._pick_non_banned_fallback("forward")
        assert fallback != "forward"

    def test_returns_stop_as_last_resort(self, base_behavior):
        # Ban everything except stop
        for action in ["forward", "backward", "turn_left", "turn_right"]:
            base_behavior._ban(action, seconds=10.0)
        fallback = base_behavior._pick_non_banned_fallback("forward")
        assert fallback == "stop"

    def test_returns_first_available_in_order(self, base_behavior):
        # For "forward", preferred fallbacks are backward, turn_left, turn_right, stop
        # Ban only backward
        base_behavior._ban("backward", seconds=10.0)
        fallback = base_behavior._pick_non_banned_fallback("forward")
        assert fallback == "turn_left"


# ---------------------------------------------------------------------------
# postprocess_action — normal flow
# ---------------------------------------------------------------------------

class TestPostprocessNormal:
    def test_clean_action_passes_through(self, base_behavior):
        result = base_behavior.postprocess_action("forward")
        assert result == "forward" or (isinstance(result, tuple) and result[0] == "forward")

    def test_stop_passes_through_immediately(self, base_behavior):
        result = base_behavior.postprocess_action("stop")
        assert result == "stop"

    def test_banned_action_is_replaced(self, base_behavior):
        base_behavior._ban("forward", seconds=10.0)
        # Add enough history to pass threshold check
        fill_actions(base_behavior, "forward", base_behavior._repeat_threshold)
        result = base_behavior.postprocess_action("forward")
        action = result[0] if isinstance(result, tuple) else result
        assert action != "forward"


# ---------------------------------------------------------------------------
# postprocess_action — repeat detection triggers escape
# ---------------------------------------------------------------------------

class TestPostprocessRepeat:
    def test_repeat_triggers_escape_action(self, base_behavior):
        # Fill history so the action is already at threshold
        fill_actions(base_behavior, "turn_left", base_behavior._repeat_threshold)
        result = base_behavior.postprocess_action("turn_left")
        # Escape should return a tuple (action, duration)
        assert isinstance(result, tuple)
        escape_action, duration = result
        assert escape_action in ("backward", "forward", "turn_right", "turn_left", "stop")
        assert duration > 0

    def test_repeat_bans_turn_actions(self, base_behavior):
        fill_actions(base_behavior, "turn_right", base_behavior._repeat_threshold)
        base_behavior.postprocess_action("turn_right")
        assert base_behavior._is_banned("turn_right")


# ---------------------------------------------------------------------------
# postprocess_action — oscillation detection
# ---------------------------------------------------------------------------

class TestPostprocessOscillation:
    def test_oscillation_triggers_escape(self, base_behavior):
        for a in ["turn_left", "turn_right", "turn_left", "turn_right"]:
            base_behavior._recent_actions.append(a)
        result = base_behavior.postprocess_action("turn_left")
        assert isinstance(result, tuple)

    def test_oscillation_bans_both_turns(self, base_behavior):
        for a in ["turn_left", "turn_right", "turn_left", "turn_right"]:
            base_behavior._recent_actions.append(a)
        base_behavior.postprocess_action("turn_left")
        assert base_behavior._is_banned("turn_left")
        assert base_behavior._is_banned("turn_right")


# ---------------------------------------------------------------------------
# postprocess_action — obstacle override
# ---------------------------------------------------------------------------

class TestPostprocessObstacleOverride:
    def test_obstacle_overrides_forward(self, obstacle_robot, minimal_config, mock_camera, mock_ai):
        from behaviors.base_behavior import BaseBehavior

        class _C(BaseBehavior):
            name = "test"

        b = _C(
            config=minimal_config,
            robot=obstacle_robot,
            camera=mock_camera,
            ai=mock_ai,
            duration_minutes=0.01,
        )
        result = b.postprocess_action("forward")
        # Should redirect to a turn, not continue forward
        action = result[0] if isinstance(result, tuple) else result
        assert action in ("turn_left", "turn_right")

    def test_no_obstacle_passes_forward(self, base_behavior):
        result = base_behavior.postprocess_action("forward")
        action = result[0] if isinstance(result, tuple) else result
        assert action == "forward"

    def test_consecutive_overrides_trigger_escape(self, obstacle_robot, minimal_config, mock_camera, mock_ai):
        from behaviors.base_behavior import BaseBehavior

        class _C(BaseBehavior):
            name = "test"

        b = _C(
            config=minimal_config,
            robot=obstacle_robot,
            camera=mock_camera,
            ai=mock_ai,
            duration_minutes=0.01,
        )
        max_overrides = b._max_obstacle_overrides
        # Trigger max_overrides times to activate escape
        for _ in range(max_overrides):
            b.postprocess_action("forward")
        # Counter should reset after escape
        assert b._consecutive_obstacle_overrides == 0


# ---------------------------------------------------------------------------
# _escape
# ---------------------------------------------------------------------------

class TestEscape:
    def test_escape_returns_tuple(self, base_behavior):
        result = base_behavior._escape("forward", reason="test")
        assert isinstance(result, tuple)
        action, duration = result
        assert action in ("stop", "backward", "forward", "turn_left", "turn_right")
        assert duration > 0

    def test_escape_clears_recent_actions(self, base_behavior):
        fill_actions(base_behavior, "forward", 5)
        base_behavior._escape("forward", reason="test")
        assert len(base_behavior._recent_actions) == 1  # only the escape action

    def test_escape_clears_scene_signatures(self, base_behavior):
        for _ in range(5):
            base_behavior._recent_scene_sigs.append("sig")
        base_behavior._escape("forward", reason="test")
        assert len(base_behavior._recent_scene_sigs) == 0

    def test_escape_increments_stage(self, base_behavior):
        stage_before = base_behavior._escape_stage
        base_behavior._escape("forward", reason="test")
        assert base_behavior._escape_stage == stage_before + 1

    def test_escape_cooldown_returns_simple_action(self, base_behavior):
        base_behavior._last_escape_at = time.time() + 100  # Far in the future
        result = base_behavior._escape("turn_left", reason="test")
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Scene stagnation
# ---------------------------------------------------------------------------

class TestSceneStagnation:
    def test_not_stagnant_with_varied_scenes(self, base_behavior):
        for sig in ["a", "b", "c", "d"]:
            base_behavior._recent_scene_sigs.append(sig)
        assert not base_behavior._is_scene_stagnant()

    def test_stagnant_with_repeated_scene(self, base_behavior):
        for _ in range(base_behavior._scene_repeat_threshold):
            base_behavior._recent_scene_sigs.append("same_scene")
        assert base_behavior._is_scene_stagnant()

    def test_not_stagnant_below_threshold(self, base_behavior):
        for _ in range(base_behavior._scene_repeat_threshold - 1):
            base_behavior._recent_scene_sigs.append("same_scene")
        assert not base_behavior._is_scene_stagnant()


# ---------------------------------------------------------------------------
# _scene_signature
# ---------------------------------------------------------------------------

class TestSceneSignature:
    def test_signature_is_string(self, base_behavior):
        analysis = MagicMock()
        analysis.description = "clear hallway"
        analysis.hazards = ["wall"]
        analysis.objects = ["chair", "table"]
        sig = base_behavior._scene_signature(analysis)
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_same_scene_same_signature(self, base_behavior):
        class _A:
            description = "open room"
            hazards = ["wall"]
            objects = ["door"]

        sig1 = base_behavior._scene_signature(_A())
        sig2 = base_behavior._scene_signature(_A())
        assert sig1 == sig2

    def test_different_scenes_different_signatures(self, base_behavior):
        class _A:
            description = "blocked corridor"
            hazards = ["wall"]
            objects = []

        class _B:
            description = "open space"
            hazards = []
            objects = ["chair"]

        assert base_behavior._scene_signature(_A()) != base_behavior._scene_signature(_B())
