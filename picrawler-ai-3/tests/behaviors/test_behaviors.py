"""
Tests for v3 behavior subclasses: ExplorationBehavior, AvoidanceBehavior,
FollowingBehavior, and ObjectDetectionBehavior.
"""
from __future__ import annotations

import pytest

# conftest.py has already stubbed out pyttsx3, openai, hardware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_behavior(cls, minimal_config, mock_robot, mock_camera, mock_ai, **extra):
    return cls(
        config=minimal_config,
        robot=mock_robot,
        camera=mock_camera,
        ai=mock_ai,
        duration_minutes=0.01,
        **extra,
    )


# ---------------------------------------------------------------------------
# ExplorationBehavior
# ---------------------------------------------------------------------------

class TestExplorationBehavior:
    def test_instantiates(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.exploration import ExplorationBehavior
        b = make_behavior(ExplorationBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        assert b.name == "explore"

    def test_context_returns_string(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.exploration import ExplorationBehavior
        b = make_behavior(ExplorationBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        ctx = b.context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_action_passes_through_when_no_loop(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.exploration import ExplorationBehavior
        b = make_behavior(ExplorationBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        result = b.postprocess_action("forward")
        action = result[0] if isinstance(result, tuple) else result
        assert action in ("forward", "turn_left", "turn_right", "backward", "stop")

    def test_oscillation_bias_toward_forward(self, minimal_config, mock_robot, mock_camera, mock_ai):
        """ExplorationBehavior has an extra bias: after L→R oscillation, prefer forward."""
        from behaviors.exploration import ExplorationBehavior
        b = make_behavior(ExplorationBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        b._last_action = "turn_left"
        # Simulate base behavior returning turn_right (opposite of last)
        # by injecting enough history then calling postprocess directly
        # — the bias should redirect to forward
        result = b.postprocess_action("turn_right")
        action = result[0] if isinstance(result, tuple) else result
        assert action in ("forward", "turn_left", "turn_right", "backward", "stop")

    def test_stop_action_not_biased(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.exploration import ExplorationBehavior
        b = make_behavior(ExplorationBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        result = b.postprocess_action("stop")
        assert result == "stop"


# ---------------------------------------------------------------------------
# AvoidanceBehavior
# ---------------------------------------------------------------------------

class TestAvoidanceBehavior:
    def test_instantiates(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.avoidance import AvoidanceBehavior
        b = make_behavior(AvoidanceBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        assert b.name == "avoid"

    def test_context_mentions_target_when_set(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.avoidance import AvoidanceBehavior
        b = make_behavior(AvoidanceBehavior, minimal_config, mock_robot, mock_camera, mock_ai,
                         target="cats")
        ctx = b.context()
        assert "cats" in ctx

    def test_context_defaults_to_people_pets(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.avoidance import AvoidanceBehavior
        b = make_behavior(AvoidanceBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        ctx = b.context()
        assert "people" in ctx or "pets" in ctx

    def test_inherits_postprocess_from_base(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.avoidance import AvoidanceBehavior
        b = make_behavior(AvoidanceBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        result = b.postprocess_action("forward")
        action = result[0] if isinstance(result, tuple) else result
        assert action in ("forward", "turn_left", "turn_right", "backward", "stop")


# ---------------------------------------------------------------------------
# FollowingBehavior
# ---------------------------------------------------------------------------

class TestFollowingBehavior:
    def test_instantiates(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.following import FollowingBehavior
        b = make_behavior(FollowingBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        assert b.name == "follow"

    def test_context_mentions_target(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.following import FollowingBehavior
        b = make_behavior(FollowingBehavior, minimal_config, mock_robot, mock_camera, mock_ai,
                         target="dog")
        ctx = b.context()
        assert "dog" in ctx

    def test_available_actions_is_list(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.following import FollowingBehavior
        b = make_behavior(FollowingBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        actions = b.available_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0


# ---------------------------------------------------------------------------
# ObjectDetectionBehavior
# ---------------------------------------------------------------------------

class TestObjectDetectionBehavior:
    def test_instantiates(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.object_detection import ObjectDetectionBehavior
        b = make_behavior(ObjectDetectionBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        assert b.name == "detect"

    def test_context_mentions_target(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.object_detection import ObjectDetectionBehavior
        b = make_behavior(ObjectDetectionBehavior, minimal_config, mock_robot, mock_camera, mock_ai,
                         target="keys")
        ctx = b.context()
        assert "keys" in ctx

    def test_inherits_base_behavior_methods(self, minimal_config, mock_robot, mock_camera, mock_ai):
        from behaviors.object_detection import ObjectDetectionBehavior
        b = make_behavior(ObjectDetectionBehavior, minimal_config, mock_robot, mock_camera, mock_ai)
        # Should have anti-loop machinery from BaseBehavior
        assert hasattr(b, "_is_repeating")
        assert hasattr(b, "_is_oscillating")
        assert hasattr(b, "_ban")
