"""
Tests for WorldModel sensor fusion logic.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.world_model import WorldModel, ObstacleInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_depth_map(front: float = 0.8, left: float = 0.8, right: float = 0.8) -> MagicMock:
    """Create a mock DepthMap with configurable directional values."""
    dm = MagicMock()
    dm.get_directional_depths.return_value = {
        "front": front,
        "left": left,
        "right": right,
    }
    return dm


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_all_directions_unknown(self, world_model):
        for direction in ("front", "left", "right", "rear"):
            obs = world_model.obstacles[direction]
            assert not obs.detected
            assert obs.confidence == 0.0

    def test_free_space_score_starts_at_half(self, world_model):
        assert world_model.free_space_score == 0.5


# ---------------------------------------------------------------------------
# update_ultrasonic
# ---------------------------------------------------------------------------

class TestUpdateUltrasonic:
    def test_obstacle_detected_below_threshold(self, world_model):
        world_model.update_ultrasonic(10.0)  # threshold is 20 cm
        assert world_model.obstacles["front"].detected
        assert world_model.obstacles["front"].distance_cm == pytest.approx(10.0)

    def test_no_obstacle_above_threshold(self, world_model):
        world_model.update_ultrasonic(50.0)
        assert not world_model.obstacles["front"].detected

    def test_exactly_at_threshold_not_detected(self, world_model):
        world_model.update_ultrasonic(20.0)
        assert not world_model.obstacles["front"].detected

    def test_high_confidence_from_ultrasonic(self, world_model):
        world_model.update_ultrasonic(30.0)
        assert world_model.obstacles["front"].confidence == pytest.approx(0.95)

    def test_source_set_to_ultrasonic(self, world_model):
        world_model.update_ultrasonic(30.0)
        assert world_model.obstacles["front"].source == "ultrasonic"

    def test_none_distance_sets_unavailable(self, world_model):
        world_model.update_ultrasonic(None)
        assert world_model.obstacles["front"].source == "ultrasonic_unavailable"
        assert not world_model.obstacles["front"].detected

    def test_last_update_advances(self, world_model):
        before = world_model.last_update
        world_model.update_ultrasonic(30.0)
        assert world_model.last_update >= before


# ---------------------------------------------------------------------------
# update_vision
# ---------------------------------------------------------------------------

class TestUpdateVision:
    def test_vision_only_detects_obstacle_from_hazards(self, world_model):
        world_model.update_vision([], ["wall"], "blocked")
        assert world_model.obstacles["front"].detected

    def test_vision_only_no_obstacle_with_no_hazards(self, world_model):
        world_model.update_vision(["chair"], [], "clear path")
        assert not world_model.obstacles["front"].detected

    def test_vision_confidence_higher_with_obstacle(self, world_model):
        world_model.update_vision([], ["wall"], "obstacle")
        no_obstacle_confidence = world_model.obstacles["front"].confidence

        world_model2 = WorldModel()
        world_model2.update_vision([], [], "clear")
        clear_confidence = world_model2.obstacles["front"].confidence

        assert no_obstacle_confidence > clear_confidence

    def test_vision_recognised_hazard_types(self, world_model):
        for hazard in ("wall", "obstacle", "barrier", "furniture"):
            wm = WorldModel()
            wm.update_vision([], [hazard], "desc")
            assert wm.obstacles["front"].detected, f"Should detect hazard '{hazard}'"

    def test_ultrasonic_takes_priority_over_vision(self, world_model):
        world_model.update_ultrasonic(30.0)  # no obstacle
        world_model.update_vision([], ["wall"], "blocked")  # vision says obstacle
        # Ultrasonic already set to no-obstacle; should stay with ultrasonic source
        assert world_model.obstacles["front"].source in ("ultrasonic", "fused")

    def test_fusion_increases_confidence_when_both_agree(self, world_model):
        world_model.update_ultrasonic(10.0)   # obstacle at 10 cm
        world_model.update_vision([], ["wall"], "blocked")
        assert world_model.obstacles["front"].confidence > 0.95
        assert "fused" in world_model.obstacles["front"].source

    def test_visual_description_stored(self, world_model):
        world_model.update_vision(["chair"], [], "open room with chair")
        assert world_model.visual_description == "open room with chair"

    def test_visual_objects_stored(self, world_model):
        world_model.update_vision(["chair", "table"], [], "")
        assert "chair" in world_model.visual_objects
        assert "table" in world_model.visual_objects


# ---------------------------------------------------------------------------
# update_depth
# ---------------------------------------------------------------------------

class TestUpdateDepth:
    def test_none_depth_map_does_nothing(self, world_model):
        world_model.update_depth(None)  # should not raise

    def test_close_depth_sets_obstacle(self, world_model):
        # depth value 0.0 → distance ~10 cm → below 20 cm threshold
        dm = make_depth_map(front=0.0)
        world_model.update_depth(dm)
        assert world_model.obstacles["front"].detected

    def test_far_depth_no_obstacle(self, world_model):
        # depth value 1.0 → distance ~200 cm → above 20 cm threshold
        dm = make_depth_map(front=1.0)
        world_model.update_depth(dm)
        assert not world_model.obstacles["front"].detected

    def test_depth_updates_left_and_right(self, world_model):
        dm = make_depth_map(front=1.0, left=0.0, right=0.0)
        world_model.update_depth(dm)
        assert world_model.obstacles["left"].detected
        assert world_model.obstacles["right"].detected
        assert not world_model.obstacles["front"].detected

    def test_ultrasonic_takes_priority_over_depth(self, world_model):
        world_model.update_ultrasonic(50.0)  # clear front
        dm = make_depth_map(front=0.0)       # depth says obstacle
        world_model.update_depth(dm)
        # Source should remain ultrasonic-based
        assert "ultrasonic" in world_model.obstacles["front"].source


# ---------------------------------------------------------------------------
# is_safe_to_move
# ---------------------------------------------------------------------------

class TestIsSafeToMove:
    def test_safe_when_no_obstacle(self, world_model):
        world_model.update_ultrasonic(50.0)
        assert world_model.is_safe_to_move("forward")

    def test_not_safe_when_obstacle_detected(self, world_model):
        world_model.update_ultrasonic(10.0)
        assert not world_model.is_safe_to_move("forward")

    def test_direction_aliases_resolve(self, world_model):
        world_model.update_ultrasonic(10.0)
        for alias in ("forward", "ahead"):
            assert not world_model.is_safe_to_move(alias)

    def test_unknown_direction_defaults_to_front(self, world_model):
        world_model.update_ultrasonic(10.0)
        # "fly" is not a known direction — should default to front
        assert not world_model.is_safe_to_move("fly")

    def test_low_confidence_obstacle_considered_safe(self, world_model):
        # Confidence 0.3 < 0.5 threshold → safe
        world_model.obstacles["front"] = ObstacleInfo(
            distance_cm=5.0, confidence=0.3, source="vision", detected=True
        )
        assert world_model.is_safe_to_move("forward")


# ---------------------------------------------------------------------------
# get_best_direction
# ---------------------------------------------------------------------------

class TestGetBestDirection:
    def test_prefers_forward_when_all_clear(self, world_model):
        for d in ("front", "left", "right", "rear"):
            world_model.obstacles[d] = ObstacleInfo(
                distance_cm=100.0, confidence=0.9, source="ultrasonic", detected=False
            )
        assert world_model.get_best_direction() == "forward"

    def test_avoids_blocked_front(self, world_model):
        world_model.obstacles["front"] = ObstacleInfo(
            distance_cm=5.0, confidence=0.9, source="ultrasonic", detected=True
        )
        world_model.obstacles["left"] = ObstacleInfo(
            distance_cm=200.0, confidence=0.9, source="ultrasonic", detected=False
        )
        world_model.obstacles["right"] = ObstacleInfo(
            distance_cm=200.0, confidence=0.9, source="ultrasonic", detected=False
        )
        world_model.obstacles["rear"] = ObstacleInfo(
            distance_cm=5.0, confidence=0.9, source="ultrasonic", detected=True
        )
        direction = world_model.get_best_direction()
        assert direction in ("turn_left", "turn_right")


# ---------------------------------------------------------------------------
# calculate_free_space
# ---------------------------------------------------------------------------

class TestCalculateFreeSpace:
    def test_all_clear_gives_full_score(self, world_model):
        for d in ("front", "left", "right", "rear"):
            world_model.obstacles[d] = ObstacleInfo(None, 0.0, "unknown", False)
        score = world_model.calculate_free_space()
        assert score == pytest.approx(1.0)

    def test_all_blocked_gives_zero_score(self, world_model):
        for d in ("front", "left", "right", "rear"):
            world_model.obstacles[d] = ObstacleInfo(5.0, 0.9, "ultrasonic", True)
        score = world_model.calculate_free_space()
        assert score == pytest.approx(0.0)

    def test_partial_block_gives_partial_score(self, world_model):
        world_model.obstacles["front"] = ObstacleInfo(5.0, 0.9, "ultrasonic", True)
        world_model.obstacles["left"] = ObstacleInfo(None, 0.0, "unknown", False)
        world_model.obstacles["right"] = ObstacleInfo(None, 0.0, "unknown", False)
        world_model.obstacles["rear"] = ObstacleInfo(None, 0.0, "unknown", False)
        score = world_model.calculate_free_space()
        assert score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# to_dict / __str__
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_has_required_keys(self, world_model):
        d = world_model.to_dict()
        assert "obstacles" in d
        assert "vision" in d
        assert "free_space_score" in d
        assert "best_direction" in d

    def test_str_representation_is_string(self, world_model):
        assert isinstance(str(world_model), str)
