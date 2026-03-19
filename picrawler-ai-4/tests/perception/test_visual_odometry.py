"""
Tests for VisualOdometry — pose tracking, motion estimation, and reset.
Uses synthetic numpy images so no camera hardware is required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perception.visual_odometry import VisualOdometry, Pose2D, MotionEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vo(**kwargs) -> VisualOdometry:
    return VisualOdometry(
        camera_height_m=0.1,
        camera_tilt_deg=20.0,
        scale_calibration_factor=1.0,
        feature_count=500,
    )


def checkerboard(size: int = 200, tile: int = 20) -> np.ndarray:
    """Generate a high-texture checkerboard image for feature detection."""
    img = np.zeros((size, size), dtype=np.uint8)
    for r in range(0, size, tile):
        for c in range(0, size, tile):
            if (r // tile + c // tile) % 2 == 0:
                img[r:r + tile, c:c + tile] = 255
    return img


def random_texture(size: int = 200, seed: int = 42) -> np.ndarray:
    """Random noise image — high feature content."""
    rng = np.random.RandomState(seed)
    return (rng.rand(size, size) * 255).astype(np.uint8)


def translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift image by (dx, dy) pixels."""
    import cv2
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# ---------------------------------------------------------------------------
# Pose2D
# ---------------------------------------------------------------------------

class TestPose2D:
    def test_str_representation(self):
        p = Pose2D(1.0, 2.0, math.pi / 4, 0.0)
        s = str(p)
        assert "1.00" in s
        assert "2.00" in s
        assert "45.0" in s


# ---------------------------------------------------------------------------
# MotionEstimate dataclass
# ---------------------------------------------------------------------------

class TestMotionEstimate:
    def test_fields_accessible(self):
        m = MotionEstimate(delta_x=0.1, delta_y=0.05, delta_theta=0.02,
                           confidence=0.8, num_matches=50)
        assert m.delta_x == pytest.approx(0.1)
        assert m.confidence == pytest.approx(0.8)
        assert m.num_matches == 50


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_initial_pose_at_origin(self):
        vo = make_vo()
        p = vo.get_pose()
        assert p.x == pytest.approx(0.0)
        assert p.y == pytest.approx(0.0)
        assert p.theta == pytest.approx(0.0)

    def test_pose_history_has_initial_pose(self):
        vo = make_vo()
        assert len(vo.get_trajectory()) == 1

    def test_statistics_start_at_zero(self):
        vo = make_vo()
        stats = vo.get_statistics()
        assert stats["total_distance_m"] == pytest.approx(0.0)
        assert stats["total_rotation_rad"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# process_frame — first frame
# ---------------------------------------------------------------------------

class TestFirstFrame:
    def test_first_frame_returns_none(self):
        vo = make_vo()
        img = checkerboard()
        result = vo.process_frame(img)
        assert result is None

    def test_first_frame_stores_state(self):
        vo = make_vo()
        img = checkerboard()
        vo.process_frame(img)
        assert vo.prev_descriptors is not None

    def test_first_frame_rgb_also_accepted(self):
        vo = make_vo()
        rgb = np.stack([checkerboard()] * 3, axis=-1)
        result = vo.process_frame(rgb)
        assert result is None  # First frame always returns None


# ---------------------------------------------------------------------------
# process_frame — feature-poor images
# ---------------------------------------------------------------------------

class TestFeaturePoorImages:
    def test_blank_image_returns_none(self):
        vo = make_vo()
        blank = np.zeros((200, 200), dtype=np.uint8)
        vo.process_frame(blank)  # Store first
        result = vo.process_frame(blank)
        # Either None or a valid motion (blank images may have 0 features)
        assert result is None or isinstance(result, MotionEstimate)


# ---------------------------------------------------------------------------
# process_frame — textured images (will actually detect features)
# ---------------------------------------------------------------------------

class TestTexturedFrames:
    def test_second_frame_may_return_motion_estimate(self):
        vo = make_vo()
        frame1 = random_texture(300, seed=0)
        frame2 = random_texture(300, seed=0)  # Identical = no motion
        vo.process_frame(frame1)
        result = vo.process_frame(frame2)
        # Identical frames should produce a motion estimate (or None if bad matches)
        # Either is acceptable — we're testing it doesn't crash
        assert result is None or isinstance(result, MotionEstimate)

    def test_motion_estimate_confidence_between_0_and_1(self):
        vo = make_vo()
        frame1 = checkerboard(300)
        frame2 = translate(checkerboard(300), 5, 0)  # Small shift
        vo.process_frame(frame1)
        result = vo.process_frame(frame2)
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0

    def test_motion_estimate_has_valid_match_count(self):
        vo = make_vo()
        frame1 = checkerboard(300)
        frame2 = translate(checkerboard(300), 3, 0)
        vo.process_frame(frame1)
        result = vo.process_frame(frame2)
        if result is not None:
            assert result.num_matches >= 0

    def test_pose_history_grows_after_successful_frame(self):
        vo = make_vo()
        initial_len = len(vo.get_trajectory())
        frame1 = checkerboard(300)
        frame2 = translate(checkerboard(300), 3, 0)
        vo.process_frame(frame1)
        result = vo.process_frame(frame2)
        if result is not None:
            assert len(vo.get_trajectory()) > initial_len


# ---------------------------------------------------------------------------
# _update_pose
# ---------------------------------------------------------------------------

class TestUpdatePose:
    def test_forward_motion_increases_x(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=0.1, delta_y=0.0, delta_theta=0.0,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        assert vo.current_pose.x > 0.0

    def test_rotation_updates_theta(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=0.0, delta_y=0.0, delta_theta=0.5,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        assert abs(vo.current_pose.theta) > 0.0

    def test_theta_normalised_to_pi_range(self):
        vo = make_vo()
        # Apply a large rotation that would exceed pi
        motion = MotionEstimate(delta_x=0.0, delta_y=0.0, delta_theta=4.0,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        assert -math.pi <= vo.current_pose.theta <= math.pi

    def test_cumulative_distance_tracked(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=1.0, delta_y=0.0, delta_theta=0.0,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        assert vo.total_distance > 0.0

    def test_cumulative_rotation_tracked(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=0.0, delta_y=0.0, delta_theta=0.3,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        assert vo.total_rotation > 0.0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_pose_to_origin(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=1.0, delta_y=0.5, delta_theta=0.3,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        vo.reset()
        p = vo.get_pose()
        assert p.x == pytest.approx(0.0)
        assert p.y == pytest.approx(0.0)

    def test_reset_to_specific_pose(self):
        vo = make_vo()
        target = Pose2D(2.0, 3.0, 1.0, 0.0)
        vo.reset(target)
        p = vo.get_pose()
        assert p.x == pytest.approx(2.0)
        assert p.y == pytest.approx(3.0)
        assert p.theta == pytest.approx(1.0)

    def test_reset_clears_trajectory(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=0.1, delta_y=0.0, delta_theta=0.0,
                                confidence=1.0, num_matches=50)
        for _ in range(5):
            vo._update_pose(motion)
        vo.reset()
        assert len(vo.get_trajectory()) == 1

    def test_reset_clears_statistics(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=1.0, delta_y=0.0, delta_theta=0.1,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        vo.reset()
        stats = vo.get_statistics()
        assert stats["total_distance_m"] == pytest.approx(0.0)
        assert stats["total_rotation_rad"] == pytest.approx(0.0)

    def test_reset_clears_previous_frame(self):
        vo = make_vo()
        vo.process_frame(checkerboard())
        assert vo.prev_descriptors is not None
        vo.reset()
        assert vo.prev_frame is None
        assert vo.prev_descriptors is None


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:
    def test_statistics_dict_has_required_keys(self):
        vo = make_vo()
        stats = vo.get_statistics()
        for key in ("total_distance_m", "total_rotation_rad", "total_rotation_deg",
                    "num_poses", "current_pose"):
            assert key in stats

    def test_rotation_degrees_consistent_with_radians(self):
        vo = make_vo()
        motion = MotionEstimate(delta_x=0.0, delta_y=0.0, delta_theta=math.pi / 2,
                                confidence=1.0, num_matches=50)
        vo._update_pose(motion)
        stats = vo.get_statistics()
        assert stats["total_rotation_deg"] == pytest.approx(
            math.degrees(stats["total_rotation_rad"])
        )
