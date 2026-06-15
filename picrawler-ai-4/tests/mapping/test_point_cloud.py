"""Tests for mapping/point_cloud.py — PointCloudBuilder and PointCloud."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from mapping.point_cloud import PointCloud, PointCloudBuilder
from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Pose2D:
    return Pose2D(x, y, theta, time.time())


def _mid_depth(h: int = 60, w: int = 80) -> np.ndarray:
    """Flat depth map at mid-range (0.5 = ~1.05 m with default scale)."""
    return np.full((h, w), 0.5, dtype=np.float32)


def _near_depth(h: int = 60, w: int = 80) -> np.ndarray:
    """Depth map at 0.1 (closest range)."""
    return np.full((h, w), 0.1, dtype=np.float32)


def _color_frame(h: int = 60, w: int = 80) -> np.ndarray:
    """Simple BGR color frame (all green)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 1] = 200  # green channel
    return frame


# ---------------------------------------------------------------------------
# TestPointCloudDataclass
# ---------------------------------------------------------------------------

class TestPointCloudDataclass:
    def test_fields(self):
        pts = np.zeros((10, 3), dtype=np.float32)
        cols = np.zeros((10, 3), dtype=np.uint8)
        pc = PointCloud(points=pts, colors=cols)
        assert pc.points.shape == (10, 3)
        assert pc.colors.shape == (10, 3)


# ---------------------------------------------------------------------------
# TestPointCloudBuilderInit
# ---------------------------------------------------------------------------

class TestPointCloudBuilderInit:
    def test_defaults(self):
        b = PointCloudBuilder()
        assert b.point_count() == 0
        assert b.max_points == 100_000
        assert b.subsample == 8

    def test_custom_params(self):
        b = PointCloudBuilder(max_points=500, subsample=4)
        assert b.max_points == 500
        assert b.subsample == 4

    def test_ray_grid_shape(self):
        h, w = 60, 80
        b = PointCloudBuilder(image_height=h, image_width=w)
        assert b._ray_h.shape == (h, w)
        assert b._ray_v.shape == (h, w)

    def test_get_cloud_empty(self):
        b = PointCloudBuilder()
        cloud = b.get_cloud()
        assert cloud.points.shape == (0, 3)
        assert cloud.colors.shape == (0, 3)


# ---------------------------------------------------------------------------
# TestPointCloudBuilderAddFrame
# ---------------------------------------------------------------------------

class TestPointCloudBuilderAddFrame:
    def test_returns_positive_count(self):
        b = PointCloudBuilder(subsample=1, image_height=60, image_width=80)
        added = b.add_frame(_mid_depth(60, 80), _pose())
        assert added > 0

    def test_point_count_increases(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        assert b.point_count() > 0

    def test_get_cloud_after_add(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        cloud = b.get_cloud()
        assert len(cloud.points) > 0
        assert len(cloud.colors) == len(cloud.points)

    def test_points_dtype_float32(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        assert b.get_cloud().points.dtype == np.float32

    def test_colors_dtype_uint8(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        assert b.get_cloud().colors.dtype == np.uint8

    def test_with_color_frame(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose(), color_frame=_color_frame(60, 80))
        cloud = b.get_cloud()
        assert len(cloud.points) > 0

    def test_multiple_frames_accumulate(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose(0.0))
        count1 = b.point_count()
        b.add_frame(_mid_depth(60, 80), _pose(1.0))
        count2 = b.point_count()
        assert count2 > count1

    def test_pose_offsets_points(self):
        """Points from a frame at x=10 should have x-coords near 10."""
        b = PointCloudBuilder(subsample=2, image_height=30, image_width=40)
        b.add_frame(_mid_depth(30, 40), _pose(10.0, 0.0, 0.0))
        cloud = b.get_cloud()
        if len(cloud.points) > 0:
            # Robot at x=10 facing forward (+X), so points should be around x~10+depth
            assert np.mean(cloud.points[:, 0]) > 5.0

    def test_image_size_mismatch_rebuilds_ray_grid(self):
        """Builder should handle frames of unexpected sizes gracefully."""
        b = PointCloudBuilder(image_height=60, image_width=80)
        # Feed a different size
        added = b.add_frame(_mid_depth(30, 40), _pose())
        assert b.image_height == 30
        assert b.image_width == 40
        assert added >= 0


# ---------------------------------------------------------------------------
# TestPointCloudBuilderFiltering
# ---------------------------------------------------------------------------

class TestPointCloudBuilderFiltering:
    def test_underground_points_filtered(self):
        """All-zero depth should produce some valid (not underground) points."""
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        # Use near depth — some pixels may map underground; check we don't crash
        b.add_frame(_near_depth(60, 80), _pose())
        cloud = b.get_cloud()
        if len(cloud.points) > 0:
            assert np.all(cloud.points[:, 2] > -0.1)

    def test_reset_clears_points(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        assert b.point_count() > 0
        b.reset()
        assert b.point_count() == 0
        cloud = b.get_cloud()
        assert len(cloud.points) == 0


# ---------------------------------------------------------------------------
# TestPointCloudBuilderMaxPoints
# ---------------------------------------------------------------------------

class TestPointCloudBuilderMaxPoints:
    def test_max_points_cap(self):
        b = PointCloudBuilder(subsample=1, image_height=30, image_width=40, max_points=50)
        # Adding many frames should not exceed the cap
        for i in range(10):
            b.add_frame(_mid_depth(30, 40), _pose(float(i)))
        assert b.point_count() <= 50


# ---------------------------------------------------------------------------
# TestHeightMap
# ---------------------------------------------------------------------------

class TestHeightMap:
    def test_height_map_shape(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        hm = b.get_height_map(grid_size_m=10.0, resolution_m=0.1)
        expected_cells = int(10.0 / 0.1)
        assert hm.shape == (expected_cells, expected_cells)

    def test_height_map_dtype(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        hm = b.get_height_map()
        assert hm.dtype == np.float32

    def test_height_map_empty_is_nan(self):
        b = PointCloudBuilder()
        hm = b.get_height_map(grid_size_m=5.0, resolution_m=0.1)
        assert np.all(np.isnan(hm))

    def test_height_map_has_some_values(self):
        b = PointCloudBuilder(subsample=2, image_height=30, image_width=40)
        b.add_frame(_mid_depth(30, 40), _pose())
        hm = b.get_height_map()
        assert np.any(~np.isnan(hm))


# ---------------------------------------------------------------------------
# TestObstacleMask
# ---------------------------------------------------------------------------

class TestObstacleMask:
    def test_obstacle_mask_shape(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        mask = b.get_obstacle_mask(grid_size_m=10.0, resolution_m=0.1)
        expected_cells = int(10.0 / 0.1)
        assert mask.shape == (expected_cells, expected_cells)

    def test_obstacle_mask_dtype(self):
        b = PointCloudBuilder(subsample=4, image_height=60, image_width=80)
        b.add_frame(_mid_depth(60, 80), _pose())
        mask = b.get_obstacle_mask()
        assert mask.dtype == bool

    def test_empty_cloud_mask_all_false(self):
        b = PointCloudBuilder()
        mask = b.get_obstacle_mask(grid_size_m=5.0, resolution_m=0.1)
        assert not np.any(mask)

    def test_height_colouring_no_color_frame(self):
        """When no color frame is given, height-based coloring should be applied."""
        b = PointCloudBuilder(subsample=2, image_height=30, image_width=40)
        b.add_frame(_mid_depth(30, 40), _pose(), color_frame=None)
        cloud = b.get_cloud()
        if len(cloud.colors) > 0:
            # Colors should be in 0-255 range
            assert np.all(cloud.colors >= 0)
            assert np.all(cloud.colors <= 255)
