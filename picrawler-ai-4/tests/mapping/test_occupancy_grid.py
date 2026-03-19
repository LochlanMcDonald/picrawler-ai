"""
Tests for OccupancyGrid — pure algorithmic code, no hardware required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mapping.occupancy_grid import OccupancyGrid
from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grid(width_m: float = 5.0, height_m: float = 5.0, res: float = 0.1) -> OccupancyGrid:
    return OccupancyGrid(width_m=width_m, height_m=height_m, resolution_m=res)


def at_origin() -> Pose2D:
    return Pose2D(0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

class TestCoordinateConversion:
    def test_world_to_grid_origin(self):
        g = make_grid()
        gx, gy = g.world_to_grid(0.0, 0.0)
        assert gx == g.origin_x
        assert gy == g.origin_y

    def test_round_trip_near_origin(self):
        g = make_grid()
        for wx, wy in [(0.5, 0.3), (-0.4, 1.0), (0.0, 0.0)]:
            gx, gy = g.world_to_grid(wx, wy)
            rx, ry = g.grid_to_world(gx, gy)
            # Round-trip should be within one cell size
            assert abs(rx - wx) <= g.resolution_m + 1e-9
            assert abs(ry - wy) <= g.resolution_m + 1e-9

    def test_positive_x_maps_right_of_origin(self):
        g = make_grid()
        gx, _ = g.world_to_grid(1.0, 0.0)
        assert gx > g.origin_x

    def test_negative_x_maps_left_of_origin(self):
        g = make_grid()
        gx, _ = g.world_to_grid(-1.0, 0.0)
        assert gx < g.origin_x


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------

class TestBoundsChecking:
    def test_origin_is_valid(self):
        g = make_grid()
        assert g.is_valid_cell(g.origin_x, g.origin_y)

    def test_negative_indices_are_invalid(self):
        g = make_grid()
        assert not g.is_valid_cell(-1, 0)
        assert not g.is_valid_cell(0, -1)

    def test_out_of_bounds_indices_are_invalid(self):
        g = make_grid()
        assert not g.is_valid_cell(g.grid_width, 0)
        assert not g.is_valid_cell(0, g.grid_height)

    def test_last_valid_cell(self):
        g = make_grid()
        assert g.is_valid_cell(g.grid_width - 1, g.grid_height - 1)


# ---------------------------------------------------------------------------
# Obstacle updates
# ---------------------------------------------------------------------------

class TestObstacleUpdates:
    def test_obstacle_increases_log_odds(self):
        g = make_grid()
        pose = at_origin()
        gx, gy = g.world_to_grid(1.0, 0.0)
        before = g.grid[gy, gx]
        g.update_obstacle(pose, distance_m=1.0, bearing_rad=0.0)
        assert g.grid[gy, gx] > before

    def test_ray_marks_intermediate_cells_free(self):
        """Cells between robot and obstacle should become more free (negative log-odds)."""
        g = make_grid()
        pose = at_origin()
        g.update_obstacle(pose, distance_m=2.0, bearing_rad=0.0)
        # A cell halfway along should be free (log-odds < 0)
        half_gx, half_gy = g.world_to_grid(1.0, 0.0)
        assert g.grid[half_gy, half_gx] < 0.0

    def test_free_space_decreases_log_odds(self):
        g = make_grid()
        pose = at_origin()
        gx, gy = g.world_to_grid(1.0, 0.0)
        g.update_free_space(pose, max_range_m=2.0, bearing_rad=0.0)
        assert g.grid[gy, gx] < 0.0

    def test_repeated_obstacles_clamp_at_max(self):
        g = make_grid()
        pose = at_origin()
        for _ in range(100):
            g.update_obstacle(pose, distance_m=1.0, bearing_rad=0.0)
        gx, gy = g.world_to_grid(1.0, 0.0)
        assert g.grid[gy, gx] <= g.log_odds_max

    def test_repeated_free_clamps_at_min(self):
        g = make_grid()
        pose = at_origin()
        for _ in range(100):
            g.update_free_space(pose, max_range_m=2.0, bearing_rad=0.0)
        gx, gy = g.world_to_grid(1.0, 0.0)
        assert g.grid[gy, gx] >= g.log_odds_min

    def test_out_of_bounds_obstacle_does_not_crash(self):
        """Rays that leave the grid should be ignored gracefully."""
        g = make_grid(width_m=2.0, height_m=2.0)
        pose = at_origin()
        g.update_obstacle(pose, distance_m=50.0, bearing_rad=0.0)  # way outside grid


# ---------------------------------------------------------------------------
# Robot position marking
# ---------------------------------------------------------------------------

class TestMarkRobotPosition:
    def test_visited_set_at_robot_position(self):
        g = make_grid()
        pose = at_origin()
        g.mark_robot_position(pose, radius_m=0.1)
        gx, gy = g.world_to_grid(0.0, 0.0)
        assert g.visited[gy, gx]

    def test_visited_marks_radius(self):
        g = make_grid(res=0.05)
        pose = at_origin()
        g.mark_robot_position(pose, radius_m=0.2)
        visited_count = g.visited.sum()
        assert visited_count > 1  # More than just the centre cell


# ---------------------------------------------------------------------------
# Probability map
# ---------------------------------------------------------------------------

class TestProbabilityMap:
    def test_unknown_cells_near_half(self):
        g = make_grid()
        prob = g.get_probability_map()
        # Fresh grid: all cells log-odds=0 → probability exactly 0.5
        assert np.allclose(prob, 0.5)

    def test_occupied_cell_high_probability(self):
        g = make_grid()
        pose = at_origin()
        for _ in range(10):
            g.update_obstacle(pose, distance_m=1.0, bearing_rad=0.0)
        prob = g.get_probability_map()
        gx, gy = g.world_to_grid(1.0, 0.0)
        assert prob[gy, gx] > 0.7

    def test_free_cell_low_probability(self):
        g = make_grid()
        pose = at_origin()
        for _ in range(10):
            g.update_free_space(pose, max_range_m=2.0, bearing_rad=0.0)
        prob = g.get_probability_map()
        gx, gy = g.world_to_grid(1.0, 0.0)
        assert prob[gy, gx] < 0.3


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_fresh_grid_all_unknown(self):
        g = make_grid()
        stats = g.get_statistics()
        assert stats["free_cells"] == 0
        assert stats["occupied_cells"] == 0
        assert stats["visited_cells"] == 0

    def test_statistics_after_obstacle(self):
        g = make_grid()
        pose = at_origin()
        for _ in range(10):
            g.update_obstacle(pose, distance_m=1.0, bearing_rad=0.0)
        stats = g.get_statistics()
        assert stats["occupied_cells"] > 0
        assert stats["free_cells"] > 0  # cells along the ray are free

    def test_explored_percent_increases(self):
        g = make_grid()
        stats_before = g.get_statistics()
        pose = at_origin()
        # Multiple updates are needed to push cells past the 0.3/0.7 thresholds
        for _ in range(5):
            g.update_obstacle(pose, distance_m=1.0, bearing_rad=0.0)
        stats_after = g.get_statistics()
        assert stats_after["explored_percent"] > stats_before["explored_percent"]


# ---------------------------------------------------------------------------
# Frontier detection
# ---------------------------------------------------------------------------

class TestFrontierDetection:
    def test_no_frontiers_in_fresh_grid(self):
        """All cells unknown — boundary between free and unknown needs free cells."""
        g = make_grid()
        frontiers = g.find_frontiers()
        assert frontiers == []

    def test_frontier_exists_at_free_unknown_boundary(self):
        g = make_grid()
        pose = at_origin()
        # Repeat enough times to push cells below the 0.3 probability threshold
        for _ in range(10):
            g.update_free_space(pose, max_range_m=1.5, bearing_rad=0.0)
        frontiers = g.find_frontiers()
        assert len(frontiers) > 0


# ---------------------------------------------------------------------------
# Bresenham line algorithm
# ---------------------------------------------------------------------------

class TestBresenhamLine:
    def test_horizontal_line(self):
        g = make_grid()
        cells = g._bresenham_line(0, 0, 5, 0)
        assert cells[0] == (0, 0)
        assert cells[-1] == (5, 0)
        assert len(cells) == 6

    def test_diagonal_line_start_end(self):
        g = make_grid()
        cells = g._bresenham_line(0, 0, 3, 3)
        assert cells[0] == (0, 0)
        assert cells[-1] == (3, 3)

    def test_single_point(self):
        g = make_grid()
        cells = g._bresenham_line(2, 3, 2, 3)
        assert cells == [(2, 3)]
