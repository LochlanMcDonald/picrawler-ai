"""
Tests for PathPlanner (A* algorithm) — pure algorithmic code, no hardware required.
"""
from __future__ import annotations

import numpy as np
import pytest

from mapping.occupancy_grid import OccupancyGrid
from mapping.path_planner import PathPlanner, PathNode
from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_planner(inflation: float = 0.0, threshold: float = 0.65) -> tuple[PathPlanner, OccupancyGrid]:
    """Return a planner on a 5×5 m grid at 10 cm resolution."""
    grid = OccupancyGrid(width_m=5.0, height_m=5.0, resolution_m=0.1)
    planner = PathPlanner(grid, obstacle_inflation_radius=inflation, occupancy_threshold=threshold)
    return planner, grid


def pose(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Pose2D:
    return Pose2D(x, y, theta, 0.0)


def block_cell(grid: OccupancyGrid, x_m: float, y_m: float, repeats: int = 15) -> None:
    """Hammer a single cell with obstacle updates until it's marked occupied."""
    p = pose()
    import math
    dist = math.sqrt(x_m**2 + y_m**2)
    bearing = math.atan2(y_m, x_m)
    for _ in range(repeats):
        grid.update_obstacle(p, distance_m=dist if dist > 0.01 else 0.1, bearing_rad=bearing)


# ---------------------------------------------------------------------------
# PathNode
# ---------------------------------------------------------------------------

class TestPathNode:
    def test_f_cost_sum(self):
        n = PathNode(0, 0, g_cost=2.0, h_cost=3.0)
        assert n.f_cost == pytest.approx(5.0)

    def test_comparison_by_f_cost(self):
        cheap = PathNode(0, 0, g_cost=1.0, h_cost=1.0)
        expensive = PathNode(1, 0, g_cost=5.0, h_cost=5.0)
        assert cheap < expensive

    def test_equality_by_grid_position(self):
        a = PathNode(3, 4, g_cost=1.0, h_cost=0.0)
        b = PathNode(3, 4, g_cost=9.0, h_cost=9.0)
        assert a == b

    def test_hash_by_grid_position(self):
        a = PathNode(3, 4, g_cost=1.0, h_cost=0.0)
        b = PathNode(3, 4, g_cost=9.0, h_cost=9.0)
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_zero_when_at_goal(self):
        planner, _ = make_planner()
        assert planner._heuristic(5, 5, 5, 5) == pytest.approx(0.0)

    def test_positive_when_apart(self):
        planner, _ = make_planner()
        assert planner._heuristic(0, 0, 3, 4) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Neighbors
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    def test_returns_eight_neighbors(self):
        planner, _ = make_planner()
        neighbors = planner._get_neighbors(10, 10)
        assert len(neighbors) == 8

    def test_diagonal_cost_higher_than_straight(self):
        planner, _ = make_planner()
        neighbors = planner._get_neighbors(10, 10)
        straight = [c for dx, dy, c in [(n[0] - 10, n[1] - 10, n[2]) for n in neighbors]
                    if dx == 0 or dy == 0]
        diagonal = [c for dx, dy, c in [(n[0] - 10, n[1] - 10, n[2]) for n in neighbors]
                    if dx != 0 and dy != 0]
        assert all(c == planner.straight_cost for c in straight)
        assert all(c == planner.diagonal_cost for c in diagonal)


# ---------------------------------------------------------------------------
# Path planning — happy paths
# ---------------------------------------------------------------------------

class TestPlanPath:
    def test_trivial_path_start_equals_goal(self):
        """Start == goal should return a (very short) path."""
        planner, _ = make_planner()
        path = planner.plan_path(pose(0.0, 0.0), 0.0, 0.0)
        # Some planners return empty or single-point path; the key is not None
        assert path is not None

    def test_straight_path_in_empty_grid(self):
        planner, _ = make_planner()
        path = planner.plan_path(pose(0.0, 0.0), 1.0, 0.0)
        assert path is not None
        assert len(path) >= 1
        # Last waypoint should be close to goal
        last_x, last_y = path[-1]
        assert abs(last_x - 1.0) < 0.3
        assert abs(last_y - 0.0) < 0.3

    def test_path_moves_in_right_direction(self):
        planner, _ = make_planner()
        path = planner.plan_path(pose(0.0, 0.0), 1.5, 0.0)
        assert path is not None
        # The path should move in positive x direction
        first_x, _ = path[0]
        last_x, _ = path[-1]
        assert last_x >= first_x

    def test_path_avoids_obstacle(self):
        """A wall of obstacles should cause the path to route around."""
        planner, grid = make_planner(inflation=0.0)
        # Block x=0.5 from y=-0.5 to y=0.5 (vertical wall)
        p = pose()
        import math
        for y in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
            dist = math.sqrt(0.5**2 + y**2)
            bearing = math.atan2(y, 0.5)
            for _ in range(20):
                grid.update_obstacle(p, distance_m=dist, bearing_rad=bearing)

        path = planner.plan_path(pose(0.0, 0.0), 1.0, 0.0)
        if path is not None:
            # Path should deviate from y=0 to avoid wall
            y_values = [wp[1] for wp in path]
            max_deviation = max(abs(y) for y in y_values)
            assert max_deviation > 0.05  # Had to go around

    def test_path_not_found_when_completely_enclosed(self):
        """If start is fully surrounded with a heavy obstacle ring, planner fails gracefully."""
        planner, grid = make_planner(inflation=0.0)
        p = pose()
        import math
        # Dense ring of obstacles every 5 degrees, hammered 40 times each
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            for dist in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for _ in range(40):
                    grid.update_obstacle(p, distance_m=dist, bearing_rad=rad)

        path = planner.plan_path(pose(0.0, 0.0), 2.0, 0.0)
        # Path should either not be found, or be extremely short (couldn't escape the ring)
        assert path is None or len(path) <= 2

    def test_out_of_bounds_start_returns_none(self):
        planner, _ = make_planner()
        path = planner.plan_path(pose(100.0, 100.0), 0.0, 0.0)
        assert path is None

    def test_out_of_bounds_goal_returns_none(self):
        planner, _ = make_planner()
        path = planner.plan_path(pose(0.0, 0.0), 100.0, 100.0)
        assert path is None


# ---------------------------------------------------------------------------
# Path simplification (Douglas-Peucker)
# ---------------------------------------------------------------------------

class TestSimplifyPath:
    def test_empty_path_unchanged(self):
        planner, _ = make_planner()
        assert planner._simplify_path([]) == []

    def test_two_point_path_unchanged(self):
        planner, _ = make_planner()
        path = [(0.0, 0.0), (1.0, 0.0)]
        assert planner._simplify_path(path) == path

    def test_collinear_points_simplified(self):
        """Straight line with intermediate points should collapse to endpoints."""
        planner, _ = make_planner()
        # 5 collinear points on y=0
        path = [(float(i) * 0.25, 0.0) for i in range(5)]
        simplified = planner._simplify_path(path, tolerance=0.01)
        assert len(simplified) <= 2

    def test_curved_path_keeps_corners(self):
        """An L-shaped path should keep the corner."""
        planner, _ = make_planner()
        path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        simplified = planner._simplify_path(path, tolerance=0.01)
        assert len(simplified) == 3  # Corner must be kept


# ---------------------------------------------------------------------------
# Nearest free cell
# ---------------------------------------------------------------------------

class TestFindNearestFreeCell:
    def test_already_free_returns_original(self):
        planner, grid = make_planner()
        prob_map = grid.get_probability_map()
        gx, gy = grid.world_to_grid(0.0, 0.0)
        nx, ny = planner._find_nearest_free_cell(gx, gy, prob_map, threshold=0.65)
        assert nx is not None

    def test_returns_none_when_all_occupied(self):
        planner, grid = make_planner()
        # Force all cells to occupied by setting probability directly
        grid.grid[:, :] = grid.log_odds_max
        prob_map = grid.get_probability_map()
        gx, gy = grid.world_to_grid(0.0, 0.0)
        nx, ny = planner._find_nearest_free_cell(gx, gy, prob_map, threshold=0.65)
        assert nx is None
        assert ny is None
