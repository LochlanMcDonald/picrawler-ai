"""
Path planning algorithms for navigation on occupancy grids.

Implements A* algorithm for finding optimal collision-free paths.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import numpy as np

from mapping.occupancy_grid import OccupancyGrid
from perception.visual_odometry import Pose2D


@dataclass
class PathNode:
    """Node in A* search."""
    grid_x: int
    grid_y: int
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic to goal
    parent: Optional[PathNode] = None

    @property
    def f_cost(self) -> float:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: PathNode) -> bool:
        """For heap comparison."""
        return self.f_cost < other.f_cost

    def __eq__(self, other) -> bool:
        """For set comparison."""
        if not isinstance(other, PathNode):
            return False
        return self.grid_x == other.grid_x and self.grid_y == other.grid_y

    def __hash__(self) -> int:
        """For set hashing."""
        return hash((self.grid_x, self.grid_y))


class PathPlanner:
    """A* path planner for occupancy grids."""

    def __init__(self, occupancy_grid: OccupancyGrid):
        """
        Args:
            occupancy_grid: The occupancy grid to plan on
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.grid = occupancy_grid

        # A* parameters
        self.diagonal_cost = 1.414  # sqrt(2)
        self.straight_cost = 1.0

    def plan_path(self, start_pose: Pose2D, goal_x_m: float, goal_y_m: float,
                  occupancy_threshold: float = 0.65) -> Optional[List[Tuple[float, float]]]:
        """Plan a path from start to goal using A*.

        Args:
            start_pose: Starting robot pose
            goal_x_m, goal_y_m: Goal position in meters
            occupancy_threshold: Cells above this probability are obstacles

        Returns:
            List of (x_m, y_m) waypoints from start to goal, or None if no path
        """
        # Convert to grid coordinates
        start_gx, start_gy = self.grid.world_to_grid(start_pose.x, start_pose.y)
        goal_gx, goal_gy = self.grid.world_to_grid(goal_x_m, goal_y_m)

        # Validate start and goal
        if not self.grid.is_valid_cell(start_gx, start_gy):
            self.logger.error(f"Start position out of bounds: ({start_pose.x}, {start_pose.y})")
            return None

        if not self.grid.is_valid_cell(goal_gx, goal_gy):
            self.logger.error(f"Goal position out of bounds: ({goal_x_m}, {goal_y_m})")
            return None

        # Get probability map
        prob_map = self.grid.get_probability_map()

        # Check if goal is occupied
        if prob_map[goal_gy, goal_gx] > occupancy_threshold:
            self.logger.warning(f"Goal position is occupied (prob={prob_map[goal_gy, goal_gx]:.2f})")
            # Try to find nearby free cell
            goal_gx, goal_gy = self._find_nearest_free_cell(goal_gx, goal_gy, prob_map, occupancy_threshold)
            if goal_gx is None:
                self.logger.error("Could not find free cell near goal")
                return None
            goal_x_m, goal_y_m = self.grid.grid_to_world(goal_gx, goal_gy)
            self.logger.info(f"Adjusted goal to nearest free cell: ({goal_x_m:.2f}, {goal_y_m:.2f})")

        # A* search
        start_node = PathNode(start_gx, start_gy, 0.0, self._heuristic(start_gx, start_gy, goal_gx, goal_gy))

        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        g_scores = {(start_gx, start_gy): 0.0}

        iterations = 0
        max_iterations = 10000

        while open_set and iterations < max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            # Check if reached goal
            if current.grid_x == goal_gx and current.grid_y == goal_gy:
                path = self._reconstruct_path(current)
                self.logger.info(f"Path found with {len(path)} waypoints ({iterations} iterations)")
                return path

            # Mark as visited
            closed_set.add((current.grid_x, current.grid_y))

            # Check neighbors
            for neighbor_gx, neighbor_gy, move_cost in self._get_neighbors(current.grid_x, current.grid_y):
                # Skip if out of bounds
                if not self.grid.is_valid_cell(neighbor_gx, neighbor_gy):
                    continue

                # Skip if already visited
                if (neighbor_gx, neighbor_gy) in closed_set:
                    continue

                # Skip if occupied
                if prob_map[neighbor_gy, neighbor_gx] > occupancy_threshold:
                    continue

                # Calculate g_cost
                tentative_g = current.g_cost + move_cost

                # Check if this path is better
                if (neighbor_gx, neighbor_gy) not in g_scores or tentative_g < g_scores[(neighbor_gx, neighbor_gy)]:
                    g_scores[(neighbor_gx, neighbor_gy)] = tentative_g

                    h_cost = self._heuristic(neighbor_gx, neighbor_gy, goal_gx, goal_gy)
                    neighbor_node = PathNode(neighbor_gx, neighbor_gy, tentative_g, h_cost, current)

                    heapq.heappush(open_set, neighbor_node)

        # No path found
        self.logger.warning(f"No path found after {iterations} iterations")
        return None

    def _heuristic(self, gx1: int, gy1: int, gx2: int, gy2: int) -> float:
        """Calculate heuristic cost (Euclidean distance).

        Args:
            gx1, gy1: Start grid position
            gx2, gy2: Goal grid position

        Returns:
            Heuristic cost
        """
        return np.sqrt((gx2 - gx1)**2 + (gy2 - gy1)**2)

    def _get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int, float]]:
        """Get valid neighbors and their costs.

        Args:
            gx, gy: Current grid position

        Returns:
            List of (neighbor_gx, neighbor_gy, cost) tuples
        """
        neighbors = []

        # 8-connected grid (includes diagonals)
        for dx, dy in [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
        ]:
            nx, ny = gx + dx, gy + dy
            cost = self.diagonal_cost if (dx != 0 and dy != 0) else self.straight_cost
            neighbors.append((nx, ny, cost))

        return neighbors

    def _find_nearest_free_cell(self, gx: int, gy: int, prob_map: np.ndarray,
                                threshold: float) -> Tuple[Optional[int], Optional[int]]:
        """Find nearest free cell to a given position.

        Args:
            gx, gy: Target grid position
            prob_map: Probability map
            threshold: Occupancy threshold

        Returns:
            (nearest_gx, nearest_gy) or (None, None) if not found
        """
        max_search_radius = 20  # Search up to 20 cells away

        for radius in range(1, max_search_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check cells on the perimeter
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = gx + dx, gy + dy

                        if not self.grid.is_valid_cell(nx, ny):
                            continue

                        if prob_map[ny, nx] < threshold:
                            return nx, ny

        return None, None

    def _reconstruct_path(self, goal_node: PathNode) -> List[Tuple[float, float]]:
        """Reconstruct path from goal node back to start.

        Args:
            goal_node: The goal node with parent chain

        Returns:
            List of (x_m, y_m) waypoints from start to goal
        """
        path_grid = []
        current = goal_node

        while current is not None:
            path_grid.append((current.grid_x, current.grid_y))
            current = current.parent

        # Reverse to go from start to goal
        path_grid.reverse()

        # Convert to world coordinates and simplify
        path_world = [self.grid.grid_to_world(gx, gy) for gx, gy in path_grid]

        # Simplify path (remove redundant waypoints)
        simplified_path = self._simplify_path(path_world)

        return simplified_path

    def _simplify_path(self, path: List[Tuple[float, float]],
                      tolerance: float = 0.1) -> List[Tuple[float, float]]:
        """Simplify path by removing redundant waypoints.

        Uses Douglas-Peucker algorithm to reduce waypoints while preserving shape.

        Args:
            path: Original path
            tolerance: Maximum deviation allowed (meters)

        Returns:
            Simplified path
        """
        if len(path) <= 2:
            return path

        # Douglas-Peucker algorithm
        def perpendicular_distance(point: Tuple[float, float],
                                  line_start: Tuple[float, float],
                                  line_end: Tuple[float, float]) -> float:
            """Calculate perpendicular distance from point to line."""
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end

            num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            if den == 0:
                return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

            return num / den

        def douglas_peucker(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
            """Recursive Douglas-Peucker simplification."""
            if len(points) <= 2:
                return points

            # Find point with maximum distance
            dmax = 0.0
            index = 0
            for i in range(1, len(points) - 1):
                d = perpendicular_distance(points[i], points[0], points[-1])
                if d > dmax:
                    index = i
                    dmax = d

            # If max distance > epsilon, recursively simplify
            if dmax > epsilon:
                # Recursive call
                rec1 = douglas_peucker(points[:index + 1], epsilon)
                rec2 = douglas_peucker(points[index:], epsilon)

                # Build result
                result = rec1[:-1] + rec2
            else:
                result = [points[0], points[-1]]

            return result

        return douglas_peucker(path, tolerance)
