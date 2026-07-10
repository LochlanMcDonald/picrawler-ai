"""
Occupancy grid mapping for spatial representation.

Maintains a 2D grid map of explored space, marking cells as:
- Free (navigable)
- Occupied (obstacle)
- Unknown (not yet observed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from perception.visual_odometry import Pose2D


@dataclass
class GridCell:
    """Single cell in occupancy grid."""
    log_odds: float  # Log-odds occupancy (>0 = occupied, <0 = free)
    visited: bool  # Has robot visited this cell?


class OccupancyGrid:
    """2D occupancy grid map."""

    def __init__(self, width_m: float = 10.0, height_m: float = 10.0,
                 resolution_m: float = 0.05):
        """
        Args:
            width_m: Map width in meters
            height_m: Map height in meters
            resolution_m: Cell size in meters
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.width_m = width_m
        self.height_m = height_m
        self.resolution_m = resolution_m

        # Grid dimensions
        self.grid_width = int(width_m / resolution_m)
        self.grid_height = int(height_m / resolution_m)

        # Initialize grid (log-odds = 0 means unknown/50% probability)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.visited = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        # Origin is at center of map
        self.origin_x = self.grid_width // 2
        self.origin_y = self.grid_height // 2

        # Log-odds parameters
        self.log_odds_occupied = 0.7  # Increase when occupied
        self.log_odds_free = -0.4  # Decrease when free
        self.log_odds_max = 5.0  # Clamp maximum
        self.log_odds_min = -5.0  # Clamp minimum

        self.logger.info(
            f"Occupancy grid initialized: {self.grid_width}x{self.grid_height} cells "
            f"({width_m:.1f}m x {height_m:.1f}m at {resolution_m*100:.1f}cm resolution)"
        )

    def world_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices.

        Args:
            x_m, y_m: Position in meters (world frame)

        Returns:
            (grid_x, grid_y) indices
        """
        grid_x = int(x_m / self.resolution_m) + self.origin_x
        grid_y = int(y_m / self.resolution_m) + self.origin_y
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates.

        Args:
            grid_x, grid_y: Grid indices

        Returns:
            (x_m, y_m) in meters
        """
        x_m = (grid_x - self.origin_x) * self.resolution_m
        y_m = (grid_y - self.origin_y) * self.resolution_m
        return x_m, y_m

    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are within map bounds."""
        return (0 <= grid_x < self.grid_width and
                0 <= grid_y < self.grid_height)

    def update_obstacle(self, pose: Pose2D, distance_m: float, bearing_rad: float) -> None:
        """Update grid with obstacle observation.

        Args:
            pose: Robot pose when observation was made
            distance_m: Distance to obstacle
            bearing_rad: Bearing to obstacle (relative to robot heading)
        """
        # Calculate obstacle position in world frame
        abs_bearing = pose.theta + bearing_rad
        obs_x = pose.x + distance_m * np.cos(abs_bearing)
        obs_y = pose.y + distance_m * np.sin(abs_bearing)

        # Mark cells along ray as free, endpoint as occupied
        self._ray_trace(pose.x, pose.y, obs_x, obs_y, mark_endpoint_occupied=True)

    def update_free_space(self, pose: Pose2D, max_range_m: float, bearing_rad: float) -> None:
        """Update grid with free space observation (no obstacle detected).

        Args:
            pose: Robot pose
            max_range_m: Maximum sensor range
            bearing_rad: Bearing scanned
        """
        # Calculate endpoint at max range
        abs_bearing = pose.theta + bearing_rad
        end_x = pose.x + max_range_m * np.cos(abs_bearing)
        end_y = pose.y + max_range_m * np.sin(abs_bearing)

        # Mark entire ray as free
        self._ray_trace(pose.x, pose.y, end_x, end_y, mark_endpoint_occupied=False)

    def _ray_trace(self, x0_m: float, y0_m: float, x1_m: float, y1_m: float,
                   mark_endpoint_occupied: bool) -> None:
        """Trace a ray and update cells (Bresenham algorithm).

        Args:
            x0_m, y0_m: Start position (meters)
            x1_m, y1_m: End position (meters)
            mark_endpoint_occupied: If True, mark endpoint as occupied
        """
        # Convert to grid coordinates
        gx0, gy0 = self.world_to_grid(x0_m, y0_m)
        gx1, gy1 = self.world_to_grid(x1_m, y1_m)

        # Bresenham line algorithm
        cells = self._bresenham_line(gx0, gy0, gx1, gy1)

        for i, (gx, gy) in enumerate(cells):
            if not self.is_valid_cell(gx, gy):
                continue

            is_endpoint = (i == len(cells) - 1)

            if is_endpoint and mark_endpoint_occupied:
                # Endpoint is occupied
                self.grid[gy, gx] = np.clip(
                    self.grid[gy, gx] + self.log_odds_occupied,
                    self.log_odds_min,
                    self.log_odds_max
                )
            else:
                # Cell along ray is free
                self.grid[gy, gx] = np.clip(
                    self.grid[gy, gx] + self.log_odds_free,
                    self.log_odds_min,
                    self.log_odds_max
                )

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for ray tracing."""
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

    def mark_robot_position(self, pose: Pose2D, radius_m: float = 0.1) -> None:
        """Mark robot's current position as visited/free.

        Args:
            pose: Robot pose
            radius_m: Radius around robot to mark as free
        """
        gx, gy = self.world_to_grid(pose.x, pose.y)

        radius_cells = int(radius_m / self.resolution_m)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nx, ny = gx + dx, gy + dy

                if not self.is_valid_cell(nx, ny):
                    continue

                # Distance check
                dist = np.sqrt(dx**2 + dy**2) * self.resolution_m
                if dist <= radius_m:
                    self.visited[ny, nx] = True
                    # Mark as free
                    self.grid[ny, nx] = np.clip(
                        self.grid[ny, nx] + self.log_odds_free * 2,
                        self.log_odds_min,
                        self.log_odds_max
                    )

    def get_probability_map(self) -> np.ndarray:
        """Convert log-odds to probability map (0-1).

        Returns:
            Probability map (0 = free, 1 = occupied)
        """
        # Convert log-odds to probability: p = 1 / (1 + exp(-log_odds))
        prob_map = 1.0 / (1.0 + np.exp(-self.grid))
        return prob_map

    def get_visualization(self, robot_pose: Optional[Pose2D] = None) -> np.ndarray:
        """Create RGB visualization of the map.

        Args:
            robot_pose: Optional robot pose to draw on map

        Returns:
            RGB image (uint8)
        """
        prob_map = self.get_probability_map()

        # Create RGB image
        # Free = white, Occupied = black, Unknown = gray
        vis = np.ones((self.grid_height, self.grid_width, 3), dtype=np.uint8) * 127  # Gray

        # Free space (low probability) = white
        vis[prob_map < 0.3] = [255, 255, 255]

        # Occupied (high probability) = black
        vis[prob_map > 0.7] = [0, 0, 0]

        # Visited cells = slight blue tint
        vis[self.visited] = np.minimum(vis[self.visited] + [0, 0, 30], 255)

        # Draw robot position
        if robot_pose is not None:
            gx, gy = self.world_to_grid(robot_pose.x, robot_pose.y)
            if self.is_valid_cell(gx, gy):
                # Draw robot as red circle
                cv2.circle(vis, (gx, gy), 3, (0, 0, 255), -1)

                # Draw heading indicator
                heading_len = 8
                end_x = int(gx + heading_len * np.cos(robot_pose.theta))
                end_y = int(gy + heading_len * np.sin(robot_pose.theta))
                cv2.line(vis, (gx, gy), (end_x, end_y), (0, 0, 255), 2)

        # Flip vertically for correct orientation
        vis = np.flipud(vis)

        return vis

    def find_frontiers(self) -> List[Tuple[float, float]]:
        """Find frontier cells (boundary between free and unknown space).

        Returns:
            List of (x_m, y_m) frontier points
        """
        prob_map = self.get_probability_map()

        frontiers = []

        for gy in range(1, self.grid_height - 1):
            for gx in range(1, self.grid_width - 1):
                # Cell is free
                if prob_map[gy, gx] < 0.3:
                    # Check neighbors for unknown cells
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = gx + dx, gy + dy

                        # Neighbor is unknown (near 0.5 probability)
                        if 0.4 < prob_map[ny, nx] < 0.6:
                            x_m, y_m = self.grid_to_world(gx, gy)
                            frontiers.append((x_m, y_m))
                            break

        return frontiers

    def get_statistics(self) -> dict:
        """Get map statistics."""
        prob_map = self.get_probability_map()

        total_cells = self.grid_width * self.grid_height
        free_cells = (prob_map < 0.3).sum()
        occupied_cells = (prob_map > 0.7).sum()
        unknown_cells = total_cells - free_cells - occupied_cells
        visited_cells = self.visited.sum()

        return {
            'total_cells': total_cells,
            'free_cells': int(free_cells),
            'occupied_cells': int(occupied_cells),
            'unknown_cells': int(unknown_cells),
            'visited_cells': int(visited_cells),
            'explored_percent': (free_cells + occupied_cells) / total_cells * 100,
            'visited_percent': visited_cells / total_cells * 100
        }
