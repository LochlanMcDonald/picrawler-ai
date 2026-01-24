"""
SLAM Controller integrating visual odometry and occupancy mapping.

Coordinates:
- Visual odometry for pose tracking
- Depth estimation for obstacle detection
- Occupancy grid mapping for spatial representation
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np

from perception.visual_odometry import VisualOdometry, Pose2D, MotionEstimate
from perception.depth_estimator import DepthMap
from mapping.occupancy_grid import OccupancyGrid
from mapping.path_planner import PathPlanner
from mapping.waypoint_navigator import WaypointNavigator, NavigationCommand


class SLAMController:
    """Simultaneous Localization and Mapping controller."""

    def __init__(self, map_size_m: float = 10.0, resolution_m: float = 0.05):
        """
        Args:
            map_size_m: Size of map (square)
            resolution_m: Grid resolution in meters
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.visual_odometry = VisualOdometry(
            camera_height_m=0.1,
            camera_tilt_deg=20.0
        )

        self.occupancy_grid = OccupancyGrid(
            width_m=map_size_m,
            height_m=map_size_m,
            resolution_m=resolution_m
        )

        # Path planning and navigation
        self.path_planner = PathPlanner(self.occupancy_grid)
        self.waypoint_navigator = WaypointNavigator(
            position_tolerance_m=0.15,
            heading_tolerance_deg=15.0
        )

        # State
        self.initialized = False
        self.last_map_update = 0.0
        self.map_update_interval = 0.5  # Update map every 0.5s
        self.current_planned_path: Optional[List[Tuple[float, float]]] = None

        self.logger.info("SLAM controller initialized with path planning")

    def process_frame(self, image: np.ndarray,
                     depth_map: Optional[DepthMap] = None,
                     action_hint: Optional[str] = None) -> Tuple[Pose2D, np.ndarray]:
        """Process a new frame with SLAM.

        Args:
            image: RGB or grayscale image
            depth_map: Optional depth estimation
            action_hint: Robot action taken (for scale estimation)

        Returns:
            (current_pose, map_visualization)
        """
        # Update visual odometry
        motion = self.visual_odometry.process_frame(image, action_hint)

        # Get current pose estimate
        pose = self.visual_odometry.get_pose()

        # Update map if enough time passed
        now = time.time()
        if now - self.last_map_update >= self.map_update_interval:
            self._update_map(pose, depth_map)
            self.last_map_update = now

        # Generate map visualization
        map_vis = self.occupancy_grid.get_visualization(robot_pose=pose)

        if not self.initialized:
            self.initialized = True
            self.logger.info("SLAM initialized with first frame")

        return pose, map_vis

    def _update_map(self, pose: Pose2D, depth_map: Optional[DepthMap]) -> None:
        """Update occupancy grid with current observations.

        Args:
            pose: Current robot pose
            depth_map: Depth estimation (if available)
        """
        # Mark robot's current position
        self.occupancy_grid.mark_robot_position(pose, radius_m=0.15)

        # Update map using depth information
        if depth_map:
            depths = depth_map.get_directional_depths()

            # Convert depth values to distances (0=close, 1=far)
            # Map to real distances (approximate)
            def depth_to_distance(d: float) -> float:
                return 0.1 + d * 1.9  # 0.1m to 2.0m range

            # Front sensor
            front_dist = depth_to_distance(depths['front'])
            if depths['front'] < 0.3:  # Close obstacle
                self.occupancy_grid.update_obstacle(pose, front_dist, bearing_rad=0.0)
            else:  # Free space
                self.occupancy_grid.update_free_space(pose, max_range_m=2.0, bearing_rad=0.0)

            # Left sensor
            left_dist = depth_to_distance(depths['left'])
            if depths['left'] < 0.3:
                self.occupancy_grid.update_obstacle(pose, left_dist, bearing_rad=np.pi/4)
            else:
                self.occupancy_grid.update_free_space(pose, max_range_m=2.0, bearing_rad=np.pi/4)

            # Right sensor
            right_dist = depth_to_distance(depths['right'])
            if depths['right'] < 0.3:
                self.occupancy_grid.update_obstacle(pose, right_dist, bearing_rad=-np.pi/4)
            else:
                self.occupancy_grid.update_free_space(pose, max_range_m=2.0, bearing_rad=-np.pi/4)

    def get_current_pose(self) -> Pose2D:
        """Get current robot pose estimate."""
        return self.visual_odometry.get_pose()

    def get_trajectory(self) -> List[Pose2D]:
        """Get full trajectory history."""
        return self.visual_odometry.get_trajectory()

    def get_map_visualization(self, include_trajectory: bool = False,
                            include_planned_path: bool = True) -> np.ndarray:
        """Get map visualization with optional trajectory and path overlay.

        Args:
            include_trajectory: If True, draw full trajectory on map
            include_planned_path: If True, draw planned path on map

        Returns:
            RGB visualization
        """
        pose = self.visual_odometry.get_pose()
        map_vis = self.occupancy_grid.get_visualization(robot_pose=pose)

        if include_trajectory and len(self.visual_odometry.pose_history) > 1:
            # Draw trajectory (blue)
            for i in range(len(self.visual_odometry.pose_history) - 1):
                p1 = self.visual_odometry.pose_history[i]
                p2 = self.visual_odometry.pose_history[i + 1]

                gx1, gy1 = self.occupancy_grid.world_to_grid(p1.x, p1.y)
                gx2, gy2 = self.occupancy_grid.world_to_grid(p2.x, p2.y)

                # Flip y for visualization
                gy1_vis = self.occupancy_grid.grid_height - gy1
                gy2_vis = self.occupancy_grid.grid_height - gy2

                cv2.line(map_vis, (gx1, gy1_vis), (gx2, gy2_vis), (255, 0, 0), 1)

        if include_planned_path and self.current_planned_path:
            # Draw planned path (green)
            for i in range(len(self.current_planned_path) - 1):
                x1, y1 = self.current_planned_path[i]
                x2, y2 = self.current_planned_path[i + 1]

                gx1, gy1 = self.occupancy_grid.world_to_grid(x1, y1)
                gx2, gy2 = self.occupancy_grid.world_to_grid(x2, y2)

                # Flip y for visualization
                gy1_vis = self.occupancy_grid.grid_height - gy1
                gy2_vis = self.occupancy_grid.grid_height - gy2

                cv2.line(map_vis, (gx1, gy1_vis), (gx2, gy2_vis), (0, 255, 0), 2)

            # Draw waypoints as green circles
            for x, y in self.current_planned_path:
                gx, gy = self.occupancy_grid.world_to_grid(x, y)
                gy_vis = self.occupancy_grid.grid_height - gy
                cv2.circle(map_vis, (gx, gy_vis), 2, (0, 255, 0), -1)

        return map_vis

    def find_exploration_targets(self, num_targets: int = 3) -> List[Tuple[float, float]]:
        """Find promising locations to explore (frontiers).

        Args:
            num_targets: Number of targets to return

        Returns:
            List of (x_m, y_m) target locations
        """
        frontiers = self.occupancy_grid.find_frontiers()

        if not frontiers:
            return []

        # Sort frontiers by distance from robot
        pose = self.visual_odometry.get_pose()
        frontiers_with_dist = []

        for fx, fy in frontiers:
            dist = np.sqrt((fx - pose.x)**2 + (fy - pose.y)**2)
            frontiers_with_dist.append((dist, fx, fy))

        frontiers_with_dist.sort()

        # Return closest frontiers (skip very close ones)
        targets = []
        for dist, fx, fy in frontiers_with_dist:
            if dist > 0.3 and len(targets) < num_targets:  # At least 30cm away
                targets.append((fx, fy))

        return targets

    def get_navigation_waypoint(self) -> Optional[Tuple[float, float]]:
        """Get next waypoint for exploration.

        Returns:
            (x_m, y_m) waypoint or None
        """
        targets = self.find_exploration_targets(num_targets=1)
        return targets[0] if targets else None

    def plan_path_to_goal(self, goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """Plan a path to goal coordinates.

        Args:
            goal_x, goal_y: Goal position in meters

        Returns:
            List of waypoints or None if no path found
        """
        current_pose = self.visual_odometry.get_pose()
        path = self.path_planner.plan_path(current_pose, goal_x, goal_y)

        if path:
            self.current_planned_path = path
            self.waypoint_navigator.set_path(path)
            self.logger.info(f"Planned path to ({goal_x:.2f}, {goal_y:.2f}) with {len(path)} waypoints")
        else:
            self.logger.warning(f"Could not find path to ({goal_x:.2f}, {goal_y:.2f})")

        return path

    def get_navigation_command(self, obstacle_detected: bool = False) -> Optional[NavigationCommand]:
        """Get next navigation command for waypoint following.

        Args:
            obstacle_detected: Whether obstacle is currently blocking

        Returns:
            NavigationCommand or None if not navigating
        """
        if not self.waypoint_navigator.is_active():
            return None

        current_pose = self.visual_odometry.get_pose()
        return self.waypoint_navigator.get_next_command(current_pose, obstacle_detected)

    def is_navigating(self) -> bool:
        """Check if currently following a planned path."""
        return self.waypoint_navigator.is_active()

    def get_navigation_progress(self) -> dict:
        """Get navigation progress information."""
        return self.waypoint_navigator.get_progress()

    def cancel_navigation(self) -> None:
        """Cancel current navigation."""
        self.waypoint_navigator.reset()
        self.current_planned_path = None
        self.logger.info("Navigation cancelled")

    def get_statistics(self) -> dict:
        """Get SLAM statistics."""
        vo_stats = self.visual_odometry.get_statistics()
        map_stats = self.occupancy_grid.get_statistics()

        return {
            'slam': {
                'initialized': self.initialized,
                'current_pose': str(self.visual_odometry.get_pose())
            },
            'odometry': vo_stats,
            'map': map_stats
        }

    def reset(self) -> None:
        """Reset SLAM system."""
        self.visual_odometry.reset()
        self.occupancy_grid = OccupancyGrid(
            width_m=self.occupancy_grid.width_m,
            height_m=self.occupancy_grid.height_m,
            resolution_m=self.occupancy_grid.resolution_m
        )
        self.initialized = False
        self.logger.info("SLAM system reset")

    def save_map(self, filepath: str) -> None:
        """Save current map visualization to file.

        Args:
            filepath: Path to save image
        """
        map_vis = self.get_map_visualization(include_trajectory=True)
        cv2.imwrite(filepath, cv2.cvtColor(map_vis, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Map saved to {filepath}")
