"""
Waypoint navigation using SLAM pose estimates.

Follows planned paths by tracking robot position and controlling movements
toward waypoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from perception.visual_odometry import Pose2D


class NavigationStatus(Enum):
    """Status of waypoint navigation."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class NavigationCommand:
    """Command for robot to execute."""
    action: str  # forward, left, right, stop
    duration: float  # How long to execute (seconds)
    reason: str  # Why this command was chosen


class WaypointNavigator:
    """Navigate along a planned path using waypoints."""

    def __init__(self, position_tolerance_m: float = 0.15,
                 heading_tolerance_deg: float = 15.0):
        """
        Args:
            position_tolerance_m: Distance to waypoint to consider "reached"
            heading_tolerance_deg: Heading error tolerance in degrees
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.position_tolerance_m = position_tolerance_m
        self.heading_tolerance_rad = np.radians(heading_tolerance_deg)

        # Navigation state
        self.waypoints: List[Tuple[float, float]] = []
        self.current_waypoint_index = 0
        self.status = NavigationStatus.NOT_STARTED

        # Stuck detection
        self.last_progress_time = 0.0
        self.last_distance_to_goal = float('inf')
        self.stuck_timeout_s = 10.0  # Consider stuck after 10s with no progress

    def set_path(self, waypoints: List[Tuple[float, float]]) -> None:
        """Set a new path to follow.

        Args:
            waypoints: List of (x_m, y_m) waypoints
        """
        if not waypoints:
            self.logger.warning("Empty waypoint list provided")
            return

        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.status = NavigationStatus.IN_PROGRESS
        self.last_progress_time = time.time()
        self.last_distance_to_goal = float('inf')

        self.logger.info(f"Path set with {len(waypoints)} waypoints")
        self.logger.debug(f"Waypoints: {waypoints}")

    def get_next_command(self, current_pose: Pose2D,
                        obstacle_detected: bool = False) -> Optional[NavigationCommand]:
        """Get next navigation command based on current pose.

        Args:
            current_pose: Current robot pose from SLAM
            obstacle_detected: Whether obstacle is blocking path

        Returns:
            NavigationCommand or None if navigation complete/failed
        """
        if self.status != NavigationStatus.IN_PROGRESS:
            return None

        if not self.waypoints:
            self.status = NavigationStatus.FAILED
            self.logger.error("No waypoints set")
            return None

        # Check for obstacles
        if obstacle_detected:
            self.logger.warning("Obstacle detected - stopping")
            self.status = NavigationStatus.BLOCKED
            return NavigationCommand("stop", 0.5, "obstacle_blocking_path")

        # Get current target waypoint
        if self.current_waypoint_index >= len(self.waypoints):
            # Reached all waypoints!
            self.status = NavigationStatus.COMPLETED
            self.logger.info("Navigation completed - reached final waypoint")
            return NavigationCommand("stop", 0.5, "goal_reached")

        target_x, target_y = self.waypoints[self.current_waypoint_index]

        # Calculate distance and bearing to target
        dx = target_x - current_pose.x
        dy = target_y - current_pose.y
        distance = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx)

        # Calculate heading error (angle to turn)
        heading_error = bearing - current_pose.theta
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Check if waypoint reached
        if distance < self.position_tolerance_m:
            self.logger.info(
                f"Reached waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} "
                f"at ({target_x:.2f}, {target_y:.2f})"
            )
            self.current_waypoint_index += 1
            self.last_progress_time = time.time()
            self.last_distance_to_goal = float('inf')

            # If that was the last waypoint, we're done
            if self.current_waypoint_index >= len(self.waypoints):
                self.status = NavigationStatus.COMPLETED
                return NavigationCommand("stop", 0.5, "goal_reached")

            # Otherwise, get command for next waypoint
            return self.get_next_command(current_pose, obstacle_detected)

        # Check for progress (stuck detection)
        if distance < self.last_distance_to_goal - 0.05:  # Made progress
            self.last_progress_time = time.time()
            self.last_distance_to_goal = distance
        elif time.time() - self.last_progress_time > self.stuck_timeout_s:
            self.logger.error("Navigation stuck - no progress for 10 seconds")
            self.status = NavigationStatus.FAILED
            return NavigationCommand("stop", 0.5, "stuck_no_progress")

        # Decide action based on heading error and distance
        if abs(heading_error) > self.heading_tolerance_rad:
            # Need to turn toward target
            if heading_error > 0:
                # Turn left
                turn_duration = min(abs(heading_error) / 2.0, 0.8)  # Proportional turning
                return NavigationCommand(
                    "left", turn_duration,
                    f"turn_left_{np.degrees(heading_error):.1f}deg_to_waypoint"
                )
            else:
                # Turn right
                turn_duration = min(abs(heading_error) / 2.0, 0.8)
                return NavigationCommand(
                    "right", turn_duration,
                    f"turn_right_{np.degrees(abs(heading_error)):.1f}deg_to_waypoint"
                )
        else:
            # Heading is good, move forward
            # Move proportionally to distance, but cap at reasonable values
            forward_duration = min(distance / 0.3, 1.5)  # Assume ~0.3m/s speed

            return NavigationCommand(
                "forward", forward_duration,
                f"forward_{distance:.2f}m_to_waypoint"
            )

    def get_status(self) -> NavigationStatus:
        """Get current navigation status."""
        return self.status

    def get_progress(self) -> dict:
        """Get navigation progress information.

        Returns:
            Dict with progress stats
        """
        if not self.waypoints:
            return {
                'status': self.status.value,
                'progress_percent': 0.0,
                'waypoints_reached': 0,
                'waypoints_total': 0,
                'current_target': None
            }

        progress_percent = (self.current_waypoint_index / len(self.waypoints)) * 100

        current_target = None
        if self.current_waypoint_index < len(self.waypoints):
            current_target = self.waypoints[self.current_waypoint_index]

        return {
            'status': self.status.value,
            'progress_percent': progress_percent,
            'waypoints_reached': self.current_waypoint_index,
            'waypoints_total': len(self.waypoints),
            'current_target': current_target
        }

    def reset(self) -> None:
        """Reset navigation state."""
        self.waypoints = []
        self.current_waypoint_index = 0
        self.status = NavigationStatus.NOT_STARTED
        self.last_progress_time = 0.0
        self.last_distance_to_goal = float('inf')
        self.logger.info("Waypoint navigator reset")

    def is_active(self) -> bool:
        """Check if navigation is currently active."""
        return self.status == NavigationStatus.IN_PROGRESS

    def is_complete(self) -> bool:
        """Check if navigation completed successfully."""
        return self.status == NavigationStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if navigation failed or blocked."""
        return self.status in [NavigationStatus.FAILED, NavigationStatus.BLOCKED]
