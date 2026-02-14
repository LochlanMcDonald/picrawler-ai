"""
Unified world model combining all sensor inputs.

This provides a single source of truth about the robot's environment,
fusing ultrasonic sensor, vision AI, depth estimation, and other data sources.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from perception.depth_estimator import DepthMap


@dataclass
class ObstacleInfo:
    """Information about obstacles in a specific direction."""
    distance_cm: Optional[float]
    confidence: float  # 0.0 to 1.0
    source: str  # 'ultrasonic', 'vision', 'fused'
    detected: bool


class WorldModel:
    """Unified representation of robot's environment."""

    def __init__(self, obstacle_threshold_cm: float = 20.0):
        self.obstacle_threshold_cm = obstacle_threshold_cm

        # Obstacle distances by direction
        self.obstacles: Dict[str, ObstacleInfo] = {
            'front': ObstacleInfo(None, 0.0, 'unknown', False),
            'left': ObstacleInfo(None, 0.0, 'unknown', False),
            'right': ObstacleInfo(None, 0.0, 'unknown', False),
            'rear': ObstacleInfo(None, 0.0, 'unknown', False),
        }

        # Vision AI features
        self.visual_objects: List[str] = []
        self.visual_hazards: List[str] = []
        self.visual_description: str = ""

        # Spatial assessment
        self.free_space_score: float = 0.5  # 0=trapped, 1=open
        self.last_update: float = 0.0

    def update_ultrasonic(self, distance_cm: Optional[float]) -> None:
        """Update front obstacle from ultrasonic sensor."""
        if distance_cm is not None:
            self.obstacles['front'] = ObstacleInfo(
                distance_cm=distance_cm,
                confidence=0.95,  # High confidence - physical sensor
                source='ultrasonic',
                detected=distance_cm < self.obstacle_threshold_cm
            )
        else:
            self.obstacles['front'] = ObstacleInfo(
                None, 0.0, 'ultrasonic_unavailable', False
            )
        self.last_update = time.time()

    def update_vision(self, objects: List[str], hazards: List[str],
                     description: str) -> None:
        """Update visual information from camera AI."""
        self.visual_objects = objects
        self.visual_hazards = hazards
        self.visual_description = description

        # Fuse vision with ultrasonic for front obstacle
        vision_sees_obstacle = any(
            hazard in ['wall', 'obstacle', 'barrier', 'furniture']
            for hazard in hazards
        )

        # If we have ultrasonic, use it. Otherwise, use vision.
        if self.obstacles['front'].source == 'ultrasonic':
            # Ultrasonic takes priority, but vision can add confidence
            if vision_sees_obstacle and self.obstacles['front'].detected:
                # Both agree - high confidence!
                self.obstacles['front'].confidence = 0.99
                self.obstacles['front'].source = 'fused'
        else:
            # No ultrasonic, use vision only (lower confidence)
            self.obstacles['front'] = ObstacleInfo(
                distance_cm=None,
                confidence=0.6 if vision_sees_obstacle else 0.3,
                source='vision',
                detected=vision_sees_obstacle
            )

        self.last_update = time.time()

    def update_depth(self, depth_map: 'DepthMap') -> None:
        """Update obstacle information from learned depth map.

        Args:
            depth_map: Monocular depth estimation result
        """
        if depth_map is None:
            return

        # Get depth in cardinal directions (0=close, 1=far)
        depths = depth_map.get_directional_depths()

        # Convert depth to distance estimate
        # Depth 0.0 (close) -> ~10cm, Depth 1.0 (far) -> ~200cm
        # This is approximate - learned depth has no absolute scale
        def depth_to_distance(depth_value: float) -> float:
            """Convert normalized depth to estimated distance in cm."""
            # Inverse relationship: close objects have low depth value
            # Map 0-1 depth to 10-200cm range
            return 10 + (depth_value * 190)

        # Update obstacles with depth information
        for direction in ['front', 'left', 'right']:
            depth_value = depths[direction]
            estimated_distance = depth_to_distance(depth_value)

            # Determine if this is an obstacle
            is_obstacle = estimated_distance < self.obstacle_threshold_cm

            # Create or update obstacle info
            current_obs = self.obstacles.get(
                direction,
                ObstacleInfo(None, 0.0, 'unknown', False)
            )

            # If we have ultrasonic for front, fuse it with depth
            if direction == 'front' and current_obs.source == 'ultrasonic':
                # Ultrasonic has priority, but depth can confirm
                if is_obstacle and current_obs.detected:
                    # Both agree - increase confidence
                    current_obs.confidence = 0.98
                    current_obs.source = 'fused_ultrasonic_depth'
                elif not is_obstacle and not current_obs.detected:
                    # Both agree it's clear - increase confidence
                    current_obs.confidence = 0.95
                    current_obs.source = 'fused_ultrasonic_depth'
                # If they disagree, trust ultrasonic more (keep as is)
            else:
                # No ultrasonic, use depth estimation
                # Depth has lower confidence than physical sensors
                self.obstacles[direction] = ObstacleInfo(
                    distance_cm=estimated_distance,
                    confidence=0.7,  # Good but not perfect
                    source='depth_estimation',
                    detected=is_obstacle
                )

        self.last_update = time.time()

    def is_safe_to_move(self, direction: str) -> bool:
        """Check if it's safe to move in given direction.

        Args:
            direction: 'forward', 'backward', 'left', 'right'

        Returns:
            True if safe, False if blocked
        """
        # Map action to obstacle direction
        direction_map = {
            'forward': 'front',
            'ahead': 'front',
            'backward': 'rear',
            'back': 'rear',
            'reverse': 'rear',
            'turn_left': 'left',
            'left': 'left',
            'turn_right': 'right',
            'right': 'right',
        }

        obstacle_dir = direction_map.get(direction, 'front')
        obstacle = self.obstacles[obstacle_dir]

        # If obstacle detected with high confidence, not safe
        if obstacle.detected and obstacle.confidence > 0.5:
            return False

        return True

    def get_best_direction(self) -> str:
        """Suggest best direction to move based on all sensors.

        Returns:
            Direction name ('forward', 'turn_left', 'turn_right', 'backward')
        """
        scores = {}

        # Score each direction based on obstacle info
        for action, direction in [
            ('forward', 'front'),
            ('turn_left', 'left'),
            ('turn_right', 'right'),
            ('backward', 'rear')
        ]:
            obstacle = self.obstacles[direction]

            if obstacle.distance_cm is not None:
                # Higher distance = better score
                scores[action] = obstacle.distance_cm / 100.0
            elif obstacle.detected:
                # Detected but no distance = avoid
                scores[action] = 0.0
            else:
                # Unknown = medium score
                scores[action] = 0.5

        # Prefer forward over turns, turns over backward
        scores['forward'] *= 1.5
        scores['turn_left'] *= 1.0
        scores['turn_right'] *= 1.0
        scores['backward'] *= 0.3

        return max(scores, key=scores.get)

    def calculate_free_space(self) -> float:
        """Calculate how trapped vs open the robot is.

        Returns:
            Score from 0.0 (trapped) to 1.0 (wide open)
        """
        clear_directions = sum(
            1 for obs in self.obstacles.values()
            if not obs.detected
        )

        # 0-4 clear directions -> 0.0 to 1.0
        self.free_space_score = clear_directions / 4.0
        return self.free_space_score

    def to_dict(self) -> Dict:
        """Export as dictionary for logging/AI."""
        return {
            'obstacles': {
                direction: {
                    'distance_cm': obs.distance_cm,
                    'detected': obs.detected,
                    'confidence': obs.confidence,
                    'source': obs.source
                }
                for direction, obs in self.obstacles.items()
            },
            'vision': {
                'objects': self.visual_objects,
                'hazards': self.visual_hazards,
                'description': self.visual_description
            },
            'free_space_score': self.free_space_score,
            'best_direction': self.get_best_direction(),
            'last_update': self.last_update
        }

    def __str__(self) -> str:
        front = self.obstacles['front']
        if front.distance_cm:
            front_str = f"{front.distance_cm:.1f}cm"
        else:
            front_str = "blocked" if front.detected else "unknown"

        return (
            f"WorldModel(front={front_str}, "
            f"free_space={self.free_space_score:.2f}, "
            f"best_dir={self.get_best_direction()})"
        )
