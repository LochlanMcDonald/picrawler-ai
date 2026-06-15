"""
Keyframe management for SLAM loop closure.

Stores camera frames at key positions along the trajectory,
along with their ORB descriptors and pose estimates.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from perception.visual_odometry import Pose2D


@dataclass
class Keyframe:
    """A single keyframe stored for loop closure detection."""
    id: int
    timestamp: float
    pose: Pose2D                         # Pose when frame was captured
    keypoints: list                      # ORB keypoints
    descriptors: np.ndarray             # ORB descriptors (Nx32 uint8)
    thumbnail: np.ndarray               # Small grayscale image for debugging


class KeyframeStore:
    """
    Manages keyframes for loop closure.

    A new keyframe is added when the robot has moved far enough from
    the last keyframe, ensuring the database stays sparse and searchable.
    """

    def __init__(self,
                 min_distance_m: float = 0.3,
                 min_rotation_rad: float = 0.3,
                 max_keyframes: int = 500,
                 feature_count: int = 500):
        """
        Args:
            min_distance_m: Minimum travel before adding a new keyframe
            min_rotation_rad: Minimum rotation before adding a new keyframe
            max_keyframes: Cap on stored keyframes (oldest are pruned)
            feature_count: ORB features per keyframe
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_distance_m = min_distance_m
        self.min_rotation_rad = min_rotation_rad
        self.max_keyframes = max_keyframes

        self.detector = cv2.ORB_create(
            nfeatures=feature_count,
            scaleFactor=1.2,
            nlevels=8,
        )

        self.keyframes: List[Keyframe] = []
        self._next_id = 0
        self._last_kf_pose: Optional[Pose2D] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_add(self, frame: np.ndarray, pose: Pose2D) -> Optional[Keyframe]:
        """Add a keyframe if the robot has moved enough since the last one.

        Args:
            frame: Grayscale image
            pose: Current pose estimate

        Returns:
            The new Keyframe if one was created, else None
        """
        if not self._should_add(pose):
            return None

        gray = self._to_gray(frame)
        kps, descs = self.detector.detectAndCompute(gray, None)

        if descs is None or len(kps) < 20:
            return None

        thumbnail = cv2.resize(gray, (80, 60))

        kf = Keyframe(
            id=self._next_id,
            timestamp=time.time(),
            pose=Pose2D(pose.x, pose.y, pose.theta, pose.timestamp),
            keypoints=kps,
            descriptors=descs.astype(np.uint8),
            thumbnail=thumbnail,
        )

        self.keyframes.append(kf)
        self._next_id += 1
        self._last_kf_pose = pose

        # Prune oldest if over capacity
        if len(self.keyframes) > self.max_keyframes:
            self.keyframes.pop(0)

        self.logger.debug(
            f"Keyframe {kf.id} added at ({pose.x:.2f}, {pose.y:.2f}) "
            f"[total={len(self.keyframes)}]"
        )
        return kf

    def get_all(self) -> List[Keyframe]:
        return self.keyframes

    def get_by_id(self, kf_id: int) -> Optional[Keyframe]:
        for kf in self.keyframes:
            if kf.id == kf_id:
                return kf
        return None

    def __len__(self) -> int:
        return len(self.keyframes)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _should_add(self, pose: Pose2D) -> bool:
        if self._last_kf_pose is None:
            return True
        dx = pose.x - self._last_kf_pose.x
        dy = pose.y - self._last_kf_pose.y
        dist = np.sqrt(dx * dx + dy * dy)
        dtheta = abs(np.arctan2(
            np.sin(pose.theta - self._last_kf_pose.theta),
            np.cos(pose.theta - self._last_kf_pose.theta)
        ))
        return dist >= self.min_distance_m or dtheta >= self.min_rotation_rad

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
