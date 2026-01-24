"""
Visual odometry for tracking camera/robot motion.

Uses feature detection and matching to estimate camera movement between frames,
enabling position tracking without external sensors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np


@dataclass
class Pose2D:
    """2D robot pose (position + orientation)."""
    x: float  # meters
    y: float  # meters
    theta: float  # radians (heading)
    timestamp: float

    def __str__(self) -> str:
        return f"Pose(x={self.x:.2f}m, y={self.y:.2f}m, θ={np.degrees(self.theta):.1f}°)"


@dataclass
class MotionEstimate:
    """Estimated motion between frames."""
    delta_x: float  # meters
    delta_y: float  # meters
    delta_theta: float  # radians
    confidence: float  # 0-1
    num_matches: int


class VisualOdometry:
    """Track robot motion using visual features."""

    def __init__(self, camera_height_m: float = 0.1, camera_tilt_deg: float = 20.0):
        """
        Args:
            camera_height_m: Camera height above ground
            camera_tilt_deg: Camera tilt angle (down from horizontal)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.camera_height_m = camera_height_m
        self.camera_tilt_rad = np.radians(camera_tilt_deg)

        # Feature detector (ORB - fast and free)
        self.detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_timestamp = None

        # Current pose estimate
        self.current_pose = Pose2D(0.0, 0.0, 0.0, time.time())
        self.pose_history: List[Pose2D] = [self.current_pose]

        # Statistics
        self.total_distance = 0.0
        self.total_rotation = 0.0

    def process_frame(self, frame: np.ndarray,
                     action_hint: Optional[str] = None) -> Optional[MotionEstimate]:
        """Process a new frame and estimate motion.

        Args:
            frame: Grayscale or RGB image
            action_hint: Robot action taken (helps with scale ambiguity)

        Returns:
            MotionEstimate if successful, None otherwise
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            self.logger.warning(f"Insufficient features detected: {len(keypoints) if keypoints else 0}")
            return None

        # First frame - just store
        if self.prev_descriptors is None:
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_timestamp = time.time()
            return None

        # Match features with previous frame
        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 8:
            self.logger.warning(f"Insufficient good matches: {len(good_matches)}")
            # Still update for next iteration
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None

        # Extract matched point coordinates
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

        # Estimate motion using Essential matrix (for calibrated camera)
        # Simplified: Use homography for 2D motion (assumes planar ground)
        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)

        if H is None:
            self.logger.warning("Homography estimation failed")
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None

        # Decompose homography to extract rotation and translation
        # Simplified approach: Extract from homography matrix
        inliers = mask.sum()
        confidence = inliers / len(good_matches)

        # Estimate rotation (theta) from homography
        # H encodes rotation + translation for planar motion
        delta_theta = np.arctan2(H[1, 0], H[0, 0])

        # Estimate translation - this is scale-ambiguous without depth
        # Use action hint and typical robot speeds for scale estimation
        tx_pixels = H[0, 2]
        ty_pixels = H[1, 2]

        # Convert pixel motion to metric motion
        # Rough calibration: assume ~200 pixels = 0.3m forward motion
        scale = 0.3 / 200.0  # meters per pixel (approximate)

        # Apply action hint for better scale estimation
        if action_hint in ['forward', 'ahead']:
            scale *= 1.2  # Robot moved forward
        elif action_hint in ['backward', 'reverse']:
            scale *= 1.2
            tx_pixels *= -1
        elif action_hint in ['left', 'turn_left', 'right', 'turn_right']:
            scale *= 0.5  # Mostly rotation, little translation

        delta_x_cam = tx_pixels * scale
        delta_y_cam = ty_pixels * scale

        # Transform camera frame to robot frame
        # Camera looks forward and down, robot frame is ground plane
        delta_x = delta_x_cam * np.cos(self.camera_tilt_rad)
        delta_y = delta_y_cam

        motion = MotionEstimate(
            delta_x=delta_x,
            delta_y=delta_y,
            delta_theta=delta_theta,
            confidence=confidence,
            num_matches=inliers
        )

        # Update pose
        self._update_pose(motion)

        # Update state for next iteration
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_timestamp = time.time()

        self.logger.debug(
            f"Motion: dx={delta_x:.3f}m, dy={delta_y:.3f}m, dθ={np.degrees(delta_theta):.1f}°, "
            f"confidence={confidence:.2f}, matches={inliers}"
        )

        return motion

    def _update_pose(self, motion: MotionEstimate) -> None:
        """Update current pose estimate from motion."""
        # Rotate delta by current heading
        cos_theta = np.cos(self.current_pose.theta)
        sin_theta = np.sin(self.current_pose.theta)

        dx_global = motion.delta_x * cos_theta - motion.delta_y * sin_theta
        dy_global = motion.delta_x * sin_theta + motion.delta_y * cos_theta

        # Update pose
        new_pose = Pose2D(
            x=self.current_pose.x + dx_global,
            y=self.current_pose.y + dy_global,
            theta=self.current_pose.theta + motion.delta_theta,
            timestamp=time.time()
        )

        # Normalize theta to [-pi, pi]
        new_pose.theta = np.arctan2(np.sin(new_pose.theta), np.cos(new_pose.theta))

        self.current_pose = new_pose
        self.pose_history.append(new_pose)

        # Update statistics
        distance = np.sqrt(dx_global**2 + dy_global**2)
        self.total_distance += distance
        self.total_rotation += abs(motion.delta_theta)

    def get_pose(self) -> Pose2D:
        """Get current pose estimate."""
        return self.current_pose

    def get_trajectory(self) -> List[Pose2D]:
        """Get full pose history (trajectory)."""
        return self.pose_history

    def reset(self, pose: Optional[Pose2D] = None) -> None:
        """Reset odometry, optionally to a specific pose."""
        if pose is None:
            pose = Pose2D(0.0, 0.0, 0.0, time.time())

        self.current_pose = pose
        self.pose_history = [pose]
        self.total_distance = 0.0
        self.total_rotation = 0.0

        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        self.logger.info(f"Visual odometry reset to {pose}")

    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        return {
            'total_distance_m': self.total_distance,
            'total_rotation_rad': self.total_rotation,
            'total_rotation_deg': np.degrees(self.total_rotation),
            'num_poses': len(self.pose_history),
            'current_pose': str(self.current_pose)
        }
