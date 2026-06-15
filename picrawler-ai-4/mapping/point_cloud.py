"""
Semi-dense 3D point cloud builder.

Projects MiDaS depth maps into 3D using corrected SLAM poses,
accumulates a coloured point cloud, and provides:
  - A 3D point array for host-side visualisation
  - A 2.5D height map for the occupancy grid
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PointCloud:
    """A snapshot of the accumulated 3D map."""
    points: np.ndarray    # (N, 3) float32  — X, Y, Z in metres (world frame)
    colors: np.ndarray    # (N, 3) uint8    — R, G, B


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PointCloudBuilder:
    """
    Builds a semi-dense 3D point cloud from MiDaS depth + SLAM poses.

    Coordinate convention (world frame):
        +X = robot's initial forward direction
        +Y = robot's initial left
        +Z = up

    MiDaS depth is relative (0=close, 1=far) with no absolute scale.
    We use a linear mapping calibrated to the robot's typical operating
    range (0.1 m – 2.0 m), consistent with world_model.py.
    """

    DEPTH_MIN_M = 0.10   # metres corresponding to depth=0.0
    DEPTH_MAX_M = 2.00   # metres corresponding to depth=1.0

    def __init__(self,
                 camera_height_m: float = 0.10,
                 camera_tilt_deg: float = 20.0,
                 h_fov_deg: float = 62.0,
                 v_fov_deg: float = 48.0,
                 image_width: int = 640,
                 image_height: int = 480,
                 max_points: int = 100_000,
                 subsample: int = 8):
        """
        Args:
            camera_height_m: Camera height above ground
            camera_tilt_deg: Downward tilt from horizontal
            h_fov_deg / v_fov_deg: Horizontal/vertical field of view
            image_width / image_height: Expected depth map resolution
            max_points: Rolling cap — oldest points are pruned when exceeded
            subsample: Only project every Nth pixel (controls density vs. speed)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.camera_height_m = camera_height_m
        self.camera_tilt_rad = np.radians(camera_tilt_deg)
        self.h_fov_rad = np.radians(h_fov_deg)
        self.v_fov_rad = np.radians(v_fov_deg)
        self.image_width = image_width
        self.image_height = image_height
        self.max_points = max_points
        self.subsample = subsample

        # Pre-compute per-pixel ray angles
        self._ray_h, self._ray_v = self._build_ray_grid()

        # Accumulated point cloud
        self._points: List[np.ndarray] = []   # (3,) float32 each
        self._colors: List[np.ndarray] = []   # (3,) uint8 each
        self._total_added = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self,
                  depth_map_raw: np.ndarray,
                  pose: Pose2D,
                  color_frame: Optional[np.ndarray] = None) -> int:
        """
        Project a depth map into 3D and accumulate into the point cloud.

        Args:
            depth_map_raw: HxW float32 depth map (0=close, 1=far, MiDaS convention)
            pose: Corrected robot pose from pose graph
            color_frame: Optional HxW×3 BGR image for colouring the points

        Returns:
            Number of new points added
        """
        h, w = depth_map_raw.shape[:2]

        # Resize ray grid if image size differs from expected
        if h != self.image_height or w != self.image_width:
            self.image_height = h
            self.image_width = w
            self._ray_h, self._ray_v = self._build_ray_grid()

        # Subsample pixel grid
        rows = np.arange(0, h, self.subsample)
        cols = np.arange(0, w, self.subsample)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        rr_flat = rr.ravel()
        cc_flat = cc.ravel()

        # Depth values → metric distance
        depth_vals = depth_map_raw[rr_flat, cc_flat].astype(np.float32)
        distance_m = self.DEPTH_MIN_M + depth_vals * (self.DEPTH_MAX_M - self.DEPTH_MIN_M)

        # Ray angles for selected pixels
        ray_h = self._ray_h[rr_flat, cc_flat]
        ray_v = self._ray_v[rr_flat, cc_flat]

        # Camera-frame 3D points
        # Camera looks forward (+Z_cam) and is tilted downward
        # X_cam = right, Y_cam = up, Z_cam = forward
        x_cam = distance_m * np.tan(ray_h)
        z_cam = distance_m * np.cos(ray_v)            # forward
        y_cam_raw = distance_m * np.tan(ray_v)        # up/down component

        # Apply camera tilt to get robot-body frame
        # Rotation around X_cam by tilt angle
        cos_t = np.cos(self.camera_tilt_rad)
        sin_t = np.sin(self.camera_tilt_rad)
        y_body = -(-y_cam_raw * cos_t + z_cam * sin_t) + self.camera_height_m
        z_body = -y_cam_raw * sin_t + z_cam * cos_t   # forward in body frame
        x_body = x_cam                                 # right

        # Rotate into world frame using robot heading (theta)
        cos_p = np.cos(pose.theta)
        sin_p = np.sin(pose.theta)
        x_world = pose.x + cos_p * z_body - sin_p * x_body
        y_world = pose.y + sin_p * z_body + cos_p * x_body
        z_world = y_body  # height above ground

        # Filter: keep points above ground and within sensor range
        valid = (
            (z_world > -0.05) &          # slightly below ground OK (noise)
            (z_world < 2.5) &            # ceiling
            (distance_m > 0.15) &        # min range
            (distance_m < 1.9)           # max reliable range
        )

        new_pts = np.stack([x_world[valid], y_world[valid], z_world[valid]], axis=1)

        # Colours
        if color_frame is not None:
            cf = color_frame[rr_flat[valid], cc_flat[valid]]
            if cf.shape[1] == 3:
                new_cols = cf[:, ::-1].astype(np.uint8)  # BGR → RGB
            else:
                new_cols = np.full((len(new_pts), 3), 180, dtype=np.uint8)
        else:
            # Height-based colouring: low=blue, mid=green, high=red
            new_cols = self._height_color(z_world[valid])

        for i in range(len(new_pts)):
            self._points.append(new_pts[i])
            self._colors.append(new_cols[i])

        self._total_added += len(new_pts)

        # Prune if over capacity (drop oldest)
        if len(self._points) > self.max_points:
            excess = len(self._points) - self.max_points
            self._points = self._points[excess:]
            self._colors = self._colors[excess:]

        return len(new_pts)

    def get_cloud(self) -> PointCloud:
        """Return the current accumulated point cloud."""
        if not self._points:
            return PointCloud(
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
            )
        return PointCloud(
            points=np.array(self._points, dtype=np.float32),
            colors=np.array(self._colors, dtype=np.uint8),
        )

    def get_height_map(self,
                       grid_size_m: float = 10.0,
                       resolution_m: float = 0.05) -> np.ndarray:
        """
        Project the 3D cloud into a 2.5D height map (max Z per cell).

        Returns:
            HxW float32 array, NaN where no points observed
        """
        if not self._points:
            cells = int(grid_size_m / resolution_m)
            return np.full((cells, cells), np.nan, dtype=np.float32)

        pts = np.array(self._points, dtype=np.float32)
        cells = int(grid_size_m / resolution_m)
        half = grid_size_m / 2.0

        height_map = np.full((cells, cells), np.nan, dtype=np.float32)

        ix = ((pts[:, 0] + half) / resolution_m).astype(int)
        iy = ((pts[:, 1] + half) / resolution_m).astype(int)
        z = pts[:, 2]

        valid = (ix >= 0) & (ix < cells) & (iy >= 0) & (iy < cells)
        for i in np.where(valid)[0]:
            r, c = iy[i], ix[i]
            if np.isnan(height_map[r, c]) or z[i] > height_map[r, c]:
                height_map[r, c] = z[i]

        return height_map

    def get_obstacle_mask(self,
                          grid_size_m: float = 10.0,
                          resolution_m: float = 0.05,
                          min_height_m: float = 0.05,
                          max_height_m: float = 0.5) -> np.ndarray:
        """
        Return a boolean obstacle mask derived from point height.

        Points between min and max height are considered obstacles
        (above ground, below camera).
        """
        if not self._points:
            cells = int(grid_size_m / resolution_m)
            return np.zeros((cells, cells), dtype=bool)

        pts = np.array(self._points, dtype=np.float32)
        cells = int(grid_size_m / resolution_m)
        half = grid_size_m / 2.0

        mask = np.zeros((cells, cells), dtype=bool)

        obstacle = (pts[:, 2] >= min_height_m) & (pts[:, 2] <= max_height_m)
        pts_obs = pts[obstacle]

        ix = ((pts_obs[:, 0] + half) / resolution_m).astype(int)
        iy = ((pts_obs[:, 1] + half) / resolution_m).astype(int)

        valid = (ix >= 0) & (ix < cells) & (iy >= 0) & (iy < cells)
        mask[iy[valid], ix[valid]] = True

        return mask

    def point_count(self) -> int:
        return len(self._points)

    def reset(self) -> None:
        self._points.clear()
        self._colors.clear()
        self._total_added = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_ray_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute horizontal and vertical ray angles per pixel."""
        h, w = self.image_height, self.image_width
        cols = np.arange(w)
        rows = np.arange(h)
        cc, rr = np.meshgrid(cols, rows)

        # Horizontal angle: left edge = -h_fov/2, right = +h_fov/2
        ray_h = (cc / (w - 1) - 0.5) * self.h_fov_rad

        # Vertical angle: top = +v_fov/2, bottom = -v_fov/2
        ray_v = (0.5 - rr / (h - 1)) * self.v_fov_rad

        return ray_h.astype(np.float32), ray_v.astype(np.float32)

    @staticmethod
    def _height_color(z: np.ndarray) -> np.ndarray:
        """Map height values to RGB colours (blue→green→red)."""
        z_norm = np.clip(z / 0.5, 0.0, 1.0)
        r = (z_norm * 255).astype(np.uint8)
        g = ((1.0 - z_norm) * 128).astype(np.uint8)
        b = ((1.0 - z_norm) * 255).astype(np.uint8)
        return np.stack([r, g, b], axis=1)
