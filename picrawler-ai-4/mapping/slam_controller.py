"""
SLAM Controller — full stack with loop closure and 3D mapping.

Coordinates:
  VisualOdometry  → raw pose estimates
  KeyframeStore   → sparse keyframe database
  LoopClosureDetector → place recognition
  PoseGraph       → drift-corrected trajectory
  PointCloudBuilder   → semi-dense 3D map
  OccupancyGrid   → 2D navigation map (fed from point cloud obstacle mask)
  MapServer       → real-time UDP stream to host visualiser
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from perception.visual_odometry import VisualOdometry, Pose2D, MotionEstimate
from perception.depth_estimator import DepthMap
from mapping.occupancy_grid import OccupancyGrid
from mapping.path_planner import PathPlanner
from mapping.waypoint_navigator import WaypointNavigator, NavigationCommand
from mapping.keyframe import KeyframeStore
from mapping.loop_closure import LoopClosureDetector
from mapping.pose_graph import PoseGraph
from mapping.point_cloud import PointCloudBuilder
from mapping.map_server import MapServer


class SLAMController:
    """Full SLAM pipeline with loop closure and 3D point cloud."""

    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        self.logger = logging.getLogger(self.__class__.__name__)

        # ----------------------------------------------------------------
        # Config sections
        # ----------------------------------------------------------------
        slam_cfg = config.get("slam_settings", {})
        vo_cfg   = config.get("visual_odometry_settings", {})
        path_cfg = config.get("path_planning_settings", {})
        nav_cfg  = config.get("navigation_settings", {})
        kf_cfg   = config.get("keyframe_settings", {})
        lc_cfg   = config.get("loop_closure_settings", {})
        pc_cfg   = config.get("point_cloud_settings", {})
        srv_cfg  = config.get("map_server_settings", {})

        map_size_m   = slam_cfg.get("map_size_m", 20.0)
        resolution_m = slam_cfg.get("map_resolution_m", 0.05)

        # ----------------------------------------------------------------
        # Core pose estimation
        # ----------------------------------------------------------------
        self.visual_odometry = VisualOdometry(
            camera_height_m=vo_cfg.get("camera_height_m", 0.1),
            camera_tilt_deg=vo_cfg.get("camera_tilt_deg", 20.0),
            scale_calibration_factor=vo_cfg.get("scale_calibration_factor", 1.0),
            feature_count=vo_cfg.get("feature_count", 500),
        )

        # ----------------------------------------------------------------
        # Keyframe database
        # ----------------------------------------------------------------
        self.keyframe_store = KeyframeStore(
            min_distance_m=kf_cfg.get("min_distance_m", 0.3),
            min_rotation_rad=kf_cfg.get("min_rotation_rad", 0.3),
            max_keyframes=kf_cfg.get("max_keyframes", 500),
            feature_count=kf_cfg.get("feature_count", 500),
        )

        # ----------------------------------------------------------------
        # Loop closure
        # ----------------------------------------------------------------
        self.loop_detector = LoopClosureDetector(
            min_score=lc_cfg.get("min_score", 0.25),
            min_inliers=lc_cfg.get("min_inliers", 12),
            min_age_frames=lc_cfg.get("min_age_frames", 10),
            top_k_candidates=lc_cfg.get("top_k_candidates", 5),
        )

        # ----------------------------------------------------------------
        # Pose graph
        # ----------------------------------------------------------------
        self.pose_graph = PoseGraph(
            max_iterations=slam_cfg.get("pg_max_iterations", 100),
            convergence_tol=slam_cfg.get("pg_convergence_tol", 1e-4),
        )
        # Seed the graph with the origin node (fixed)
        self.pose_graph.add_node(0, Pose2D(0.0, 0.0, 0.0, time.time()), fixed=True)
        self._pg_node_id = 0    # Most recent node in the pose graph

        # ----------------------------------------------------------------
        # 3D point cloud
        # ----------------------------------------------------------------
        self.point_cloud = PointCloudBuilder(
            camera_height_m=vo_cfg.get("camera_height_m", 0.1),
            camera_tilt_deg=vo_cfg.get("camera_tilt_deg", 20.0),
            max_points=pc_cfg.get("max_points", 100_000),
            subsample=pc_cfg.get("subsample", 8),
        )

        # ----------------------------------------------------------------
        # 2D navigation map
        # ----------------------------------------------------------------
        self.occupancy_grid = OccupancyGrid(
            width_m=map_size_m,
            height_m=map_size_m,
            resolution_m=resolution_m,
        )
        self.path_planner = PathPlanner(
            self.occupancy_grid,
            obstacle_inflation_radius=path_cfg.get("obstacle_inflation_radius_m", 0.15),
            occupancy_threshold=path_cfg.get("occupancy_threshold", 0.65),
        )
        self.waypoint_navigator = WaypointNavigator(
            position_tolerance_m=nav_cfg.get("position_tolerance_m", 0.15),
            heading_tolerance_deg=nav_cfg.get("heading_tolerance_deg", 15.0),
        )

        # ----------------------------------------------------------------
        # Map server (host visualisation)
        # ----------------------------------------------------------------
        self._server_enabled = srv_cfg.get("enabled", False)
        self.map_server: Optional[MapServer] = None
        if self._server_enabled:
            self.map_server = MapServer(
                host=srv_cfg.get("host", "255.255.255.255"),
                port=srv_cfg.get("port", 5005),
                broadcast_hz=srv_cfg.get("broadcast_hz", 2.0),
            )
            self.map_server.start()

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        self.initialized = False
        self._last_map_update = 0.0
        self._map_update_interval = slam_cfg.get("map_update_interval_s", 0.5)
        self._last_cloud_update = 0.0
        self._cloud_update_interval = pc_cfg.get("update_interval_s", 1.0)
        self.current_planned_path: Optional[List[Tuple[float, float]]] = None

        # Corrected pose from pose graph (starts at origin)
        self._corrected_pose = Pose2D(0.0, 0.0, 0.0, time.time())

        # Loop closure stats
        self._loop_closures: List[dict] = []

        self.logger.info("SLAM controller initialised (loop closure + 3D point cloud)")

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------

    def process_frame(self,
                      image: np.ndarray,
                      depth_map: Optional[DepthMap] = None,
                      action_hint: Optional[str] = None) -> Tuple[Pose2D, np.ndarray]:
        """
        Process one camera frame through the full SLAM pipeline.

        Args:
            image: BGR or grayscale image from camera
            depth_map: MiDaS depth map (optional but enables 3D mapping)
            action_hint: Most recent robot action ('forward', 'turn_left', …)

        Returns:
            (corrected_pose, 2D_map_visualization)
        """
        # ---- 1. Visual odometry ----------------------------------------
        motion: Optional[MotionEstimate] = self.visual_odometry.process_frame(
            image, action_hint
        )
        raw_pose = self.visual_odometry.get_pose()

        # ---- 2. Pose graph — add odometry edge --------------------------
        if motion is not None:
            new_id = self._pg_node_id + 1
            self.pose_graph.add_node(new_id, raw_pose)
            dx = raw_pose.x - self._corrected_pose.x
            dy = raw_pose.y - self._corrected_pose.y
            dtheta = raw_pose.theta - self._corrected_pose.theta
            self.pose_graph.add_odometry_edge(
                self._pg_node_id, new_id, (dx, dy, dtheta)
            )
            self._pg_node_id = new_id

        # ---- 3. Keyframe + loop closure ---------------------------------
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_kf = self.keyframe_store.try_add(gray, raw_pose)

        if new_kf is not None:
            closure = self.loop_detector.detect(new_kf, self.keyframe_store)
            if closure is not None:
                # Add loop edge to pose graph
                from_pose = closure.query_kf.pose
                to_pose   = closure.match_kf.pose
                rel = (
                    to_pose.x - from_pose.x,
                    to_pose.y - from_pose.y,
                    to_pose.theta - from_pose.theta,
                )
                self.pose_graph.add_loop_edge(
                    closure.query_kf.id,
                    closure.match_kf.id,
                    rel,
                    score=closure.score,
                )

                # Optimise and update corrected pose
                corrected_poses = self.pose_graph.optimize()
                if self._pg_node_id in corrected_poses:
                    cp = corrected_poses[self._pg_node_id]
                    self._corrected_pose = cp

                # Log and broadcast the loop
                event = {
                    "from_kf": closure.query_kf.id,
                    "to_kf": closure.match_kf.id,
                    "score": round(closure.score, 3),
                    "inliers": closure.num_inliers,
                }
                self._loop_closures.append(event)
                self.logger.info(f"Loop closure confirmed: {event}")

                if self.map_server:
                    self.map_server.update_loop(from_pose, to_pose)

        # ---- 4. Use corrected pose where available ----------------------
        # Between loop closures, track raw odometry delta from last correction
        if motion is not None:
            self._corrected_pose = Pose2D(
                self._corrected_pose.x + (raw_pose.x - self.visual_odometry.pose_history[-2].x
                                          if len(self.visual_odometry.pose_history) > 1 else 0),
                self._corrected_pose.y + (raw_pose.y - self.visual_odometry.pose_history[-2].y
                                          if len(self.visual_odometry.pose_history) > 1 else 0),
                raw_pose.theta,
                raw_pose.timestamp,
            )
        pose = self._corrected_pose

        # ---- 5. 3D point cloud ------------------------------------------
        now = time.time()
        if depth_map is not None and (now - self._last_cloud_update) >= self._cloud_update_interval:
            depth_raw = depth_map.depth_map if hasattr(depth_map, 'depth_map') else None
            if depth_raw is not None:
                color_frame = image if len(image.shape) == 3 else None
                added = self.point_cloud.add_frame(depth_raw, pose, color_frame)
                self.logger.debug(f"Point cloud: +{added} points (total={self.point_cloud.point_count()})")
            self._last_cloud_update = now

        # ---- 6. 2D occupancy grid ---------------------------------------
        if (now - self._last_map_update) >= self._map_update_interval:
            self._update_occupancy(pose, depth_map)
            self._last_map_update = now

        # ---- 7. Broadcast to host ---------------------------------------
        if self.map_server:
            self.map_server.update_pose(pose)
            if self.point_cloud.point_count() > 0:
                self.map_server.update_cloud(self.point_cloud.get_cloud())

        if not self.initialized:
            self.initialized = True
            self.logger.info("SLAM initialised with first frame")

        map_vis = self.get_map_visualization(
            include_trajectory=True, include_planned_path=True
        )
        return pose, map_vis

    # ------------------------------------------------------------------
    # Occupancy grid update
    # ------------------------------------------------------------------

    def _update_occupancy(self, pose: Pose2D, depth_map: Optional[DepthMap]) -> None:
        self.occupancy_grid.mark_robot_position(pose, radius_m=0.15)

        # Feed from point cloud obstacle mask when available
        if self.point_cloud.point_count() > 50:
            mask = self.point_cloud.get_obstacle_mask(
                grid_size_m=self.occupancy_grid.width_m,
                resolution_m=self.occupancy_grid.resolution_m,
            )
            # Stamp obstacle cells directly into the log-odds grid
            gx_off = self.occupancy_grid.origin_x
            gy_off = self.occupancy_grid.origin_y
            ys, xs = np.where(mask)
            for gx, gy in zip(xs, ys):
                if self.occupancy_grid.is_valid_cell(gx, gy):
                    self.occupancy_grid.grid[gy, gx] = min(
                        self.occupancy_grid.grid[gy, gx] + self.occupancy_grid.log_odds_occupied,
                        self.occupancy_grid.log_odds_max,
                    )
            return

        # Fallback to directional depth (original behaviour)
        if depth_map is None:
            return

        depths = depth_map.get_directional_depths()

        def dist(d: float) -> float:
            return 0.1 + d * 1.9

        for bearing, key in ((0.0, 'front'), (np.pi / 4, 'left'), (-np.pi / 4, 'right')):
            d = depths[key]
            if d < 0.3:
                self.occupancy_grid.update_obstacle(pose, dist(d), bearing)
            else:
                self.occupancy_grid.update_free_space(pose, 2.0, bearing)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get_current_pose(self) -> Pose2D:
        return self._corrected_pose

    def get_trajectory(self) -> List[Pose2D]:
        return self.pose_graph.get_trajectory()

    def find_exploration_targets(self, num_targets: int = 3) -> List[Tuple[float, float]]:
        frontiers = self.occupancy_grid.find_frontiers()
        if not frontiers:
            return []
        pose = self._corrected_pose
        by_dist = sorted(
            frontiers,
            key=lambda p: (p[0] - pose.x) ** 2 + (p[1] - pose.y) ** 2
        )
        return [(fx, fy) for fx, fy in by_dist if
                (fx - pose.x) ** 2 + (fy - pose.y) ** 2 > 0.09][:num_targets]

    def get_navigation_waypoint(self) -> Optional[Tuple[float, float]]:
        targets = self.find_exploration_targets(num_targets=1)
        return targets[0] if targets else None

    def plan_path_to_goal(self, goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        path = self.path_planner.plan_path(self._corrected_pose, goal_x, goal_y)
        if path:
            self.current_planned_path = path
            self.waypoint_navigator.set_path(path)
            self.logger.info(f"Path to ({goal_x:.2f}, {goal_y:.2f}): {len(path)} waypoints")
        else:
            self.logger.warning(f"No path found to ({goal_x:.2f}, {goal_y:.2f})")
        return path

    def get_navigation_command(self, obstacle_detected: bool = False) -> Optional[NavigationCommand]:
        if not self.waypoint_navigator.is_active():
            return None
        return self.waypoint_navigator.get_next_command(self._corrected_pose, obstacle_detected)

    def is_navigating(self) -> bool:
        return self.waypoint_navigator.is_active()

    def get_navigation_progress(self) -> dict:
        return self.waypoint_navigator.get_progress()

    def cancel_navigation(self) -> None:
        self.waypoint_navigator.reset()
        self.current_planned_path = None

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def get_map_visualization(self,
                               include_trajectory: bool = False,
                               include_planned_path: bool = True) -> np.ndarray:
        pose = self._corrected_pose
        vis = self.occupancy_grid.get_visualization(robot_pose=pose)

        if include_trajectory:
            traj = self.pose_graph.get_trajectory()
            for i in range(len(traj) - 1):
                p1, p2 = traj[i], traj[i + 1]
                gx1, gy1 = self.occupancy_grid.world_to_grid(p1.x, p1.y)
                gx2, gy2 = self.occupancy_grid.world_to_grid(p2.x, p2.y)
                gy1v = self.occupancy_grid.grid_height - gy1
                gy2v = self.occupancy_grid.grid_height - gy2
                cv2.line(vis, (gx1, gy1v), (gx2, gy2v), (255, 100, 0), 1)

            # Draw loop closure arcs
            for lc in self._loop_closures:
                kf_from = self.keyframe_store.get_by_id(lc["from_kf"])
                kf_to   = self.keyframe_store.get_by_id(lc["to_kf"])
                if kf_from and kf_to:
                    gx1, gy1 = self.occupancy_grid.world_to_grid(kf_from.pose.x, kf_from.pose.y)
                    gx2, gy2 = self.occupancy_grid.world_to_grid(kf_to.pose.x, kf_to.pose.y)
                    gy1v = self.occupancy_grid.grid_height - gy1
                    gy2v = self.occupancy_grid.grid_height - gy2
                    cv2.line(vis, (gx1, gy1v), (gx2, gy2v), (0, 0, 255), 1)  # Red = loop closure

        if include_planned_path and self.current_planned_path:
            for i in range(len(self.current_planned_path) - 1):
                x1, y1 = self.current_planned_path[i]
                x2, y2 = self.current_planned_path[i + 1]
                gx1, gy1 = self.occupancy_grid.world_to_grid(x1, y1)
                gx2, gy2 = self.occupancy_grid.world_to_grid(x2, y2)
                gy1v = self.occupancy_grid.grid_height - gy1
                gy2v = self.occupancy_grid.grid_height - gy2
                cv2.line(vis, (gx1, gy1v), (gx2, gy2v), (0, 255, 0), 2)

        return vis

    # ------------------------------------------------------------------
    # Stats / persistence
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        return {
            "slam": {
                "initialized": self.initialized,
                "current_pose": str(self._corrected_pose),
                "loop_closures": len(self._loop_closures),
                "pose_graph_nodes": self.pose_graph.node_count(),
                "pose_graph_edges": self.pose_graph.edge_count(),
            },
            "keyframes": len(self.keyframe_store),
            "point_cloud": {
                "points": self.point_cloud.point_count(),
            },
            "odometry": self.visual_odometry.get_statistics(),
            "map": self.occupancy_grid.get_statistics(),
        }

    def save_map(self, filepath: str) -> None:
        vis = self.get_map_visualization(include_trajectory=True)
        cv2.imwrite(filepath, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Map saved to {filepath}")

    def reset(self) -> None:
        self.visual_odometry.reset()
        self.keyframe_store.keyframes.clear()
        self.loop_detector.total_closures = 0
        self.pose_graph.__init__()
        self.pose_graph.add_node(0, Pose2D(0.0, 0.0, 0.0, time.time()), fixed=True)
        self._pg_node_id = 0
        self.point_cloud.reset()
        self.occupancy_grid = OccupancyGrid(
            width_m=self.occupancy_grid.width_m,
            height_m=self.occupancy_grid.height_m,
            resolution_m=self.occupancy_grid.resolution_m,
        )
        self._corrected_pose = Pose2D(0.0, 0.0, 0.0, time.time())
        self._loop_closures.clear()
        self.initialized = False
        self.logger.info("SLAM system reset")

    def __del__(self) -> None:
        if self.map_server:
            self.map_server.stop()
