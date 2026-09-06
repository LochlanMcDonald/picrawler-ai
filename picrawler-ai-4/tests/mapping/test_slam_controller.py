"""Integration tests for mapping/slam_controller.py — the full SLAM pipeline."""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import pytest

from mapping.slam_controller import SLAMController
from perception.depth_estimator import DepthMap

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "..", "..", "config", "config.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(shift: int = 0, h: int = 240, w: int = 320) -> np.ndarray:
    """Textured BGR frame; `shift` translates the texture to simulate motion."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, (h, w + 200, 3), dtype=np.uint8)
    for i in range(0, h, 30):
        for j in range(0, w + 200, 40):
            cv2.circle(base, (j, i), 6, (255, 255, 255), -1)
            cv2.rectangle(base, (j + 5, i + 5), (j + 20, i + 20), (0, 0, 0), 2)
    return np.ascontiguousarray(base[:, shift:shift + w])


def _depth(h: int = 240, w: int = 320, value: float = 0.3) -> DepthMap:
    return DepthMap(
        depth_array=np.full((h, w), value, dtype=np.float32),
        resolution=(w, h),
        inference_time_ms=1.0,
        timestamp=time.time(),
    )


def _config(**overrides) -> dict:
    cfg = {
        "slam_settings": {"map_size_m": 10.0, "map_resolution_m": 0.1,
                          "map_update_interval_s": 0.0},
        "point_cloud_settings": {"max_points": 20_000, "subsample": 8,
                                 "update_interval_s": 0.0},
        "map_server_settings": {"enabled": False},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_config(self):
        c = SLAMController()
        assert c.pose_graph.node_count() == 1          # origin node
        assert c.pose_graph.nodes[0].fixed
        assert c.map_server is None
        assert not c.initialized

    def test_repo_config_json_loads(self):
        """The shipped config.json must construct a controller without error."""
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        cfg["map_server_settings"]["enabled"] = False   # no network in tests
        c = SLAMController(cfg)
        assert c.keyframe_store.max_keyframes == cfg["keyframe_settings"]["max_keyframes"]
        assert c.loop_detector.min_inliers == cfg["loop_closure_settings"]["min_inliers"]
        assert c.point_cloud.subsample == cfg["point_cloud_settings"]["subsample"]

    def test_map_server_started_when_enabled(self):
        cfg = _config(map_server_settings={"enabled": True, "host": "127.0.0.1",
                                           "port": 5098, "broadcast_hz": 50.0})
        c = SLAMController(cfg)
        try:
            assert c.map_server is not None
            assert c.map_server._running
        finally:
            c.shutdown()
        assert c.map_server is None

    def test_shutdown_idempotent(self):
        c = SLAMController(_config())
        c.shutdown()
        c.shutdown()


# ---------------------------------------------------------------------------
# TestProcessFrame
# ---------------------------------------------------------------------------

class TestProcessFrame:
    def test_first_frame_initialises(self):
        c = SLAMController(_config())
        pose, vis = c.process_frame(_frame(0))
        assert c.initialized
        assert pose.x == 0.0 and pose.y == 0.0
        assert vis.ndim == 3 and vis.shape[2] == 3

    def test_pil_image_accepted(self):
        """Regression: CameraSystem.capture() returns a PIL Image, which used to
        raise "'Image' object has no attribute 'shape'" inside process_frame."""
        from PIL import Image
        c = SLAMController(_config())
        bgr = _frame(0)
        pil = Image.fromarray(bgr[:, :, ::-1])       # PIL is RGB
        pose, vis = c.process_frame(pil, _depth())
        assert c.initialized
        assert c.point_cloud.point_count() > 0
        # Colours must come through as RGB after the BGR round-trip
        cloud = c.point_cloud.get_cloud()
        assert cloud.colors.shape[1] == 3

    def test_unsupported_image_type_raises(self):
        c = SLAMController(_config())
        with pytest.raises(TypeError):
            c.process_frame("not an image")

    def test_grayscale_frame_accepted(self):
        c = SLAMController(_config())
        gray = cv2.cvtColor(_frame(0), cv2.COLOR_BGR2GRAY)
        pose, _ = c.process_frame(gray)
        assert pose is not None

    def test_depth_map_populates_point_cloud(self):
        """Regression: DepthMap exposes `depth_array`, and the cloud must fill from it."""
        c = SLAMController(_config())
        c.process_frame(_frame(0), _depth())
        assert c.point_cloud.point_count() > 0

    def test_raw_ndarray_depth_accepted(self):
        c = SLAMController(_config())
        c.process_frame(_frame(0), np.full((240, 320), 0.3, dtype=np.float32))
        assert c.point_cloud.point_count() > 0

    def test_no_depth_leaves_cloud_empty(self):
        c = SLAMController(_config())
        c.process_frame(_frame(0), None)
        assert c.point_cloud.point_count() == 0

    def test_motion_adds_pose_graph_nodes(self):
        c = SLAMController(_config())
        for s in range(0, 60, 6):
            c.process_frame(_frame(s), action_hint="forward")
        assert c.pose_graph.node_count() > 1
        assert c.pose_graph.edge_count() == c.pose_graph.node_count() - 1

    def test_first_keyframe_added(self):
        c = SLAMController(_config())
        c.process_frame(_frame(0))
        assert len(c.keyframe_store) == 1

    def test_occupancy_grid_marks_obstacles_from_cloud(self):
        c = SLAMController(_config())
        before = float(np.abs(c.occupancy_grid.grid).sum())
        for _ in range(3):
            c.process_frame(_frame(0), _depth(value=0.2))
        after = float(np.abs(c.occupancy_grid.grid).sum())
        assert after > before

    def test_statistics_shape(self):
        c = SLAMController(_config())
        c.process_frame(_frame(0), _depth())
        s = c.get_statistics()
        assert s["slam"]["initialized"] is True
        assert s["slam"]["pose_graph_nodes"] >= 1
        assert s["keyframes"] == 1
        assert s["point_cloud"]["points"] > 0
        assert "explored_percent" in s["map"]

    def test_stats_broadcast_to_map_server(self):
        cfg = _config(map_server_settings={"enabled": True, "host": "127.0.0.1",
                                           "port": 5097, "broadcast_hz": 0.1})
        c = SLAMController(cfg)
        try:
            c.process_frame(_frame(0), _depth())
            with c.map_server._lock:
                stats = c.map_server._pending_stats
                pose = c.map_server._pending_pose
                cloud = c.map_server._pending_cloud
            assert stats["keyframes"] == 1
            assert pose is not None
            assert cloud is not None and len(cloud.points) > 0
        finally:
            c.shutdown()


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_everything(self):
        c = SLAMController(_config())
        for s in range(0, 30, 6):
            c.process_frame(_frame(s), _depth(), action_hint="forward")
        c.reset()
        assert not c.initialized
        assert len(c.keyframe_store) == 0
        assert c.point_cloud.point_count() == 0
        assert c.pose_graph.node_count() == 1
        assert c.get_current_pose().x == 0.0

    def test_save_map_writes_file(self, tmp_path):
        c = SLAMController(_config())
        c.process_frame(_frame(0))
        out = tmp_path / "map.jpg"
        c.save_map(str(out))
        assert out.exists() and out.stat().st_size > 0
