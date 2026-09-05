"""Tests for tools/map_viewer.py — MapDecoder, driven end-to-end by MapServer packets."""
from __future__ import annotations

import json
import struct
import time

import numpy as np
import pytest

from mapping.map_server import MapServer, _POINTS_PER_PKT
from mapping.point_cloud import PointCloud
from perception.visual_odometry import Pose2D
from tools.map_viewer import (
    CLOUD_HEADER, MSG_CLOUD, MSG_LOOP, MSG_POSE, MSG_STATS, MapDecoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose(x=0.0, y=0.0, theta=0.0) -> Pose2D:
    return Pose2D(x, y, theta, time.time())


def _cloud(n: int, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3)).astype(np.float32)
    cols = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    return PointCloud(points=pts, colors=cols)


@pytest.fixture
def server_and_packets():
    """MapServer whose _send is captured into a list instead of hitting the network."""
    server = MapServer(host="127.0.0.1", port=5099)
    packets: list = []
    server._send = packets.append
    yield server, packets
    server._sock.close()


# ---------------------------------------------------------------------------
# TestFeedValidation
# ---------------------------------------------------------------------------

class TestFeedValidation:
    def test_empty_datagram_is_bad(self):
        d = MapDecoder()
        assert d.feed(b"") is None
        assert d.bad_packets == 1
        assert d.packets == 0

    def test_unknown_type_is_bad(self):
        d = MapDecoder()
        assert d.feed(b"\x09abc") is None
        assert d.bad_packets == 1

    def test_truncated_pose_is_bad(self):
        d = MapDecoder()
        assert d.feed(bytes([MSG_POSE]) + b"\x00\x00") is None
        assert d.bad_packets == 1
        assert d.pose is None

    def test_malformed_stats_json_is_bad(self):
        d = MapDecoder()
        assert d.feed(bytes([MSG_STATS]) + b"{not json") is None
        assert d.bad_packets == 1

    def test_packet_counter_and_timestamp(self):
        d = MapDecoder()
        before = time.time()
        d.feed(bytes([MSG_POSE]) + struct.pack("!3f", 0, 0, 0))
        assert d.packets == 1
        assert d.last_rx >= before


# ---------------------------------------------------------------------------
# TestPose
# ---------------------------------------------------------------------------

class TestPose:
    def test_pose_decoded_from_server_packet(self, server_and_packets):
        server, packets = server_and_packets
        server._send_pose(_pose(1.5, -2.0, 0.75))
        d = MapDecoder()
        assert d.feed(packets[0]) == MSG_POSE
        x, y, th = d.pose
        assert abs(x - 1.5) < 1e-5
        assert abs(y + 2.0) < 1e-5
        assert abs(th - 0.75) < 1e-5

    def test_trajectory_grows_on_movement(self, server_and_packets):
        server, packets = server_and_packets
        for i in range(4):
            server._send_pose(_pose(float(i), 0.0))
        d = MapDecoder()
        for p in packets:
            d.feed(p)
        assert d.trajectory == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]

    def test_stationary_pose_not_duplicated(self, server_and_packets):
        server, packets = server_and_packets
        for _ in range(5):
            server._send_pose(_pose(1.0, 1.0))
        d = MapDecoder()
        for p in packets:
            d.feed(p)
        assert len(d.trajectory) == 1

    def test_trajectory_capped(self):
        d = MapDecoder(max_trajectory=10)
        for i in range(25):
            d.feed(bytes([MSG_POSE]) + struct.pack("!3f", float(i), 0.0, 0.0))
        assert len(d.trajectory) == 10
        assert d.trajectory[-1] == (24.0, 0.0)


# ---------------------------------------------------------------------------
# TestLoopAndStats
# ---------------------------------------------------------------------------

class TestLoopAndStats:
    def test_loop_decoded(self, server_and_packets):
        server, packets = server_and_packets
        server._send_loop(0.0, 0.0, 3.0, 4.0)
        d = MapDecoder()
        assert d.feed(packets[0]) == MSG_LOOP
        assert len(d.loops) == 1
        fx, fy, tx, ty = d.loops[0]
        assert (fx, fy) == (0.0, 0.0)
        assert abs(tx - 3.0) < 1e-5 and abs(ty - 4.0) < 1e-5

    def test_loops_accumulate(self, server_and_packets):
        server, packets = server_and_packets
        server._send_loop(0, 0, 1, 1)
        server._send_loop(2, 2, 3, 3)
        d = MapDecoder()
        for p in packets:
            d.feed(p)
        assert len(d.loops) == 2

    def test_stats_decoded(self, server_and_packets):
        server, packets = server_and_packets
        server._send_stats({"keyframes": 12, "pose_graph_nodes": 40})
        d = MapDecoder()
        assert d.feed(packets[0]) == MSG_STATS
        assert d.stats == {"keyframes": 12, "pose_graph_nodes": 40}


# ---------------------------------------------------------------------------
# TestCloud
# ---------------------------------------------------------------------------

class TestCloud:
    def test_empty_by_default(self):
        d = MapDecoder()
        assert d.cloud().shape == (0, 6)
        assert d.points_xyz().shape == (0, 3)
        assert d.colors_rgb01().shape == (0, 3)

    def test_single_batch_roundtrip(self, server_and_packets):
        server, packets = server_and_packets
        cloud = _cloud(50)
        server._send_cloud(cloud)
        assert len(packets) == 1
        d = MapDecoder()
        assert d.feed(packets[0]) == MSG_CLOUD
        np.testing.assert_allclose(d.points_xyz(), cloud.points, atol=1e-6)
        np.testing.assert_array_equal(d.cloud()[:, 3:].astype(np.uint8), cloud.colors)

    def test_colors_scaled_to_unit_range(self, server_and_packets):
        server, packets = server_and_packets
        server._send_cloud(_cloud(20))
        d = MapDecoder()
        d.feed(packets[0])
        c = d.colors_rgb01()
        assert c.min() >= 0.0 and c.max() <= 1.0

    def test_multi_batch_reassembled_in_order(self, server_and_packets):
        server, packets = server_and_packets
        n = _POINTS_PER_PKT * 2 + 7
        cloud = _cloud(n, seed=3)
        server._send_cloud(cloud)
        assert len(packets) == 3
        d = MapDecoder()
        # Deliver out of order to make sure batch_idx ordering is honoured
        for p in (packets[2], packets[0], packets[1]):
            d.feed(p)
        assert len(d.cloud()) == n
        np.testing.assert_allclose(d.points_xyz(), cloud.points, atol=1e-6)

    def test_new_seq_replaces_old_snapshot(self, server_and_packets):
        server, packets = server_and_packets
        server._send_cloud(_cloud(30, seed=1))
        server._send_cloud(_cloud(10, seed=2))
        d = MapDecoder()
        d.feed(packets[0])
        assert len(d.cloud()) == 30
        d.feed(packets[1])
        assert len(d.cloud()) == 10

    def test_partial_snapshot_is_shown_progressively(self, server_and_packets):
        server, packets = server_and_packets
        n = _POINTS_PER_PKT + 1
        server._send_cloud(_cloud(n))
        d = MapDecoder()
        d.feed(packets[0])
        assert len(d.cloud()) == _POINTS_PER_PKT
        d.feed(packets[1])
        assert len(d.cloud()) == n

    def test_trailing_garbage_bytes_ignored(self):
        d = MapDecoder()
        header = CLOUD_HEADER.pack(0, 0, 1)
        one_point = np.array([[1, 2, 3, 4, 5, 6]], dtype="<f4").tobytes()
        d.feed(bytes([MSG_CLOUD]) + header + one_point + b"\x00\x01\x02")
        assert len(d.cloud()) == 1


# ---------------------------------------------------------------------------
# TestExport
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_ply(self, tmp_path, server_and_packets):
        server, packets = server_and_packets
        server._send_cloud(_cloud(5))
        d = MapDecoder()
        d.feed(packets[0])
        out = tmp_path / "map.ply"
        n = d.export_ply(str(out))
        assert n == 5
        text = out.read_text().splitlines()
        assert text[0] == "ply"
        assert "element vertex 5" in text
        assert text.index("end_header") == 9
        assert len(text) == 10 + 5

    def test_export_empty_cloud(self, tmp_path):
        d = MapDecoder()
        out = tmp_path / "empty.ply"
        assert d.export_ply(str(out)) == 0
        assert "element vertex 0" in out.read_text()
