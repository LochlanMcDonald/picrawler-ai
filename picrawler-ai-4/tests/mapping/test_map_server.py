"""Tests for mapping/map_server.py — MapServer UDP broadcast."""
from __future__ import annotations

import socket
import struct
import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mapping.map_server import (
    CLOUD_HEADER, POINT_BYTES, MapServer, _MSG_POSE, _MSG_CLOUD, _MSG_LOOP,
    _MSG_STATS, _POINTS_PER_PKT,
)
from mapping.point_cloud import PointCloud
from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Pose2D:
    return Pose2D(x, y, theta, time.time())


def _small_cloud(n: int = 5) -> PointCloud:
    pts = np.random.rand(n, 3).astype(np.float32)
    cols = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
    return PointCloud(points=pts, colors=cols)


def _empty_cloud() -> PointCloud:
    return PointCloud(
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# TestMessageTypeConstants
# ---------------------------------------------------------------------------

class TestMessageTypeConstants:
    def test_pose_byte(self):
        assert _MSG_POSE == b'\x01'

    def test_cloud_byte(self):
        assert _MSG_CLOUD == b'\x02'

    def test_loop_byte(self):
        assert _MSG_LOOP == b'\x03'

    def test_stats_byte(self):
        assert _MSG_STATS == b'\x04'


# ---------------------------------------------------------------------------
# TestMapServerInit
# ---------------------------------------------------------------------------

class TestMapServerInit:
    def test_defaults(self):
        server = MapServer()
        assert server.host == "0.0.0.0"
        assert server.port == 5005
        assert not server._running

    def test_custom_params(self):
        server = MapServer(host="127.0.0.1", port=9999, broadcast_hz=5.0)
        assert server.host == "127.0.0.1"
        assert server.port == 9999
        assert abs(server.interval - 0.2) < 1e-6

    def test_minimum_hz_clamped(self):
        server = MapServer(broadcast_hz=0.0)
        assert server.interval == 1.0 / 0.1   # clamped to 0.1 Hz

    def test_not_running_on_init(self):
        server = MapServer()
        assert not server._running
        assert server._thread is None


# ---------------------------------------------------------------------------
# TestMapServerLifecycle
# ---------------------------------------------------------------------------

class TestMapServerLifecycle:
    def test_start_sets_running(self):
        server = MapServer(host="127.0.0.1", broadcast_hz=100.0)
        try:
            server.start()
            assert server._running
            assert server._thread is not None
        finally:
            server.stop()

    def test_stop_clears_running(self):
        server = MapServer(host="127.0.0.1", broadcast_hz=100.0)
        server.start()
        server.stop()
        assert not server._running

    def test_double_start_is_safe(self):
        server = MapServer(host="127.0.0.1", broadcast_hz=100.0)
        try:
            server.start()
            server.start()  # Second start should be no-op
            assert server._running
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# TestUpdateMethods
# ---------------------------------------------------------------------------

class TestUpdateMethods:
    def setup_method(self):
        self.server = MapServer(host="127.0.0.1", broadcast_hz=1000.0)

    def test_update_pose_stores_pending(self):
        p = _pose(1.0, 2.0, 0.5)
        self.server.update_pose(p)
        with self.server._lock:
            assert self.server._pending_pose is p

    def test_update_cloud_stores_pending(self):
        cloud = _small_cloud()
        self.server.update_cloud(cloud)
        with self.server._lock:
            assert self.server._pending_cloud is cloud

    def test_update_loop_stores_tuple(self):
        from_p = _pose(0.0, 0.0)
        to_p = _pose(1.0, 1.0)
        self.server.update_loop(from_p, to_p)
        with self.server._lock:
            assert self.server._pending_loop == (0.0, 0.0, 1.0, 1.0)

    def test_update_stats_stores_dict(self):
        stats = {"nodes": 5, "loops": 1}
        self.server.update_stats(stats)
        with self.server._lock:
            assert self.server._pending_stats == stats

    def test_update_loop_coordinates(self):
        from_p = _pose(3.0, 4.0)
        to_p = _pose(7.0, 8.0)
        self.server.update_loop(from_p, to_p)
        with self.server._lock:
            fx, fy, tx, ty = self.server._pending_loop
        assert fx == 3.0
        assert fy == 4.0
        assert tx == 7.0
        assert ty == 8.0


# ---------------------------------------------------------------------------
# TestPacketBuilders (via _send captured)
# ---------------------------------------------------------------------------

class TestPacketBuilders:
    """Test the packet building methods by intercepting _send."""

    def setup_method(self):
        self.server = MapServer(host="127.0.0.1", port=5099)
        self.packets: list = []
        self.server._send = lambda data: self.packets.append(data)

    def test_send_pose_format(self):
        self.server._send_pose(_pose(1.0, 2.0, 0.5))
        assert len(self.packets) == 1
        pkt = self.packets[0]
        assert pkt[0:1] == _MSG_POSE
        x, y, t = struct.unpack("!3f", pkt[1:])
        assert abs(x - 1.0) < 1e-5
        assert abs(y - 2.0) < 1e-5
        assert abs(t - 0.5) < 1e-5

    def test_send_loop_format(self):
        self.server._send_loop(1.0, 2.0, 3.0, 4.0)
        assert len(self.packets) == 1
        pkt = self.packets[0]
        assert pkt[0:1] == _MSG_LOOP
        values = struct.unpack("!4f", pkt[1:])
        assert abs(values[0] - 1.0) < 1e-5
        assert abs(values[3] - 4.0) < 1e-5

    def test_send_cloud_type_byte(self):
        cloud = _small_cloud(3)
        self.server._send_cloud(cloud)
        assert len(self.packets) >= 1
        assert self.packets[0][0:1] == _MSG_CLOUD

    def test_send_cloud_empty_no_send(self):
        """Empty cloud should produce no packets (guarded in _flush)."""
        # _send_cloud is called with a non-empty cloud — test that it sends
        cloud = _empty_cloud()
        self.server._send_cloud(cloud)
        # 0 points → n=0 → range(0,0,N) = empty loop → no packets
        assert len(self.packets) == 0

    def test_send_stats_type_byte(self):
        self.server._send_stats({"foo": "bar"})
        assert len(self.packets) == 1
        pkt = self.packets[0]
        assert pkt[0:1] == _MSG_STATS

    def test_send_stats_json_content(self):
        import json
        self.server._send_stats({"nodes": 7})
        pkt = self.packets[0]
        payload = json.loads(pkt[1:].decode())
        assert payload["nodes"] == 7

    def test_cloud_point_data_layout(self):
        """Each point should be encoded as 6 float32s (x, y, z, r, g, b)."""
        cloud = PointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            colors=np.array([[255, 128, 64]], dtype=np.uint8),
        )
        self.server._send_cloud(cloud)
        pkt = self.packets[0]
        # 1 byte type + 12 byte header + 6 float32 = 37 bytes
        assert len(pkt) == 1 + CLOUD_HEADER.size + 6 * 4
        seq, batch_idx, batch_count = CLOUD_HEADER.unpack(pkt[1:1 + CLOUD_HEADER.size])
        assert seq == 0
        assert batch_idx == 0
        assert batch_count == 1
        vals = np.frombuffer(pkt[1 + CLOUD_HEADER.size:], dtype="<f4")
        assert abs(vals[0] - 1.0) < 1e-5  # x
        assert abs(vals[1] - 2.0) < 1e-5  # y
        assert abs(vals[2] - 3.0) < 1e-5  # z
        assert abs(vals[3] - 255.0) < 1e-4  # r
        assert abs(vals[4] - 128.0) < 1e-4  # g
        assert abs(vals[5] - 64.0) < 1e-4   # b

    def test_cloud_seq_increments_per_snapshot(self):
        self.server._send_cloud(_small_cloud(2))
        self.server._send_cloud(_small_cloud(2))
        seqs = [CLOUD_HEADER.unpack(p[1:1 + CLOUD_HEADER.size])[0] for p in self.packets]
        assert seqs == [0, 1]

    def test_large_cloud_split_into_batches(self):
        n = _POINTS_PER_PKT * 2 + 5
        self.server._send_cloud(_small_cloud(n))
        assert len(self.packets) == 3
        headers = [CLOUD_HEADER.unpack(p[1:1 + CLOUD_HEADER.size]) for p in self.packets]
        assert [h[1] for h in headers] == [0, 1, 2]
        assert all(h[2] == 3 for h in headers)
        assert all(h[0] == headers[0][0] for h in headers)
        total_points = sum(
            (len(p) - 1 - CLOUD_HEADER.size) // POINT_BYTES for p in self.packets
        )
        assert total_points == n


# ---------------------------------------------------------------------------
# TestFlush
# ---------------------------------------------------------------------------

class TestFlush:
    def setup_method(self):
        self.server = MapServer(host="127.0.0.1", port=5099)
        self.packets: list = []
        self.server._send = lambda data: self.packets.append(data)

    def test_flush_sends_pose_and_clears_nothing(self):
        self.server._pending_pose = _pose(1.0)
        self.server._flush()
        assert any(p[0:1] == _MSG_POSE for p in self.packets)
        # pose is NOT cleared (persists for next flush)
        assert self.server._pending_pose is not None

    def test_flush_sends_loop_then_clears_it(self):
        self.server._pending_loop = (0.0, 0.0, 1.0, 1.0)
        self.server._flush()
        assert any(p[0:1] == _MSG_LOOP for p in self.packets)
        assert self.server._pending_loop is None

    def test_flush_sends_stats_then_clears(self):
        self.server._pending_stats = {"x": 1}
        self.server._flush()
        assert any(p[0:1] == _MSG_STATS for p in self.packets)
        assert self.server._pending_stats is None

    def test_flush_skips_empty_cloud(self):
        self.server._pending_cloud = _empty_cloud()
        self.server._flush()
        assert not any(p[0:1] == _MSG_CLOUD for p in self.packets)

    def test_flush_sends_nonempty_cloud(self):
        self.server._pending_cloud = _small_cloud(5)
        self.server._flush()
        assert any(p[0:1] == _MSG_CLOUD for p in self.packets)


# ---------------------------------------------------------------------------
# TestUDPSendError
# ---------------------------------------------------------------------------

class TestUDPSendError:
    def test_oserror_logged_not_raised(self):
        server = MapServer(host="127.0.0.1", port=5099)
        server._sock = MagicMock()
        server._sock.sendto.side_effect = OSError("network error")
        # Should not raise
        server._send(b'\x01\x00\x00\x00\x00')
