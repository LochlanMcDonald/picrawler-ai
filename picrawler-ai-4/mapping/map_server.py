"""
Real-time map server — streams point cloud and pose graph to a host machine.

Sends compact binary UDP datagrams so the host can visualise the map
as the robot builds it.  No dependencies beyond stdlib + numpy.

Protocol
--------
Each datagram starts with a 1-byte message type:

  0x01  POSE     — current robot pose (x, y, theta) as 3× float32
  0x02  CLOUD    — batch of point cloud points (x, y, z, r, g, b) per point
  0x03  LOOP     — loop closure event (from_x, from_y, to_x, to_y) as 4× float32
  0x04  STATS    — ASCII JSON stats string

The host can receive these with the companion `map_viewer.py` script
or any UDP listener on the configured port.
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from mapping.point_cloud import PointCloud
from perception.visual_odometry import Pose2D

# Message type bytes
_MSG_POSE  = b'\x01'
_MSG_CLOUD = b'\x02'
_MSG_LOOP  = b'\x03'
_MSG_STATS = b'\x04'

# Max UDP payload (stay under typical MTU)
_MAX_UDP = 60_000
# Points per UDP packet  (6 × float32 = 24 bytes each)
_POINTS_PER_PKT = _MAX_UDP // 24


class MapServer:
    """
    Broadcasts SLAM map data to a host viewer over UDP.

    Thread-safe: call update_*() from the robot's SLAM loop;
    the server sends asynchronously on a background thread.
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 5005,
                 broadcast_hz: float = 2.0):
        """
        Args:
            host: Destination IP (use "255.255.255.255" for LAN broadcast,
                  or the host machine's IP for direct send)
            port: UDP port
            broadcast_hz: How often to send cloud updates
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.host = host
        self.port = port
        self.interval = 1.0 / max(broadcast_hz, 0.1)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._lock = threading.Lock()
        self._pending_pose: Optional[Pose2D] = None
        self._pending_cloud: Optional[PointCloud] = None
        self._pending_loop: Optional[Tuple[float, float, float, float]] = None
        self._pending_stats: Optional[dict] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="MapServer")
        self._thread.start()
        self.logger.info(f"Map server started → {self.host}:{self.port} @ {1/self.interval:.1f} Hz")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()
        self.logger.info("Map server stopped")

    # ------------------------------------------------------------------
    # Update API (called from SLAM loop)
    # ------------------------------------------------------------------

    def update_pose(self, pose: Pose2D) -> None:
        with self._lock:
            self._pending_pose = pose

    def update_cloud(self, cloud: PointCloud) -> None:
        with self._lock:
            self._pending_cloud = cloud

    def update_loop(self,
                    from_pose: Pose2D,
                    to_pose: Pose2D) -> None:
        with self._lock:
            self._pending_loop = (from_pose.x, from_pose.y, to_pose.x, to_pose.y)

    def update_stats(self, stats: dict) -> None:
        with self._lock:
            self._pending_stats = stats

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            t0 = time.time()
            self._flush()
            elapsed = time.time() - t0
            remaining = self.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _flush(self) -> None:
        with self._lock:
            pose = self._pending_pose
            cloud = self._pending_cloud
            loop = self._pending_loop
            stats = self._pending_stats
            self._pending_loop = None   # Send loops once
            self._pending_stats = None

        if pose is not None:
            self._send_pose(pose)

        if loop is not None:
            self._send_loop(*loop)

        if cloud is not None and len(cloud.points) > 0:
            self._send_cloud(cloud)

        if stats is not None:
            self._send_stats(stats)

    # ------------------------------------------------------------------
    # Packet builders
    # ------------------------------------------------------------------

    def _send(self, data: bytes) -> None:
        try:
            self._sock.sendto(data, (self.host, self.port))
        except OSError as e:
            self.logger.debug(f"UDP send failed: {e}")

    def _send_pose(self, pose: Pose2D) -> None:
        payload = struct.pack("!3f", pose.x, pose.y, pose.theta)
        self._send(_MSG_POSE + payload)

    def _send_loop(self, fx: float, fy: float, tx: float, ty: float) -> None:
        payload = struct.pack("!4f", fx, fy, tx, ty)
        self._send(_MSG_LOOP + payload)

    def _send_cloud(self, cloud: PointCloud) -> None:
        pts = cloud.points.astype(np.float32)
        cols = cloud.colors.astype(np.uint8)
        n = len(pts)

        for start in range(0, n, _POINTS_PER_PKT):
            end = min(start + _POINTS_PER_PKT, n)
            batch_pts = pts[start:end]
            batch_cols = cols[start:end]

            # Interleave x,y,z (float32) and r,g,b (uint8 packed into float32 slot)
            # Format per point: x y z r g b  — 6× float32 (b/g/r as floats 0-255)
            chunk = np.empty((end - start, 6), dtype=np.float32)
            chunk[:, :3] = batch_pts
            chunk[:, 3] = batch_cols[:, 0].astype(np.float32)
            chunk[:, 4] = batch_cols[:, 1].astype(np.float32)
            chunk[:, 5] = batch_cols[:, 2].astype(np.float32)

            self._send(_MSG_CLOUD + chunk.tobytes())

    def _send_stats(self, stats: dict) -> None:
        try:
            payload = json.dumps(stats, default=str).encode()[:_MAX_UDP]
            self._send(_MSG_STATS + payload)
        except Exception:
            pass
