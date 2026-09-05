#!/usr/bin/env python3
"""
Host-side live SLAM map viewer.

Run this on your laptop / desktop (NOT on the robot).  It listens for the
UDP stream broadcast by ``mapping/map_server.py`` and renders:

  * Left  — top-down 2D map: coloured point cloud, robot trajectory (blue),
            current pose (green arrow), loop closures (red lines)
  * Right — 3D point cloud (disable with --no-3d on slow machines)

Usage
-----
    pip install numpy matplotlib
    python tools/map_viewer.py                 # listen on UDP 5005
    python tools/map_viewer.py --port 6000     # custom port
    python tools/map_viewer.py --no-3d         # 2D only (faster)
    python tools/map_viewer.py --export map.ply  # save cloud on exit

Only numpy + matplotlib are needed on the host — nothing else from this repo.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants (mirrors mapping/map_server.py — kept local so the
# viewer can run on a host without the robot code installed)
# ---------------------------------------------------------------------------

MSG_POSE = 0x01
MSG_CLOUD = 0x02
MSG_LOOP = 0x03
MSG_STATS = 0x04

POSE_STRUCT = struct.Struct("!3f")
LOOP_STRUCT = struct.Struct("!4f")
CLOUD_HEADER = struct.Struct("!III")
POINT_DTYPE = np.dtype("<f4")
POINT_FLOATS = 6


# ---------------------------------------------------------------------------
# Decoder — pure data, no networking or plotting, so it is unit-testable
# ---------------------------------------------------------------------------

class MapDecoder:
    """Accumulates map state from raw datagrams."""

    def __init__(self, max_trajectory: int = 5000):
        self.max_trajectory = max_trajectory
        self.pose: Optional[Tuple[float, float, float]] = None
        self.trajectory: List[Tuple[float, float]] = []
        self.loops: List[Tuple[float, float, float, float]] = []
        self.stats: Dict = {}

        # Cloud reassembly
        self._cloud_seq: Optional[int] = None
        self._cloud_batches: Dict[int, np.ndarray] = {}
        self._cloud_batch_count = 0
        self._latest_cloud: np.ndarray = np.zeros((0, POINT_FLOATS), dtype=np.float32)

        self.packets = 0
        self.bad_packets = 0
        self.last_rx: float = 0.0

    # -- public -----------------------------------------------------------

    def feed(self, data: bytes) -> Optional[int]:
        """Decode one datagram. Returns the message type or None if invalid."""
        if not data:
            self.bad_packets += 1
            return None
        self.packets += 1
        self.last_rx = time.time()
        mtype = data[0]
        payload = data[1:]
        try:
            if mtype == MSG_POSE:
                self._on_pose(payload)
            elif mtype == MSG_CLOUD:
                self._on_cloud(payload)
            elif mtype == MSG_LOOP:
                self._on_loop(payload)
            elif mtype == MSG_STATS:
                self._on_stats(payload)
            else:
                self.bad_packets += 1
                return None
        except (struct.error, ValueError, json.JSONDecodeError):
            self.bad_packets += 1
            return None
        return mtype

    def cloud(self) -> np.ndarray:
        """Return the most recent complete-or-partial cloud as (N, 6) float32."""
        return self._latest_cloud

    def points_xyz(self) -> np.ndarray:
        return self._latest_cloud[:, :3]

    def colors_rgb01(self) -> np.ndarray:
        """Colours scaled to 0-1 for matplotlib."""
        if len(self._latest_cloud) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.clip(self._latest_cloud[:, 3:6] / 255.0, 0.0, 1.0)

    # -- handlers ---------------------------------------------------------

    def _on_pose(self, payload: bytes) -> None:
        x, y, theta = POSE_STRUCT.unpack(payload[:POSE_STRUCT.size])
        self.pose = (x, y, theta)
        if not self.trajectory or (
            abs(self.trajectory[-1][0] - x) > 1e-4
            or abs(self.trajectory[-1][1] - y) > 1e-4
        ):
            self.trajectory.append((x, y))
            if len(self.trajectory) > self.max_trajectory:
                self.trajectory = self.trajectory[-self.max_trajectory:]

    def _on_loop(self, payload: bytes) -> None:
        self.loops.append(LOOP_STRUCT.unpack(payload[:LOOP_STRUCT.size]))

    def _on_stats(self, payload: bytes) -> None:
        self.stats = json.loads(payload.decode("utf-8", errors="replace"))

    def _on_cloud(self, payload: bytes) -> None:
        seq, batch_idx, batch_count = CLOUD_HEADER.unpack(payload[:CLOUD_HEADER.size])
        body = payload[CLOUD_HEADER.size:]
        usable = (len(body) // (POINT_FLOATS * 4)) * POINT_FLOATS * 4
        pts = np.frombuffer(body[:usable], dtype=POINT_DTYPE).reshape(-1, POINT_FLOATS)

        if seq != self._cloud_seq:
            # New snapshot begins — discard partial previous one
            self._cloud_seq = seq
            self._cloud_batches = {}
            self._cloud_batch_count = batch_count

        self._cloud_batches[batch_idx] = pts
        ordered = [self._cloud_batches[i] for i in sorted(self._cloud_batches)]
        self._latest_cloud = (
            np.concatenate(ordered, axis=0) if ordered
            else np.zeros((0, POINT_FLOATS), dtype=np.float32)
        )

    # -- export -----------------------------------------------------------

    def export_ply(self, path: str) -> int:
        """Write the current cloud as an ASCII PLY file. Returns point count."""
        cloud = self._latest_cloud
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(cloud)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for x, y, z, r, g, b in cloud:
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
        return len(cloud)


# ---------------------------------------------------------------------------
# UDP receiver thread
# ---------------------------------------------------------------------------

class UdpReceiver(threading.Thread):
    def __init__(self, decoder: MapDecoder, port: int, bind: str = "0.0.0.0"):
        super().__init__(daemon=True, name="MapViewerRx")
        self.decoder = decoder
        self.lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        self._sock.bind((bind, port))
        self._sock.settimeout(0.5)
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                self.decoder.feed(data)

    def stop(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Matplotlib viewer
# ---------------------------------------------------------------------------

def run_viewer(port: int, show_3d: bool, refresh_hz: float,
               export_path: Optional[str], max_draw_points: int) -> int:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib is required:  pip install matplotlib", file=sys.stderr)
        return 1

    decoder = MapDecoder()
    rx = UdpReceiver(decoder, port)
    rx.start()
    print(f"Listening for robot map stream on UDP :{port}  (Ctrl+C to quit)")

    if show_3d:
        fig = plt.figure(figsize=(14, 7))
        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig = plt.figure(figsize=(8, 8))
        ax2d = fig.add_subplot(1, 1, 1)
        ax3d = None

    fig.canvas.manager.set_window_title("PiCrawler SLAM — live map")

    def _subsample(pts: np.ndarray, cols: np.ndarray):
        if len(pts) > max_draw_points:
            idx = np.linspace(0, len(pts) - 1, max_draw_points).astype(int)
            return pts[idx], cols[idx]
        return pts, cols

    def _draw(_frame):
        with rx.lock:
            pts = decoder.points_xyz().copy()
            cols = decoder.colors_rgb01().copy()
            traj = list(decoder.trajectory)
            loops = list(decoder.loops)
            pose = decoder.pose
            stats = dict(decoder.stats)
            age = time.time() - decoder.last_rx if decoder.last_rx else None

        pts, cols = _subsample(pts, cols)

        # ---- 2D top-down -------------------------------------------------
        ax2d.cla()
        ax2d.set_title("Top-down map (m)")
        ax2d.set_xlabel("X (forward at start)")
        ax2d.set_ylabel("Y (left at start)")
        ax2d.set_aspect("equal", adjustable="datalim")
        ax2d.grid(True, alpha=0.3)

        if len(pts):
            ax2d.scatter(pts[:, 0], pts[:, 1], c=cols, s=2, linewidths=0)
        if len(traj) > 1:
            t = np.asarray(traj)
            ax2d.plot(t[:, 0], t[:, 1], color="#1f77ff", linewidth=1.5, label="trajectory")
        for fx, fy, tx, ty in loops:
            ax2d.plot([fx, tx], [fy, ty], color="red", linewidth=2.0, alpha=0.9)
            ax2d.scatter([fx, tx], [fy, ty], color="red", s=30, zorder=5)
        if pose is not None:
            x, y, th = pose
            ax2d.arrow(x, y, 0.25 * np.cos(th), 0.25 * np.sin(th),
                       head_width=0.08, color="#00c853", zorder=6)
        if loops:
            ax2d.plot([], [], color="red", linewidth=2, label=f"loop closures ({len(loops)})")
        if traj or loops:
            ax2d.legend(loc="upper right", fontsize=8)

        status = f"points: {len(decoder.cloud()):,}   poses: {len(traj)}   loops: {len(loops)}"
        if age is not None:
            status += f"   last packet: {age:.1f}s ago"
        else:
            status += "   waiting for robot…"
        if stats:
            status += (f"   kf: {stats.get('keyframes', '?')}"
                       f"  nodes: {stats.get('pose_graph_nodes', '?')}")
        ax2d.text(0.01, 0.01, status, transform=ax2d.transAxes, fontsize=8,
                  family="monospace", va="bottom",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        # ---- 3D --------------------------------------------------------
        if ax3d is not None:
            ax3d.cla()
            ax3d.set_title("3D point cloud")
            ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
            if len(pts):
                ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1, linewidths=0)
                span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                ax3d.set_xlim(cx - span / 2, cx + span / 2)
                ax3d.set_ylim(cy - span / 2, cy + span / 2)
                ax3d.set_zlim(0, max(1.0, float(pts[:, 2].max())))
            if len(traj) > 1:
                t = np.asarray(traj)
                ax3d.plot(t[:, 0], t[:, 1], np.zeros(len(t)), color="#1f77ff", linewidth=1.5)
            for fx, fy, tx, ty in loops:
                ax3d.plot([fx, tx], [fy, ty], [0, 0], color="red", linewidth=2.0)

    interval_ms = int(1000 / max(refresh_hz, 0.2))
    anim = FuncAnimation(fig, _draw, interval=interval_ms, cache_frame_data=False)  # noqa: F841

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        if export_path:
            with rx.lock:
                n = decoder.export_ply(export_path)
            print(f"Saved {n:,} points to {export_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Live viewer for the PiCrawler SLAM map stream")
    p.add_argument("--port", type=int, default=5005, help="UDP port (default 5005)")
    p.add_argument("--no-3d", action="store_true", help="Disable the 3D panel")
    p.add_argument("--hz", type=float, default=2.0, help="Redraw rate (default 2 Hz)")
    p.add_argument("--max-points", type=int, default=60_000,
                   help="Max points drawn per frame (subsampled above this)")
    p.add_argument("--export", type=str, default=None,
                   help="Write the point cloud to this .ply file on exit")
    args = p.parse_args(argv)
    return run_viewer(args.port, not args.no_3d, args.hz, args.export, args.max_points)


if __name__ == "__main__":
    sys.exit(main())
