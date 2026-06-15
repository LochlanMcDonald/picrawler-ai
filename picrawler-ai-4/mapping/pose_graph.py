"""
Pose graph SLAM with Gauss-Seidel optimization.

Maintains a graph of robot poses connected by:
  - Odometry edges (from visual odometry, between consecutive poses)
  - Loop closure edges (from LoopClosureDetector, between revisited places)

When a loop closure is added, runs a lightweight iterative optimizer to
distribute the accumulated drift across all poses in the loop.

No external libraries required — pure numpy.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PoseNode:
    """A node in the pose graph."""
    id: int
    pose: Pose2D
    fixed: bool = False   # Fixed nodes (e.g. origin) are not optimised


@dataclass
class PoseEdge:
    """A directed edge between two pose nodes."""
    from_id: int
    to_id: int
    dx: float          # Relative translation x (metres)
    dy: float          # Relative translation y
    dtheta: float      # Relative rotation (radians)
    info_xx: float = 100.0   # Information matrix diagonal (translation)
    info_yy: float = 100.0
    info_tt: float = 50.0    # Information matrix diagonal (rotation)
    is_loop: bool = False    # True for loop closure edges


# ---------------------------------------------------------------------------
# Pose graph
# ---------------------------------------------------------------------------

class PoseGraph:
    """
    Lightweight pose graph for 2D SLAM.

    Usage:
        graph = PoseGraph()
        graph.add_node(0, initial_pose, fixed=True)

        # After each odometry step:
        graph.add_node(new_id, estimated_pose)
        graph.add_odometry_edge(prev_id, new_id, relative_measurement)

        # When loop closure detected:
        graph.add_loop_edge(current_id, matched_id, relative_measurement)
        corrected = graph.optimize()          # Returns corrected poses
    """

    def __init__(self, max_iterations: int = 100, convergence_tol: float = 1e-4):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

        self.nodes: Dict[int, PoseNode] = {}
        self.edges: List[PoseEdge] = []
        self._loop_count = 0

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node_id: int, pose: Pose2D, fixed: bool = False) -> None:
        self.nodes[node_id] = PoseNode(
            id=node_id,
            pose=Pose2D(pose.x, pose.y, pose.theta, pose.timestamp),
            fixed=fixed,
        )

    def add_odometry_edge(self,
                          from_id: int,
                          to_id: int,
                          relative: Tuple[float, float, float]) -> None:
        """Add an odometry constraint between consecutive poses.

        Args:
            from_id, to_id: Node IDs
            relative: (dx, dy, dtheta) in the from-node's frame
        """
        dx, dy, dtheta = relative
        self.edges.append(PoseEdge(
            from_id=from_id,
            to_id=to_id,
            dx=dx, dy=dy, dtheta=dtheta,
            is_loop=False,
        ))

    def add_loop_edge(self,
                      from_id: int,
                      to_id: int,
                      relative: Tuple[float, float, float],
                      score: float = 0.5) -> None:
        """Add a loop closure constraint.

        Args:
            from_id, to_id: Node IDs (from current, to previously seen)
            relative: (dx, dy, dtheta) in the from-node's frame
            score: Detection confidence 0–1 (used to weight the edge)
        """
        dx, dy, dtheta = relative
        weight = max(0.1, score)
        self.edges.append(PoseEdge(
            from_id=from_id,
            to_id=to_id,
            dx=dx, dy=dy, dtheta=dtheta,
            info_xx=100.0 * weight,
            info_yy=100.0 * weight,
            info_tt=50.0 * weight,
            is_loop=True,
        ))
        self._loop_count += 1
        self.logger.info(
            f"Loop edge added: {from_id} → {to_id}  "
            f"score={score:.3f}  total_loops={self._loop_count}"
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self) -> Dict[int, Pose2D]:
        """
        Run Gauss-Seidel pose graph optimization.

        Minimises the sum of squared errors across all edges.
        Each non-fixed node is updated to reduce the residual on all
        its incident edges.

        Returns:
            Dict mapping node_id → corrected Pose2D
        """
        if len(self.nodes) < 2 or len(self.edges) < 1:
            return {nid: n.pose for nid, n in self.nodes.items()}

        # Work on mutable copies
        poses: Dict[int, List[float]] = {
            nid: [n.pose.x, n.pose.y, n.pose.theta]
            for nid, n in self.nodes.items()
        }

        # Build adjacency: node_id → [(edge, is_from)]
        adj: Dict[int, List[Tuple[PoseEdge, bool]]] = {
            nid: [] for nid in self.nodes
        }
        for e in self.edges:
            if e.from_id in adj:
                adj[e.from_id].append((e, True))
            if e.to_id in adj:
                adj[e.to_id].append((e, False))

        prev_total_err = float("inf")

        for iteration in range(self.max_iterations):
            total_err = 0.0

            for nid, node in self.nodes.items():
                if node.fixed:
                    continue

                grad_x = grad_y = grad_t = 0.0
                weight_sum = 0.0

                for edge, is_from in adj[nid]:
                    fi, ti = edge.from_id, edge.to_id
                    fx, fy, ft = poses[fi]
                    tx, ty, tt = poses[ti]

                    # Predict the edge measurement from current poses
                    c, s = math.cos(ft), math.sin(ft)
                    pred_dx = c * (tx - fx) + s * (ty - fy)
                    pred_dy = -s * (tx - fx) + c * (ty - fy)
                    pred_dt = _angle_diff(tt, ft)

                    # Residual
                    res_x = edge.dx - pred_dx
                    res_y = edge.dy - pred_dy
                    res_t = _angle_diff(edge.dtheta, pred_dt)

                    total_err += (
                        edge.info_xx * res_x ** 2 +
                        edge.info_yy * res_y ** 2 +
                        edge.info_tt * res_t ** 2
                    )

                    # Gradient of error w.r.t. the current node's pose
                    if is_from:
                        # Moving from-node: nid == fi
                        grad_x += edge.info_xx * res_x * (-c) + edge.info_yy * res_y * s
                        grad_y += edge.info_xx * res_x * (-s) + edge.info_yy * res_y * (-c)
                        grad_t += edge.info_tt * res_t * (-1.0)
                    else:
                        # Moving to-node: nid == ti
                        grad_x += edge.info_xx * res_x * c + edge.info_yy * res_y * (-s)
                        grad_y += edge.info_xx * res_x * s + edge.info_yy * res_y * c
                        grad_t += edge.info_tt * res_t

                    weight_sum += edge.info_xx + edge.info_yy + edge.info_tt

                if weight_sum < 1e-9:
                    continue

                # Gradient step (dampened)
                lr = 0.1
                poses[nid][0] += lr * grad_x / weight_sum
                poses[nid][1] += lr * grad_y / weight_sum
                poses[nid][2] = _wrap_angle(
                    poses[nid][2] + lr * grad_t / weight_sum
                )

            # Convergence check
            delta = abs(prev_total_err - total_err)
            if delta < self.convergence_tol and iteration > 5:
                self.logger.debug(
                    f"Pose graph converged at iteration {iteration}, "
                    f"error={total_err:.4f}"
                )
                break
            prev_total_err = total_err

        # Write results back to nodes and return
        result: Dict[int, Pose2D] = {}
        for nid, node in self.nodes.items():
            x, y, t = poses[nid]
            corrected = Pose2D(x, y, t, node.pose.timestamp)
            self.nodes[nid].pose = corrected
            result[nid] = corrected

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_poses(self) -> Dict[int, Pose2D]:
        return {nid: n.pose for nid, n in self.nodes.items()}

    def get_trajectory(self) -> List[Pose2D]:
        """Return poses in node-id order."""
        return [self.nodes[nid].pose for nid in sorted(self.nodes)]

    def loop_count(self) -> int:
        return self._loop_count

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    """Shortest signed difference between two angles."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))
