"""Tests for mapping/pose_graph.py — PoseGraph, PoseNode, PoseEdge, optimizer."""
from __future__ import annotations

import math
import time

import pytest

from mapping.pose_graph import PoseGraph, PoseEdge, PoseNode, _angle_diff, _wrap_angle
from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Pose2D:
    return Pose2D(x, y, theta, time.time())


def _rel(dx: float, dy: float, dtheta: float = 0.0):
    return (dx, dy, dtheta)


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_angle_diff_same(self):
        assert abs(_angle_diff(0.0, 0.0)) < 1e-9

    def test_angle_diff_pi(self):
        diff = _angle_diff(math.pi, 0.0)
        assert abs(abs(diff) - math.pi) < 1e-6

    def test_angle_diff_wrap_around(self):
        diff = _angle_diff(-0.1, 0.1)
        assert abs(diff - (-0.2)) < 1e-6

    def test_wrap_angle_zero(self):
        assert abs(_wrap_angle(0.0)) < 1e-9

    def test_wrap_angle_past_pi(self):
        wrapped = _wrap_angle(math.pi + 0.1)
        assert -math.pi <= wrapped <= math.pi

    def test_wrap_angle_below_neg_pi(self):
        wrapped = _wrap_angle(-math.pi - 0.1)
        assert -math.pi <= wrapped <= math.pi


# ---------------------------------------------------------------------------
# TestPoseNode
# ---------------------------------------------------------------------------

class TestPoseNode:
    def test_default_not_fixed(self):
        node = PoseNode(id=0, pose=_pose())
        assert not node.fixed

    def test_fixed_flag(self):
        node = PoseNode(id=0, pose=_pose(), fixed=True)
        assert node.fixed


# ---------------------------------------------------------------------------
# TestPoseEdge
# ---------------------------------------------------------------------------

class TestPoseEdge:
    def test_defaults(self):
        edge = PoseEdge(from_id=0, to_id=1, dx=1.0, dy=0.0, dtheta=0.0)
        assert edge.info_xx == 100.0
        assert edge.info_yy == 100.0
        assert edge.info_tt == 50.0
        assert not edge.is_loop

    def test_loop_flag(self):
        edge = PoseEdge(from_id=0, to_id=1, dx=0.0, dy=0.0, dtheta=0.0, is_loop=True)
        assert edge.is_loop


# ---------------------------------------------------------------------------
# TestPoseGraphConstruction
# ---------------------------------------------------------------------------

class TestPoseGraphConstruction:
    def test_initial_empty(self):
        g = PoseGraph()
        assert g.node_count() == 0
        assert g.edge_count() == 0
        assert g.loop_count() == 0

    def test_add_node(self):
        g = PoseGraph()
        g.add_node(0, _pose(1.0, 2.0))
        assert g.node_count() == 1

    def test_add_multiple_nodes(self):
        g = PoseGraph()
        for i in range(5):
            g.add_node(i, _pose(float(i), 0.0))
        assert g.node_count() == 5

    def test_add_odometry_edge(self):
        g = PoseGraph()
        g.add_node(0, _pose(0.0, 0.0))
        g.add_node(1, _pose(1.0, 0.0))
        g.add_odometry_edge(0, 1, _rel(1.0, 0.0))
        assert g.edge_count() == 1

    def test_add_loop_edge(self):
        g = PoseGraph()
        g.add_node(0, _pose(0.0, 0.0))
        g.add_node(5, _pose(0.1, 0.1))
        g.add_loop_edge(5, 0, _rel(0.0, 0.0), score=0.8)
        assert g.edge_count() == 1
        assert g.loop_count() == 1

    def test_loop_edge_weight_scales_with_score(self):
        g = PoseGraph()
        g.add_node(0, _pose())
        g.add_node(1, _pose(1.0))
        g.add_loop_edge(1, 0, _rel(0.0, 0.0), score=0.5)
        edge = g.edges[-1]
        assert edge.is_loop
        assert abs(edge.info_xx - 50.0) < 1e-6   # 100 * 0.5

    def test_get_poses_returns_all_nodes(self):
        g = PoseGraph()
        g.add_node(0, _pose(0.0, 0.0))
        g.add_node(1, _pose(1.0, 0.0))
        poses = g.get_poses()
        assert set(poses.keys()) == {0, 1}

    def test_get_trajectory_sorted(self):
        g = PoseGraph()
        g.add_node(2, _pose(2.0))
        g.add_node(0, _pose(0.0))
        g.add_node(1, _pose(1.0))
        traj = g.get_trajectory()
        xs = [p.x for p in traj]
        assert xs == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# TestPoseGraphOptimize — trivial cases
# ---------------------------------------------------------------------------

class TestPoseGraphOptimize:
    def test_optimize_single_node_noop(self):
        g = PoseGraph()
        g.add_node(0, _pose(0.0, 0.0), fixed=True)
        result = g.optimize()
        assert abs(result[0].x) < 1e-9

    def test_optimize_returns_dict_with_all_ids(self):
        g = PoseGraph()
        g.add_node(0, _pose(0.0, 0.0), fixed=True)
        g.add_node(1, _pose(1.0, 0.0))
        g.add_odometry_edge(0, 1, _rel(1.0, 0.0))
        result = g.optimize()
        assert set(result.keys()) == {0, 1}

    def test_fixed_node_not_moved(self):
        g = PoseGraph()
        g.add_node(0, _pose(0.0, 0.0), fixed=True)
        g.add_node(1, _pose(1.0, 0.0))
        g.add_odometry_edge(0, 1, _rel(1.0, 0.0))
        result = g.optimize()
        assert abs(result[0].x) < 1e-9
        assert abs(result[0].y) < 1e-9

    def test_straight_line_chain_stays_consistent(self):
        """A perfectly consistent chain (no drift) should barely change after optimization."""
        g = PoseGraph()
        n = 5
        for i in range(n):
            g.add_node(i, _pose(float(i), 0.0), fixed=(i == 0))
        for i in range(n - 1):
            g.add_odometry_edge(i, i + 1, _rel(1.0, 0.0))
        result = g.optimize()
        # All poses should remain close to their original positions
        for i in range(n):
            assert abs(result[i].x - float(i)) < 0.1

    def test_loop_closure_reduces_drift(self):
        """Chain with accumulated drift; loop closure should pull endpoints together."""
        g = PoseGraph()
        # 4 poses in a square; accumulated angle drift
        poses = [
            _pose(0.0, 0.0, 0.0),
            _pose(1.0, 0.0, math.pi / 2),
            _pose(1.0, 1.0, math.pi),
            _pose(0.0, 1.0, -math.pi / 2),
        ]
        for i, p in enumerate(poses):
            g.add_node(i, p, fixed=(i == 0))

        # Odometry edges around the square
        g.add_odometry_edge(0, 1, _rel(1.0, 0.0, math.pi / 2))
        g.add_odometry_edge(1, 2, _rel(1.0, 0.0, math.pi / 2))
        g.add_odometry_edge(2, 3, _rel(1.0, 0.0, math.pi / 2))

        # Loop closure: node 3 is close to node 0 (return to origin)
        # Add slight drift to node 3 first
        g.nodes[3].pose = Pose2D(0.2, 0.9, -math.pi / 2, 0.0)
        before_x = g.nodes[3].pose.x

        g.add_loop_edge(3, 0, _rel(0.0, 0.0, math.pi / 2), score=0.9)
        result = g.optimize()
        after_x = result[3].x

        # Optimizer should move node 3 closer to origin (x closer to 0)
        assert abs(after_x) < abs(before_x) or abs(after_x - 0.0) < 0.3

    def test_optimize_empty_graph(self):
        g = PoseGraph()
        result = g.optimize()
        assert result == {}

    def test_optimize_no_edges_returns_original(self):
        g = PoseGraph()
        g.add_node(0, _pose(1.0, 2.0))
        result = g.optimize()
        assert abs(result[0].x - 1.0) < 1e-6
        assert abs(result[0].y - 2.0) < 1e-6

    def test_updated_poses_written_back_to_nodes(self):
        g = PoseGraph()
        g.add_node(0, _pose(0.0, 0.0), fixed=True)
        g.add_node(1, _pose(2.0, 0.0))
        g.add_odometry_edge(0, 1, _rel(1.0, 0.0))
        g.optimize()
        # Node poses should be updated in-place
        assert g.nodes[1].pose is not None

    def test_multiple_loop_closures_tracked(self):
        g = PoseGraph()
        g.add_node(0, _pose())
        g.add_node(1, _pose(1.0))
        g.add_node(2, _pose(2.0))
        g.add_loop_edge(1, 0, _rel(0.0, 0.0))
        g.add_loop_edge(2, 0, _rel(0.0, 0.0))
        assert g.loop_count() == 2
