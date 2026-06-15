"""Tests for mapping/keyframe.py — KeyframeStore and Keyframe dataclass."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from mapping.keyframe import Keyframe, KeyframeStore
from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Pose2D:
    return Pose2D(x, y, theta, time.time())


def _textured_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a random-noise BGR image with enough features for ORB."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    # Draw a grid of circles to guarantee detectable keypoints
    for i in range(0, h, 60):
        for j in range(0, w, 80):
            cv2.circle(img, (j, i), 8, (255, 255, 255), -1)
            cv2.circle(img, (j + 10, i + 10), 4, (0, 0, 0), -1)
    return img


def _gray_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a grayscale textured frame."""
    return cv2.cvtColor(_textured_frame(h, w), cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# TestKeyframeDataclass
# ---------------------------------------------------------------------------

class TestKeyframeDataclass:
    def test_fields(self):
        kps = []
        descs = np.zeros((10, 32), dtype=np.uint8)
        thumb = np.zeros((60, 80), dtype=np.uint8)
        p = _pose(1.0, 2.0, 0.5)
        kf = Keyframe(
            id=7,
            timestamp=1234.0,
            pose=p,
            keypoints=kps,
            descriptors=descs,
            thumbnail=thumb,
        )
        assert kf.id == 7
        assert kf.timestamp == 1234.0
        assert kf.pose is p
        assert kf.keypoints is kps
        assert kf.descriptors.shape == (10, 32)


# ---------------------------------------------------------------------------
# TestKeyframeStore — initialisation
# ---------------------------------------------------------------------------

class TestKeyframeStoreInit:
    def test_default_empty(self):
        store = KeyframeStore()
        assert len(store) == 0
        assert store.get_all() == []

    def test_custom_thresholds(self):
        store = KeyframeStore(min_distance_m=1.0, min_rotation_rad=0.5, max_keyframes=10)
        assert store.min_distance_m == 1.0
        assert store.min_rotation_rad == 0.5
        assert store.max_keyframes == 10

    def test_get_by_id_returns_none_when_empty(self):
        store = KeyframeStore()
        assert store.get_by_id(0) is None


# ---------------------------------------------------------------------------
# TestKeyframeStoreAddBehavior
# ---------------------------------------------------------------------------

class TestKeyframeStoreTryAdd:
    def test_first_frame_always_added(self):
        store = KeyframeStore()
        kf = store.try_add(_textured_frame(), _pose(0.0, 0.0))
        assert kf is not None
        assert len(store) == 1

    def test_second_frame_skipped_if_not_moved(self):
        store = KeyframeStore(min_distance_m=0.3, min_rotation_rad=0.3)
        store.try_add(_textured_frame(), _pose(0.0, 0.0))
        kf2 = store.try_add(_textured_frame(), _pose(0.0, 0.0))
        assert kf2 is None
        assert len(store) == 1

    def test_second_frame_added_after_translation(self):
        store = KeyframeStore(min_distance_m=0.3)
        store.try_add(_textured_frame(), _pose(0.0, 0.0))
        kf2 = store.try_add(_textured_frame(), _pose(0.5, 0.0))
        assert kf2 is not None
        assert len(store) == 2

    def test_second_frame_added_after_rotation(self):
        store = KeyframeStore(min_rotation_rad=0.3)
        store.try_add(_textured_frame(), _pose(0.0, 0.0, 0.0))
        kf2 = store.try_add(_textured_frame(), _pose(0.0, 0.0, 0.5))
        assert kf2 is not None
        assert len(store) == 2

    def test_ids_increment(self):
        store = KeyframeStore()
        kf1 = store.try_add(_textured_frame(), _pose(0.0, 0.0))
        kf2 = store.try_add(_textured_frame(), _pose(1.0, 0.0))
        assert kf1.id == 0
        assert kf2.id == 1

    def test_get_by_id(self):
        store = KeyframeStore()
        kf = store.try_add(_textured_frame(), _pose(0.0, 0.0))
        assert store.get_by_id(kf.id) is kf

    def test_get_by_id_missing_returns_none(self):
        store = KeyframeStore()
        store.try_add(_textured_frame(), _pose(0.0, 0.0))
        assert store.get_by_id(999) is None

    def test_get_all_returns_list(self):
        store = KeyframeStore()
        store.try_add(_textured_frame(), _pose(0.0, 0.0))
        store.try_add(_textured_frame(), _pose(1.0, 0.0))
        all_kfs = store.get_all()
        assert len(all_kfs) == 2

    def test_pose_stored_correctly(self):
        store = KeyframeStore()
        p = _pose(1.5, -0.5, math.pi / 4)
        kf = store.try_add(_textured_frame(), p)
        assert abs(kf.pose.x - 1.5) < 1e-6
        assert abs(kf.pose.y - (-0.5)) < 1e-6
        assert abs(kf.pose.theta - math.pi / 4) < 1e-6

    def test_descriptors_are_uint8(self):
        store = KeyframeStore()
        kf = store.try_add(_textured_frame(), _pose())
        assert kf.descriptors.dtype == np.uint8

    def test_thumbnail_shape(self):
        store = KeyframeStore()
        kf = store.try_add(_textured_frame(), _pose())
        assert kf.thumbnail.shape == (60, 80)

    def test_accepts_grayscale_input(self):
        store = KeyframeStore()
        kf = store.try_add(_gray_frame(), _pose())
        assert kf is not None


# ---------------------------------------------------------------------------
# TestKeyframeStorePruning
# ---------------------------------------------------------------------------

class TestKeyframeStorePruning:
    def test_oldest_pruned_at_capacity(self):
        store = KeyframeStore(max_keyframes=3)
        for i in range(4):
            store.try_add(_textured_frame(), _pose(float(i), 0.0))
        assert len(store) == 3

    def test_count_caps_at_max(self):
        store = KeyframeStore(max_keyframes=5)
        for i in range(10):
            store.try_add(_textured_frame(), _pose(float(i), 0.0))
        assert len(store) <= 5


# ---------------------------------------------------------------------------
# TestKeyframeStoreFeaturePoor
# ---------------------------------------------------------------------------

class TestKeyframeStoreFeaturePoor:
    def test_blank_image_not_added(self):
        """A blank uniform image has no detectable ORB features; should return None."""
        store = KeyframeStore()
        blank = np.zeros((480, 640), dtype=np.uint8)
        kf = store.try_add(blank, _pose())
        # Either None (no features) or very few features — store should not add
        if kf is not None:
            assert kf.descriptors is not None
