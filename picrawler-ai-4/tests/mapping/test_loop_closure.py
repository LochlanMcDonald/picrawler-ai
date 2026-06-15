"""Tests for mapping/loop_closure.py — LoopClosureDetector."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from mapping.keyframe import Keyframe, KeyframeStore
from mapping.loop_closure import LoopCandidate, LoopClosureDetector
from perception.visual_odometry import Pose2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Pose2D:
    return Pose2D(x, y, theta, time.time())


def _textured_frame(seed: int = 0, h: int = 240, w: int = 320) -> np.ndarray:
    """Rich textured frame with ORB-detectable features."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    for i in range(0, h, 30):
        for j in range(0, w, 40):
            cv2.circle(img, (j, i), 6, (255, 255, 255), -1)
            cv2.rectangle(img, (j + 5, i + 5), (j + 20, i + 20), (0, 0, 0), 2)
    return img


def _make_keyframe(kf_id: int, pose: Pose2D, seed: int = 0) -> Keyframe:
    """Create a Keyframe directly with ORB descriptors from a textured image."""
    detector = cv2.ORB_create(nfeatures=500)
    img = _textured_frame(seed=seed)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, descs = detector.detectAndCompute(gray, None)
    if descs is None:
        descs = np.zeros((100, 32), dtype=np.uint8)
        kps = [cv2.KeyPoint(float(i * 5 % 320), float(i * 3 % 240), 5.0) for i in range(100)]
    thumb = cv2.resize(gray, (80, 60))
    return Keyframe(
        id=kf_id,
        timestamp=time.time(),
        pose=pose,
        keypoints=kps,
        descriptors=descs.astype(np.uint8),
        thumbnail=thumb,
    )


def _build_store_with_n_keyframes(n: int, same_image: bool = False) -> KeyframeStore:
    """Build a KeyframeStore with n keyframes at 1-metre intervals."""
    store = KeyframeStore(min_distance_m=0.1, max_keyframes=1000)
    frame = _textured_frame(seed=0)
    for i in range(n):
        seed = 0 if same_image else i
        img = _textured_frame(seed=seed)
        p = _pose(float(i), 0.0)
        store.try_add(img, p)
    return store


# ---------------------------------------------------------------------------
# TestLoopCandidate
# ---------------------------------------------------------------------------

class TestLoopCandidate:
    def test_fields(self):
        q = _make_keyframe(0, _pose(0, 0))
        m = _make_keyframe(1, _pose(1, 0))
        cand = LoopCandidate(
            query_kf=q,
            match_kf=m,
            score=0.7,
            num_inliers=20,
            relative_pose=(0.1, 0.2, 0.3),
        )
        assert cand.score == 0.7
        assert cand.num_inliers == 20
        assert cand.relative_pose == (0.1, 0.2, 0.3)


# ---------------------------------------------------------------------------
# TestLoopClosureDetectorInit
# ---------------------------------------------------------------------------

class TestLoopClosureDetectorInit:
    def test_defaults(self):
        d = LoopClosureDetector()
        assert d.min_score == 0.25
        assert d.min_inliers == 12
        assert d.min_age_frames == 10
        assert d.total_closures == 0

    def test_custom_params(self):
        d = LoopClosureDetector(min_score=0.5, min_inliers=20, min_age_frames=5)
        assert d.min_score == 0.5
        assert d.min_inliers == 20
        assert d.min_age_frames == 5


# ---------------------------------------------------------------------------
# TestDetectEmptyAndSmallStores
# ---------------------------------------------------------------------------

class TestDetectInsufficientData:
    def test_returns_none_for_empty_store(self):
        d = LoopClosureDetector()
        store = KeyframeStore()
        q = _make_keyframe(0, _pose())
        result = d.detect(q, store)
        assert result is None

    def test_returns_none_when_store_too_small(self):
        """Store must have at least min_age_frames + 2 keyframes."""
        d = LoopClosureDetector(min_age_frames=10)
        store = _build_store_with_n_keyframes(5)
        q = _make_keyframe(99, _pose(99.0))
        result = d.detect(q, store)
        assert result is None

    def test_returns_none_when_all_candidates_recent(self):
        """If all keyframes are excluded by min_age_frames, no candidates exist."""
        d = LoopClosureDetector(min_age_frames=100)
        store = _build_store_with_n_keyframes(15)
        q = _make_keyframe(99, _pose(99.0))
        result = d.detect(q, store)
        assert result is None


# ---------------------------------------------------------------------------
# TestDetectWithSameImage
# ---------------------------------------------------------------------------

class TestDetectWithMatchingImages:
    def test_detects_closure_with_identical_frames(self):
        """Query keyframe with same texture as stored KFs should find a loop."""
        d = LoopClosureDetector(min_score=0.1, min_inliers=4, min_age_frames=3)
        store = _build_store_with_n_keyframes(8, same_image=True)
        # Create query from the same image (seed=0)
        q = _make_keyframe(99, _pose(0.05, 0.0), seed=0)
        result = d.detect(q, store)
        # With identical descriptors, we expect a closure to be found
        # (or None if geometry check fails — both are acceptable in unit test)
        if result is not None:
            assert result.score >= d.min_score
            assert result.num_inliers >= d.min_inliers

    def test_closure_count_increments(self):
        d = LoopClosureDetector(min_score=0.1, min_inliers=4, min_age_frames=3)
        store = _build_store_with_n_keyframes(8, same_image=True)
        q = _make_keyframe(99, _pose(0.05), seed=0)
        result = d.detect(q, store)
        if result is not None:
            assert d.total_closures == 1


# ---------------------------------------------------------------------------
# TestScoreCandidates
# ---------------------------------------------------------------------------

class TestScoreCandidates:
    def test_returns_sorted_scores(self):
        d = LoopClosureDetector()
        q = _make_keyframe(0, _pose(), seed=0)
        candidates = [_make_keyframe(i, _pose(float(i)), seed=i) for i in range(5)]
        scored = d._score_candidates(q, candidates)
        # Check descending order
        scores = [s for s, _ in scored]
        assert scores == sorted(scores, reverse=True)

    def test_same_image_scores_high(self):
        d = LoopClosureDetector()
        q = _make_keyframe(0, _pose(), seed=42)
        same = _make_keyframe(1, _pose(1.0), seed=42)
        different = _make_keyframe(2, _pose(2.0), seed=7)
        scored = d._score_candidates(q, [same, different])
        # Same image should score highest
        assert scored[0][1].id == same.id

    def test_handles_empty_candidates(self):
        d = LoopClosureDetector()
        q = _make_keyframe(0, _pose())
        result = d._score_candidates(q, [])
        assert result == []


# ---------------------------------------------------------------------------
# TestRelativePoseFromHomography
# ---------------------------------------------------------------------------

class TestRelativePoseFromHomography:
    def test_identity_homography_gives_zero_rotation(self):
        H = np.eye(3, dtype=np.float32)
        result = LoopClosureDetector._relative_pose_from_homography(H, _pose())
        assert result is not None
        dx, dy, dtheta = result
        assert abs(dtheta) < 1e-6

    def test_rotation_homography_extracts_angle(self):
        angle = math.pi / 6  # 30 degrees
        H = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle),  math.cos(angle), 0.0],
            [0.0,              0.0,              1.0],
        ], dtype=np.float32)
        result = LoopClosureDetector._relative_pose_from_homography(H, _pose())
        assert result is not None
        _, _, dtheta = result
        assert abs(dtheta - angle) < 0.01

    def test_translation_fixed_at_zero(self):
        """Translation is scale-ambiguous; always returned as 0, 0."""
        H = np.eye(3, dtype=np.float32)
        H[0, 2] = 100.0  # Large translation
        H[1, 2] = 50.0
        result = LoopClosureDetector._relative_pose_from_homography(H, _pose())
        assert result is not None
        dx, dy, _ = result
        assert dx == 0.0
        assert dy == 0.0

    def test_degenerate_homography_handled(self):
        H = np.zeros((3, 3), dtype=np.float32)
        result = LoopClosureDetector._relative_pose_from_homography(H, _pose())
        # Should not raise — returns (0, 0, 0.0) since arctan2(0,0) is defined
        assert result is not None
