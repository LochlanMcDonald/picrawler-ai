"""
Loop closure detection using ORB descriptor matching.

When the robot revisits a previously seen place, this module detects the
match and returns the candidate keyframe + a relative pose estimate between
the two viewpoints.  No vocabulary training required — uses brute-force
Hamming matching with Lowe's ratio test, then geometric verification via
homography RANSAC.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mapping.keyframe import Keyframe, KeyframeStore
from perception.visual_odometry import Pose2D


@dataclass
class LoopCandidate:
    """A detected loop closure candidate."""
    query_kf: Keyframe       # Current (query) keyframe
    match_kf: Keyframe       # Previously seen (match) keyframe
    score: float             # Match quality  0–1  (higher = better)
    num_inliers: int         # RANSAC geometric inliers
    relative_pose: Optional[Tuple[float, float, float]]  # (dx, dy, dtheta) or None


class LoopClosureDetector:
    """
    Detects when the robot revisits a previously mapped location.

    Pipeline per query keyframe:
      1. Fast descriptor matching against all stored keyframes
      2. Geometric verification (homography RANSAC) on top candidates
      3. Return the best candidate above thresholds
    """

    def __init__(self,
                 min_score: float = 0.25,
                 min_inliers: int = 12,
                 min_age_frames: int = 10,
                 top_k_candidates: int = 5,
                 lowe_ratio: float = 0.75):
        """
        Args:
            min_score: Minimum descriptor match score to accept a loop
            min_inliers: Minimum RANSAC inliers for geometric verification
            min_age_frames: Ignore matches to keyframes added in the last N frames
                            (prevents matching against the immediately preceding KF)
            top_k_candidates: How many descriptor-match candidates to geometrically verify
            lowe_ratio: Lowe's ratio test threshold
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_score = min_score
        self.min_inliers = min_inliers
        self.min_age_frames = min_age_frames
        self.top_k = top_k_candidates
        self.lowe_ratio = lowe_ratio

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.total_closures = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self,
               query_kf: Keyframe,
               store: KeyframeStore) -> Optional[LoopCandidate]:
        """Search for a loop closure against all stored keyframes.

        Args:
            query_kf: The freshly added keyframe to match against the database
            store: The keyframe store to search

        Returns:
            Best LoopCandidate if found, else None
        """
        keyframes = store.get_all()

        # Need enough keyframes to even attempt closure
        if len(keyframes) < self.min_age_frames + 2:
            return None

        # Exclude recent keyframes (they're just the neighbourhood, not a loop)
        candidates = keyframes[:-self.min_age_frames]

        if not candidates:
            return None

        # Stage 1 — fast descriptor matching to rank candidates
        scored = self._score_candidates(query_kf, candidates)
        if not scored:
            return None

        # Stage 2 — geometric verification on top-K
        top = scored[: self.top_k]
        best: Optional[LoopCandidate] = None

        for score, kf in top:
            if score < self.min_score * 0.5:
                break  # Remaining candidates are worse, no point checking

            candidate = self._verify_geometry(query_kf, kf, score)
            if candidate is not None:
                if best is None or candidate.score > best.score:
                    best = candidate

        if best is not None:
            self.total_closures += 1
            self.logger.info(
                f"Loop closure #{self.total_closures}: "
                f"KF {query_kf.id} → KF {best.match_kf.id}  "
                f"score={best.score:.3f}  inliers={best.num_inliers}"
            )

        return best

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_candidates(
            self,
            query_kf: Keyframe,
            candidates: List[Keyframe]) -> List[Tuple[float, Keyframe]]:
        """Score each candidate by descriptor similarity."""
        results: List[Tuple[float, Keyframe]] = []

        for kf in candidates:
            try:
                matches = self.matcher.knnMatch(
                    query_kf.descriptors, kf.descriptors, k=2
                )
            except cv2.error:
                continue

            good = 0
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.lowe_ratio * n.distance:
                        good += 1

            total = min(len(query_kf.descriptors), len(kf.descriptors))
            score = good / total if total > 0 else 0.0
            results.append((score, kf))

        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def _verify_geometry(
            self,
            query_kf: Keyframe,
            match_kf: Keyframe,
            raw_score: float) -> Optional[LoopCandidate]:
        """Geometrically verify a candidate using RANSAC homography."""
        try:
            matches = self.matcher.knnMatch(
                query_kf.descriptors, match_kf.descriptors, k=2
            )
        except cv2.error:
            return None

        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.lowe_ratio * n.distance:
                    good.append(m)

        if len(good) < self.min_inliers:
            return None

        # Extract matched point coordinates
        q_pts = np.float32([
            query_kf.keypoints[m.queryIdx].pt for m in good
        ])
        m_pts = np.float32([
            match_kf.keypoints[m.trainIdx].pt for m in good
        ])

        # RANSAC homography
        H, mask = cv2.findHomography(q_pts, m_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())
        if inliers < self.min_inliers:
            return None

        # Geometric score = fraction of inliers
        geom_score = inliers / len(good)
        combined_score = 0.5 * raw_score + 0.5 * geom_score

        if combined_score < self.min_score:
            return None

        # Estimate relative pose from homography
        relative_pose = self._relative_pose_from_homography(H, match_kf.pose)

        return LoopCandidate(
            query_kf=query_kf,
            match_kf=match_kf,
            score=combined_score,
            num_inliers=inliers,
            relative_pose=relative_pose,
        )

    @staticmethod
    def _relative_pose_from_homography(
            H: np.ndarray,
            match_pose: Pose2D) -> Optional[Tuple[float, float, float]]:
        """
        Estimate relative 2D pose from a homography.

        This is approximate — homography encodes rotation and translation
        but at unknown scale.  We extract the rotation component reliably;
        translation is scale-ambiguous and returned as zero so the pose
        graph uses the direct pose difference instead.
        """
        try:
            dtheta = float(np.arctan2(H[1, 0], H[0, 0]))
            return (0.0, 0.0, dtheta)
        except Exception:
            return None
