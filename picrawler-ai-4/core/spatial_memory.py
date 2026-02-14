"""
Spatial memory system for tracking history and learning from outcomes.

This allows the robot to remember what it's tried, what failed,
and avoid repeating unsuccessful strategies.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class ActionRecord:
    """Record of a single action and its outcome."""
    action: str
    timestamp: float
    success: bool
    distance_before: Optional[float]
    distance_after: Optional[float]
    reason: str  # Why it failed/succeeded


class SpatialMemory:
    """Tracks robot's action history and learns from outcomes."""

    def __init__(self, history_size: int = 50):
        # Action history (recent actions and outcomes)
        self.action_history: Deque[ActionRecord] = deque(maxlen=history_size)

        # Failure tracking per action type
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)

        # Pattern detection
        self.last_escape_time: float = 0.0
        self.escape_count: int = 0

        # Recovery escalation tracking
        self.recovery_level: int = 0  # 0=none, 1=light, 2=medium, 3=aggressive
        self.consecutive_stuck_count: int = 0
        self.last_stuck_time: float = 0.0

        # Directional preferences (which turns work better)
        self.turn_outcomes: Dict[str, List[bool]] = {
            'turn_left': [],
            'turn_right': []
        }

    def record_action(self, action: str, success: bool,
                     distance_before: Optional[float] = None,
                     distance_after: Optional[float] = None,
                     reason: str = "") -> None:
        """Record an action and its outcome."""
        record = ActionRecord(
            action=action,
            timestamp=time.time(),
            success=success,
            distance_before=distance_before,
            distance_after=distance_after,
            reason=reason
        )

        self.action_history.append(record)

        if success:
            self.success_counts[action] += 1
        else:
            self.failure_counts[action] += 1

        # Track turn outcomes specifically
        if action in ['turn_left', 'turn_right']:
            self.turn_outcomes[action].append(success)
            # Keep only recent 20 outcomes
            if len(self.turn_outcomes[action]) > 20:
                self.turn_outcomes[action].pop(0)

    def get_action_scores(self) -> Dict[str, float]:
        """Calculate preference scores for each action.

        Returns:
            Dict mapping action to score (0.0 to 1.0)
        """
        scores = {
            'forward': 1.0,
            'turn_left': 1.0,
            'turn_right': 1.0,
            'backward': 0.5  # Less preferred by default
        }

        # Penalize actions that have been failing
        for action in scores:
            total = self.success_counts[action] + self.failure_counts[action]
            if total > 0:
                success_rate = self.success_counts[action] / total
                # Exponential decay based on success rate
                scores[action] *= (0.5 + 0.5 * success_rate)

        # Heavy penalty for recently repeated failures
        if len(self.action_history) >= 5:
            recent = list(self.action_history)[-5:]
            recent_actions = [r.action for r in recent]

            # If same action repeated with failures
            for action in set(recent_actions):
                count = recent_actions.count(action)
                failures = sum(1 for r in recent if r.action == action and not r.success)

                if count >= 3 and failures >= 2:
                    # Heavily penalize stuck patterns
                    scores[action] *= 0.1

        return scores

    def get_best_turn_direction(self) -> str:
        """Decide which turn direction has been more successful.

        Returns:
            'turn_left' or 'turn_right'
        """
        left_rate = self._get_success_rate('turn_left')
        right_rate = self._get_success_rate('turn_right')

        if left_rate > right_rate:
            return 'turn_left'
        elif right_rate > left_rate:
            return 'turn_right'
        else:
            # Equal or unknown - alternate based on history
            return 'turn_left' if len(self.turn_outcomes['turn_right']) > len(self.turn_outcomes['turn_left']) else 'turn_right'

    def _get_success_rate(self, action: str) -> float:
        """Get success rate for an action."""
        outcomes = self.turn_outcomes.get(action, [])
        if not outcomes:
            return 0.5  # Unknown
        return sum(outcomes) / len(outcomes)

    def is_stuck(self) -> bool:
        """Detect if robot is stuck based on action patterns.

        Returns:
            True if stuck pattern detected
        """
        if len(self.action_history) < 10:
            return False

        recent = list(self.action_history)[-10:]

        # Pattern 1: Same action repeating with failures
        actions = [r.action for r in recent]
        if len(set(actions)) == 1:  # All same action
            failures = sum(1 for r in recent if not r.success)
            if failures >= 7:  # 7/10 failures
                return True

        # Pattern 2: Oscillation (turn left, turn right, repeat...)
        if self._is_oscillating(actions):
            return True

        # Pattern 3: No progress (distance not changing)
        if self._no_spatial_progress(recent):
            return True

        # Pattern 4: Recent escape didn't help
        if self.last_escape_time > 0:
            time_since_escape = time.time() - self.last_escape_time
            if time_since_escape < 20:  # Within 20s of last escape
                recent_failures = sum(
                    1 for r in recent if not r.success
                )
                if recent_failures >= 6:  # Still failing after escape
                    return True

        return False

    def _is_oscillating(self, actions: List[str]) -> bool:
        """Detect left-right oscillation pattern."""
        if len(actions) < 6:
            return False

        # Look for alternating turn pattern
        turns = [a for a in actions if 'turn' in a]
        if len(turns) < 4:
            return False

        # Check if alternating
        alternating_count = 0
        for i in range(len(turns) - 1):
            if turns[i] != turns[i + 1]:
                alternating_count += 1

        # If >75% of turns are alternating, it's oscillation
        return alternating_count / len(turns) > 0.75

    def _no_spatial_progress(self, records: List[ActionRecord]) -> bool:
        """Check if robot isn't making spatial progress."""
        # Get distance measurements
        distances = [
            r.distance_before for r in records
            if r.distance_before is not None
        ]

        if len(distances) < 5:
            return False

        # Check if variance is very low (stuck in same place)
        avg = sum(distances) / len(distances)
        variance = sum((d - avg) ** 2 for d in distances) / len(distances)

        # Low variance = not moving much
        return variance < 10  # Less than 10cm² variance

    def record_escape(self) -> None:
        """Record that an escape sequence was executed."""
        self.last_escape_time = time.time()
        self.escape_count += 1

        # Escalate recovery level if stuck again soon
        time_since_last_stuck = time.time() - self.last_stuck_time
        if time_since_last_stuck < 30:  # Stuck again within 30s
            self.consecutive_stuck_count += 1
            self.recovery_level = min(3, self.consecutive_stuck_count)
        else:
            # Been a while since last stuck, reset escalation
            self.consecutive_stuck_count = 1
            self.recovery_level = 1

        self.last_stuck_time = time.time()

    def reset_failure_counts(self) -> None:
        """Reset failure tracking (after successful escape)."""
        self.failure_counts.clear()

    def reset_recovery_escalation(self) -> None:
        """Reset recovery escalation after successful navigation."""
        self.recovery_level = 0
        self.consecutive_stuck_count = 0

    def get_recovery_level(self) -> int:
        """Get current recovery escalation level (0-3).

        Returns:
            0 = No recovery needed
            1 = Light recovery (first time stuck)
            2 = Medium recovery (stuck again within 30s)
            3 = Aggressive recovery (stuck multiple times)
        """
        return self.recovery_level

    def get_recent_actions(self, n: int = 10) -> List[str]:
        """Get list of N most recent actions."""
        return [r.action for r in list(self.action_history)[-n:]]

    def get_recent_failures(self, n: int = 10) -> List[ActionRecord]:
        """Get N most recent failed actions."""
        failures = [r for r in self.action_history if not r.success]
        return list(failures)[-n:]

    def to_dict(self) -> Dict:
        """Export as dictionary for logging/AI context."""
        return {
            'recent_actions': self.get_recent_actions(10),
            'failure_counts': dict(self.failure_counts),
            'success_counts': dict(self.success_counts),
            'action_scores': self.get_action_scores(),
            'is_stuck': self.is_stuck(),
            'escape_count': self.escape_count,
            'best_turn': self.get_best_turn_direction()
        }

    def __str__(self) -> str:
        return (
            f"SpatialMemory(actions={len(self.action_history)}, "
            f"stuck={self.is_stuck()}, "
            f"best_turn={self.get_best_turn_direction()})"
        )
