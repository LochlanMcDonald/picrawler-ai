"""
Behavior tree implementation for structured robot behaviors.

Behavior trees provide hierarchical decision making with clear priorities
and fallback strategies, replacing the old linear decision flow.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from core.world_model import WorldModel
from core.spatial_memory import SpatialMemory


class Status(Enum):
    """Status returned by behavior nodes."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorContext:
    """Shared context passed to all behavior nodes."""

    def __init__(self, world_model: WorldModel, memory: SpatialMemory,
                 robot_controller, logger: Optional[logging.Logger] = None):
        self.world_model = world_model
        self.memory = memory
        self.robot = robot_controller
        self.logger = logger or logging.getLogger(__name__)
        self.blackboard: dict = {}  # Shared state between nodes


class BehaviorNode:
    """Base class for behavior tree nodes."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"BehaviorTree.{name}")

    def execute(self, context: BehaviorContext) -> Status:
        """Execute this node. Must be implemented by subclasses."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


# ============================================================================
# Composite Nodes (have children)
# ============================================================================

class SequenceNode(BehaviorNode):
    """Execute children in order. Fail if any child fails."""

    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children

    def execute(self, context: BehaviorContext) -> Status:
        for child in self.children:
            status = child.execute(context)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS


class FallbackNode(BehaviorNode):
    """Try children in order until one succeeds. Fail if all fail."""

    def __init__(self, name: str, children: List[BehaviorNode]):
        super().__init__(name)
        self.children = children

    def execute(self, context: BehaviorContext) -> Status:
        for child in self.children:
            status = child.execute(context)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE


# ============================================================================
# Condition Nodes (checks, no actions)
# ============================================================================

class CheckStuckCondition(BehaviorNode):
    """Check if robot is stuck."""

    def __init__(self):
        super().__init__("CheckStuck")

    def execute(self, context: BehaviorContext) -> Status:
        if context.memory.is_stuck():
            self.logger.warning("Robot is stuck!")
            return Status.SUCCESS
        return Status.FAILURE


class CheckPathClear(BehaviorNode):
    """Check if path ahead is clear."""

    def __init__(self):
        super().__init__("CheckPathClear")

    def execute(self, context: BehaviorContext) -> Status:
        if context.world_model.is_safe_to_move('forward'):
            self.logger.debug("Path is clear")
            return Status.SUCCESS
        else:
            self.logger.debug("Path is blocked")
            return Status.FAILURE


class CheckFreeSpace(BehaviorNode):
    """Check if robot has free space around it."""

    def __init__(self, min_score: float = 0.3):
        super().__init__("CheckFreeSpace")
        self.min_score = min_score

    def execute(self, context: BehaviorContext) -> Status:
        score = context.world_model.calculate_free_space()
        if score >= self.min_score:
            self.logger.debug(f"Free space OK: {score:.2f}")
            return Status.SUCCESS
        else:
            self.logger.warning(f"Low free space: {score:.2f}")
            return Status.FAILURE


# ============================================================================
# Action Nodes (do things)
# ============================================================================

class MoveForwardAction(BehaviorNode):
    """Move forward for specified duration."""

    def __init__(self, duration: float = 1.0):
        super().__init__("MoveForward")
        self.duration = duration

    def execute(self, context: BehaviorContext) -> Status:
        # Pre-check: still safe?
        if not context.world_model.is_safe_to_move('forward'):
            self.logger.warning("Path became blocked, aborting")
            return Status.FAILURE

        self.logger.info(f"Moving forward for {self.duration}s")

        distance_before = context.robot.get_distance()
        context.robot.execute('forward', self.duration)
        distance_after = context.robot.get_distance()

        # Update world model after movement
        context.world_model.update_ultrasonic(distance_after)

        # Record outcome
        success = distance_after is None or distance_after > 15
        context.memory.record_action(
            'forward',
            success=success,
            distance_before=distance_before,
            distance_after=distance_after,
            reason="completed" if success else "blocked"
        )

        # Reset recovery escalation after 3 successful forward movements
        if success:
            recent_successes = sum(
                1 for r in list(context.memory.action_history)[-5:]
                if r.action == 'forward' and r.success
            )
            if recent_successes >= 3:
                context.memory.reset_recovery_escalation()
                self.logger.debug("Reset recovery escalation after successful navigation")

        return Status.SUCCESS if success else Status.FAILURE


class TurnAction(BehaviorNode):
    """Turn in specified direction."""

    def __init__(self, direction: str, duration: float = 0.8):
        super().__init__(f"Turn{direction.title()}")
        self.direction = direction  # 'left' or 'right'
        self.duration = duration

    def execute(self, context: BehaviorContext) -> Status:
        action = f"turn_{self.direction}"
        self.logger.info(f"Executing {action} for {self.duration}s")

        distance_before = context.robot.get_distance()
        context.robot.execute(action, self.duration)
        distance_after = context.robot.get_distance()

        # Update world model
        context.world_model.update_ultrasonic(distance_after)

        # Turns almost always succeed
        context.memory.record_action(
            action,
            success=True,
            distance_before=distance_before,
            distance_after=distance_after,
            reason="turn_completed"
        )

        return Status.SUCCESS


class BackupAction(BehaviorNode):
    """Back up for specified duration."""

    def __init__(self, duration: float = 1.5):
        super().__init__("Backup")
        self.duration = duration

    def execute(self, context: BehaviorContext) -> Status:
        self.logger.info(f"Backing up for {self.duration}s")

        distance_before = context.robot.get_distance()
        context.robot.execute('backward', self.duration)
        distance_after = context.robot.get_distance()

        context.world_model.update_ultrasonic(distance_after)

        context.memory.record_action(
            'backward',
            success=True,
            distance_before=distance_before,
            distance_after=distance_after,
            reason="backup_completed"
        )

        return Status.SUCCESS


class SmartTurnAction(BehaviorNode):
    """Turn in the direction that memory suggests is best."""

    def __init__(self, duration: float = 0.8):
        super().__init__("SmartTurn")
        self.duration = duration

    def execute(self, context: BehaviorContext) -> Status:
        # Use memory to pick best direction
        best_direction = context.memory.get_best_turn_direction()

        self.logger.info(f"Smart turn chose: {best_direction}")

        # Execute the turn
        direction = 'left' if 'left' in best_direction else 'right'
        turn_action = TurnAction(direction, self.duration)
        return turn_action.execute(context)


class RecoverySequence(BehaviorNode):
    """Execute stuck recovery sequence with escalating strategies."""

    def __init__(self, backup_dur: float = 2.0, turn_dur: float = 1.2, forward_dur: float = 1.5):
        super().__init__("RecoverySequence")
        self.backup_dur = backup_dur
        self.turn_dur = turn_dur
        self.forward_dur = forward_dur

    def execute(self, context: BehaviorContext) -> Status:
        # Get recovery escalation level
        recovery_level = context.memory.get_recovery_level()

        if recovery_level == 0:
            # Light recovery (first time stuck)
            self.logger.warning("Executing LIGHT recovery (level 0)")
            strategy = self._light_recovery()
        elif recovery_level == 1:
            # Medium recovery
            self.logger.warning("Executing MEDIUM recovery (level 1)")
            strategy = self._medium_recovery()
        elif recovery_level == 2:
            # Aggressive recovery
            self.logger.warning("Executing AGGRESSIVE recovery (level 2)")
            strategy = self._aggressive_recovery()
        else:
            # Maximum recovery (very stuck)
            self.logger.error("Executing MAXIMUM recovery (level 3) - robot is very stuck!")
            strategy = self._maximum_recovery()

        context.memory.record_escape()

        result = strategy.execute(context)

        if result == Status.SUCCESS:
            # Reset failure counts after successful recovery
            context.memory.reset_failure_counts()

        return result

    def _light_recovery(self) -> BehaviorNode:
        """Light recovery - minimal maneuver."""
        return SequenceNode("LightRecovery", [
            StopAction(),
            BackupAction(duration=self.backup_dur * 0.75),  # Back up a bit
            SmartTurnAction(duration=self.turn_dur * 0.8),  # Small turn
            MoveForwardAction(duration=self.forward_dur * 0.8),
        ])

    def _medium_recovery(self) -> BehaviorNode:
        """Medium recovery - standard escape."""
        return SequenceNode("MediumRecovery", [
            StopAction(),
            BackupAction(duration=self.backup_dur),  # Standard backup
            SmartTurnAction(duration=self.turn_dur),  # Standard turn
            MoveForwardAction(duration=self.forward_dur),
        ])

    def _aggressive_recovery(self) -> BehaviorNode:
        """Aggressive recovery - larger movements."""
        return SequenceNode("AggressiveRecovery", [
            StopAction(),
            BackupAction(duration=self.backup_dur * 1.5),  # Back up more
            SmartTurnAction(duration=self.turn_dur * 1.5),  # Turn more
            BackupAction(duration=self.backup_dur * 0.5),  # Another backup
            SmartTurnAction(duration=self.turn_dur),  # Another turn (opposite direction)
            MoveForwardAction(duration=self.forward_dur * 1.2),
        ])

    def _maximum_recovery(self) -> BehaviorNode:
        """Maximum recovery - multi-step escape with exploration."""
        return FallbackNode("MaximumRecovery", [
            # Try 1: Aggressive backward + turn sequence
            SequenceNode("MaxRecovery1", [
                StopAction(),
                BackupAction(duration=self.backup_dur * 2.0),  # Back way up
                TurnAction('left', duration=self.turn_dur * 1.5),  # Hard left
                MoveForwardAction(duration=self.forward_dur),
            ]),
            # Try 2: Opposite direction
            SequenceNode("MaxRecovery2", [
                BackupAction(duration=self.backup_dur * 1.5),
                TurnAction('right', duration=self.turn_dur * 1.5),  # Hard right
                MoveForwardAction(duration=self.forward_dur),
            ]),
            # Try 3: Just get somewhere different
            SequenceNode("MaxRecovery3", [
                BackupAction(duration=self.backup_dur * 2.0),
                TurnAction('left', duration=self.turn_dur * 2.0),  # Complete turn
                MoveForwardAction(duration=self.forward_dur * 1.5),
            ])
        ])


class StopAction(BehaviorNode):
    """Stop the robot."""

    def __init__(self):
        super().__init__("Stop")

    def execute(self, context: BehaviorContext) -> Status:
        self.logger.info("Stopping")
        context.robot.execute('stop', 0.1)
        return Status.SUCCESS


# ============================================================================
# Pre-built Behavior Trees
# ============================================================================

def build_exploration_tree(config: dict = None) -> BehaviorNode:
    """Build the standard exploration behavior tree.

    Args:
        config: Configuration dictionary with behavior_tree_settings
    """
    if config is None:
        config = {}

    bt_settings = config.get("behavior_tree_settings", {})

    # Extract configurable durations
    move_forward_dur = bt_settings.get("move_forward_duration_s", 1.2)
    turn_dur = bt_settings.get("turn_duration_s", 0.8)
    smart_turn_dur = bt_settings.get("smart_turn_duration_s", 0.8)
    backup_dur = bt_settings.get("backup_duration_s", 1.5)
    aggressive_backup_dur = bt_settings.get("aggressive_backup_duration_s", 1.5)
    aggressive_turn_dur = bt_settings.get("aggressive_turn_duration_s", 1.2)
    aggressive_forward_dur = bt_settings.get("aggressive_forward_duration_s", 0.8)
    recovery_backup_dur = bt_settings.get("recovery_backup_duration_s", 2.0)
    recovery_turn_dur = bt_settings.get("recovery_turn_duration_s", 1.2)
    recovery_forward_dur = bt_settings.get("recovery_forward_duration_s", 1.5)

    return FallbackNode("ExplorationRoot", [
        # Priority 1: If stuck, try recovery
        SequenceNode("StuckRecovery", [
            CheckStuckCondition(),
            RecoverySequence(
                backup_dur=recovery_backup_dur,
                turn_dur=recovery_turn_dur,
                forward_dur=recovery_forward_dur
            )
        ]),

        # Priority 2: If path clear, move forward
        SequenceNode("MoveForward", [
            CheckPathClear(),
            MoveForwardAction(duration=move_forward_dur)
        ]),

        # Priority 3: Path blocked, find alternative
        SequenceNode("FindAlternative", [
            SmartTurnAction(duration=smart_turn_dur),
            # Check if turn helped
            CheckPathClear(),
            MoveForwardAction(duration=move_forward_dur)
        ]),

        # Priority 4: Still blocked, aggressive maneuver
        SequenceNode("AggressiveManeuver", [
            BackupAction(duration=aggressive_backup_dur),
            SmartTurnAction(duration=aggressive_turn_dur),
            MoveForwardAction(duration=aggressive_forward_dur)
        ]),

        # Priority 5: Last resort - random exploration
        SequenceNode("RandomExplore", [
            TurnAction('left', duration=turn_dur),  # Just turn and hope
            MoveForwardAction(duration=aggressive_forward_dur)
        ])
    ])


def build_cautious_exploration_tree(config: dict = None) -> BehaviorNode:
    """Build a more cautious exploration tree (shorter movements).

    Args:
        config: Configuration dictionary with behavior_tree_settings
    """
    if config is None:
        config = {}

    bt_settings = config.get("behavior_tree_settings", {})

    # Extract configurable parameters
    cautious_forward_dur = bt_settings.get("cautious_forward_duration_s", 0.8)
    smart_turn_dur = bt_settings.get("smart_turn_duration_s", 0.7)
    cautious_free_space_min = bt_settings.get("cautious_free_space_min_score", 0.5)
    recovery_backup_dur = bt_settings.get("recovery_backup_duration_s", 2.0)
    recovery_turn_dur = bt_settings.get("recovery_turn_duration_s", 1.2)
    recovery_forward_dur = bt_settings.get("recovery_forward_duration_s", 1.5)

    return FallbackNode("CautiousExplorationRoot", [
        SequenceNode("StuckRecovery", [
            CheckStuckCondition(),
            RecoverySequence(
                backup_dur=recovery_backup_dur,
                turn_dur=recovery_turn_dur,
                forward_dur=recovery_forward_dur
            )
        ]),

        SequenceNode("CautiousForward", [
            CheckPathClear(),
            CheckFreeSpace(min_score=cautious_free_space_min),  # Need more free space
            MoveForwardAction(duration=cautious_forward_dur)  # Shorter movements
        ]),

        SequenceNode("FindAlternative", [
            SmartTurnAction(duration=smart_turn_dur),
            MoveForwardAction(duration=cautious_forward_dur * 0.75)
        ]),

        RecoverySequence(
            backup_dur=recovery_backup_dur,
            turn_dur=recovery_turn_dur,
            forward_dur=recovery_forward_dur
        )  # Fallback to recovery
    ])
