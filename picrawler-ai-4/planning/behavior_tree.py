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
    """Execute full stuck recovery sequence."""

    def __init__(self):
        super().__init__("RecoverySequence")

    def execute(self, context: BehaviorContext) -> Status:
        self.logger.warning("Executing recovery sequence!")

        context.memory.record_escape()

        # Recovery: Stop, back up, turn significantly, try forward
        sequence = SequenceNode("Recovery", [
            StopAction(),
            BackupAction(duration=2.0),  # Back up more
            SmartTurnAction(duration=1.2),  # Turn more
            MoveForwardAction(duration=1.5),  # Try forward
        ])

        result = sequence.execute(context)

        if result == Status.SUCCESS:
            # Reset failure counts after successful recovery
            context.memory.reset_failure_counts()

        return result


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

def build_exploration_tree() -> BehaviorNode:
    """Build the standard exploration behavior tree."""

    return FallbackNode("ExplorationRoot", [
        # Priority 1: If stuck, try recovery
        SequenceNode("StuckRecovery", [
            CheckStuckCondition(),
            RecoverySequence()
        ]),

        # Priority 2: If path clear, move forward
        SequenceNode("MoveForward", [
            CheckPathClear(),
            MoveForwardAction(duration=1.2)
        ]),

        # Priority 3: Path blocked, find alternative
        SequenceNode("FindAlternative", [
            SmartTurnAction(duration=0.9),
            # Check if turn helped
            CheckPathClear(),
            MoveForwardAction(duration=1.0)
        ]),

        # Priority 4: Still blocked, aggressive maneuver
        SequenceNode("AggressiveManeuver", [
            BackupAction(duration=1.5),
            SmartTurnAction(duration=1.2),
            MoveForwardAction(duration=0.8)
        ]),

        # Priority 5: Last resort - random exploration
        SequenceNode("RandomExplore", [
            TurnAction('left', duration=1.0),  # Just turn and hope
            MoveForwardAction(duration=0.5)
        ])
    ])


def build_cautious_exploration_tree() -> BehaviorNode:
    """Build a more cautious exploration tree (shorter movements)."""

    return FallbackNode("CautiousExplorationRoot", [
        SequenceNode("StuckRecovery", [
            CheckStuckCondition(),
            RecoverySequence()
        ]),

        SequenceNode("CautiousForward", [
            CheckPathClear(),
            CheckFreeSpace(min_score=0.5),  # Need more free space
            MoveForwardAction(duration=0.8)  # Shorter movements
        ]),

        SequenceNode("FindAlternative", [
            SmartTurnAction(duration=0.7),
            MoveForwardAction(duration=0.6)
        ]),

        RecoverySequence()  # Fallback to recovery
    ])
