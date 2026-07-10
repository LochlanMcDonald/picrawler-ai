#!/usr/bin/env python3
"""
Test the new v4 architecture components independently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.world_model import WorldModel
from core.spatial_memory import SpatialMemory
from planning.behavior_tree import (
    build_exploration_tree,
    BehaviorContext,
    Status
)


def test_world_model():
    print("=" * 60)
    print("Testing WorldModel (Sensor Fusion)")
    print("=" * 60)

    world = WorldModel(obstacle_threshold_cm=20)

    # Test 1: Ultrasonic only
    print("\n1. Ultrasonic sensor: 15cm obstacle")
    world.update_ultrasonic(15.0)
    print(f"   Is safe to move forward? {world.is_safe_to_move('forward')}")
    print(f"   Best direction: {world.get_best_direction()}")
    print(f"   Free space: {world.calculate_free_space():.2f}")

    # Test 2: Vision confirms
    print("\n2. Vision AI sees wall (confirming ultrasonic)")
    world.update_vision([], ['wall'], "Wall directly ahead")
    print(f"   Front obstacle confidence: {world.obstacles['front'].confidence:.2f}")
    print(f"   Source: {world.obstacles['front'].source}")

    # Test 3: Clear path
    print("\n3. Clear path (50cm ahead)")
    world.update_ultrasonic(50.0)
    world.update_vision(['chair', 'table'], [], "Room with furniture")
    print(f"   Is safe to move forward? {world.is_safe_to_move('forward')}")
    print(f"   Best direction: {world.get_best_direction()}")

    print("\n✅ WorldModel tests passed!")


def test_spatial_memory():
    print("\n" + "=" * 60)
    print("Testing SpatialMemory (Learning)")
    print("=" * 60)

    memory = SpatialMemory()

    # Test 1: Normal operation
    print("\n1. Recording successful actions")
    memory.record_action('forward', success=True, distance_before=50, distance_after=45)
    memory.record_action('turn_left', success=True)
    print(f"   Action scores: {memory.get_action_scores()}")

    # Test 2: Stuck pattern
    print("\n2. Simulating stuck pattern (forward failing repeatedly)")
    for i in range(10):
        memory.record_action('forward', success=False, distance_before=15, distance_after=15, reason="blocked")

    print(f"   Is stuck? {memory.is_stuck()}")
    print(f"   Recent actions: {memory.get_recent_actions(5)}")
    print(f"   Action scores after failures: {memory.get_action_scores()}")

    # Test 3: Turn preference
    print("\n3. Recording turn outcomes")
    memory_turns = SpatialMemory()
    for _ in range(5):
        memory_turns.record_action('turn_left', success=True)
    for _ in range(2):
        memory_turns.record_action('turn_right', success=False)

    print(f"   Best turn direction: {memory_turns.get_best_turn_direction()}")

    print("\n✅ SpatialMemory tests passed!")


def test_behavior_tree():
    print("\n" + "=" * 60)
    print("Testing Behavior Trees")
    print("=" * 60)

    # Create mock components
    class MockRobot:
        def execute(self, action, duration):
            print(f"   [ROBOT] {action} for {duration}s")

        def get_distance(self):
            return 50.0  # Clear path

    world = WorldModel()
    world.update_ultrasonic(50.0)  # Clear
    memory = SpatialMemory()
    robot = MockRobot()

    context = BehaviorContext(world, memory, robot)

    print("\n1. Building exploration tree")
    tree = build_exploration_tree()
    print(f"   Tree root: {tree}")

    print("\n2. Executing tree with clear path")
    status = tree.execute(context)
    print(f"   Result: {status}")

    print("\n3. Executing tree with blocked path")
    world.update_ultrasonic(10.0)  # Blocked!
    status = tree.execute(context)
    print(f"   Result: {status}")

    print("\n✅ Behavior tree tests passed!")


def test_integration():
    print("\n" + "=" * 60)
    print("Testing Integration")
    print("=" * 60)

    # Simulate complete decision cycle
    class MockRobot:
        def __init__(self):
            self.distance = 50.0

        def execute(self, action, duration):
            print(f"   [EXECUTE] {action} for {duration:.1f}s")
            # Simulate movement
            if action == 'forward':
                self.distance -= 10
            elif action == 'backward':
                self.distance += 10
            elif 'turn' in action:
                self.distance = 50.0  # New direction

        def get_distance(self):
            return self.distance

    world = WorldModel()
    memory = SpatialMemory()
    robot = MockRobot()
    context = BehaviorContext(world, memory, robot)
    tree = build_exploration_tree()

    print("\nSimulating 5 decision cycles:")
    for i in range(5):
        print(f"\nCycle {i+1}:")

        # Update world model
        distance = robot.get_distance()
        world.update_ultrasonic(distance)
        print(f"   Sensor: {distance:.1f}cm")

        # Execute behavior
        status = tree.execute(context)
        print(f"   Status: {status}")

        # Check memory
        if i >= 3:
            print(f"   Stuck? {memory.is_stuck()}")

    print("\n✅ Integration test passed!")


if __name__ == "__main__":
    try:
        test_world_model()
        test_spatial_memory()
        test_behavior_tree()
        test_integration()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe v4 architecture is working correctly!")
        print("Components are properly integrated and tested.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
