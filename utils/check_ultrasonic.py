#!/usr/bin/env python3
"""
Test ultrasonic sensor to verify it's connected and working properly.
Shows real-time distance readings and obstacle detection.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.robot_controller import RobotController


def main():
    print("=" * 60)
    print("ULTRASONIC SENSOR TEST")
    print("=" * 60)
    print()

    # Load config
    config = load_config("config/config.json")
    threshold = config.get("robot_settings", {}).get("obstacle_distance_threshold_cm", 20.0)

    print(f"Initializing robot controller...")
    print(f"Obstacle threshold: {threshold}cm")
    print()

    try:
        robot = RobotController(config)
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize robot: {e}")
        return 1

    # Check if sensor is available
    initial_distance = robot.get_distance()
    if initial_distance is None:
        print("❌ ULTRASONIC SENSOR NOT AVAILABLE")
        print()
        print("Possible issues:")
        print("  1. Sensor not physically connected to robot")
        print("  2. SunFounder library doesn't support ultrasonic on this hardware")
        print("  3. Running in MockRobot mode (dry run)")
        print()
        print("Check hardware connections and verify sensor is plugged in.")
        return 1

    print("✅ ULTRASONIC SENSOR DETECTED")
    print()
    print("Reading distance continuously...")
    print("Press Ctrl+C to stop")
    print()
    print(f"{'Time':<12} {'Distance (cm)':<15} {'Status':<20} {'Obstacle?'}")
    print("-" * 70)

    try:
        reading_count = 0
        obstacle_count = 0
        clear_count = 0

        while True:
            distance = robot.get_distance()
            obstacle_info = robot.get_obstacle_info()

            if distance is None:
                status = "⚠️  SENSOR READ FAILED"
                obstacle_str = "Unknown"
            elif distance < threshold:
                status = f"🚨 TOO CLOSE"
                obstacle_str = "YES"
                obstacle_count += 1
            elif distance < threshold * 1.5:
                status = f"⚠️  Getting close"
                obstacle_str = "No (but near)"
            else:
                status = "✅ Clear"
                obstacle_str = "No"
                clear_count += 1

            reading_count += 1
            timestamp = time.strftime("%H:%M:%S")
            distance_str = f"{distance:.1f}" if distance is not None else "N/A"

            print(f"{timestamp:<12} {distance_str:<15} {status:<20} {obstacle_str}")

            # Show statistics every 20 readings
            if reading_count % 20 == 0:
                print()
                print(f"📊 Statistics after {reading_count} readings:")
                print(f"   Obstacles detected: {obstacle_count} ({obstacle_count/reading_count*100:.1f}%)")
                print(f"   Clear path: {clear_count} ({clear_count/reading_count*100:.1f}%)")
                print()

            time.sleep(0.5)

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f"Total readings: {reading_count}")
        print(f"Obstacles detected: {obstacle_count} ({obstacle_count/reading_count*100:.1f}% of time)")
        print(f"Clear path: {clear_count} ({clear_count/reading_count*100:.1f}% of time)")
        print()

        if obstacle_count > reading_count * 0.8:
            print("⚠️  WARNING: Obstacle detected >80% of time")
            print("   Robot may be blocked or sensor pointed at obstacle")
        elif clear_count == reading_count:
            print("✅ Sensor working - all readings show clear path")
        else:
            print("✅ Sensor working - detecting obstacles when present")

        return 0


if __name__ == "__main__":
    sys.exit(main())
