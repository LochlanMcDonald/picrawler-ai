#!/usr/bin/env python3

import time

from core.config_loader import load_config
from core.logger import setup_logging
from core.robot_controller import RobotController


def main() -> int:
    setup_logging("INFO")
    config = load_config("config/config.json")
    robot = RobotController(config)

    print("Forward")
    robot.execute("forward", 0.8)
    time.sleep(0.5)

    print("Turn left")
    robot.execute("turn_left", 0.7)
    time.sleep(0.5)

    print("Turn right")
    robot.execute("turn_right", 0.7)
    time.sleep(0.5)

    print("Backward")
    robot.execute("backward", 0.8)
    time.sleep(0.5)

    print("Stop")
    robot.execute("stop", 0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
