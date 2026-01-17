"""Robot hardware abstraction.

This project is designed to run on a SunFounder PiCrawler, but it should also
run on a dev machine or a Pi without the robot attached.

- If the SunFounder 'picrawler' library is installed, we drive real motors.
- Otherwise we fall back to a MockRobot that logs actions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Motion:
    action: str
    value: Optional[float] = None


class BaseRobot:
    def forward(self, speed: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def backward(self, speed: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def turn_left(self, speed: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def turn_right(self, speed: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def posture(self, name: str, speed: int) -> None:  # pragma: no cover
        raise NotImplementedError


class MockRobot(BaseRobot):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, speed: int) -> None:
        self.logger.info(f"MOCK: forward speed={speed}")

    def backward(self, speed: int) -> None:
        self.logger.info(f"MOCK: backward speed={speed}")

    def turn_left(self, speed: int) -> None:
        self.logger.info(f"MOCK: turn_left speed={speed}")

    def turn_right(self, speed: int) -> None:
        self.logger.info(f"MOCK: turn_right speed={speed}")

    def stop(self) -> None:
        self.logger.info("MOCK: stop")

    def posture(self, name: str, speed: int) -> None:
        self.logger.info(f"MOCK: posture {name} speed={speed}")


class SunFounderPiCrawlerRobot(BaseRobot):
    """Adapter around SunFounder picrawler library."""

    def __init__(self) -> None:
        from picrawler import Picrawler  # type: ignore

        self.logger = logging.getLogger(self.__class__.__name__)
        self.crawler = Picrawler()
        # Safe default posture
        try:
            self.crawler.do_step("sit", 60)
        except Exception:
            pass

    def forward(self, speed: int) -> None:
        self.crawler.forward(speed)

    def backward(self, speed: int) -> None:
        self.crawler.backward(speed)

    def turn_left(self, speed: int) -> None:
        self.crawler.turn_left(speed)

    def turn_right(self, speed: int) -> None:
        self.crawler.turn_right(speed)

    def stop(self) -> None:
        self.crawler.stop()

    def posture(self, name: str, speed: int) -> None:
        self.crawler.do_step(name, speed)


class RobotController:
    """High-level control used by behaviors."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        rs = config.get("robot_settings", {})
        self.move_speed = int(rs.get("movement_speed", 50))
        self.turn_speed = int(rs.get("turn_speed", 45))
        self.step_delay = float(rs.get("step_delay", 0.05))
        self.dry_run_if_no_hardware = bool(rs.get("dry_run_if_no_hardware", True))

        self.robot: BaseRobot = self._init_robot()

    def _init_robot(self) -> BaseRobot:
        try:
            robot = SunFounderPiCrawlerRobot()
            self.logger.info("Hardware backend: SunFounder picrawler")
            return robot
        except Exception as e:
            if not self.dry_run_if_no_hardware:
                raise
            self.logger.warning(f"Falling back to MockRobot (reason: {e})")
            return MockRobot()

    def execute(self, action: str, duration_s: float = 0.6) -> None:
        """Execute a simple motion primitive."""
        action = action.lower().strip()
        self.logger.info(f"ACTION: {action} duration={duration_s}")

        if action in {"forward", "ahead"}:
            self.robot.forward(self.move_speed)
            time.sleep(duration_s)
            self.robot.stop()
        elif action in {"back", "backward", "reverse"}:
            self.robot.backward(self.move_speed)
            time.sleep(duration_s)
            self.robot.stop()
        elif action in {"turn_left", "left"}:
            self.robot.turn_left(self.turn_speed)
            time.sleep(duration_s)
            self.robot.stop()
        elif action in {"turn_right", "right"}:
            self.robot.turn_right(self.turn_speed)
            time.sleep(duration_s)
            self.robot.stop()
        elif action in {"stop", "halt"}:
            self.robot.stop()
        elif action.startswith("posture:"):
            name = action.split(":", 1)[1].strip()
            self.robot.posture(name, max(self.move_speed, 40))
        else:
            self.logger.warning(f"Unknown action '{action}', stopping")
            self.robot.stop()
