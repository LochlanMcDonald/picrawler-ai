"""
Robot hardware abstraction.

This project is designed to run on a SunFounder PiCrawler, but it should also
run on a dev machine or a Pi without the robot attached.

- If the SunFounder 'picrawler' library is installed, we drive real motors.
- Otherwise we fall back to a MockRobot that logs actions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional


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
    """
    Adapter around SunFounder picrawler library.

    SunFounder variants differ slightly. Many expose:
      - do_action(action_name, speed)
      - do_step(step_or_pose_name, speed)

    and DO NOT expose forward()/turn_left() methods.
    This adapter normalizes to BaseRobot primitives.
    """

    def __init__(self) -> None:
        from picrawler import Picrawler  # type: ignore

        self.logger = logging.getLogger(self.__class__.__name__)
        self.crawler = Picrawler()

        # Safe default posture if available (log failures but continue)
        for pose in ("sit", "stand"):
            try:
                self.posture(pose, 60)
                self.logger.info(f"Initialized with posture: {pose}")
                break
            except Exception as e:
                self.logger.debug(f"Posture '{pose}' not available: {e}")

    def _do(self, name: str, speed: int) -> None:
        """
        Call into the underlying SunFounder API using the best available method.
        Tries do_action first, then do_step, then a direct callable attribute.
        """
        # Most common in SunFounder PiCrawler libs
        if hasattr(self.crawler, "do_action"):
            try:
                self.crawler.do_action(name, speed)  # type: ignore[attr-defined]
                return
            except TypeError:
                # some versions may not take speed for some actions
                self.crawler.do_action(name)  # type: ignore[attr-defined]
                return

        # Some versions treat actions as named steps
        if hasattr(self.crawler, "do_step"):
            try:
                self.crawler.do_step(name, speed)  # type: ignore[attr-defined]
                return
            except TypeError:
                self.crawler.do_step(name)  # type: ignore[attr-defined]
                return

        # Last resort: direct method if it exists
        fn = getattr(self.crawler, name, None)
        if callable(fn):
            try:
                fn(speed)
            except TypeError:
                fn()
            return

        raise AttributeError(
            f"Picrawler backend does not support action '{name}' "
            f"(no do_action/do_step and no callable attribute)."
        )

    def forward(self, speed: int) -> None:
        # common naming in SunFounder actions
        self._do("forward", speed)

    def backward(self, speed: int) -> None:
        # some libs use backward; some use back
        for name in ("backward", "back"):
            try:
                self._do(name, speed)
                return
            except Exception as e:
                self.logger.debug(f"Action '{name}' not available: {e}")
                continue
        # last try (will raise if not found):
        self._do("backward", speed)

    def turn_left(self, speed: int) -> None:
        # common naming variants
        for name in ("turn_left", "left"):
            try:
                self._do(name, speed)
                return
            except Exception as e:
                self.logger.debug(f"Action '{name}' not available: {e}")
                continue
        # last try (will raise if not found):
        self._do("turn_left", speed)

    def turn_right(self, speed: int) -> None:
        for name in ("turn_right", "right"):
            try:
                self._do(name, speed)
                return
            except Exception as e:
                self.logger.debug(f"Action '{name}' not available: {e}")
                continue
        # last try (will raise if not found):
        self._do("turn_right", speed)

    def stop(self) -> None:
        # stop often ignores speed; try both call shapes
        for name in ("stop", "halt"):
            try:
                self._do(name, 0)
                return
            except Exception as e:
                self.logger.debug(f"Action '{name}' not available: {e}")
                continue
        # If stop isn't a known action, try direct attr without speed
        fn = getattr(self.crawler, "stop", None)
        if callable(fn):
            fn()
            return
        # As a last resort, log critical error (cannot stop robot!)
        self.logger.error("CRITICAL: No stop() method or stop action available on hardware!")

    def posture(self, name: str, speed: int) -> None:
        # Postures are typically steps
        self._do(name, speed)


class RobotController:
    """High-level control used by behaviors."""

    # Valid action prefixes and keywords
    VALID_ACTIONS = {
        "forward", "ahead",
        "back", "backward", "reverse",
        "turn_left", "left",
        "turn_right", "right",
        "stop", "halt",
    }

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

        # Validate action (posture: prefix is also valid)
        is_posture = action.startswith("posture:")
        if not is_posture and action not in self.VALID_ACTIONS:
            self.logger.warning(f"Invalid action '{action}', defaulting to stop for safety")
            action = "stop"

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
