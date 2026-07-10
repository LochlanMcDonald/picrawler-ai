"""
Robot hardware abstraction.

This project is designed to run on a SunFounder PiCrawler, but it should also
run on a dev machine or a Pi without the robot attached.

- If the SunFounder 'picrawler' library is installed, we drive real motors.
- Otherwise we fall back to a MockRobot that logs actions.
"""

from __future__ import annotations

import logging
import threading
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

    def get_distance(self) -> Optional[float]:  # pragma: no cover
        """Get distance reading from ultrasonic sensor in cm."""
        return None  # Default: no sensor


class MockRobot(BaseRobot):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._mock_distance = 50.0  # Start with safe distance

    def forward(self, speed: int) -> None:
        self.logger.info(f"MOCK: forward speed={speed}")
        # Simulate getting closer to obstacle when moving forward
        if self._mock_distance > 10:
            self._mock_distance -= 5

    def backward(self, speed: int) -> None:
        self.logger.info(f"MOCK: backward speed={speed}")
        # Simulate moving away from obstacle
        self._mock_distance = min(100, self._mock_distance + 5)

    def turn_left(self, speed: int) -> None:
        self.logger.info(f"MOCK: turn_left speed={speed}")
        # Simulate different distance after turning
        import random
        self._mock_distance = random.uniform(20, 80)

    def turn_right(self, speed: int) -> None:
        self.logger.info(f"MOCK: turn_right speed={speed}")
        # Simulate different distance after turning
        import random
        self._mock_distance = random.uniform(20, 80)

    def stop(self) -> None:
        self.logger.info("MOCK: stop")

    def posture(self, name: str, speed: int) -> None:
        self.logger.info(f"MOCK: posture {name} speed={speed}")

    def get_distance(self) -> Optional[float]:
        """Return mock distance reading."""
        return self._mock_distance


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

        # Try to access ultrasonic sensor if available
        self.ultrasonic = None
        try:
            if hasattr(self.crawler, "ultrasonic"):
                self.ultrasonic = self.crawler.ultrasonic
                self.logger.info("Ultrasonic sensor available")
            elif hasattr(self.crawler, "get_distance"):
                # Some versions have direct distance method
                self.ultrasonic = self.crawler
                self.logger.info("Distance sensor available")
        except Exception as e:
            self.logger.debug(f"Ultrasonic sensor not available: {e}")

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

    def get_distance(self) -> Optional[float]:
        """Get distance reading from ultrasonic sensor in cm.

        Returns None if sensor not available or reading fails.
        """
        if self.ultrasonic is None:
            return None

        try:
            # Try common SunFounder API patterns
            if hasattr(self.ultrasonic, "get_distance"):
                distance = self.ultrasonic.get_distance()
            elif hasattr(self.ultrasonic, "read"):
                distance = self.ultrasonic.read()
            elif callable(self.ultrasonic):
                distance = self.ultrasonic()
            else:
                return None

            # Validate reading (sensor sometimes returns -1 or very large values on error)
            if distance is not None and 0 < distance < 400:  # 4 meters max
                return float(distance)
            return None

        except Exception as e:
            self.logger.debug(f"Distance sensor read failed: {e}")
            return None


class ActionWatchdog:
    """External watchdog that guarantees actions cannot exceed maximum duration.

    Runs in a separate thread and force-stops robot if action exceeds timeout,
    even if main thread hangs or blocks.
    """

    def __init__(self, max_duration_s: float = 2.0):
        """
        Args:
            max_duration_s: Maximum allowed action duration (hard limit)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_duration_s = max_duration_s

        self.robot: Optional[BaseRobot] = None
        self.timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()
        self.current_action: Optional[str] = None

        self.logger.info(f"Action watchdog initialized (max duration: {max_duration_s:.1f}s)")

    def set_robot(self, robot: BaseRobot) -> None:
        """Set robot instance for emergency stop."""
        self.robot = robot

    def start_action(self, action: str, requested_duration: float) -> float:
        """Register action start and return capped duration.

        Args:
            action: Action being executed
            requested_duration: Requested duration

        Returns:
            Actual duration to use (capped at max)
        """
        with self.lock:
            # Cancel any existing timer
            if self.timer is not None:
                self.timer.cancel()

            # Cap duration at maximum
            actual_duration = min(requested_duration, self.max_duration_s)

            if requested_duration > self.max_duration_s:
                self.logger.warning(
                    f"Action {action} duration {requested_duration:.2f}s capped to {self.max_duration_s:.1f}s"
                )

            self.current_action = action

            # Start watchdog timer with small buffer for cleanup
            self.timer = threading.Timer(
                self.max_duration_s + 0.5,  # Extra 0.5s grace period
                self._timeout_callback
            )
            self.timer.daemon = True
            self.timer.start()

            self.logger.debug(f"Watchdog started for {action} ({actual_duration:.2f}s)")

            return actual_duration

    def end_action(self) -> None:
        """Register action completion (cancels watchdog)."""
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None

            self.current_action = None
            self.logger.debug("Watchdog stopped normally")

    def _timeout_callback(self) -> None:
        """Called by watchdog timer if action exceeds max duration."""
        with self.lock:
            if self.robot is None:
                self.logger.error("Watchdog timeout but no robot registered!")
                return

            action = self.current_action or "unknown"

            self.logger.error(
                f"⚠️  WATCHDOG TIMEOUT: Action '{action}' exceeded {self.max_duration_s:.1f}s - "
                "executing emergency stop from watchdog thread"
            )

            try:
                # Emergency stop from watchdog thread
                self.robot.stop()
                self.logger.info("Emergency stop executed by watchdog")
            except Exception as e:
                self.logger.error(f"Watchdog emergency stop failed: {e}")
            finally:
                self.timer = None
                self.current_action = None


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
        self.obstacle_threshold_cm = float(rs.get("obstacle_distance_threshold_cm", 20.0))
        self.dry_run_if_no_hardware = bool(rs.get("dry_run_if_no_hardware", True))

        # Safety: Maximum action duration (hard limit)
        max_action_duration = float(rs.get("max_action_duration_s", 2.0))

        self.robot: BaseRobot = self._init_robot()

        # Initialize watchdog for action timeout enforcement
        self.watchdog = ActionWatchdog(max_duration_s=max_action_duration)
        self.watchdog.set_robot(self.robot)

        # Check if ultrasonic sensor is available
        distance = self.robot.get_distance()
        if distance is not None:
            self.logger.info(f"Ultrasonic sensor initialized (current: {distance:.1f}cm, threshold: {self.obstacle_threshold_cm}cm)")
        else:
            self.logger.info("Ultrasonic sensor not available - using vision only")

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

    def get_distance(self) -> Optional[float]:
        """Get current distance reading from ultrasonic sensor.

        Returns:
            Distance in cm, or None if sensor not available
        """
        return self.robot.get_distance()

    def has_obstacle(self) -> bool:
        """Check if there's an obstacle within threshold distance.

        Returns:
            True if obstacle detected, False otherwise
        """
        distance = self.get_distance()
        if distance is None:
            return False
        return distance < self.obstacle_threshold_cm

    def get_obstacle_info(self) -> dict:
        """Get obstacle detection info for logging/decision making.

        Returns:
            Dict with distance, has_obstacle, threshold
        """
        distance = self.get_distance()
        return {
            "distance_cm": distance,
            "has_obstacle": distance is not None and distance < self.obstacle_threshold_cm,
            "threshold_cm": self.obstacle_threshold_cm,
            "sensor_available": distance is not None,
        }

    def execute(self, action: str, duration_s: float = 0.6) -> None:
        """Execute a simple motion primitive with watchdog timeout enforcement."""
        action = action.lower().strip()

        # Validate action (posture: prefix is also valid)
        is_posture = action.startswith("posture:")
        if not is_posture and action not in self.VALID_ACTIONS:
            self.logger.warning(f"Invalid action '{action}', defaulting to stop for safety")
            action = "stop"

        # Check for obstacles before forward movement
        if action in {"forward", "ahead"}:
            if self.has_obstacle():
                distance = self.get_distance()
                self.logger.warning(
                    f"Obstacle detected at {distance:.1f}cm (< {self.obstacle_threshold_cm}cm) - "
                    "blocking forward movement"
                )
                action = "stop"

        # Start watchdog and get capped duration
        capped_duration = self.watchdog.start_action(action, duration_s)

        self.logger.info(f"ACTION: {action} duration={capped_duration:.2f}s (requested={duration_s:.2f}s)")

        try:
            if action in {"forward", "ahead"}:
                self.robot.forward(self.move_speed)
                time.sleep(capped_duration)
                self.robot.stop()

            elif action in {"back", "backward", "reverse"}:
                self.robot.backward(self.move_speed)
                time.sleep(capped_duration)
                self.robot.stop()

            elif action in {"turn_left", "left"}:
                self.robot.turn_left(self.turn_speed)
                time.sleep(capped_duration)
                self.robot.stop()

            elif action in {"turn_right", "right"}:
                self.robot.turn_right(self.turn_speed)
                time.sleep(capped_duration)
                self.robot.stop()

            elif action in {"stop", "halt"}:
                self.robot.stop()

            elif action.startswith("posture:"):
                name = action.split(":", 1)[1].strip()
                self.robot.posture(name, max(self.move_speed, 40))

            else:
                self.logger.warning(f"Unknown action '{action}', stopping")
                self.robot.stop()

        finally:
            # Always stop watchdog when action completes
            self.watchdog.end_action()
