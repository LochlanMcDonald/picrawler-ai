"""Tests for the safety gate between decisions and physical motors."""

from typing import List, Optional

import pytest

import core.robot_controller as rc
from core.robot_controller import BaseRobot, MockRobot, RobotController


class FakeRobot(BaseRobot):
    """Records motor calls; distance readings come from a scripted sequence."""

    def __init__(self, distances: Optional[List[Optional[float]]] = None):
        self.calls: List[str] = []
        self._distances = list(distances) if distances else []
        self._last: Optional[float] = self._distances[-1] if self._distances else None

    def forward(self, speed):
        self.calls.append(f"forward:{speed}")

    def backward(self, speed):
        self.calls.append(f"backward:{speed}")

    def turn_left(self, speed):
        self.calls.append(f"turn_left:{speed}")

    def turn_right(self, speed):
        self.calls.append(f"turn_right:{speed}")

    def stop(self):
        self.calls.append("stop")

    def posture(self, name, speed):
        self.calls.append(f"posture:{name}:{speed}")

    def get_distance(self):
        if self._distances:
            self._last = self._distances.pop(0)
        return self._last


@pytest.fixture
def controller(config, monkeypatch):
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    ctl = RobotController(config)
    ctl.robot = FakeRobot(distances=[100.0])
    return ctl


def test_falls_back_to_mock_without_hardware(config):
    ctl = RobotController(config)
    assert isinstance(ctl.robot, MockRobot)


def test_missing_hardware_raises_when_dry_run_disabled(config):
    config["robot_settings"]["dry_run_if_no_hardware"] = False
    with pytest.raises(Exception):
        RobotController(config)


def test_invalid_action_coerced_to_stop(controller):
    controller.execute("self_destruct", 1.0)
    assert controller.robot.calls == ["stop"]


def test_forward_runs_then_stops_when_clear(controller):
    controller.execute("forward", 0.3)
    assert controller.robot.calls[0] == "forward:50"
    assert controller.robot.calls[-1] == "stop"


def test_forward_blocked_by_obstacle(controller):
    controller.robot = FakeRobot(distances=[10.0])
    controller.execute("forward", 0.5)
    assert not any(c.startswith("forward") for c in controller.robot.calls)
    assert "stop" in controller.robot.calls


def test_forward_aborts_mid_motion_when_obstacle_appears(controller):
    # Pre-check sees 100cm (clear); first mid-motion poll sees 5cm.
    controller.robot = FakeRobot(distances=[100.0, 5.0])
    controller.execute("forward", 5.0)
    assert controller.robot.calls[0] == "forward:50"
    assert controller.robot.calls[-1] == "stop"


def test_backward_not_blocked_by_forward_obstacle(controller):
    controller.robot = FakeRobot(distances=[5.0])
    controller.execute("backward", 0.3)
    assert controller.robot.calls[0] == "backward:50"
    assert controller.robot.calls[-1] == "stop"


@pytest.mark.parametrize(
    "action,expected",
    [("turn_left", "turn_left:45"), ("left", "turn_left:45"), ("turn_right", "turn_right:45"), ("right", "turn_right:45")],
)
def test_turns_use_turn_speed(controller, action, expected):
    controller.execute(action, 0.3)
    assert controller.robot.calls[0] == expected
    assert controller.robot.calls[-1] == "stop"


def test_posture_action_parsing(controller):
    controller.execute("posture: stand ", 0.3)
    assert controller.robot.calls == ["posture:stand:50"]


def test_action_normalized_case_and_whitespace(controller):
    controller.execute("  STOP  ", 0.3)
    assert controller.robot.calls == ["stop"]


def test_has_obstacle_threshold(controller):
    controller.robot = FakeRobot(distances=[19.9])
    assert controller.has_obstacle() is True
    controller.robot = FakeRobot(distances=[20.0])
    assert controller.has_obstacle() is False


def test_has_obstacle_false_without_sensor(controller):
    controller.robot = FakeRobot(distances=[None])
    assert controller.has_obstacle() is False


def test_get_obstacle_info_shape(controller):
    controller.robot = FakeRobot(distances=[15.0])
    info = controller.get_obstacle_info()
    assert info == {
        "distance_cm": 15.0,
        "has_obstacle": True,
        "threshold_cm": 20.0,
        "sensor_available": True,
    }


def test_get_obstacle_info_without_sensor(controller):
    controller.robot = FakeRobot(distances=[None])
    info = controller.get_obstacle_info()
    assert info["sensor_available"] is False
    assert info["has_obstacle"] is False


def test_timed_move_stops_even_if_sensor_raises(controller):
    class ExplodingRobot(FakeRobot):
        def get_distance(self):
            raise RuntimeError("sensor died")

    robot = ExplodingRobot()
    controller.robot = robot
    with pytest.raises(RuntimeError):
        controller._timed_move(lambda: robot.forward(50), 1.0, monitor_obstacles=True)
    assert robot.calls[-1] == "stop"


class TestMockRobot:
    def test_distance_decreases_moving_forward(self):
        m = MockRobot()
        d0 = m.get_distance()
        m.forward(50)
        assert m.get_distance() < d0

    def test_distance_increases_moving_backward(self):
        m = MockRobot()
        d0 = m.get_distance()
        m.backward(50)
        assert m.get_distance() > d0

    def test_distance_capped(self):
        m = MockRobot()
        for _ in range(30):
            m.backward(50)
        assert m.get_distance() <= 100


# ------------------------------------------------------- SunFounder adapter


from core.robot_controller import SunFounderPiCrawlerRobot


def make_adapter(crawler, ultrasonic=None):
    """Build the adapter around a fake crawler, skipping hardware init."""
    import logging

    r = SunFounderPiCrawlerRobot.__new__(SunFounderPiCrawlerRobot)
    r.logger = logging.getLogger("test-adapter")
    r.crawler = crawler
    r.ultrasonic = ultrasonic
    return r


class DoActionCrawler:
    def __init__(self):
        self.calls = []

    def do_action(self, name, speed=None):
        self.calls.append((name, speed))


class DoStepCrawler:
    def __init__(self):
        self.calls = []

    def do_step(self, name, speed=None):
        self.calls.append((name, speed))


def test_do_prefers_do_action():
    c = DoActionCrawler()
    make_adapter(c)._do("forward", 50)
    assert c.calls == [("forward", 50)]


def test_do_falls_back_to_do_step():
    c = DoStepCrawler()
    make_adapter(c)._do("forward", 50)
    assert c.calls == [("forward", 50)]


def test_do_retries_without_speed_on_type_error():
    class NoSpeedCrawler:
        def __init__(self):
            self.calls = []

        def do_action(self, name):  # no speed parameter
            self.calls.append(name)

    c = NoSpeedCrawler()
    make_adapter(c)._do("forward", 50)
    assert c.calls == ["forward"]


def test_do_uses_direct_callable_as_last_resort():
    class DirectCrawler:
        def __init__(self):
            self.calls = []

        def forward(self, speed):
            self.calls.append(("forward", speed))

    c = DirectCrawler()
    make_adapter(c)._do("forward", 50)
    assert c.calls == [("forward", 50)]


def test_do_raises_when_unsupported():
    class EmptyCrawler:
        pass

    with pytest.raises(AttributeError):
        make_adapter(EmptyCrawler())._do("forward", 50)


def test_backward_tries_back_alias():
    class BackOnlyCrawler:
        def __init__(self):
            self.calls = []

        def do_action(self, name, speed=None):
            if name != "back":
                raise ValueError(f"unknown action {name}")
            self.calls.append((name, speed))

    c = BackOnlyCrawler()
    make_adapter(c).backward(50)
    assert ("back", 50) in c.calls


class TestAdapterDistance:
    def test_no_sensor_returns_none(self):
        assert make_adapter(object(), ultrasonic=None).get_distance() is None

    def test_reads_via_read_method(self):
        class Sensor:
            def read(self):
                return 42.0

        assert make_adapter(object(), ultrasonic=Sensor()).get_distance() == 42.0

    @pytest.mark.parametrize("bad", [-1.0, 0.0, 400.0, 9999.0])
    def test_out_of_range_readings_rejected(self, bad):
        class Sensor:
            def __init__(self, v):
                self.v = v

            def read(self):
                return self.v

        assert make_adapter(object(), ultrasonic=Sensor(bad)).get_distance() is None

    def test_sensor_exception_returns_none(self):
        class Sensor:
            def read(self):
                raise OSError("sensor fault")

        assert make_adapter(object(), ultrasonic=Sensor()).get_distance() is None
