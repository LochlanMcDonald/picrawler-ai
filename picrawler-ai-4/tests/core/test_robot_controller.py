"""Tests for core/robot_controller.py — SunFounder adapter and motion execution.

These pin the *real* SunFounder picrawler API:
    Picrawler.do_action(motion_name, step=1, speed=50)
with motion names such as "forward" and "turn left" (with a space), and no
"stop" motion at all. Getting any of that wrong makes the robot either not
move, or run dozens of gait cycles per command.
"""
from __future__ import annotations

import sys
import time
import types
from typing import List, Tuple

import pytest

import core.robot_controller as rc
from core.robot_controller import (
    ActionWatchdog, MockRobot, RobotController, SunFounderPiCrawlerRobot,
)


# ---------------------------------------------------------------------------
# Fake SunFounder library
# ---------------------------------------------------------------------------

class FakePicrawler:
    """Mimics sunfounder/picrawler's Picrawler.do_action exactly."""
    KNOWN = {"forward", "backward", "turn left", "turn right",
             "turn left angle", "turn right angle", "stand", "sit"}

    def __init__(self, cycle_s: float = 0.0):
        self.calls: List[Tuple[str, int, int]] = []
        self.unknown: List[str] = []
        self.cycle_s = cycle_s

    def do_action(self, motion_name, step=1, speed=50):
        if motion_name not in self.KNOWN:
            self.unknown.append(motion_name)   # library only prints here
            return
        for _ in range(step):
            self.calls.append((motion_name, step, speed))
            if self.cycle_s:
                time.sleep(self.cycle_s)


@pytest.fixture
def fake_lib(monkeypatch):
    """Install a fake `picrawler` module and return the instance the adapter will get."""
    inst = FakePicrawler()
    mod = types.ModuleType("picrawler")
    mod.Picrawler = lambda *a, **k: inst
    monkeypatch.setitem(sys.modules, "picrawler", mod)
    return inst


def _config(**robot_settings) -> dict:
    rs = {"movement_speed": 50, "turn_speed": 45, "dry_run_if_no_hardware": True,
          "max_action_duration_s": 2.0, "action_grace_s": 2.0}
    rs.update(robot_settings)
    return {"robot_settings": rs}


# ---------------------------------------------------------------------------
# SunFounderPiCrawlerRobot adapter
# ---------------------------------------------------------------------------

class TestSunFounderAdapter:
    def test_init_sits_once_not_sixty_times(self, fake_lib):
        SunFounderPiCrawlerRobot()
        sits = [c for c in fake_lib.calls if c[0] == "sit"]
        assert len(sits) == 1
        assert sits[0][1] == 1          # step=1

    def test_forward_is_one_cycle_with_speed_in_speed_slot(self, fake_lib):
        r = SunFounderPiCrawlerRobot()
        fake_lib.calls.clear()
        r.forward(50)
        assert fake_lib.calls == [("forward", 1, 50)]

    def test_backward(self, fake_lib):
        r = SunFounderPiCrawlerRobot()
        fake_lib.calls.clear()
        r.backward(40)
        assert fake_lib.calls == [("backward", 1, 40)]

    def test_turns_use_library_names_with_spaces(self, fake_lib):
        r = SunFounderPiCrawlerRobot()
        fake_lib.calls.clear()
        r.turn_left(45)
        r.turn_right(45)
        assert [c[0] for c in fake_lib.calls] == ["turn left", "turn right"]
        assert fake_lib.unknown == []

    def test_stop_does_not_send_unknown_motion(self, fake_lib):
        r = SunFounderPiCrawlerRobot()
        fake_lib.calls.clear()
        r.stop()
        assert fake_lib.calls == []
        assert fake_lib.unknown == []

    def test_stop_uses_direct_method_when_available(self, fake_lib):
        fake_lib.stopped = 0
        fake_lib.stop = lambda: setattr(fake_lib, "stopped", fake_lib.stopped + 1)
        r = SunFounderPiCrawlerRobot()
        r.stop()
        assert fake_lib.stopped == 1

    def test_is_blocking(self, fake_lib):
        assert SunFounderPiCrawlerRobot().is_blocking() is True

    def test_old_library_without_step_kwarg(self, monkeypatch):
        class OldLib:
            def __init__(self):
                self.calls = []
            def do_action(self, motion_name, speed=50):
                self.calls.append((motion_name, speed))
        inst = OldLib()
        mod = types.ModuleType("picrawler")
        mod.Picrawler = lambda *a, **k: inst
        monkeypatch.setitem(sys.modules, "picrawler", mod)
        r = SunFounderPiCrawlerRobot()
        r.forward(30)
        assert ("forward", 30) in inst.calls


# ---------------------------------------------------------------------------
# RobotController.execute with a blocking backend
# ---------------------------------------------------------------------------

class TestExecuteBlockingBackend:
    def test_forward_repeats_gait_cycles_to_fill_duration(self, monkeypatch):
        inst = FakePicrawler(cycle_s=0.05)
        mod = types.ModuleType("picrawler")
        mod.Picrawler = lambda *a, **k: inst
        monkeypatch.setitem(sys.modules, "picrawler", mod)

        ctl = RobotController(_config())
        inst.calls.clear()
        ctl.execute("forward", 0.22)
        fwd = [c for c in inst.calls if c[0] == "forward"]
        # ~0.22 s / 0.05 s per cycle → 5 cycles (allow scheduler slop)
        assert 4 <= len(fwd) <= 6
        assert all(c[1] == 1 and c[2] == 50 for c in fwd)

    def test_turn_left_maps_to_library_name(self, monkeypatch):
        inst = FakePicrawler()
        mod = types.ModuleType("picrawler")
        mod.Picrawler = lambda *a, **k: inst
        monkeypatch.setitem(sys.modules, "picrawler", mod)

        ctl = RobotController(_config())
        inst.calls.clear()
        ctl.execute("turn_left", 0.01)
        assert inst.calls and inst.calls[0][0] == "turn left"
        assert inst.calls[0][2] == 45           # turn_speed
        assert inst.unknown == []

    def test_watchdog_does_not_fire_for_normal_overrun(self, monkeypatch, caplog):
        """One gait cycle may overrun the requested duration; that's not a fault."""
        inst = FakePicrawler(cycle_s=0.3)
        mod = types.ModuleType("picrawler")
        mod.Picrawler = lambda *a, **k: inst
        monkeypatch.setitem(sys.modules, "picrawler", mod)

        ctl = RobotController(_config(max_action_duration_s=0.2, action_grace_s=1.0))
        with caplog.at_level("ERROR"):
            ctl.execute("forward", 0.2)      # 1 cycle of 0.3 s > 0.2 s cap
        assert "WATCHDOG TIMEOUT" not in caplog.text


# ---------------------------------------------------------------------------
# RobotController.execute with a continuous (mock) backend
# ---------------------------------------------------------------------------

class TestExecuteContinuousBackend:
    def _controller(self, monkeypatch) -> RobotController:
        monkeypatch.setitem(sys.modules, "picrawler", None)   # force ImportError
        ctl = RobotController(_config())
        assert isinstance(ctl.robot, MockRobot)
        return ctl

    def test_mock_backend_is_non_blocking(self, monkeypatch):
        ctl = self._controller(monkeypatch)
        assert ctl.robot.is_blocking() is False

    def test_forward_then_sleep_then_stop(self, monkeypatch):
        ctl = self._controller(monkeypatch)
        slept = []
        monkeypatch.setattr(rc.time, "sleep", lambda s: slept.append(s))
        ctl.execute("forward", 0.4)
        assert slept == [0.4]

    def test_last_action_recorded(self, monkeypatch):
        ctl = self._controller(monkeypatch)
        monkeypatch.setattr(rc.time, "sleep", lambda s: None)
        assert ctl.last_action is None
        ctl.execute("turn_right", 0.1)
        assert ctl.last_action == "turn_right"
        ctl.execute("bogus", 0.1)          # invalid → coerced to stop
        assert ctl.last_action == "stop"

    def test_duration_capped_at_max(self, monkeypatch):
        ctl = self._controller(monkeypatch)
        slept = []
        monkeypatch.setattr(rc.time, "sleep", lambda s: slept.append(s))
        ctl.execute("forward", 10.0)
        assert slept == [2.0]


# ---------------------------------------------------------------------------
# ActionWatchdog grace period
# ---------------------------------------------------------------------------

class TestWatchdogGrace:
    def test_grace_added_to_timer(self):
        wd = ActionWatchdog(max_duration_s=1.0, grace_s=0.25)
        wd.set_robot(MockRobot())
        wd.start_action("forward", 1.0)
        try:
            assert abs(wd.timer.interval - 1.25) < 1e-9
        finally:
            wd.end_action()

    def test_default_grace(self):
        assert ActionWatchdog().grace_s == 2.0
