"""Tests for the BaseBehavior.run() main loop: normal ticks, camera failure
circuit breaker, panic handling, and the AI-unavailable short-circuit."""

import json

import pytest

import behaviors.base_behavior as bb
from ai.vision_ai import SceneAnalysis
from behaviors.base_behavior import BaseBehavior
from vision.camera import CameraPanicException

from tests.test_base_behavior import StubRobot


class LoopClock:
    """Simulated time: only sleep() advances it."""

    def __init__(self, start=100.0):
        self.now = start

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(float(seconds), 0.01)


class StubCamera:
    """Yields scripted frames: a str (base64), None (failure), or an Exception."""

    def __init__(self, frames=None):
        self.frames = list(frames) if frames else []

    def capture(self, save=True):
        frame = self.frames.pop(0) if self.frames else "b64frame"
        if isinstance(frame, Exception):
            raise frame
        if frame is None:
            return None, None, None
        return "pil-image", frame, "/fake/frame.jpg"


class StubAI:
    def __init__(self, description="a room", action="forward"):
        self.voice = None
        self.analyze_calls = 0
        self.decide_calls = 0
        self.description = description
        self.action = action

    def analyze_scene(self, b64, context):
        self.analyze_calls += 1
        return SceneAnalysis(
            description=f"{self.description} {self.analyze_calls}",
            objects=[],
            hazards=[],
            suggested_actions=[],
            raw="",
            processing_time_s=0.0,
        )

    def decide_action(self, **kwargs):
        self.decide_calls += 1
        return {"action": self.action, "duration_s": 1.0, "reasoning": "test"}

    def generate_dialogue(self, **kwargs):
        return ""


@pytest.fixture
def loop_clock(monkeypatch):
    clock = LoopClock()
    monkeypatch.setattr(bb.time, "time", clock.time)
    monkeypatch.setattr(bb.time, "sleep", clock.sleep)
    return clock


def make_behavior(config, dummy_voice, *, camera, ai, duration_minutes=0.08):
    return BaseBehavior(
        config=config,
        robot=StubRobot(),
        camera=camera,
        ai=ai,
        duration_minutes=duration_minutes,
        voice=dummy_voice,
    )


def test_normal_ticks_execute_decisions(config, dummy_voice, loop_clock):
    ai = StubAI(action="forward")
    b = make_behavior(config, dummy_voice, camera=StubCamera(), ai=ai)
    b.run()
    # ~4.8s at a 2.5s capture interval → two full ticks
    assert ai.analyze_calls == 2
    assert ai.decide_calls == 2
    assert b.robot.executed == [("forward", 1.0), ("forward", 1.0)]


def test_run_announces_start_and_end(config, dummy_voice, loop_clock):
    cfg = dict(config)
    cfg["voice_settings"] = {"enabled": True, "narration_enabled": True, "narration_min_interval_s": 0.0}
    b = make_behavior(cfg, dummy_voice, camera=StubCamera(), ai=StubAI())
    b.run()
    spoken = [t for t, _, _ in dummy_voice.spoken]
    assert any("Starting" in t for t in spoken)
    assert any("complete" in t for t in spoken)


def test_camera_circuit_breaker_stops_after_five_failures(config, dummy_voice, loop_clock):
    ai = StubAI()
    camera = StubCamera(frames=[None] * 20)
    b = make_behavior(config, dummy_voice, camera=camera, ai=ai, duration_minutes=5.0)
    b.run()
    # Loop must end long before the 5-minute deadline
    assert loop_clock.now - 100.0 < 60.0
    # Every executed action was a safety stop; the AI was never consulted
    assert b.robot.executed
    assert all(action == "stop" for action, _ in b.robot.executed)
    assert ai.analyze_calls == 0
    # Exactly 5 failures consumed
    assert len(camera.frames) == 15


def test_camera_failure_counter_resets_on_success(config, dummy_voice, loop_clock):
    ai = StubAI()
    # 4 failures (below breaker), one good frame, 4 more failures, good frame...
    frames = [None] * 4 + ["b64"] + [None] * 4 + ["b64"]
    b = make_behavior(config, dummy_voice, camera=StubCamera(frames=frames), ai=ai, duration_minutes=0.5)
    b.run()
    # The breaker (5 consecutive) never trips, so the AI does get consulted
    assert ai.analyze_calls >= 1


def test_camera_panic_stops_and_continues(config, dummy_voice, loop_clock):
    ai = StubAI(action="forward")
    camera = StubCamera(frames=[CameraPanicException("sudden change"), "b64"])
    b = make_behavior(config, dummy_voice, camera=camera, ai=ai, duration_minutes=0.15)
    b.run()
    # First reaction is an emergency stop, then the loop recovers
    assert b.robot.executed[0] == ("stop", 0.5)
    assert ("forward", 1.0) in b.robot.executed


def test_ai_unavailable_short_circuits_decide(config, dummy_voice, loop_clock):
    ai = StubAI(description="AI unavailable")

    def broken_analyze(b64, context):
        ai.analyze_calls += 1
        return SceneAnalysis(
            description="AI unavailable",
            objects=[],
            hazards=[],
            suggested_actions=["stop"],
            raw="err",
            processing_time_s=0.0,
        )

    ai.analyze_scene = broken_analyze
    b = make_behavior(config, dummy_voice, camera=StubCamera(), ai=ai)
    b.run()
    assert ai.analyze_calls >= 1
    assert ai.decide_calls == 0  # never burns a doomed decision call
    assert all(action == "stop" for action, _ in b.robot.executed)


def test_decisions_logged_as_jsonl(config, dummy_voice, loop_clock, tmp_path):
    cfg = dict(config)
    cfg["logging_settings"] = {"save_images": False, "save_decisions": True}
    ai = StubAI(action="forward")
    b = make_behavior(cfg, dummy_voice, camera=StubCamera(), ai=ai)
    b.decisions_path = tmp_path / "logs" / "decisions.jsonl"
    b.run()
    lines = b.decisions_path.read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["mode"] == "base"
    assert record["executed_action"] == "forward"
    assert record["decision"]["action"] == "forward"
    assert "obstacle" in record


def test_tuple_choice_from_postprocess_overrides_duration(config, dummy_voice, loop_clock, monkeypatch):
    monkeypatch.setattr(bb.random, "choice", lambda seq: seq[0])
    ai = StubAI(action="forward")
    b = make_behavior(config, dummy_voice, camera=StubCamera(), ai=ai)
    b.robot.obstacle = True  # ultrasonic override kicks in
    b.robot.distance = 10.0
    b.run()
    assert b.robot.executed
    for action, duration in b.robot.executed:
        assert action != "forward"
