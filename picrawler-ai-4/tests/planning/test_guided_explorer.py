"""Tests for planning/guided_explorer.py — look, explain, propose, ask, act."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from core.spatial_memory import SpatialMemory
from core.world_model import WorldModel
from planning.guided_explorer import (
    Answer, Decision, GuidedDecider, GuidedExplorer, VALID_ACTIONS,
    interpret_answer, _parse_json,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeAnalysis:
    description: str = "a hallway with a chair on the left"
    objects: List[str] = field(default_factory=lambda: ["chair"])
    hazards: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    processing_time: float = 0.1


class FakeVision:
    def __init__(self, analysis=None):
        self.analysis = analysis or FakeAnalysis()
        self.calls = 0
    def analyze_scene(self, b64):
        self.calls += 1
        return self.analysis


class FakeCamera:
    def __init__(self, ok=True):
        self.ok = ok
        self.voice = None
    def capture(self, save=False):
        if not self.ok:
            return None, None, None
        return object(), "b64data", None


class FakeVoice:
    def __init__(self):
        self.spoken: List[str] = []
    def say(self, text, *, level="normal", force=False):
        self.spoken.append(text)


class FakeRobot:
    def __init__(self, distance=80.0, obstacle=False):
        self.calls: List[tuple] = []
        self.distance = distance
        self.obstacle = obstacle
    def execute(self, action, duration=0.5):
        self.calls.append((action, duration))
    def get_distance(self):
        return self.distance
    def has_obstacle(self):
        return self.obstacle


class ScriptedListener:
    """Returns answers in order; None = timeout."""
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts: List[str] = []
    def ask(self, prompt="", timeout_s=0):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else None


class ScriptedDecider:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.rejected_seen: List[List[str]] = []
    def decide(self, world_state, description, hazards, recent, rejected):
        self.rejected_seen.append(list(rejected))
        if self.decisions:
            return self.decisions.pop(0)
        return Decision("stop", "nothing left", 0.1, source="rule")


def _explorer(answers, decisions=None, robot=None, vision=None, camera=None, config=None):
    robot = robot or FakeRobot()
    voice = FakeVoice()
    listener = ScriptedListener(answers)
    decider = ScriptedDecider(decisions or [Decision("forward", "the way ahead is clear", 0.9)])
    ex = GuidedExplorer(
        camera=camera or FakeCamera(), vision_ai=vision or FakeVision(), decider=decider,
        voice=voice, robot=robot, world_model=WorldModel(obstacle_threshold_cm=20.0),
        memory=SpatialMemory(history_size=20), listener=listener, config=config or {},
        clock=lambda: 0.0, sleep=lambda s: None,
    )
    return ex, voice, robot, listener, decider


# ---------------------------------------------------------------------------
# interpret_answer
# ---------------------------------------------------------------------------

class TestInterpretAnswer:
    @pytest.mark.parametrize("t", ["y", "yes", "Yes.", "ok", "okay", "sure", "go ahead", "yeah"])
    def test_yes(self, t):
        assert interpret_answer(t).kind == "yes"

    @pytest.mark.parametrize("t", ["n", "no", "No!", "nope", "don't"])
    def test_no(self, t):
        assert interpret_answer(t).kind == "no"

    @pytest.mark.parametrize("t", ["q", "quit", "exit", "goodbye"])
    def test_quit(self, t):
        assert interpret_answer(t).kind == "quit"

    @pytest.mark.parametrize("t,action", [
        ("left", "turn_left"), ("turn left", "turn_left"), ("no, turn right instead", "turn_right"),
        ("go forward", "forward"), ("back up", "backward"), ("reverse", "backward"),
        ("stay", "stop"), ("please don't move", "stop"),
    ])
    def test_instruction(self, t, action):
        a = interpret_answer(t)
        assert a.kind == "instruction"
        assert a.action == action

    def test_timeout(self):
        assert interpret_answer(None).kind == "timeout"
        assert interpret_answer("   ").kind == "timeout"

    def test_unknown(self):
        assert interpret_answer("purple monkey dishwasher").kind == "unknown"

    def test_leading_yes_with_chatter(self):
        assert interpret_answer("yes that sounds great").kind == "yes"


# ---------------------------------------------------------------------------
# Decision / JSON parsing / rule-based fallback
# ---------------------------------------------------------------------------

class TestDecision:
    def test_phrases_are_speakable(self):
        for a in VALID_ACTIONS:
            assert "_" not in Decision(a, "r").phrase()

    def test_parse_plain_json(self):
        assert _parse_json('{"action": "forward"}')["action"] == "forward"

    def test_parse_fenced_json(self):
        assert _parse_json('```json\n{"action": "stop"}\n```')["action"] == "stop"

    def test_parse_json_with_prose(self):
        assert _parse_json('Sure! {"action": "turn_left", "reasoning": "x"} ok')["action"] == "turn_left"

    def test_parse_garbage(self):
        assert _parse_json("nothing here") is None


class TestRuleBased:
    def test_forward_when_clear(self):
        d = GuidedDecider.rule_based({"obstacles": {"front": False}, "best_direction": "forward"}, [])
        assert d.action == "forward"

    def test_never_forward_when_blocked(self):
        d = GuidedDecider.rule_based({"obstacles": {"front": True}, "best_direction": "forward"}, [])
        assert d.action != "forward"

    def test_prefers_best_direction_when_blocked(self):
        d = GuidedDecider.rule_based({"obstacles": {"front": True}, "best_direction": "right"}, [])
        assert d.action == "turn_right"

    def test_skips_rejected(self):
        d = GuidedDecider.rule_based({"obstacles": {"front": False}}, ["forward", "turn_left"])
        assert d.action == "turn_right"

    def test_everything_rejected(self):
        d = GuidedDecider.rule_based({"obstacles": {"front": False}}, list(VALID_ACTIONS))
        assert d.action == "stop"


class TestGuidedDeciderAI:
    def _decider(self, content):
        client = MagicMock()
        msg = MagicMock(); msg.content = content
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=msg)])
        return GuidedDecider({"ai_settings": {"model": "m"}}, client=client), client

    def test_uses_ai_answer(self):
        d, _ = self._decider(json.dumps({"action": "turn_left", "reasoning": "more room", "confidence": 0.8}))
        out = d.decide({"obstacles": {}}, "a room", [], [], [])
        assert out.action == "turn_left"
        assert out.reasoning == "more room"
        assert out.source == "ai"

    def test_normalises_left_right(self):
        d, _ = self._decider('{"action": "right", "reasoning": "r"}')
        assert d.decide({"obstacles": {}}, "", [], [], []).action == "turn_right"

    def test_falls_back_when_ai_picks_rejected(self):
        d, _ = self._decider('{"action": "forward", "reasoning": "r"}')
        out = d.decide({"obstacles": {"front": False}}, "", [], [], ["forward"])
        assert out.action != "forward"
        assert out.source == "rule"

    def test_falls_back_on_garbage(self):
        d, _ = self._decider("I think you should go forward!")
        assert d.decide({"obstacles": {"front": False}}, "", [], [], []).source == "rule"

    def test_falls_back_on_exception(self):
        d, client = self._decider("{}")
        client.chat.completions.create.side_effect = RuntimeError("boom")
        assert d.decide({"obstacles": {"front": False}}, "", [], [], []).source == "rule"

    def test_prompt_mentions_rejected_and_scene(self):
        d, client = self._decider('{"action": "stop", "reasoning": "r"}')
        d.decide({"obstacles": {}}, "a red sofa", ["stairs"], ["forward"], ["forward"])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "a red sofa" in prompt and "stairs" in prompt and "forward" in prompt


# ---------------------------------------------------------------------------
# GuidedExplorer.step — the conversation loop
# ---------------------------------------------------------------------------

class TestStepConversation:
    def test_explains_then_asks_then_acts_on_yes(self):
        ex, voice, robot, listener, _ = _explorer(["yes"])
        assert ex.step() is True
        assert voice.spoken[0] == "I see a hallway with a chair on the left"
        assert "Is that a good choice?" in voice.spoken[1]
        assert "walk forward" in voice.spoken[1]
        assert "the way ahead is clear" in voice.spoken[1]
        assert robot.calls == [("forward", 1.2)]
        assert ex.executed == 1

    def test_mentions_hazards(self):
        vision = FakeVision(FakeAnalysis(hazards=["stairs", "a cable"]))
        ex, voice, robot, _, _ = _explorer(["yes"], vision=vision)
        ex.step()
        assert any("careful of stairs and a cable" in s for s in voice.spoken)

    def test_no_then_alternative_accepted(self):
        decisions = [Decision("forward", "clear", 0.9), Decision("turn_left", "more space left", 0.7)]
        ex, voice, robot, _, decider = _explorer(["no", "yes"], decisions)
        ex.step()
        assert robot.calls == [("turn_left", 0.8)]
        assert decider.rejected_seen == [[], ["forward"]]
        assert any("How about I turn left instead" in s for s in voice.spoken)

    def test_user_instruction_overrides_proposal(self):
        ex, voice, robot, _, _ = _explorer(["no, turn right"])
        ex.step()
        assert robot.calls == [("turn_right", 0.8)]
        assert any("I'll turn right" in s for s in voice.spoken)

    def test_quit_stops_loop(self):
        ex, voice, robot, _, _ = _explorer(["quit"])
        assert ex.step() is False
        assert robot.calls == []

    def test_timeout_stays_put(self):
        ex, voice, robot, _, _ = _explorer([None])
        assert ex.step() is True
        assert robot.calls == []
        assert any("didn't hear" in s for s in voice.spoken)

    def test_unknown_treated_as_no_and_reproposes(self):
        decisions = [Decision("forward", "a", 0.9), Decision("turn_left", "b", 0.7)]
        ex, voice, robot, _, _ = _explorer(["blorp", "yes"], decisions)
        ex.step()
        assert robot.calls == [("turn_left", 0.8)]
        assert any("didn't catch that" in s for s in voice.spoken)

    def test_all_proposals_rejected_then_asks_for_instruction(self):
        decisions = [Decision("forward", "a"), Decision("turn_left", "b"), Decision("turn_right", "c")]
        ex, voice, robot, _, _ = _explorer(["no", "no", "no", "back"], decisions,
                                           config={"guided_settings": {"max_reproposals": 2}})
        ex.step()
        assert any("What would you like me to do?" in s for s in voice.spoken)
        assert robot.calls == [("backward", 1.0)]

    def test_all_rejected_then_quit(self):
        decisions = [Decision("forward", "a"), Decision("turn_left", "b"), Decision("turn_right", "c")]
        ex, _, robot, _, _ = _explorer(["no", "no", "no", "quit"], decisions)
        assert ex.step() is False
        assert robot.calls == []


class TestSafety:
    def test_forward_blocked_by_ultrasonic_becomes_stop(self):
        robot = FakeRobot(distance=10.0, obstacle=True)
        ex, voice, robot, _, _ = _explorer(["yes"], robot=robot)
        ex.step()
        assert robot.calls == [("stop", 0.2)]
        assert any("too close in front" in s for s in voice.spoken)

    def test_user_forward_instruction_still_safety_checked(self):
        robot = FakeRobot(distance=5.0, obstacle=True)
        ex, _, robot, _, _ = _explorer(["forward"], robot=robot)
        ex.step()
        assert robot.calls == [("stop", 0.2)]

    def test_turn_not_blocked_by_front_obstacle(self):
        robot = FakeRobot(distance=5.0, obstacle=True)
        ex, _, robot, _, _ = _explorer(["yes"], [Decision("turn_left", "r")], robot=robot)
        ex.step()
        assert robot.calls == [("turn_left", 0.8)]

    def test_robot_failure_is_spoken_not_fatal(self):
        robot = FakeRobot()
        robot.execute = MagicMock(side_effect=RuntimeError("servo"))
        ex, voice, _, _, _ = _explorer(["yes"], robot=robot)
        ex.step()
        assert any("didn't work" in s for s in voice.spoken)


class TestLookRobustness:
    def test_camera_failure_still_asks(self):
        ex, voice, robot, _, _ = _explorer(["yes"], camera=FakeCamera(ok=False))
        ex.step()
        assert voice.spoken[0] == "I can't make out much right now."
        assert robot.calls == [("forward", 1.2)]

    def test_no_vision_ai(self):
        ex, voice, robot, _, _ = _explorer(["yes"], vision=None)
        ex.vision_ai = None
        ex.step()
        assert voice.spoken[0] == "I can't make out much right now."

    def test_vision_unavailable_fallback_not_spoken_as_scene(self):
        vision = FakeVision(FakeAnalysis(description="Vision unavailable", hazards=["unknown"]))
        ex, voice, _, _, _ = _explorer(["yes"], vision=vision)
        ex.step()
        assert voice.spoken[0] == "I can't make out much right now."
        assert not any("careful" in s for s in voice.spoken)

    def test_memory_records_action(self):
        ex, _, _, _, _ = _explorer(["yes"])
        ex.step()
        recent = ex.memory.get_recent_actions(5)
        assert recent and recent[-1] == "forward"


class TestRun:
    def test_run_greets_and_signs_off(self):
        ex, voice, robot, _, _ = _explorer(["quit"])
        assert ex.run(duration_min=1) == 0
        assert voice.spoken[0].startswith("Hello")
        assert voice.spoken[-1] == "Okay, I'm done for now."
        assert robot.calls[-1][0] == "stop"

    def test_run_ends_on_duration(self):
        t = {"now": 0.0}
        ex, voice, robot, _, _ = _explorer(["yes", "yes", "yes"])
        ex.clock = lambda: t["now"]
        def tick(prompt="", timeout_s=0):
            t["now"] += 40.0
            return "yes"
        ex.listener.ask = tick
        ex.run(duration_min=1)     # 60 s → 2 cycles before the clock passes the end
        assert 1 <= len(robot.calls) - 1 <= 2


# ---------------------------------------------------------------------------
# Autonomy policy
# ---------------------------------------------------------------------------

class PollingListener(ScriptedListener):
    """Adds poll(): returns queued interjections without blocking."""
    def __init__(self, answers=(), typed=()):
        super().__init__(answers)
        self.typed = list(typed)
    def poll(self):
        return self.typed.pop(0) if self.typed else None


def _auto(autonomy, decisions, typed=(), answers=(), config=None, robot=None):
    robot = robot or FakeRobot()
    voice = FakeVoice()
    listener = PollingListener(answers, typed)
    decider = ScriptedDecider(decisions)
    cfg = dict(config or {})
    ex = GuidedExplorer(
        camera=FakeCamera(), vision_ai=FakeVision(), decider=decider, voice=voice,
        robot=robot, world_model=WorldModel(obstacle_threshold_cm=20.0),
        memory=SpatialMemory(history_size=20), listener=listener, config=cfg,
        autonomy=autonomy, clock=lambda: 0.0, sleep=lambda s: None,
    )
    return ex, voice, robot, listener


class TestAutonomy:
    def test_unknown_mode_falls_back_to_ask_always(self):
        ex, *_ = _auto("yolo", [])
        assert ex.autonomy == "ask_always"

    def test_config_autonomy_used_when_no_override(self):
        ex, *_ = _auto(None, [], config={"guided_settings": {"autonomy": "never_ask"}})
        assert ex.autonomy == "never_ask"

    def test_never_ask_narrates_and_acts(self):
        ex, voice, robot, listener = _auto("never_ask", [Decision("turn_left", "more room", 0.9)])
        assert ex.step() is True
        assert robot.calls == [("turn_left", 0.8)]
        assert any(s.startswith("I'll turn left because more room") for s in voice.spoken)
        assert listener.prompts == []                       # never waited for an answer

    def test_ask_forward_only_asks_for_forward(self):
        ex, voice, robot, listener = _auto("ask_forward_only", [Decision("forward", "clear", 0.9)], answers=["yes"])
        ex.step()
        assert len(listener.prompts) == 1
        assert robot.calls == [("forward", 1.2)]

    def test_ask_forward_only_does_not_ask_for_turns(self):
        ex, voice, robot, listener = _auto("ask_forward_only", [Decision("turn_right", "r", 0.9)])
        ex.step()
        assert listener.prompts == []
        assert robot.calls == [("turn_right", 0.8)]

    def test_ask_when_unsure_confident_ai_acts(self):
        ex, _, robot, listener = _auto("ask_when_unsure", [Decision("forward", "r", 0.95, source="ai")])
        ex.step()
        assert listener.prompts == []
        assert robot.calls == [("forward", 1.2)]

    def test_ask_when_unsure_low_confidence_asks(self):
        ex, _, robot, listener = _auto("ask_when_unsure", [Decision("forward", "r", 0.3, source="ai")], answers=["no", "quit"])
        assert ex.step() is False
        assert len(listener.prompts) >= 1
        assert robot.calls == []

    def test_ask_when_unsure_rule_based_asks(self):
        ex, _, robot, listener = _auto("ask_when_unsure", [Decision("forward", "r", 0.99, source="rule")], answers=["yes"])
        ex.step()
        assert len(listener.prompts) == 1

    def test_ask_when_unsure_threshold_from_config(self):
        ex, _, _, listener = _auto("ask_when_unsure", [Decision("forward", "r", 0.75, source="ai")],
                                   answers=["yes"], config={"guided_settings": {"confidence_threshold": 0.9}})
        ex.step()
        assert len(listener.prompts) == 1

    def test_autonomous_forward_is_still_safety_gated(self):
        robot = FakeRobot(distance=5.0, obstacle=True)
        ex, voice, robot, _ = _auto("never_ask", [Decision("forward", "r", 0.9)], robot=robot)
        ex.step()
        assert robot.calls == [("stop", 0.2)]
        assert any("too close" in s for s in voice.spoken)


class TestInterjections:
    def test_typed_quit_stops_autonomous_loop(self):
        ex, _, robot, _ = _auto("never_ask", [Decision("forward", "r", 0.9)], typed=["quit"])
        assert ex.step() is False
        assert robot.calls == []

    def test_typed_instruction_overrides(self):
        ex, voice, robot, _ = _auto("never_ask", [Decision("forward", "r", 0.9)], typed=["left"])
        ex.step()
        assert robot.calls == [("turn_left", 0.8)]
        assert any("turn left instead" in s for s in voice.spoken)

    def test_typed_no_pauses_and_asks(self):
        ex, voice, robot, listener = _auto("never_ask", [Decision("forward", "r", 0.9)],
                                           typed=["no"], answers=["yes"])
        ex.step()
        assert any("ask you first" in s for s in voice.spoken)
        assert len(listener.prompts) == 1
        assert robot.calls == [("forward", 1.2)]
        assert ex.autonomy == "never_ask"          # restored after the one-off ask

    def test_typed_no_then_no_stays_put(self):
        ex, voice, robot, _ = _auto("never_ask", [Decision("forward", "r", 0.9)], typed=["no"], answers=["no"])
        ex.step()
        assert robot.calls == []
        assert any("stay put" in s for s in voice.spoken)

    def test_listener_without_poll_is_fine(self):
        ex, _, robot, _ = _auto("never_ask", [Decision("forward", "r", 0.9)])
        ex.listener = ScriptedListener([])        # no poll() method at all
        ex.step()
        assert robot.calls == [("forward", 1.2)]


# ---------------------------------------------------------------------------
# Mapping underneath
# ---------------------------------------------------------------------------

class FakeSlam:
    def __init__(self, fail=False):
        self.frames = []
        self.saved = None
        self.shut = False
        self.fail = fail
    def process_frame(self, image, depth, action_hint=None):
        if self.fail:
            raise RuntimeError("vo exploded")
        self.frames.append(action_hint)
        return None, None
    def save_map(self, path):
        self.saved = path
    def get_statistics(self):
        return {"map": {"explored_percent": 3.0}, "slam": {"loop_closures": 0}, "point_cloud": {"points": 12}}
    def shutdown(self):
        self.shut = True


class TestMapping:
    def test_slam_receives_each_frame_with_last_action(self):
        ex, _, robot, _, _ = _explorer(["yes", "yes"])
        slam = FakeSlam()
        ex.slam = slam
        robot.last_action = None
        ex.step()
        robot.last_action = "forward"
        ex.step()
        assert slam.frames == [None, "forward"]

    def test_slam_failure_does_not_break_cycle(self):
        ex, _, robot, _, _ = _explorer(["yes"])
        ex.slam = FakeSlam(fail=True)
        ex.step()
        assert robot.calls == [("forward", 1.2)]

    def test_run_saves_map_and_shuts_down(self, tmp_path):
        ex, _, _, _, _ = _explorer(["quit"], config={"guided_settings": {"map_path": str(tmp_path / "m.jpg")}})
        slam = FakeSlam()
        ex.slam = slam
        ex.run(1)
        assert slam.saved == str(tmp_path / "m.jpg")
        assert slam.shut is True

    def test_no_slam_is_default(self):
        ex, *_ = _explorer(["quit"])
        assert ex.slam is None
