"""Tests for the anti-loop / safety-override state machine in BaseBehavior."""

import pytest

import behaviors.base_behavior as bb
from ai.vision_ai import SceneAnalysis
from behaviors.base_behavior import BaseBehavior
from behaviors.exploration import ExplorationBehavior


class StubRobot:
    """Stands in for RobotController."""

    def __init__(self):
        self.executed = []
        self.obstacle = False
        self.distance = 100.0
        self.threshold = 20.0

    def get_obstacle_info(self):
        return {
            "distance_cm": self.distance,
            "has_obstacle": self.obstacle,
            "threshold_cm": self.threshold,
            "sensor_available": True,
        }

    def get_distance(self):
        return self.distance

    def has_obstacle(self):
        return self.obstacle

    def execute(self, action, duration_s=0.6):
        self.executed.append((action, duration_s))


class StubAI:
    def __init__(self):
        self.voice = None


def scene(description="a room", hazards=(), objects=()):
    return SceneAnalysis(
        description=description,
        objects=list(objects),
        hazards=list(hazards),
        suggested_actions=[],
        raw="",
        processing_time_s=0.0,
    )


@pytest.fixture
def behavior(config, dummy_voice):
    return BaseBehavior(
        config=config,
        robot=StubRobot(),
        camera=None,
        ai=StubAI(),
        duration_minutes=0.1,
        voice=dummy_voice,
    )


def changing_scenes():
    """Generator of always-distinct analyses (never triggers stagnation)."""
    i = 0
    while True:
        yield scene(description=f"scene {i}")
        i += 1


def act(behavior, action, analysis=None):
    choice = behavior.postprocess_action(action, analysis=analysis)
    if isinstance(choice, tuple):
        return choice[0]
    return choice


# ---------------------------------------------------------------- detectors


def test_is_repeating_detects_runs(behavior):
    for _ in range(3):
        behavior._recent_actions.append("turn_left")
    assert behavior._is_repeating("turn_left") is True
    assert behavior._is_repeating("forward") is False


def test_is_oscillating_detects_abab(behavior):
    for a in ["turn_left", "turn_right", "turn_left", "turn_right"]:
        behavior._recent_actions.append(a)
    assert behavior._is_oscillating() is True


def test_is_oscillating_ignores_aaaa(behavior):
    for _ in range(4):
        behavior._recent_actions.append("forward")
    assert behavior._is_oscillating() is False


def test_scene_stagnation_requires_full_window(behavior):
    sig = behavior._scene_signature(scene("same"))
    for _ in range(behavior._scene_repeat_threshold - 1):
        behavior._recent_scene_sigs.append(sig)
    assert behavior._is_scene_stagnant() is False
    behavior._recent_scene_sigs.append(sig)
    assert behavior._is_scene_stagnant() is True


def test_scene_signature_includes_hazards_and_objects(behavior):
    a = behavior._scene_signature(scene("room", hazards=["stairs"]))
    b = behavior._scene_signature(scene("room", hazards=["cliff"]))
    assert a != b


# ---------------------------------------------------------------- ban list


def test_banned_action_replaced_with_fallback(behavior):
    behavior._ban("forward")
    result = act(behavior, "forward")
    assert result != "forward"
    assert result in {"backward", "turn_left", "turn_right", "stop"}


def test_ban_expires(behavior, monkeypatch):
    behavior._ban("forward", seconds=5.0)
    assert behavior._is_banned("forward") is True
    real_time = bb.time.time()
    monkeypatch.setattr(bb.time, "time", lambda: real_time + 6.0)
    assert behavior._is_banned("forward") is False


def test_fallback_skips_banned_alternatives(behavior):
    for a in ("backward", "turn_left", "turn_right", "forward"):
        behavior._ban(a)
    assert behavior._pick_non_banned_fallback("forward") == "stop"


# ---------------------------------------------------------------- safety override


def test_obstacle_override_turns_instead_of_forward(behavior, monkeypatch):
    monkeypatch.setattr(bb.random, "choice", lambda seq: seq[0])
    behavior.robot.obstacle = True
    behavior.robot.distance = 10.0
    choice = behavior.postprocess_action("forward", analysis=scene())
    assert choice == ("turn_left", 0.8)


def test_obstacle_override_never_returns_forward(behavior):
    behavior.robot.obstacle = True
    behavior.robot.distance = 5.0
    scenes = changing_scenes()
    for _ in range(10):
        result = act(behavior, "forward", analysis=next(scenes))
        assert result != "forward"


def test_repeated_obstacle_overrides_trigger_escape_and_ban(behavior, monkeypatch):
    monkeypatch.setattr(bb.random, "choice", lambda seq: seq[0])
    behavior.robot.obstacle = True
    behavior.robot.distance = 5.0
    for _ in range(behavior._max_obstacle_overrides - 1):
        behavior.postprocess_action("forward", analysis=scene())
    assert not behavior._is_banned("forward")
    behavior.postprocess_action("forward", analysis=scene())
    assert behavior._is_banned("forward")
    assert behavior._consecutive_obstacle_overrides == 0


def test_override_counter_resets_when_path_clears(behavior, monkeypatch):
    monkeypatch.setattr(bb.random, "choice", lambda seq: seq[0])
    behavior.robot.obstacle = True
    behavior.postprocess_action("forward", analysis=scene())
    assert behavior._consecutive_obstacle_overrides == 1
    behavior.robot.obstacle = False
    behavior.postprocess_action("forward", analysis=scene("different view"))
    assert behavior._consecutive_obstacle_overrides == 0


def test_override_counter_resets_on_other_action(behavior, monkeypatch):
    monkeypatch.setattr(bb.random, "choice", lambda seq: seq[0])
    behavior.robot.obstacle = True
    behavior.postprocess_action("forward", analysis=scene())
    assert behavior._consecutive_obstacle_overrides == 1
    behavior.robot.obstacle = False
    behavior.postprocess_action("turn_left", analysis=scene("different"))
    assert behavior._consecutive_obstacle_overrides == 0


# ---------------------------------------------------------------- loop breaking


def test_stop_always_honored(behavior):
    assert act(behavior, "stop", analysis=scene()) == "stop"


def test_repeating_action_triggers_escape(behavior):
    scenes = changing_scenes()
    results = [act(behavior, "turn_left", analysis=next(scenes)) for _ in range(behavior._repeat_threshold)]
    # Once the repeat threshold is hit, the action must change
    assert results[-1] != "turn_left"
    assert behavior._is_banned("turn_left")


def test_oscillation_triggers_escape_and_bans_turns(behavior):
    scenes = changing_scenes()
    pattern = ["turn_left", "turn_right", "turn_left", "turn_right"]
    results = [act(behavior, a, analysis=next(scenes)) for a in pattern]
    assert results[-1] not in ("turn_left", "turn_right")
    assert behavior._is_banned("turn_left")
    assert behavior._is_banned("turn_right")


def test_scene_stagnation_triggers_escape(behavior):
    same = scene("identical view")
    results = [act(behavior, a, analysis=same) for a in ["forward", "backward", "forward", "backward"]]
    # By the stagnation threshold the escape plan must kick in
    assert behavior._recent_scene_sigs.maxlen is not None
    assert results[-1] in ("stop", "backward", "turn_left", "turn_right", "forward")
    assert behavior._last_control_note != ""


def test_escape_turns_opposite_of_stuck_turn(behavior):
    behavior._escape_stage = 2  # escape plan stage 2 is a turn
    choice = behavior._escape("turn_right", reason="test")
    assert choice[0] == "turn_left"


def test_escape_cooldown_returns_simple_motion(behavior):
    behavior._last_escape_at = bb.time.time()  # just escaped
    assert behavior._escape("turn_left", reason="test") == ("backward", 1.2)
    # A stuck 'forward' (obstacle loop) must never be answered with forward
    assert behavior._escape("forward", reason="test") == ("backward", 1.2)
    assert behavior._escape("backward", reason="test") == ("forward", 1.2)


def test_escape_clears_histories(behavior):
    scenes = changing_scenes()
    for _ in range(behavior._repeat_threshold):
        act(behavior, "forward", analysis=next(scenes))
    # escape happened; histories reset to just the escape action
    assert len(behavior._recent_actions) == 1


# ---------------------------------------------------------------- exploration bias


def test_exploration_biases_forward_on_immediate_turn_flip(config, dummy_voice):
    b = ExplorationBehavior(
        config=config,
        robot=StubRobot(),
        camera=None,
        ai=StubAI(),
        duration_minutes=0.1,
        voice=dummy_voice,
    )
    scenes = changing_scenes()
    assert act(b, "turn_left", analysis=next(scenes)) == "turn_left"
    assert act(b, "turn_right", analysis=next(scenes)) == "forward"


def test_exploration_keeps_stop(config, dummy_voice):
    b = ExplorationBehavior(
        config=config,
        robot=StubRobot(),
        camera=None,
        ai=StubAI(),
        duration_minutes=0.1,
        voice=dummy_voice,
    )
    assert act(b, "stop", analysis=scene()) == "stop"
