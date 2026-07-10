"""Tests for the AI decision contract: whatever the model returns, the robot
must only ever receive a valid, clamped, safe action."""

import json

import httpx
import pytest
from openai import APITimeoutError

import ai.vision_ai as va
from ai.vision_ai import AIVisionSystem, SceneAnalysis


class StubResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out

        class R:
            output_text = out

        return R()


class StubClient:
    def __init__(self, outputs):
        self.responses = StubResponses(outputs)


def make_ai(config, dummy_voice, outputs):
    ai = AIVisionSystem(config, voice=dummy_voice)
    ai.client = StubClient(outputs)
    return ai


def analysis(**kw):
    base = dict(
        description="a hallway",
        objects=["chair"],
        hazards=[],
        suggested_actions=["forward"],
        raw="",
        processing_time_s=0.1,
    )
    base.update(kw)
    return SceneAnalysis(**base)


ACTIONS = ["forward", "turn_left", "turn_right", "backward", "stop"]


def timeout_error():
    return APITimeoutError(request=httpx.Request("POST", "https://api.test"))


# ---------------------------------------------------------------- decide_action


def test_valid_decision_passes_through(config, dummy_voice):
    ai = make_ai(config, dummy_voice, [json.dumps({"action": "forward", "duration_s": 1.0, "reasoning": "clear"})])
    d = ai.decide_action(analysis(), "explore", ACTIONS)
    assert d["action"] == "forward"
    assert d["duration_s"] == 1.0
    assert d["reasoning"] == "clear"


def test_unknown_action_coerced_to_stop(config, dummy_voice):
    ai = make_ai(config, dummy_voice, [json.dumps({"action": "fly", "duration_s": 1.0})])
    assert ai.decide_action(analysis(), "explore", ACTIONS)["action"] == "stop"


def test_action_outside_available_set_coerced_to_stop(config, dummy_voice):
    ai = make_ai(config, dummy_voice, [json.dumps({"action": "forward", "duration_s": 1.0})])
    assert ai.decide_action(analysis(), "explore", ["stop"])["action"] == "stop"


@pytest.mark.parametrize("raw,expected", [(0.01, 0.2), (99.0, 3.0), ("bogus", 0.6)])
def test_duration_clamped_and_sanitized(config, dummy_voice, raw, expected):
    ai = make_ai(config, dummy_voice, [json.dumps({"action": "forward", "duration_s": raw})])
    assert ai.decide_action(analysis(), "explore", ACTIONS)["duration_s"] == expected


def test_api_failure_returns_safe_stop(config, dummy_voice, monkeypatch):
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    ai = make_ai(config, dummy_voice, [timeout_error()] * 3)
    d = ai.decide_action(analysis(), "explore", ACTIONS)
    assert d["action"] == "stop"
    assert d["reasoning"] == "AI failure"


def test_non_json_reply_returns_stop(config, dummy_voice):
    ai = make_ai(config, dummy_voice, ["I think you should go forward!"])
    assert ai.decide_action(analysis(), "explore", ACTIONS)["action"] == "stop"


def test_obstacle_info_included_in_prompt(config, dummy_voice):
    ai = make_ai(config, dummy_voice, [json.dumps({"action": "stop", "duration_s": 0.5})])
    ai.decide_action(
        analysis(),
        "explore",
        ACTIONS,
        obstacle_info={"sensor_available": True, "distance_cm": 12.0, "has_obstacle": True, "threshold_cm": 20.0},
    )
    sent = ai.client.responses.calls[0]["input"][0]["content"][0]["text"]
    prompt = json.loads(sent)
    assert prompt["ultrasonic_sensor"]["obstacle_detected"] is True
    assert "DO NOT choose 'forward'" in prompt["ultrasonic_sensor"]["note"]


# ---------------------------------------------------------------- analyze_scene


def test_analyze_scene_parses_fields(config, dummy_voice):
    reply = json.dumps(
        {"description": "kitchen", "objects": ["table"], "hazards": ["stairs"], "suggested_actions": ["stop"]}
    )
    ai = make_ai(config, dummy_voice, [reply])
    a = ai.analyze_scene("aW1n", context="test")
    assert a.description == "kitchen"
    assert a.objects == ["table"]
    assert a.hazards == ["stairs"]
    assert a.suggested_actions == ["stop"]


def test_analyze_scene_throttles_repeat_calls(config, dummy_voice):
    reply = json.dumps({"description": "kitchen", "objects": [], "hazards": [], "suggested_actions": []})
    ai = make_ai(config, dummy_voice, [reply, reply])
    first = ai.analyze_scene("aW1n", context="test")
    second = ai.analyze_scene("aW1n", context="test")  # within MIN_SECONDS_BETWEEN_CALLS
    assert second is first
    assert len(ai.client.responses.calls) == 1


def test_analyze_scene_api_failure_suggests_stop(config, dummy_voice, monkeypatch):
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    ai = make_ai(config, dummy_voice, [timeout_error()] * 3)
    a = ai.analyze_scene("aW1n", context="test")
    assert a.description == "AI unavailable"
    assert a.suggested_actions == ["stop"]


def test_analyze_scene_failure_speaks_once(config, dummy_voice, monkeypatch):
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    ai = make_ai(config, dummy_voice, [timeout_error()] * 6)
    ai.MIN_SECONDS_BETWEEN_CALLS = 0.0
    ai.analyze_scene("aW1n", context="test")
    ai.analyze_scene("aW1n", context="test")
    assert len(dummy_voice.spoken) == 1  # cooldown suppresses the repeat


# ---------------------------------------------------------------- retry logic


def test_transient_error_is_retried(config, dummy_voice, monkeypatch):
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    ai = make_ai(config, dummy_voice, [])
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise timeout_error()
        return "ok"

    assert ai._retry_api_call(flaky, max_retries=2) == "ok"
    assert calls["n"] == 2


def test_non_transient_error_not_retried(config, dummy_voice):
    ai = make_ai(config, dummy_voice, [])
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        ai._retry_api_call(broken, max_retries=2)
    assert calls["n"] == 1


# ---------------------------------------------------------------- _safe_json


@pytest.fixture
def parser(config, dummy_voice):
    return make_ai(config, dummy_voice, [])


def test_safe_json_plain(parser):
    assert parser._safe_json('{"a": 1}') == {"a": 1}


def test_safe_json_code_fence(parser):
    assert parser._safe_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_safe_json_fence_without_language(parser):
    assert parser._safe_json('```\n{"a": 1}\n```') == {"a": 1}


def test_safe_json_invalid_returns_empty_structure(parser):
    out = parser._safe_json("not json at all")
    assert out == {"description": "", "objects": [], "hazards": [], "suggested_actions": []}


def test_safe_json_non_dict_returns_empty_structure(parser):
    out = parser._safe_json("[1, 2, 3]")
    assert out == {"description": "", "objects": [], "hazards": [], "suggested_actions": []}


# ---------------------------------------------------------------- dialogue


def dialogue_config(config):
    cfg = dict(config)
    cfg["voice_settings"] = {"enabled": True, "dialogue_enabled": True, "dialogue_min_interval_s": 0.0}
    cfg["personality_settings"] = {"max_words": 5}
    return cfg


def decision():
    return {"action": "forward", "duration_s": 1.0, "reasoning": "clear path"}


def test_dialogue_word_cap_enforced(config, dummy_voice):
    ai = make_ai(dialogue_config(config), dummy_voice, ["one two three four five six seven eight"])
    line = ai.generate_dialogue(
        mode="explore", target=None, analysis=analysis(), decision=decision(), executed_action="forward"
    )
    assert len(line.split()) <= 5
    assert line.endswith(".")


def test_dialogue_disabled_returns_empty(config, dummy_voice):
    cfg = dialogue_config(config)
    cfg["voice_settings"]["dialogue_enabled"] = False
    ai = make_ai(cfg, dummy_voice, ["hello"])
    assert (
        ai.generate_dialogue(
            mode="explore", target=None, analysis=analysis(), decision=decision(), executed_action="forward"
        )
        == ""
    )


def test_dialogue_api_failure_is_non_fatal(config, dummy_voice):
    ai = make_ai(dialogue_config(config), dummy_voice, [timeout_error()])
    assert (
        ai.generate_dialogue(
            mode="explore", target=None, analysis=analysis(), decision=decision(), executed_action="forward"
        )
        == ""
    )


def test_dialogue_dedupes_exact_repeats(config, dummy_voice):
    ai = make_ai(dialogue_config(config), dummy_voice, ["same line", "same line"])
    first = ai.generate_dialogue(
        mode="explore", target=None, analysis=analysis(), decision=decision(), executed_action="forward"
    )
    second = ai.generate_dialogue(
        mode="explore", target=None, analysis=analysis(), decision=decision(), executed_action="forward"
    )
    assert first == "same line"
    assert second == ""
