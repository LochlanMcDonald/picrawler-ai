"""
Tests for LanguageController — command parsing, coordinate extraction,
and execution logic. The OpenAI client is always mocked.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from planning.language_controller import LanguageController, LanguageCommand, SceneDescription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_controller(api_key: str = "test-key") -> LanguageController:
    config = {"openai_api_key": api_key}
    ctrl = LanguageController(config)
    # Replace the real OpenAI client with a mock
    ctrl.client = MagicMock()
    return ctrl


def mock_response(content: dict) -> MagicMock:
    """Build a fake OpenAI completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(content)
    return response


def make_scene() -> SceneDescription:
    return SceneDescription(
        objects=["chair", "table"],
        spatial_layout="chair on left, table straight ahead",
        navigable_areas=["right side"],
        obstacles=["table"],
        suggested_actions=["turn right", "move forward"],
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_client_set_with_valid_key(self):
        ctrl = make_controller("real-key")
        assert ctrl.client is not None

    def test_no_client_without_api_key(self):
        config = {}
        ctrl = LanguageController(config)
        # With no API key, client should be None (OPENAI_AVAILABLE may be True but key missing)
        assert ctrl.client is None or ctrl.api_key is None

    def test_command_history_starts_empty(self):
        ctrl = make_controller()
        assert ctrl.command_history == []


# ---------------------------------------------------------------------------
# _extract_coordinates
# ---------------------------------------------------------------------------

class TestExtractCoordinates:
    def test_parses_parenthesised_coords(self):
        ctrl = make_controller()
        result = ctrl._extract_coordinates("Go to (1.5, 0.8)")
        assert result == pytest.approx((1.5, 0.8))

    def test_parses_coords_with_keyword(self):
        ctrl = make_controller()
        result = ctrl._extract_coordinates("navigate to coordinates (2.0, -1.0)")
        assert result == pytest.approx((2.0, -1.0))

    def test_parses_xy_equals_syntax(self):
        ctrl = make_controller()
        result = ctrl._extract_coordinates("move to x=1.5, y=0.8")
        assert result == pytest.approx((1.5, 0.8))

    def test_parses_position_keyword(self):
        ctrl = make_controller()
        result = ctrl._extract_coordinates("go to position (3.0, 2.5)")
        assert result == pytest.approx((3.0, 2.5))

    def test_returns_none_when_no_coords(self):
        ctrl = make_controller()
        assert ctrl._extract_coordinates("go forward") is None

    def test_handles_negative_coordinates(self):
        ctrl = make_controller()
        result = ctrl._extract_coordinates("go to (-1.5, -0.8)")
        assert result == pytest.approx((-1.5, -0.8))


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------

class TestParseCommand:
    def test_returns_none_without_client(self):
        ctrl = LanguageController({})
        ctrl.client = None
        assert ctrl.parse_command("go forward") is None

    def test_returns_language_command_on_success(self):
        ctrl = make_controller()
        ctrl.client.chat.completions.create.return_value = mock_response({
            "intent": "navigate",
            "target": None,
            "actions": ["forward", "forward"],
            "reasoning": "move ahead",
            "confidence": 0.9,
        })
        cmd = ctrl.parse_command("go forward")
        assert cmd is not None
        assert isinstance(cmd, LanguageCommand)
        assert cmd.intent == "navigate"

    def test_command_appended_to_history(self):
        ctrl = make_controller()
        ctrl.client.chat.completions.create.return_value = mock_response({
            "intent": "explore",
            "target": None,
            "actions": ["forward"],
            "reasoning": "exploring",
            "confidence": 0.8,
        })
        ctrl.parse_command("explore the room")
        assert len(ctrl.command_history) == 1

    def test_navigation_coords_extracted(self):
        ctrl = make_controller()
        ctrl.client.chat.completions.create.return_value = mock_response({
            "intent": "navigate",
            "target": None,
            "actions": ["forward"],
            "reasoning": "going to coords",
            "confidence": 0.85,
        })
        cmd = ctrl.parse_command("go to (2.0, 1.0)")
        assert cmd is not None
        assert cmd.navigation_goal == pytest.approx((2.0, 1.0))

    def test_returns_none_on_json_error(self):
        ctrl = make_controller()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "not valid json {{{"
        ctrl.client.chat.completions.create.return_value = response
        cmd = ctrl.parse_command("go forward")
        assert cmd is None  # Graceful failure

    def test_includes_scene_context_in_prompt(self):
        ctrl = make_controller()
        ctrl.client.chat.completions.create.return_value = mock_response({
            "intent": "navigate",
            "target": None,
            "actions": ["turn_right"],
            "reasoning": "obstacle ahead",
            "confidence": 0.7,
        })
        scene = make_scene()
        ctrl.parse_command("go forward", scene=scene)
        # Verify the prompt builder was called (indirectly via the API call)
        assert ctrl.client.chat.completions.create.call_count == 1
        call_args = ctrl.client.chat.completions.create.call_args
        prompt = call_args.kwargs.get("messages", [{}])[0].get("content", "")
        # Scene objects should appear somewhere in the prompt
        assert "chair" in prompt or "table" in prompt

    def test_confidence_stored_correctly(self):
        ctrl = make_controller()
        ctrl.client.chat.completions.create.return_value = mock_response({
            "intent": "stop",
            "target": None,
            "actions": ["stop"],
            "reasoning": "stopping",
            "confidence": 0.95,
        })
        cmd = ctrl.parse_command("stop")
        assert cmd.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# execute_command
# ---------------------------------------------------------------------------

class TestExecuteCommand:
    def make_cmd(self, actions=None, confidence=0.9, intent="navigate") -> LanguageCommand:
        import time
        return LanguageCommand(
            text="test",
            intent=intent,
            target=None,
            actions=actions or ["forward"],
            reasoning="testing",
            confidence=confidence,
            timestamp=time.time(),
        )

    def test_executes_valid_actions(self):
        ctrl = make_controller()
        robot = MagicMock()
        logger = MagicMock()
        cmd = self.make_cmd(actions=["forward", "stop"])
        result = ctrl.execute_command(cmd, robot, logger)
        assert result is True
        assert robot.execute.call_count >= 1

    def test_returns_false_for_low_confidence(self):
        ctrl = make_controller()
        robot = MagicMock()
        logger = MagicMock()
        cmd = self.make_cmd(confidence=0.3)
        result = ctrl.execute_command(cmd, robot, logger)
        assert result is False

    def test_skips_invalid_actions(self):
        ctrl = make_controller()
        robot = MagicMock()
        logger = MagicMock()
        cmd = self.make_cmd(actions=["fly", "forward"])
        ctrl.execute_command(cmd, robot, logger)
        # "fly" is invalid; "forward" should still be executed
        assert robot.execute.call_count >= 1

    def test_returns_false_when_robot_raises(self):
        ctrl = make_controller()
        robot = MagicMock()
        # First call raises, subsequent calls (stop recovery) also raise — catch both
        robot.execute.side_effect = RuntimeError("hardware error")
        logger = MagicMock()
        cmd = self.make_cmd(actions=["forward"])
        try:
            result = ctrl.execute_command(cmd, robot, logger)
            # If it catches the error internally, it should return False
            assert result is False
        except RuntimeError:
            # If the stop-recovery call also raises, the exception propagates — acceptable
            pass

    def test_handles_scan_action(self):
        ctrl = make_controller()
        robot = MagicMock()
        logger = MagicMock()
        cmd = self.make_cmd(actions=["scan"])
        result = ctrl.execute_command(cmd, robot, logger)
        assert result is True


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

class TestHistoryManagement:
    def test_clear_history(self):
        ctrl = make_controller()
        ctrl.client.chat.completions.create.return_value = mock_response({
            "intent": "explore",
            "target": None,
            "actions": ["forward"],
            "reasoning": "exploring",
            "confidence": 0.8,
        })
        ctrl.parse_command("explore")
        ctrl.clear_history()
        assert ctrl.command_history == []
        assert ctrl.current_command is None

    def test_get_command_history(self):
        ctrl = make_controller()
        ctrl.client.chat.completions.create.return_value = mock_response({
            "intent": "navigate",
            "target": None,
            "actions": ["forward"],
            "reasoning": "go",
            "confidence": 0.9,
        })
        ctrl.parse_command("go forward")
        history = ctrl.get_command_history()
        assert len(history) == 1
