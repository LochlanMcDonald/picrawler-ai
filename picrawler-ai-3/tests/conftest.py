"""
Shared pytest fixtures for picrawler-ai-3 tests.

All fixtures live here so individual test modules stay concise.
Heavy imports (pyttsx3, openai) are stubbed out at the module level
so tests never need real hardware or API keys.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out hardware / external modules before any project code is imported
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# pyttsx3
_pyttsx3 = _make_stub("pyttsx3")
_pyttsx3.init = MagicMock(return_value=MagicMock())

# picrawlerhat / pihat (hardware)
for _mod in ("picrawlerhat", "robot_hat", "pihat"):
    _make_stub(_mod)

# openai stub — must expose every name that vision_ai.py imports
_openai = _make_stub("openai")
_openai.OpenAI = MagicMock
_openai.APIError = type("APIError", (Exception,), {})
_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
_openai.RateLimitError = type("RateLimitError", (Exception,), {})


# ---------------------------------------------------------------------------
# Minimal config dict shared across tests
# ---------------------------------------------------------------------------

MINIMAL_CONFIG: dict = {
    "openai_api_key": "test-key",
    "camera_settings": {"capture_interval_s": 2.5},
    "logging_settings": {"save_images": False, "save_decisions": False},
    "voice_settings": {
        "enabled": False,
        "narration_enabled": False,
        "dialogue_enabled": False,
        "thoughts_enabled": False,
    },
    "behavior_settings": {
        "anti_loop": {
            "anti_loop_history": 8,
            "anti_loop_repeat_threshold": 3,
            "anti_loop_oscillation_window": 4,
            "scene_history": 10,
            "scene_repeat_threshold": 4,
            "ban_seconds": 8.0,
            "escape_cooldown_s": 0.0,   # disable cooldown in tests
            "escape_strikes_reset": 6,
            "max_obstacle_overrides_before_escape": 3,
        }
    },
}


# ---------------------------------------------------------------------------
# MockRobot
# ---------------------------------------------------------------------------

class MockRobot:
    """Minimal robot stub that records calls and returns configurable values."""

    def __init__(self, distance_cm: float = 50.0, has_obstacle: bool = False):
        self._distance_cm = distance_cm
        self._has_obstacle = has_obstacle
        self.calls: list = []

    def execute(self, action: str, duration: float = 1.0) -> None:
        self.calls.append((action, duration))

    def get_distance(self) -> float:
        return self._distance_cm

    def get_obstacle_info(self) -> dict:
        return {
            "has_obstacle": self._has_obstacle,
            "distance_cm": self._distance_cm,
            "threshold_cm": 20,
            "sensor_available": True,
        }


# ---------------------------------------------------------------------------
# MockCamera / MockAI
# ---------------------------------------------------------------------------

class MockCamera:
    def capture(self, save: bool = False):
        return (MagicMock(), "fakeb64==", "/tmp/fake.jpg")


class MockAnalysis:
    def __init__(self, description: str = "clear path", objects=None, hazards=None):
        self.description = description
        self.objects = objects or []
        self.hazards = hazards or []
        self.suggested_actions = ["forward"]
        self.processing_time_s = 0.01


class MockAI:
    def __init__(self, action: str = "forward", duration_s: float = 0.5):
        self._action = action
        self._duration_s = duration_s

    def analyze_scene(self, b64: str, context: str = "") -> MockAnalysis:
        return MockAnalysis()

    def decide_action(self, **kwargs) -> dict:
        return {"action": self._action, "duration_s": self._duration_s, "reasoning": "test"}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config() -> dict:
    return dict(MINIMAL_CONFIG)


@pytest.fixture
def mock_robot() -> MockRobot:
    return MockRobot()


@pytest.fixture
def obstacle_robot() -> MockRobot:
    return MockRobot(distance_cm=10.0, has_obstacle=True)


@pytest.fixture
def mock_camera() -> MockCamera:
    return MockCamera()


@pytest.fixture
def mock_ai() -> MockAI:
    return MockAI()


@pytest.fixture
def base_behavior(minimal_config, mock_robot, mock_camera, mock_ai):
    """Return a fully constructed BaseBehavior with all deps mocked."""
    # Import here so stubs are already registered
    from behaviors.base_behavior import BaseBehavior

    class _Concrete(BaseBehavior):
        name = "test"

    return _Concrete(
        config=minimal_config,
        robot=mock_robot,
        camera=mock_camera,
        ai=mock_ai,
        duration_minutes=0.01,
    )
