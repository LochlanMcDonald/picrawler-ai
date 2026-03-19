"""
Shared pytest fixtures for picrawler-ai-4 tests.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / hardware-only modules before project code is imported
# ---------------------------------------------------------------------------

def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# pyttsx3
_pyttsx3 = _stub("pyttsx3")
_pyttsx3.init = MagicMock(return_value=MagicMock())

# PyTorch / torchvision / timm (only needed for depth_estimator, not our tests)
for _mod in ("torch", "torchvision", "torchvision.transforms", "timm"):
    _stub(_mod)

# Hardware stubs
for _mod in ("picrawlerhat", "robot_hat", "pihat"):
    _stub(_mod)

# openai stub
_openai = _stub("openai")
_openai.OpenAI = MagicMock


# ---------------------------------------------------------------------------
# MockRobot
# ---------------------------------------------------------------------------

class MockRobot:
    """Robot stub that records calls and returns configurable sensor values."""

    def __init__(self, distance_cm: float = 50.0):
        self._distance_cm = distance_cm
        self.calls: list = []

    def execute(self, action: str, duration: float = 1.0) -> None:
        self.calls.append((action, duration))

    def get_distance(self) -> float | None:
        return self._distance_cm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_robot() -> MockRobot:
    return MockRobot()


@pytest.fixture
def close_robot() -> MockRobot:
    """Robot with an obstacle very close (10 cm)."""
    return MockRobot(distance_cm=10.0)


@pytest.fixture
def world_model():
    from core.world_model import WorldModel
    return WorldModel(obstacle_threshold_cm=20.0)


@pytest.fixture
def spatial_memory():
    from core.spatial_memory import SpatialMemory
    return SpatialMemory(history_size=50)


@pytest.fixture
def behavior_context(world_model, spatial_memory, mock_robot):
    from planning.behavior_tree import BehaviorContext
    return BehaviorContext(
        world_model=world_model,
        memory=spatial_memory,
        robot_controller=mock_robot,
    )
