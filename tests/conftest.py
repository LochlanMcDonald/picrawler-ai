import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable regardless of pytest invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Never let tests hit the real API even if a key leaks into the environment
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")


@pytest.fixture
def config() -> dict:
    """Minimal config with voice disabled and no file output."""
    return {
        "openai_api_key": "test-key-not-real",
        "ai_settings": {
            "model": "test-model",
            "max_output_tokens": 100,
            "temperature": 0.0,
            "request_timeout_s": 5,
            "min_seconds_between_calls": 3.0,
        },
        "robot_settings": {
            "movement_speed": 50,
            "turn_speed": 45,
            "obstacle_distance_threshold_cm": 20.0,
            "dry_run_if_no_hardware": True,
        },
        "voice_settings": {"enabled": False},
        "logging_settings": {"save_images": False, "save_decisions": False},
    }


class DummyVoice:
    """Records say() calls without any I/O."""

    def __init__(self):
        self.spoken = []

    def say(self, text, *, level="normal", force=False):
        self.spoken.append((text, level, force))


@pytest.fixture
def dummy_voice() -> DummyVoice:
    return DummyVoice()


class FakeClock:
    """Deterministic, manually-advanced replacement for time.time/time.sleep."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
