"""Coverage for the small pieces: logger setup, behavior contexts, CLI args."""

import sys

import pytest

import main as main_module
from behaviors.avoidance import AvoidanceBehavior
from behaviors.following import FollowingBehavior
from behaviors.object_detection import ObjectDetectionBehavior
from core.logger import setup_logging

from tests.test_base_behavior import StubAI, StubRobot


# ---------------------------------------------------------------- logger


def test_setup_logging_creates_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging("INFO", log_dir=str(log_dir))
    assert log_dir.is_dir()
    assert (log_dir / "operation.log").exists()


def test_setup_logging_invalid_level_does_not_crash(tmp_path):
    setup_logging("NOT_A_LEVEL", log_dir=str(tmp_path / "logs"))


# ---------------------------------------------------------------- contexts


def make(cls, config, dummy_voice, target=None):
    return cls(
        config=config,
        robot=StubRobot(),
        camera=None,
        ai=StubAI(),
        duration_minutes=0.1,
        target=target,
        voice=dummy_voice,
    )


def test_following_context_defaults_to_person(config, dummy_voice):
    b = make(FollowingBehavior, config, dummy_voice)
    assert "person" in b.context()
    assert b.name == "follow"


def test_following_context_uses_target(config, dummy_voice):
    b = make(FollowingBehavior, config, dummy_voice, target="dog")
    assert "dog" in b.context()


def test_detection_context_mentions_target(config, dummy_voice):
    b = make(ObjectDetectionBehavior, config, dummy_voice, target="red ball")
    assert "red ball" in b.context()
    assert b.name == "detect"


def test_avoidance_context_mentions_avoid_target(config, dummy_voice):
    b = make(AvoidanceBehavior, config, dummy_voice, target="cats")
    assert "cats" in b.context()
    assert b.name == "avoid"


# ---------------------------------------------------------------- CLI


def test_parse_args_happy_path(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--mode", "explore", "--duration", "2", "--target", "ball", "--verbose"]
    )
    args = main_module.parse_args()
    assert args.mode == "explore"
    assert args.duration == 2.0
    assert args.target == "ball"
    assert args.verbose is True


def test_parse_args_requires_mode(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit):
        main_module.parse_args()


def test_parse_args_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "attack"])
    with pytest.raises(SystemExit):
        main_module.parse_args()


def test_main_errors_cleanly_without_config(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "explore", "--config", str(tmp_path / "none.json")])
    with pytest.raises(FileNotFoundError):
        main_module.main()
