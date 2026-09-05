import json
from pathlib import Path

import pytest

from core.config_loader import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.json"))


def test_loads_valid_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"openai_api_key": "k", "ai_settings": {"model": "m"}}))
    cfg = load_config(str(p))
    assert cfg["openai_api_key"] == "k"
    assert cfg["ai_settings"]["model"] == "m"


def test_example_config_is_valid_and_complete():
    cfg = load_config(str(REPO_ROOT / "config" / "config.example.json"))
    for section in (
        "ai_settings",
        "robot_settings",
        "camera_settings",
        "voice_settings",
        "behavior_settings",
        "logging_settings",
    ):
        assert section in cfg, f"example config missing '{section}'"
    # Placeholder key must never be a real key
    assert cfg["openai_api_key"] == "your-api-key-here"
