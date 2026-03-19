"""
Tests for config_loader.load_config.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from core.config_loader import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_temp_json(data: dict) -> str:
    """Write dict to a temp JSON file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_config(self):
        config = {"openai_api_key": "abc", "camera_settings": {"fps": 10}}
        path = write_temp_json(config)
        try:
            result = load_config(path)
            assert result["openai_api_key"] == "abc"
            assert result["camera_settings"]["fps"] == 10
        finally:
            os.unlink(path)

    def test_returns_dict(self):
        path = write_temp_json({"key": "value"})
        try:
            result = load_config(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_loads_nested_keys(self):
        data = {
            "voice_settings": {
                "enabled": True,
                "narration_min_interval_s": 4.0,
            }
        }
        path = write_temp_json(data)
        try:
            result = load_config(path)
            assert result["voice_settings"]["enabled"] is True
        finally:
            os.unlink(path)

    def test_empty_config_returns_empty_dict(self):
        path = write_temp_json({})
        try:
            result = load_config(path)
            assert result == {}
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_error_message_contains_path(self):
        path = "/nonexistent/path/config.json"
        with pytest.raises(FileNotFoundError, match=path):
            load_config(path)


# ---------------------------------------------------------------------------
# Malformed JSON
# ---------------------------------------------------------------------------

class TestMalformedJson:
    def test_raises_on_invalid_json(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("{not valid json {{")
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(path)
        finally:
            os.unlink(path)

    def test_raises_on_empty_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)  # Leave file empty
        try:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                load_config(path)
        finally:
            os.unlink(path)
