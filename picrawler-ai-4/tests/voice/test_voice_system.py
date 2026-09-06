"""Tests for voice/voice_system.py — construction must never be fatal."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import voice.voice_system as vs_mod
from voice.voice_system import VoiceSystem


def _cfg(enabled: bool = True, key=None) -> dict:
    cfg = {"voice_settings": {"enabled": enabled}}
    if key is not None:
        cfg["openai_api_key"] = key
    return cfg


class TestMissingCredentials:
    def test_missing_key_disables_voice_instead_of_raising(self, monkeypatch, tmp_path):
        """Regression: SLAM mode crashed at startup because VoiceSystem built an
        OpenAI client unconditionally and the SDK raised on a missing key."""
        def _raise(**_kw):
            raise RuntimeError("Missing credentials")
        monkeypatch.setattr(vs_mod, "OpenAI", _raise)

        v = VoiceSystem(_cfg(key="your-api-key-here"), cache_dir=str(tmp_path))
        assert v.client is None
        assert v.settings.enabled is False

    def test_say_is_noop_when_disabled_by_missing_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vs_mod, "OpenAI", MagicMock(side_effect=RuntimeError("no key")))
        v = VoiceSystem(_cfg(), cache_dir=str(tmp_path))
        v.say("hello")   # must not raise

    def test_explicitly_disabled_never_builds_client(self, monkeypatch, tmp_path):
        ctor = MagicMock()
        monkeypatch.setattr(vs_mod, "OpenAI", ctor)
        v = VoiceSystem(_cfg(enabled=False), cache_dir=str(tmp_path))
        ctor.assert_not_called()
        assert v.client is None


class TestWithCredentials:
    def test_client_built_when_key_present(self, monkeypatch, tmp_path):
        ctor = MagicMock(return_value="client")
        monkeypatch.setattr(vs_mod, "OpenAI", ctor)
        v = VoiceSystem(_cfg(key="sk-test"), cache_dir=str(tmp_path))
        ctor.assert_called_once_with(api_key="sk-test")
        assert v.client == "client"
        assert v.settings.enabled is True

    def test_placeholder_key_passed_as_none(self, monkeypatch, tmp_path):
        ctor = MagicMock(return_value="client")
        monkeypatch.setattr(vs_mod, "OpenAI", ctor)
        VoiceSystem(_cfg(key="your-api-key-here"), cache_dir=str(tmp_path))
        ctor.assert_called_once_with(api_key=None)
