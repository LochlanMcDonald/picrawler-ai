"""Tests for TTS gating, caching and path-safety (no audio or network I/O)."""

from pathlib import Path

import pytest

import voice.voice_system as vv
from voice.voice_system import VoiceSystem


@pytest.fixture
def voice(config, tmp_path):
    cfg = dict(config)
    cfg["voice_settings"] = {"enabled": True, "cooldown_s": 2.5, "dedupe_window_s": 15.0, "verbosity": "normal"}
    return VoiceSystem(cfg, cache_dir=str(tmp_path / "tts_cache"))


@pytest.fixture
def played(voice, monkeypatch):
    """Stub out audio synthesis + playback, recording what would be spoken."""
    calls = []

    def fake_get(text):
        return Path(f"/fake/{len(calls)}.mp3")

    monkeypatch.setattr(voice, "_get_or_create_audio", fake_get)
    monkeypatch.setattr(voice, "_play", lambda p: calls.append(p))
    return calls


def test_disabled_voice_says_nothing(config, tmp_path, monkeypatch):
    cfg = dict(config)
    cfg["voice_settings"] = {"enabled": False}
    v = VoiceSystem(cfg, cache_dir=str(tmp_path))
    monkeypatch.setattr(v, "_play", lambda p: pytest.fail("should not play"))
    v.say("hello")


def test_say_plays_once_then_cooldown(voice, played):
    voice.say("hello")
    voice.say("world")  # within cooldown
    assert len(played) == 1


def test_force_bypasses_cooldown(voice, played):
    voice.say("hello")
    voice.say("world", force=True)
    assert len(played) == 2


def test_dedupe_suppresses_same_text(voice, played, monkeypatch):
    voice.say("hello")
    # Move past the cooldown but stay inside the dedupe window
    voice._last_spoken_at -= 10.0
    voice._last_text_at -= 10.0
    voice.say("hello")
    assert len(played) == 1


def test_empty_text_ignored(voice, played):
    voice.say("   ")
    assert played == []


def test_long_text_clipped(voice, played, monkeypatch):
    seen = {}
    monkeypatch.setattr(voice, "_get_or_create_audio", lambda t: seen.setdefault("text", t) or Path("/fake/a.mp3"))
    voice.say("x" * 1000)
    assert len(seen["text"]) <= voice.settings.max_chars + 1  # +1 for the ellipsis


def test_tts_failure_is_non_fatal(voice, monkeypatch):
    def boom(text):
        raise RuntimeError("api down")

    monkeypatch.setattr(voice, "_get_or_create_audio", boom)
    voice.say("hello")  # must not raise


# ---------------------------------------------------------------- verbosity


@pytest.mark.parametrize(
    "verbosity,level,allowed",
    [
        ("low", "low", True),
        ("low", "normal", False),
        ("normal", "normal", True),
        ("normal", "high", False),
        ("normal", "thought", False),
        ("high", "thought", True),
    ],
)
def test_verbosity_gating(config, tmp_path, verbosity, level, allowed):
    cfg = dict(config)
    cfg["voice_settings"] = {"enabled": True, "verbosity": verbosity}
    v = VoiceSystem(cfg, cache_dir=str(tmp_path))
    assert v._verbosity_allows(level) is allowed


# ---------------------------------------------------------------- cache keys


def test_hash_key_stable(voice):
    assert voice._hash_key("hello") == voice._hash_key("hello")


def test_hash_key_varies_with_voice_settings(voice):
    a = voice._hash_key("hello")
    voice.settings.voice = "other-voice"
    assert voice._hash_key("hello") != a


# ---------------------------------------------------------------- path safety


def test_path_inside_cache_accepted(voice):
    assert voice._is_in_cache_dir(voice.cache_dir / "abc.mp3") is True


def test_path_traversal_rejected(voice):
    assert voice._is_in_cache_dir(voice.cache_dir / ".." / "evil.mp3") is False


def test_sibling_prefix_dir_rejected(voice, tmp_path):
    # Regression: naive startswith() accepted '<cache>-evil/...'
    evil = Path(str(voice.cache_dir) + "-evil") / "x.mp3"
    assert voice._is_in_cache_dir(evil) is False


def test_play_rejects_outside_cache(voice):
    with pytest.raises(ValueError):
        voice._play(Path("/etc/passwd"))


def test_play_requires_wav_for_aplay(voice):
    voice._player = ["aplay", "-q"]
    with pytest.raises(RuntimeError):
        voice._play(voice.cache_dir / "sound.mp3")
