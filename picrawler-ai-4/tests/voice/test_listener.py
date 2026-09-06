"""Tests for voice/listener.py — keyboard and microphone answer capture."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

import voice.listener as listener_mod
from voice.listener import UserListener


def _cfg(**guided) -> dict:
    return {"guided_settings": guided}


class TestKeyboard:
    def test_default_mode_is_keyboard(self):
        assert UserListener({}).mode == "keyboard"

    def test_returns_stripped_text(self):
        l = UserListener({}, input_fn=lambda p: "  yes \n")
        assert l.ask("? ", 1.0) == "yes"

    def test_empty_is_none(self):
        l = UserListener({}, input_fn=lambda p: "   ")
        assert l.ask("? ", 1.0) is None

    def test_eof_is_none(self):
        def boom(p):
            raise EOFError
        l = UserListener({}, input_fn=boom)
        assert l.ask("? ", 1.0) is None

    def test_prompt_passed_through(self):
        seen = []
        l = UserListener({}, input_fn=lambda p: seen.append(p) or "ok")
        l.ask("Your answer: ", 1.0)
        assert seen == ["Your answer: "]

    def test_stdin_timeout_returns_none(self, monkeypatch):
        monkeypatch.setattr(listener_mod.select, "select", lambda *a, **k: ([], [], []))
        l = UserListener({})
        assert l.ask("> ", 0.01) is None


class TestMicrophoneFallbacks:
    def test_no_arecord_falls_back_to_keyboard(self, monkeypatch):
        monkeypatch.setattr(listener_mod.shutil, "which", lambda n: None)
        l = UserListener(_cfg(listen="microphone"), openai_client=MagicMock())
        assert l.mode == "keyboard"

    def test_no_client_falls_back_to_keyboard(self, monkeypatch):
        monkeypatch.setattr(listener_mod.shutil, "which", lambda n: "/usr/bin/arecord")
        l = UserListener(_cfg(listen="microphone"), openai_client=None)
        assert l.mode == "keyboard"


class TestMicrophone:
    def _mic(self, monkeypatch, transcript="turn left", record_ok=True):
        monkeypatch.setattr(listener_mod.shutil, "which", lambda n: "/usr/bin/arecord")
        runs = []
        def fake_run(cmd, **kw):
            runs.append(cmd)
            if not record_ok:
                raise subprocess.CalledProcessError(1, cmd)
            with open(cmd[-1], "wb") as f:
                f.write(b"RIFF")
            return MagicMock(returncode=0)
        monkeypatch.setattr(listener_mod.subprocess, "run", fake_run)
        client = MagicMock()
        client.audio.transcriptions.create.return_value = MagicMock(text=transcript)
        l = UserListener(_cfg(listen="microphone", listen_seconds=3, transcribe_model="whisper-1"),
                         openai_client=client, input_fn=lambda p: "typed fallback")
        return l, runs, client

    def test_records_and_transcribes(self, monkeypatch):
        l, runs, client = self._mic(monkeypatch)
        assert l.mode == "microphone"
        assert l.ask("> ", 5.0) == "turn left"
        assert runs[0][0] == "arecord"
        assert "3" in runs[0]                      # listen_seconds
        assert client.audio.transcriptions.create.call_args.kwargs["model"] == "whisper-1"

    def test_empty_transcript_falls_back_to_keyboard(self, monkeypatch):
        l, _, _ = self._mic(monkeypatch, transcript="")
        assert l.ask("> ", 5.0) == "typed fallback"

    def test_recording_failure_falls_back_to_keyboard(self, monkeypatch):
        l, _, _ = self._mic(monkeypatch, record_ok=False)
        assert l.ask("> ", 5.0) == "typed fallback"


class TestAutoMode:
    def test_auto_picks_microphone_when_device_and_client(self, monkeypatch):
        monkeypatch.setattr(listener_mod, "has_microphone", lambda: True)
        l = UserListener(_cfg(listen="auto"), openai_client=MagicMock())
        assert l.mode == "microphone"

    def test_auto_falls_back_without_device(self, monkeypatch):
        monkeypatch.setattr(listener_mod, "has_microphone", lambda: False)
        l = UserListener(_cfg(listen="auto"), openai_client=MagicMock())
        assert l.mode == "keyboard"

    def test_auto_falls_back_without_client(self, monkeypatch):
        monkeypatch.setattr(listener_mod, "has_microphone", lambda: True)
        l = UserListener(_cfg(listen="auto"), openai_client=None)
        assert l.mode == "keyboard"

    def test_default_is_auto(self, monkeypatch):
        monkeypatch.setattr(listener_mod, "has_microphone", lambda: False)
        assert UserListener({}).mode == "keyboard"

    def test_mode_override_beats_config(self, monkeypatch):
        monkeypatch.setattr(listener_mod, "has_microphone", lambda: True)
        l = UserListener(_cfg(listen="auto"), openai_client=MagicMock(), mode="keyboard")
        assert l.mode == "keyboard"

    def test_unknown_mode_is_keyboard(self):
        assert UserListener(_cfg(listen="telepathy")).mode == "keyboard"


class TestHasMicrophone:
    def test_no_arecord(self, monkeypatch):
        monkeypatch.setattr(listener_mod.shutil, "which", lambda n: None)
        assert listener_mod.has_microphone() is False

    def test_lists_card(self, monkeypatch):
        monkeypatch.setattr(listener_mod.shutil, "which", lambda n: "/usr/bin/arecord")
        monkeypatch.setattr(listener_mod.subprocess, "run",
                            lambda *a, **k: MagicMock(stdout="**** List of CAPTURE Hardware Devices ****\ncard 1: Device [USB Audio]\n"))
        assert listener_mod.has_microphone() is True

    def test_no_cards(self, monkeypatch):
        monkeypatch.setattr(listener_mod.shutil, "which", lambda n: "/usr/bin/arecord")
        monkeypatch.setattr(listener_mod.subprocess, "run",
                            lambda *a, **k: MagicMock(stdout="**** List of CAPTURE Hardware Devices ****\n"))
        assert listener_mod.has_microphone() is False


class TestPoll:
    def test_poll_returns_none_with_injected_input(self):
        assert UserListener({}, input_fn=lambda p: "x").poll() is None

    def test_poll_returns_none_when_nothing_typed(self, monkeypatch):
        monkeypatch.setattr(listener_mod.select, "select", lambda *a, **k: ([], [], []))
        assert UserListener(_cfg(listen="keyboard")).poll() is None

    def test_poll_returns_typed_line(self, monkeypatch):
        monkeypatch.setattr(listener_mod.select, "select", lambda *a, **k: ([listener_mod.sys.stdin], [], []))
        monkeypatch.setattr(listener_mod.sys.stdin, "readline", lambda: "quit\n")
        assert UserListener(_cfg(listen="keyboard")).poll() == "quit"
