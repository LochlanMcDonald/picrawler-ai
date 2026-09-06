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
