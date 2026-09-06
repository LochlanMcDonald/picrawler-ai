"""
User input for conversational modes — keyboard or microphone.

The robot asks a question out loud (VoiceSystem) and then waits here for the
answer.  Backends:

  keyboard    — type the answer in the terminal (always works)
  microphone  — record with `arecord` and transcribe with OpenAI; falls back
                to the keyboard if recording or transcription fails
  auto        — microphone if `arecord` finds a capture device and an OpenAI
                client is available, otherwise keyboard  (default)

Configure under "guided_settings" in config.json:

    "listen": "auto" | "keyboard" | "microphone",
    "listen_seconds": 4,            # microphone recording length
    "transcribe_model": "whisper-1"
"""
from __future__ import annotations

import logging
import os
import select
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional


def has_microphone() -> bool:
    """True if `arecord -l` lists at least one capture device."""
    if shutil.which("arecord") is None:
        return False
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "card" in out.lower()


class UserListener:
    def __init__(self,
                 config: Optional[dict] = None,
                 openai_client=None,
                 input_fn: Optional[Callable[[str], str]] = None,
                 logger: Optional[logging.Logger] = None,
                 mode: Optional[str] = None):
        cfg = (config or {}).get("guided_settings", {}) if isinstance(config, dict) else {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.mode = str(mode or cfg.get("listen", "auto")).lower()
        self.listen_seconds = float(cfg.get("listen_seconds", 4.0))
        self.transcribe_model = str(cfg.get("transcribe_model", "whisper-1"))
        self.client = openai_client
        self._input_fn = input_fn   # injectable for tests; None → stdin with timeout

        if self.mode == "auto":
            if self.client is not None and has_microphone():
                self.mode = "microphone"
                self.logger.info("Microphone detected — answers by voice (keyboard still works)")
            else:
                self.mode = "keyboard"
        elif self.mode == "microphone":
            if shutil.which("arecord") is None:
                self.logger.warning("listen=microphone but `arecord` not found — using keyboard")
                self.mode = "keyboard"
            elif self.client is None:
                self.logger.warning("listen=microphone but no OpenAI client — using keyboard")
                self.mode = "keyboard"
        elif self.mode != "keyboard":
            self.logger.warning(f"Unknown listen mode {self.mode!r} — using keyboard")
            self.mode = "keyboard"

    # ------------------------------------------------------------------

    def ask(self, prompt: str = "> ", timeout_s: float = 20.0) -> Optional[str]:
        """Wait for an answer. Returns stripped text, or None on timeout/empty."""
        if self.mode == "microphone":
            text = self._listen_microphone()
            if text:
                return text
            self.logger.info("Microphone gave nothing — falling back to keyboard")
        return self._listen_keyboard(prompt, timeout_s)

    def poll(self) -> Optional[str]:
        """Non-blocking: return a line the user has already typed, else None.

        Used by autonomous guided modes so you can still type `stop`, `quit`
        or an instruction while the robot is acting on its own. Never records
        from the microphone.
        """
        if self._input_fn is not None:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (OSError, ValueError):
            return None
        if not ready:
            return None
        line = sys.stdin.readline()
        text = (line or "").strip()
        return text or None

    # ------------------------------------------------------------------

    def _listen_keyboard(self, prompt: str, timeout_s: float) -> Optional[str]:
        if self._input_fn is not None:
            try:
                text = self._input_fn(prompt)
            except (EOFError, KeyboardInterrupt):
                return None
            text = (text or "").strip()
            return text or None

        # Real stdin with a timeout (Unix). Windows has no select() on stdin.
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        except (OSError, ValueError):
            ready = [sys.stdin]
        if not ready:
            sys.stdout.write("\n")
            return None
        line = sys.stdin.readline()
        if not line:          # EOF
            return None
        text = line.strip()
        return text or None

    def _listen_microphone(self) -> Optional[str]:
        tmp = tempfile.NamedTemporaryFile(prefix="listen_", suffix=".wav", delete=False)
        tmp.close()
        try:
            self.logger.info(f"Listening for {self.listen_seconds:.0f}s…")
            subprocess.run(
                ["arecord", "-q", "-d", str(int(self.listen_seconds)),
                 "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", tmp.name],
                check=True, timeout=self.listen_seconds + 5,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with open(tmp.name, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    model=self.transcribe_model, file=f,
                )
            text = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else None)
            text = (text or "").strip()
            if text:
                self.logger.info(f"Heard: {text!r}")
            return text or None
        except Exception as e:
            self.logger.warning(f"Microphone capture failed: {e}")
            return None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
