# PiCrawler-AI (OpenAI Vision)

[![CI](https://github.com/LochlanMcDonald/picrawler-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/LochlanMcDonald/picrawler-ai/actions/workflows/ci.yml)

An autonomous Raspberry Pi crawler robot that uses **OpenAI vision** to look at the world, decide what to do, and drive movement primitives.

This repo is designed to be:
- **Clone-and-run on a Raspberry Pi**
- **Safe by default** (stops if uncertain)
- **Hardware-optional** (falls back to a mock robot if the SunFounder library/hardware isn't present)

## Features
- Modes: **explore**, **detect**, **follow**, **avoid**
- Camera capture with **Picamera2** fallback to **OpenCV**
- OpenAI vision decisions via the **Responses API**
- Ultrasonic obstacle detection with mid-motion abort
- Anti-loop watchdog (repeat/oscillation/stagnation detection with escape sequences)
- **Voice**: spoken narration of actions, plus optional AI-generated personality dialogue
  (name, style, and humor configurable under `personality_settings`)
- Logs
  - `logs/operation.log`
  - `logs/decisions.jsonl` (one JSON record per decision tick: what was seen, decided, and executed)
  - `logs/images/` (frames)

## Quick start

```bash
./setup.sh
source .venv/bin/activate
cp config/config.example.json config/config.json
nano config/config.json   # add your API key
python main.py --mode explore --duration 5
```

## Running modes

```bash
# Explore autonomously
python main.py --mode explore --duration 10

# Detect an object
python main.py --mode detect --target "red ball" --duration 5

# Follow a target (default: person)
python main.py --mode follow --target "person" --duration 5

# Avoid specific things while exploring
python main.py --mode avoid --target "people and pets" --duration 5

# Sanity test (capture 1 frame + 1 analysis)
python main.py --mode test
```

## Where to put the API key
- Preferred: set an environment variable on the Pi:
  ```bash
  export OPENAI_API_KEY="..."
  ```
- Or: put it into `config/config.json` under `openai_api_key` (this file is gitignored).

Without a key the program still starts (useful for dry runs); AI calls fail safely and the robot stops.

## Project structure

```
main.py                 CLI entry point: wires everything together per mode
ai/vision_ai.py         OpenAI vision analysis, action decisions, dialogue generation
behaviors/              Behavior modes (explore/detect/follow/avoid) + shared
                        anti-loop/safety logic in base_behavior.py
core/robot_controller.py  Hardware abstraction + motor safety gate
                          (MockRobot / SunFounder adapter / ultrasonic)
vision/camera.py        Picamera2/OpenCV capture with panic-stop detection
voice/voice_system.py   OpenAI TTS with caching, cooldown, and dedupe
config/                 config.example.json — every setting the code reads
tests/                  Hardware-free pytest suite (runs in CI)
utils/                  Manual hardware smoke checks (run on the Pi)
scripts/                Gait recording tooling
picrawler-ai-4/         Experimental next-gen architecture (SLAM, A* path
                        planning, behavior trees) — standalone, not yet
                        integrated or covered by tests
```

Design notes for the experimental architecture live in
[ARCHITECTURE_REDESIGN.md](ARCHITECTURE_REDESIGN.md) and [IMPROVEMENTS.md](IMPROVEMENTS.md).

## Configuration

All behavior is tuned from `config/config.json` (see `config/config.example.json` for the full annotated set):

| Section | Controls |
|---|---|
| `ai_settings` | Model, token limits, request timeout, call throttling |
| `robot_settings` | Movement/turn speeds, obstacle threshold, dry-run fallback |
| `camera_settings` | Resolution, capture interval, flips, panic-stop sensitivity |
| `voice_settings` | TTS model/voice, narration and dialogue throttling, verbosity |
| `personality_settings` | Robot name, speaking style, humor, word cap |
| `behavior_settings.anti_loop` | Repeat/oscillation thresholds, ban and escape timing |
| `logging_settings` | Log level, frame saving, decision records |

## Hardware support
- If `picrawler` is installed and your SunFounder PiCrawler is connected, the project will drive real movement.
- If not, it runs in **dry-run** mode (prints actions instead of moving).

## Safety
Multiple layers keep the robot from driving into things even when the AI misbehaves:
- **Action allowlist** — anything the model returns outside the known action set is coerced to `stop`, and durations are clamped.
- **Ultrasonic override** — `forward` is blocked before it starts if an obstacle is inside the threshold, and motion is **monitored mid-move** and aborted early if an obstacle appears.
- **Anti-loop watchdog** — repeated actions, left/right oscillation, and stagnant scenes trigger an escape sequence with temporary action bans.
- **Camera panic stop** — a sudden large visual change (falling, being picked up) raises an emergency stop.
- **Camera circuit breaker** — repeated capture failures halt the robot instead of letting it run blind.
- **Fail-safe AI errors** — API failures resolve to `stop`, never to a guess.

## Running the tests
The core logic (safety gate, anti-loop watchdog, AI decision contract, panic detection, main behavior loop) is covered by a hardware-free test suite — no robot, camera, or API key required:
```bash
pip install -r requirements-dev.txt
pytest
```
Tests run automatically in CI (GitHub Actions, Python 3.11 and 3.12) on every pull request.

## Hardware smoke checks (on the Pi)
```bash
python utils/check_camera.py
python utils/check_ai.py
python utils/check_motors.py
python utils/check_ultrasonic.py
```

## Manual gait tooling (optional)
If you have the SunFounder `picrawler` library installed, you can use:
```bash
python scripts/record_new_step_by_keyboard.py
```
