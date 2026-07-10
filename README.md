# PiCrawler-AI (OpenAI Vision)

An autonomous Raspberry Pi crawler robot that uses **OpenAI vision** to look at the world, decide what to do, and drive movement primitives.

This repo is designed to be:
- **Clone-and-run on a Raspberry Pi**
- **Safe by default** (stops if uncertain)
- **Hardware-optional** (falls back to a mock robot if the SunFounder library/hardware isn’t present)

## Features
- Modes: **explore**, **detect**, **follow**, **avoid**
- Camera capture with **Picamera2** fallback to **OpenCV**
- OpenAI vision decisions via the **Responses API**
- Logs
  - `logs/operation.log`
  - `logs/decisions.jsonl`
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

## Hardware support
- If `picrawler` is installed and your SunFounder PiCrawler is connected, the project will drive real movement.
- If not, it runs in **dry-run** mode (prints actions instead of moving).

## Safety
Multiple layers keep the robot from driving into things even when the AI misbehaves:
- **Action allowlist** — anything the model returns outside the known action set is coerced to `stop`, and durations are clamped.
- **Ultrasonic override** — `forward` is blocked before it starts if an obstacle is inside the threshold, and motion is **monitored mid-move** and aborted early if an obstacle appears.
- **Anti-loop watchdog** — repeated actions, left/right oscillation, and stagnant scenes trigger an escape sequence with temporary action bans.
- **Camera panic stop** — a sudden large visual change (falling, being picked up) raises an emergency stop.
- **Fail-safe AI errors** — API failures resolve to `stop`, never to a guess.

## Running the tests
The core logic (safety gate, anti-loop watchdog, AI decision contract, panic detection) is covered by a hardware-free test suite:
```bash
pip install -r requirements-dev.txt
pytest
```
Tests run automatically in CI on every push.

## Hardware smoke checks (on the Pi)
```bash
python utils/check_camera.py
python utils/check_ai.py
python utils/check_motors.py
```

## Manual gait tooling (optional)
If you have the SunFounder `picrawler` library installed, you can use:
```bash
python scripts/record_new_step_by_keyboard.py
```
