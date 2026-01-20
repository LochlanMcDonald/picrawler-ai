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

## Developer utilities
```bash
python utils/test_camera.py
python utils/test_ai.py
python utils/test_motors.py
```

## Manual gait tooling (optional)
If you have the SunFounder `picrawler` library installed, you can use:
```bash
python scripts/record_new_step_by_keyboard.py
```
