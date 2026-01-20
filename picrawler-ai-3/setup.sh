#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] System packages (Debian/Raspberry Pi OS)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3 python3-venv python3-pip git \
    libatlas-base-dev libopenblas0 \
    espeak ffmpeg \
    python3-opencv || true
fi

echo "[2/6] Create virtual environment (.venv)"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[3/6] Install Python dependencies"
# opencv-python-headless is used by default. If you installed python3-opencv via apt,
# keeping headless is fine; apt opencv will still be available for system python.
pip install -r requirements.txt

echo "[4/6] Create config/config.json (if missing)"
mkdir -p config
if [ ! -f config/config.json ]; then
  cp config/config.example.json config/config.json
  echo "Created config/config.json from example. Edit it to add your OPENAI key."
fi

echo "[5/6] Create logs directory"
mkdir -p logs/images

echo "[6/6] Done."
echo "Activate venv: source .venv/bin/activate"
echo "Run: python main.py --mode explore --duration 2"
