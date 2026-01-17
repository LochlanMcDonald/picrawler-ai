# Quick Start (Raspberry Pi)

## 1) Clone
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd picrawler-ai
```

## 2) Install
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

## 3) Configure key
```bash
cp config/config.example.json config/config.json
nano config/config.json
```

Alternatively:
```bash
export OPENAI_API_KEY="..."
```

## 4) Test camera + AI
```bash
python utils/test_camera.py
python utils/test_ai.py
python main.py --mode test
```

## 5) Run
```bash
python main.py --mode explore --duration 10
```

## Logs
- `logs/operation.log`
- `logs/decisions.jsonl`
- `logs/images/`
