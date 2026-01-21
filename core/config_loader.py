import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}. Create it from config/config.example.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
