#!/usr/bin/env python3

import os

from core.config_loader import load_config
from core.logger import setup_logging
from ai.vision_ai import AIVisionSystem


def main() -> int:
    setup_logging("INFO")
    config = load_config("config/config.json")
    ai = AIVisionSystem(config)

    if not (os.getenv("OPENAI_API_KEY") or config.get("openai_api_key")): 
        print("No API key set. Set OPENAI_API_KEY or config/openai_api_key")

    # Text-only ping
    resp = ai.client.responses.create(
        model=ai.model,
        input="Reply with the single word OK.",
        max_output_tokens=10,
    )
    print(resp.output_text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
