import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / "operation.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
