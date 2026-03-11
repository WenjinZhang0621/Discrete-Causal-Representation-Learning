from pathlib import Path
import yaml


def load_config(path: str | Path):
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
