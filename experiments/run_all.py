from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcrl.utils import load_config


def main():
    config = load_config(ROOT / "configs" / "default.yaml")
    print("Running scaffold experiment for:", config["project_name"])
    print("This is a placeholder entry point. Add your simulation pipeline here.")


if __name__ == "__main__":
    main()
