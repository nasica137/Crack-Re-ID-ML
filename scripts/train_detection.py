from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detection.yolo import train_detection
from src.utils.config import load_config
from src.utils.io import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    weights_path = train_detection(config)
    print(f"Detection model saved at: {weights_path}")


if __name__ == "__main__":
    main()
