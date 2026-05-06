from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset_builder import build_reid_datasets
from src.utils.config import load_config
from src.utils.io import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    build_reid_datasets(config)
    print(f"Re-ID datasets created under: {config['paths']['reid_output_dir']}")


if __name__ == "__main__":
    main()
