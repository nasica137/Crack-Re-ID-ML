from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reid.features import DinoFeatureExtractor, GeometryExtractor, build_memory
from src.reid.trainer import save_model, train_attention
from src.utils.config import load_config
from src.utils.io import load_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    train_mapping_path = Path(config["paths"]["reid_output_dir"]) / "train_cropped" / "object_mapping.json"
    mapping = load_json(train_mapping_path)
    visual = DinoFeatureExtractor(config["reid"]["dino_model"])
    geometry = GeometryExtractor(config["reid"]["segmentation_weights"])
    memory = build_memory(mapping, visual, geometry)
    model = train_attention(memory, config)
    out_path = save_model(model, config["reid"]["attention"]["checkpoint"])
    print(f"Attention model checkpoint saved at: {out_path}")


if __name__ == "__main__":
    main()
