from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset_builder import build_reid_datasets
from src.detection.yolo import train_detection
from src.reid.features import DinoFeatureExtractor, GeometryExtractor, build_memory
from src.reid.trainer import save_model, train_attention
from src.utils.config import load_config
from src.utils.io import load_json, set_seed


def run_reid_training(config: dict) -> str:
    train_mapping_path = Path(config["paths"]["reid_output_dir"]) / "train_cropped" / "object_mapping.json"
    mapping = load_json(train_mapping_path)
    visual = DinoFeatureExtractor(config["reid"]["dino_model"])
    geometry = GeometryExtractor(config["reid"]["segmentation_weights"])
    memory = build_memory(mapping, visual, geometry)
    model = train_attention(memory, config)
    checkpoint = save_model(model, config["reid"]["attention"]["checkpoint"])
    return str(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training entry point for Crack Re-ID.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip_detection", action="store_true")
    parser.add_argument("--skip_dataset", action="store_true")
    parser.add_argument("--skip_reid", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    if not args.skip_detection:
        weights_path = train_detection(config)
        print(f"[1/3] Detection training complete: {weights_path}")

    if not args.skip_dataset:
        build_reid_datasets(config)
        print(f"[2/3] Re-ID dataset build complete: {config['paths']['reid_output_dir']}")

    if not args.skip_reid:
        checkpoint = run_reid_training(config)
        print(f"[3/3] Re-ID training complete: {checkpoint}")

    print("Training pipeline finished.")


if __name__ == "__main__":
    main()
