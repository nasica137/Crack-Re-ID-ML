from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.pipeline import evaluate_reid, write_evaluation_report
from src.reid.features import DinoFeatureExtractor, GeometryExtractor, build_memory
from src.reid.model import AttentionSimilarity
from src.utils.config import load_config
from src.utils.io import load_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--use_attention", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    test_mapping_path = Path(config["paths"]["reid_output_dir"]) / "test_cropped" / "object_mapping.json"
    mapping = load_json(test_mapping_path)
    visual = DinoFeatureExtractor(config["reid"]["dino_model"])
    geometry = GeometryExtractor(config["reid"]["segmentation_weights"])
    memory = build_memory(mapping, visual, geometry)

    attention_model = None
    if args.use_attention:
        input_dim = len(memory[0]["fused"])
        att_cfg = config["reid"]["attention"]
        attention_model = AttentionSimilarity(input_dim, int(att_cfg["hidden_dim"]), float(att_cfg["temperature"]))
        state = torch.load(att_cfg["checkpoint"], map_location="cpu")
        attention_model.load_state_dict(state)
        attention_model.eval()

    results = evaluate_reid(config, memory, attention_model=attention_model)
    write_evaluation_report(results, config["evaluation"]["output_dir"])
    print(f"Evaluation report saved to: {config['evaluation']['output_dir']}")


if __name__ == "__main__":
    main()
