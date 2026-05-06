from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from src.evaluation.metrics import compute_cmc, compute_map_fair
from src.reid.features import DinoFeatureExtractor, GeometryExtractor
from src.reid.model import AttentionSimilarity
from src.utils.io import ensure_dir


def _feature_for_crop(image: Image.Image, visual: DinoFeatureExtractor, geometry: GeometryExtractor) -> np.ndarray:
    visual_feat = visual.extract(image)
    geom_feat = geometry.extract(image)
    fused = np.concatenate([visual_feat, geom_feat]).astype(np.float32)
    fused /= np.linalg.norm(fused) + 1e-12
    return fused


def evaluate_reid(config: dict, memory: List[dict], attention_model: AttentionSimilarity | None = None) -> Dict[str, object]:
    detector = YOLO(config["reid"]["detection_weights"])
    query_dir = Path(config["paths"]["reid_output_dir"]) / "test_augmented"
    gallery_ids = [m["object_id"] for m in memory]
    gallery_feats = torch.tensor([m["fused"] for m in memory], dtype=torch.float32)
    visual = DinoFeatureExtractor(config["reid"]["dino_model"])
    geometry = GeometryExtractor(config["reid"]["segmentation_weights"])

    rows = []
    y_true, y_score, ranks = [], [], []
    for image_path in sorted((query_dir / "images").glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        detections = detector(str(image_path), conf=float(config["reid"]["det_conf"]))[0].boxes.xyxy.cpu().numpy()
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det[:4])
            crop = image.crop((x1, y1, x2, y2))
            feat = torch.tensor(_feature_for_crop(crop, visual, geometry), dtype=torch.float32).unsqueeze(0)

            if attention_model is None:
                sim = F.cosine_similarity(gallery_feats, feat.expand_as(gallery_feats), dim=1)
            else:
                with torch.no_grad():
                    sim = F.softmax(attention_model(feat, gallery_feats), dim=-1).squeeze(0)

            sorted_idx = torch.argsort(sim, descending=True).cpu().numpy()
            best_object = gallery_ids[int(sorted_idx[0])]
            gt_oid = f"{image_path.stem}_object{det_idx}"
            rank = int(np.where(sorted_idx == gallery_ids.index(gt_oid))[0][0] + 1) if gt_oid in gallery_ids else -1
            ranks.append(rank)

            gt = np.zeros(len(gallery_ids), dtype=np.int32)
            if gt_oid in gallery_ids:
                gt[gallery_ids.index(gt_oid)] = 1
            y_true.append(gt)
            y_score.append(sim.detach().cpu().numpy())

            rows.append({"image": image_path.name, "det_idx": det_idx, "pred": best_object, "gt": gt_oid, "rank": rank})

    cmc = compute_cmc(ranks, max_rank=10)
    map_fair = compute_map_fair(np.array(y_true), np.array(y_score))
    return {"rows": rows, "cmc": cmc, "map_fair": map_fair}


def write_evaluation_report(results: Dict[str, object], output_dir: str) -> None:
    out = ensure_dir(output_dir)
    pd.DataFrame(results["rows"]).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame({"rank": list(range(1, len(results["cmc"]) + 1)), "cmc": results["cmc"]}).to_csv(out / "cmc.csv", index=False)
    (out / "summary.txt").write_text(
        f"mAP_fair: {results['map_fair']:.4f}\nCMC@1: {results['cmc'][0]:.4f}\nCMC@5: {results['cmc'][4]:.4f}\nCMC@10: {results['cmc'][9]:.4f}\n",
        encoding="utf-8",
    )
