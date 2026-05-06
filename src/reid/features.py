from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import skeletonize
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO


class DinoFeatureExtractor:
    def __init__(self, model_name: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            batch = self.processor(images=image, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch).last_hidden_state[:, 0, :]
            out = F.normalize(out, p=2, dim=1)
            return out.squeeze(0).cpu().numpy().astype(np.float32)


class GeometryExtractor:
    def __init__(self, segmentation_weights: str, conf: float = 0.1) -> None:
        self.model = YOLO(segmentation_weights)
        self.conf = conf

    def _mask(self, image: Image.Image) -> np.ndarray:
        res = self.model.predict(source=np.array(image), conf=self.conf, verbose=False)[0]
        if res.masks is None or len(res.masks.data) == 0:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        return (res.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)

    def extract(self, image: Image.Image) -> np.ndarray:
        mask = self._mask(image)
        area = float(mask.sum()) / max(1.0, mask.shape[0] * mask.shape[1])
        skel = skeletonize(mask > 0).astype(np.uint8)
        skel_len = float(skel.sum()) / max(1.0, np.hypot(*mask.shape))
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            aspect = 0.0
        else:
            aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1 + 1e-6))
        feat = np.array([area, skel_len, aspect], dtype=np.float32)
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm


def build_memory(mapping: Dict[str, Dict[str, str]], visual: DinoFeatureExtractor, geometry: GeometryExtractor) -> List[dict]:
    memory = []
    for object_id, item in mapping.items():
        image = Image.open(item["image_file"]).convert("RGB")
        visual_feat = visual.extract(image)
        geom_feat = geometry.extract(image)
        fused = np.concatenate([visual_feat, geom_feat]).astype(np.float32)
        fused /= np.linalg.norm(fused) + 1e-12
        memory.append({"object_id": object_id, "visual": visual_feat, "geometry": geom_feat, "fused": fused})
    return memory
