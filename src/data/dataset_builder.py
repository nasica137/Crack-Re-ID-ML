from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from src.utils.io import ensure_clean_dir, ensure_dir, save_json


@dataclass
class SplitPaths:
    images: Path
    labels: Path


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        yield from sorted(images_dir.glob(ext))


def _read_polygon_labels(label_path: Path) -> List[Tuple[int, np.ndarray]]:
    polygons: List[Tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        class_id = int(parts[0])
        pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        polygons.append((class_id, pts))
    return polygons


def _polygon_to_bbox(polygon: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    px = polygon.copy()
    px[:, 0] *= width
    px[:, 1] *= height
    px = px.astype(np.int32)
    min_x, min_y = np.clip(np.min(px, axis=0), [0, 0], [width - 1, height - 1]).tolist()
    max_x, max_y = np.clip(np.max(px, axis=0), [0, 0], [width - 1, height - 1]).tolist()
    return int(min_x), int(min_y), int(max_x), int(max_y)


def build_cropped_split(input_split: SplitPaths, output_split_dir: Path) -> Path:
    output_images = ensure_dir(output_split_dir / "images")
    output_labels = ensure_dir(output_split_dir / "labels")
    mapping: Dict[str, Dict[str, object]] = {}

    for image_path in _iter_images(input_split.images):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = input_split.labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        polygons = _read_polygon_labels(label_path)
        for obj_idx, (class_id, polygon) in enumerate(polygons):
            x1, y1, x2, y2 = _polygon_to_bbox(polygon, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{image_path.stem}_object{obj_idx}"
            crop_img_path = output_images / f"{crop_name}.jpg"
            crop_lbl_path = output_labels / f"{crop_name}.txt"
            cv2.imwrite(str(crop_img_path), crop)

            points = polygon.flatten().tolist()
            crop_lbl_path.write_text(
                f"{class_id} {' '.join(map(str, points))}\n",
                encoding="utf-8",
            )
            mapping[crop_name] = {
                "object_id": crop_name,
                "image_file": str(crop_img_path),
                "label_file": str(crop_lbl_path),
                "bbox": [x1, y1, x2, y2],
            }

    mapping_path = output_split_dir / "object_mapping.json"
    save_json(mapping, mapping_path)
    return mapping_path


def augment_split(input_split: SplitPaths, output_split_dir: Path, angle_deg: float, brightness_jitter: float) -> None:
    out_images = ensure_dir(output_split_dir / "images")
    out_labels = ensure_dir(output_split_dir / "labels")

    for image_path in _iter_images(input_split.images):
        label_path = input_split.labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        angle = np.random.uniform(-angle_deg, angle_deg)
        image = image.rotate(angle, resample=Image.BILINEAR)
        enhancer = ImageEnhance.Brightness(image)
        factor = np.random.uniform(1 - brightness_jitter, 1 + brightness_jitter)
        image = enhancer.enhance(factor)
        image.save(out_images / image_path.name)

        # Keep label structure for deterministic matching.
        (out_labels / label_path.name).write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")


def build_reid_datasets(config: dict) -> None:
    out_root = Path(config["paths"]["reid_output_dir"])
    ensure_clean_dir(out_root)
    for split in ("train", "valid", "test"):
        split_cfg = config["paths"][split]
        split_paths = SplitPaths(images=Path(split_cfg["images"]), labels=Path(split_cfg["labels"]))
        cropped_dir = out_root / f"{split}_cropped"
        augmented_dir = out_root / f"{split}_augmented"
        build_cropped_split(split_paths, cropped_dir)
        augment_split(
            split_paths,
            augmented_dir,
            angle_deg=float(config["data"]["augmentation"]["max_angle_deg"]),
            brightness_jitter=float(config["data"]["augmentation"]["brightness_jitter"]),
        )
