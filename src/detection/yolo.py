from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.utils.io import ensure_dir


def train_detection(config: dict) -> Path:
    det_cfg = config["detection"]
    run_dir = ensure_dir(det_cfg["output_dir"])
    model = YOLO(det_cfg["model"])
    model.train(
        data=det_cfg["data_yaml"],
        epochs=int(det_cfg["epochs"]),
        imgsz=int(det_cfg["imgsz"]),
        batch=int(det_cfg["batch_size"]),
        lr0=float(det_cfg["lr"]),
        project=str(run_dir),
        name=det_cfg["run_name"],
        seed=int(config["seed"]),
        optimizer=det_cfg.get("optimizer", "AdamW"),
        verbose=True,
    )
    return run_dir / det_cfg["run_name"] / "weights" / "best.pt"


def load_detector(weights_path: str) -> YOLO:
    return YOLO(weights_path)
