# Crack Re-ID ML

A compact, deployable pipeline for UAV-based crack detection and single-shot crack re-identification (Re-ID).

This project is a refactored and productionized version of my original research codebase, redesigned for modularity, reproducibility, and end-to-end ML pipeline execution.

It was developed as part of my Master’s thesis at the Albert-Ludwigs-Universität Freiburg and Fraunhofer IPM.

---

## Overview

A single Ultralytics YOLO11n-seg checkpoint is used for both:

- Crack detection  
- Instance mask extraction  

Detected cracks are then passed into a Re-ID pipeline:

- Visual embeddings: DINOv2 features from cropped detections  
- Geometric embeddings: 11-D descriptor from segmentation masks  
- Matching model: Attention-based similarity network (AttentionSimilarity)  
- Training strategy: staged triplet-loss schedule with gallery-aware sampling  

---

## Goal

Robust, fair, and efficient single-shot crack Re-ID under UAV constraints, evaluated using:

- OSR (Object Success Rate)  
- mAP_fair (fair mean Average Precision under class imbalance)  
- CMC curves (Rank-k retrieval performance)

---

## Core Model Stack

- Detection / Segmentation: YOLO11n-seg (Ultralytics)
- Visual Embedding: DINOv2
- Geometry Embedding: 11-D mask-based shape descriptor
- Matching: AttentionSimilarity (gallery-aware fusion model)

---

## Key Contributions

- Refactored original research code into a clean modular ML pipeline
- Unified detection + segmentation into a single YOLO checkpoint
- Combined visual + geometric feature fusion for Re-ID
- Replaced cosine matching with attention-based similarity learning
- Introduced staged triplet training schedule
- Fully config-driven and reproducible setup

---

## Thesis Results Snapshot

### Re-ID Performance (Baseline vs Proposed)

![Re-ID Results Overview](docs/assets/reid_results_overview.svg)

- Baseline: cosine similarity on visual features only  
- Proposed: attention-based fusion of visual + geometry features  

Key improvements:
- Higher OSR  
- Better mAP_fair  
- Improved CMC@1 accuracy  

---

### CMC Curve Comparison (Top-k)

![CMC Comparison](docs/assets/cmc_comparison_baseline_vs_schedule.svg)

The proposed model consistently outperforms the baseline across all ranks, especially in early retrieval (top-1 to top-10).

---

### Qualitative Re-ID Example (Top-3 Retrieval)

![Top-k Matching Example](docs/assets/topk_success_1_top3_concat_feature_attention_similarity.png)

Shows improved ranking stability using fused visual + geometric attention-based similarity.

---

## Pipeline Flow

Detection → Cropping → Feature Extraction → Re-ID Matching → Evaluation

Steps:

1. Train YOLO11n-seg crack detector  
2. Build Re-ID dataset (query/gallery generation)  
3. Extract embeddings (DINOv2 + geometry descriptors)  
4. Train AttentionSimilarity matcher  
5. Evaluate using OSR / mAP / CMC  

---

## Project Structure

crack-reid-ml/

├── src/
│   ├── detection/
│   ├── reid/
│   ├── data/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── train_detection.py
│   ├── build_reid_dataset.py
│   ├── train_reid.py
│   ├── evaluate_reid.py
│   ├── train_pipeline.py
│   └── generate_readme_svgs.py
│
├── configs/
│   └── default.yaml
│
├── docs/assets/
├── Dockerfile
├── requirements.txt
├── crack-seg.yaml
└── README.md

---

## Quick Start

pip install -r requirements.txt

python scripts/train_pipeline.py --config configs/default.yaml

---

## Stage-wise Execution

# Skip detection training
python scripts/train_pipeline.py --config configs/default.yaml --skip_detection

# Skip dataset building
python scripts/train_pipeline.py --config configs/default.yaml --skip_dataset

# Skip Re-ID training
python scripts/train_pipeline.py --config configs/default.yaml --skip_reid

---

## Evaluation

python scripts/evaluate_reid.py \
  --config configs/default.yaml \
  --use_attention

Outputs:
- CMC curves
- mAP scores
- OSR metrics
- Retrieval visualizations

---

## Reproducibility

- All parameters defined in configs/default.yaml  
- Fixed seed ensures deterministic runs  
- No hardcoded paths  
- Regenerate figures:

python scripts/generate_readme_svgs.py

---

## Notes

- Update dataset and checkpoint paths in configs/default.yaml  
- crack-seg.yaml defines YOLO dataset config  
- Designed for UAV infrastructure inspection

---
