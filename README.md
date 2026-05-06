# Crack Re-ID ML

A compact, deployable pipeline for UAV-based crack detection and single-shot crack re-identification.

This is a refactored research codebase designed for modularity, reproducibility, and production-grade ML pipeline execution. Developed as part of a Master's thesis at Albert-Ludwigs-Universität Freiburg and Fraunhofer IPM.

## Overview

This project unifies crack detection and re-identification in a single, efficient pipeline optimized for UAV infrastructure inspection.

A single YOLO11n-seg checkpoint handles both detection and instance segmentation. Detected cracks are then processed through a Re-ID pipeline that combines visual and geometric features for robust matching.

## Architecture

- **Detection & Segmentation**: YOLO11n-seg for unified crack localization
- **Visual Embedding**: DINOv2 features from cropped detections
- **Geometric Embedding**: 11-D mask-based shape descriptor
- **Matching**: Attention-based similarity network with gallery awareness
- **Training**: Staged triplet-loss schedule with class-balanced sampling
- **Evaluation**: OSR, mAP_fair, CMC curves (Rank-k retrieval)

## Results

### Performance Improvements

Baseline (cosine similarity on visual features) vs. Proposed (attention fusion + geometry):

- Higher OSR (Object Success Rate)
- Better mAP_fair (fair evaluation under class imbalance)
- Stronger CMC@1 (top-1 retrieval accuracy)

### CMC Curve Comparison

![CMC Comparison](docs/assets/cmc_comparison_baseline_vs_schedule.svg)

The proposed model consistently outperforms the baseline across all ranks.

### Qualitative Example

![Top-k Matching Example](docs/assets/topk_success_1_top3_concat_feature_attention_similarity.png)

Improved ranking stability using fused visual and geometric attention-based similarity.

## Project Structure

```
crack-reid-ml/
├── src/
│   ├── detection/        # YOLO-based detection pipeline
│   ├── reid/             # Re-ID matching and similarity models
│   ├── data/             # Dataset utilities and loaders
│   ├── evaluation/       # Metrics (CMC, mAP, OSR)
│   └── utils/            # Common utilities
│
├── scripts/
│   ├── train_detection.py      # YOLO11n-seg training
│   ├── build_reid_dataset.py   # Query/gallery generation
│   ├── train_reid.py           # AttentionSimilarity training
│   ├── evaluate_reid.py        # Evaluation and visualization
│   ├── train_pipeline.py       # End-to-end orchestration
│   └── generate_readme_svgs.py # Figure generation
│
├── configs/
│   └── default.yaml      # All hyperparameters and paths
│
├── docs/assets/          # Figures and visualizations
├── Dockerfile            # Container image
├── requirements.txt      # Dependencies
├── crack-seg.yaml        # YOLO dataset config
└── README.md
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
python scripts/train_pipeline.py --config configs/default.yaml
```

This executes all stages:
1. Train YOLO11n-seg detector
2. Build Re-ID dataset (query/gallery)
3. Extract embeddings (DINOv2 and geometry)
4. Train AttentionSimilarity matcher
5. Evaluate and generate reports

### Stage-wise Execution

You can skip specific stages as needed:

```bash
# Skip detection training
python scripts/train_pipeline.py --config configs/default.yaml --skip_detection

# Skip dataset building
python scripts/train_pipeline.py --config configs/default.yaml --skip_dataset

# Skip Re-ID training
python scripts/train_pipeline.py --config configs/default.yaml --skip_reid
```

## Evaluation

Generate evaluation metrics and visualizations:

```bash
python scripts/evaluate_reid.py \
  --config configs/default.yaml \
  --use_attention
```

Output includes:
- CMC (Cumulative Matching Characteristic) curves
- mAP (mean Average Precision) scores
- OSR (Object Success Rate) metrics
- Top-k retrieval visualizations

## Configuration

All training parameters are defined in `configs/default.yaml`. Before running:

1. Update dataset paths:
   ```yaml
   paths:
     dataset_root: "/path/to/your/dataset"
     output_dir: "./outputs"
   ```

2. Edit `crack-seg.yaml` with your dataset directory structure

3. Set the device (GPU/CPU):
   ```yaml
   device: "cuda:0"  # or "cpu"
   ```

## Key Contributions

1. Unified architecture with single YOLO checkpoint for detection and segmentation
2. Combined visual (DINOv2) and geometric (mask descriptor) embeddings
3. Attention-based matching replacing static cosine distance
4. Staged triplet training with gallery-aware sampling
5. Full modularity with configuration-driven pipeline
6. Reproducible setup with deterministic training

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Ultralytics (YOLO11)
- timm (DINOv2)
- scikit-learn, numpy, scipy
- OpenCV, Pillow

See `requirements.txt` for pinned versions.

## Docker Deployment

```bash
docker build -t crack-reid-ml:latest .
docker run --gpus all -v /data:/data crack-reid-ml:latest \
  python scripts/train_pipeline.py --config configs/default.yaml
```

## Reproducibility

- Fixed seed ensures deterministic runs
- No hardcoded paths (configuration-driven)
- Version-pinned dependencies
- Regenerate figures: `python scripts/generate_readme_svgs.py`

## References

**Master's Thesis:**
- Institution: Albert-Ludwigs-Universität Freiburg & Fraunhofer IPM
- Topic: AI-based Crack Re-Identification for Autonomous UAV Inspection
of Concrete Structures.

**Key Works:**
- YOLO11n-seg: https://github.com/ultralytics/ultralytics
- DINOv2: https://github.com/facebookresearch/dinov2
- Triplet Loss and CMC evaluation: Reid community standards

## License

This project is provided as-is for research and educational purposes.

## Author

Developed as part of a Master's thesis in collaboration with Albert-Ludwigs-Universität Freiburg and Fraunhofer IPM (Institute for Physical Measurement Technology).
