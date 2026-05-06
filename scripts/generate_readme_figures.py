from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(
        root
        / "runs/reidentification_results/evaluation_results_mAP_baseline_run_1/best-seg_baseline/dinov2_conf_0.4/summary.csv"
    )
    selected = summary[
        ((summary["feature_set"] == "feature") & (summary["metric"] == "cosine_similarity"))
        | (
            (summary["feature_set"] == "concat_feature")
            & (summary["metric"] == "attention_similarity")
            & (summary["loss_type"] == "triplet_schedule")
        )
    ].copy()
    selected["label"] = selected.apply(
        lambda r: "Baseline (Cosine, Visual)" if r["metric"] == "cosine_similarity" else "Proposed (Attention, Fused)",
        axis=1,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plots = [
        ("success_rate_objects_%", "OSR (%)", 1.0),
        ("mAP_fair", "mAP_fair (%)", 100.0),
        ("cmc_rank1", "CMC@1 (%)", 100.0),
    ]
    for ax, (column, title, scale) in zip(axes, plots):
        values = selected[column].astype(float).tolist()
        if scale != 1.0:
            values = [v * scale for v in values]
        ax.bar(selected["label"], values, color=["#4C78A8", "#F58518"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(out / "reid_results_overview.png", dpi=220)
    plt.close(fig)

    det = pd.read_csv(
        root
        / "runs/crack_segmentation/baseline_project_yolov11/seg_model-yolo11n-seg.pt_batch-32_imgsz-416_lr-0.001_seed-47/results.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(det["epoch"], det["metrics/mAP50-95(M)"], label="Mask mAP@50:95", color="#54A24B")
    ax.plot(det["epoch"], det["metrics/mAP50-95(B)"], label="Box mAP@50:95", color="#E45756")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Detection Training Curve (YOLO11n-seg)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "detection_training_curve.png", dpi=220)
    plt.close(fig)

    base = pd.read_csv(
        root
        / "runs/reidentification_results/evaluation_results_mAP_baseline_run_1/best-seg_baseline/dinov2_conf_0.4/feature_cosine/cmc_feature_cosine_cmc.csv"
    )
    proposed = pd.read_csv(
        root
        / "runs/reidentification_results/evaluation_results_mAP_baseline_run_1/best-seg_baseline/dinov2_conf_0.4/concat_feature_attention_triplet_schedule/cmc_concat_feature_attention_triplet_schedule_cmc.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(base["rank"], base["cmc"] * 100, label="Baseline (Cosine, Visual)", color="#4C78A8")
    ax.plot(proposed["rank"], proposed["cmc"] * 100, label="Proposed (Attention, Fused)", color="#F58518")
    ax.set_xlim(1, 20)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Match Rate (%)")
    ax.set_title("CMC Comparison (Top-20)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cmc_comparison_top20.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
