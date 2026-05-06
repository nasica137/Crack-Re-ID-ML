from __future__ import annotations

import csv
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_summary(summary_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    with summary_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    baseline = next(
        r for r in rows if r["feature_set"] == "feature" and r["metric"] == "cosine_similarity"
    )
    proposed = next(
        r
        for r in rows
        if r["feature_set"] == "concat_feature"
        and r["metric"] == "attention_similarity"
        and r["loss_type"] == "triplet_schedule"
    )
    b = {
        "OSR": float(baseline["success_rate_objects_%"]),
        "mAP_fair": float(baseline["mAP_fair"]) * 100.0,
        "CMC1": float(baseline["cmc_rank1"]) * 100.0,
    }
    p = {
        "OSR": float(proposed["success_rate_objects_%"]),
        "mAP_fair": float(proposed["mAP_fair"]) * 100.0,
        "CMC1": float(proposed["cmc_rank1"]) * 100.0,
    }
    return b, p


def build_bar_svg(baseline: dict[str, float], proposed: dict[str, float]) -> str:
    metrics = ["OSR", "mAP_fair", "CMC1"]
    width, height = 960, 340
    plot_top, plot_bottom = 40, 280
    max_val = 100.0
    bars = []
    labels = []
    x = 120
    for m in metrics:
        b_val = baseline[m]
        p_val = proposed[m]
        b_h = (b_val / max_val) * (plot_bottom - plot_top)
        p_h = (p_val / max_val) * (plot_bottom - plot_top)
        bars.append(
            f'<rect x="{x}" y="{plot_bottom-b_h:.1f}" width="48" height="{b_h:.1f}" fill="#4C78A8"/>'
            f'<rect x="{x+56}" y="{plot_bottom-p_h:.1f}" width="48" height="{p_h:.1f}" fill="#F58518"/>'
            f'<text x="{x+24}" y="{plot_bottom-b_h-8:.1f}" font-size="12" text-anchor="middle">{b_val:.1f}</text>'
            f'<text x="{x+80}" y="{plot_bottom-p_h-8:.1f}" font-size="12" text-anchor="middle">{p_val:.1f}</text>'
        )
        labels.append(f'<text x="{x+52}" y="305" font-size="13" text-anchor="middle">{m}</text>')
        x += 230

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="20" y="25" font-size="18" font-weight="bold">Re-ID Result Overview (Run 1)</text>
  <line x1="90" y1="{plot_top}" x2="90" y2="{plot_bottom}" stroke="#333"/>
  <line x1="90" y1="{plot_bottom}" x2="880" y2="{plot_bottom}" stroke="#333"/>
  <text x="20" y="{plot_top+5}" font-size="12">100</text>
  <text x="30" y="{plot_bottom+5}" font-size="12">0</text>
  {''.join(bars)}
  {''.join(labels)}
  <rect x="620" y="18" width="16" height="12" fill="#4C78A8"/><text x="642" y="28" font-size="12">Baseline (Cosine, Visual)</text>
  <rect x="790" y="18" width="16" height="12" fill="#F58518"/><text x="812" y="28" font-size="12">Proposed (Attention, Fused)</text>
</svg>"""


def _read_cmc(path: Path, topk: int = 20) -> list[tuple[int, float]]:
    points = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = int(row["rank"])
            if r <= topk:
                points.append((r, float(row["cmc"]) * 100.0))
    return points


def build_cmc_svg(base_pts: list[tuple[int, float]], prop_pts: list[tuple[int, float]]) -> str:
    width, height = 960, 360
    left, top, right, bottom = 70, 40, 920, 300

    def xy(rank: int, val: float) -> tuple[float, float]:
        x = left + (rank - 1) / 19 * (right - left)
        y = bottom - (val / 100.0) * (bottom - top)
        return x, y

    def polyline(points: list[tuple[int, float]], color: str) -> str:
        coords = " ".join(f"{xy(r, v)[0]:.1f},{xy(r, v)[1]:.1f}" for r, v in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{coords}" />'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="20" y="25" font-size="18" font-weight="bold">CMC Comparison (Top-20)</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>
  <text x="15" y="{top+5}" font-size="12">100%</text>
  <text x="22" y="{bottom+5}" font-size="12">0%</text>
  <text x="{left}" y="{bottom+22}" font-size="12">1</text>
  <text x="{right-8}" y="{bottom+22}" font-size="12">20</text>
  {polyline(base_pts, "#4C78A8")}
  {polyline(prop_pts, "#F58518")}
  <rect x="620" y="18" width="16" height="12" fill="#4C78A8"/><text x="642" y="28" font-size="12">Baseline (Cosine, Visual)</text>
  <rect x="790" y="18" width="16" height="12" fill="#F58518"/><text x="812" y="28" font-size="12">Proposed (Attention, Fused)</text>
</svg>"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "runs/reidentification_results/evaluation_results_mAP_baseline_run_1/best-seg_baseline/dinov2_conf_0.4/summary.csv"
    base_cmc_path = root / "runs/reidentification_results/evaluation_results_mAP_baseline_run_1/best-seg_baseline/dinov2_conf_0.4/feature_cosine/cmc_feature_cosine_cmc.csv"
    prop_cmc_path = root / "runs/reidentification_results/evaluation_results_mAP_baseline_run_1/best-seg_baseline/dinov2_conf_0.4/concat_feature_attention_triplet_schedule/cmc_concat_feature_attention_triplet_schedule_cmc.csv"
    out = root / "docs/assets"

    baseline, proposed = _read_summary(summary_path)
    _write(out / "reid_results_overview.svg", build_bar_svg(baseline, proposed))
    _write(out / "cmc_comparison_top20.svg", build_cmc_svg(_read_cmc(base_cmc_path), _read_cmc(prop_cmc_path)))


if __name__ == "__main__":
    main()
