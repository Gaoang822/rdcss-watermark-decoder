from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "outputs" / "full"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CACHE = RESULTS / "cache"

METHOD_COLORS = {
    "PIMoG-Original": "#8C8C8C",
    "Coarse-Prior": "#4C78A8",
    "Decoder-Tail-FT": "#59A14F",
    "PMFA": "#F28E2B",
    "DCSS": "#E15759",
    "R-DCSS": "#B07AA1",
    "Oracle-Alignment": "#76B7B2",
    "Multi-View-Average": "#EDC948",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)


def summary_table() -> pd.DataFrame:
    return pd.read_csv(TABLES / "summary_results.csv")


def plot_main_results(summary: pd.DataFrame) -> None:
    methods = ["PIMoG-Original", "PMFA", "Decoder-Tail-FT", "DCSS", "R-DCSS"]
    conditions = ["crop50", "crop70", "full"]
    labels = ["50% partial", "70% partial", "100% full"]
    x = np.arange(len(conditions))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for index, method in enumerate(methods):
        values = [
            summary[(summary.condition == condition) & (summary.method == method)].bit_accuracy.iloc[0] * 100
            for condition in conditions
        ]
        ax.bar(
            x + (index - 2) * width,
            values,
            width,
            label=method,
            color=METHOD_COLORS[method],
        )
    ax.set_ylabel("Bit Accuracy (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(45, 103)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_main_results.png", bbox_inches="tight")
    plt.close(fig)


def plot_gain(summary: pd.DataFrame) -> None:
    conditions = ["crop70", "crop50", "crop50_blur", "crop50_jpeg"]
    labels = ["70%", "50%", "50% + Blur", "50% + JPEG"]
    baselines = ["PIMoG-Original", "Decoder-Tail-FT", "PMFA"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for axis, proposed in zip(axes, ["DCSS", "R-DCSS"]):
        proposed_values = np.array(
            [
                summary[(summary.condition == condition) & (summary.method == proposed)].bit_accuracy.iloc[0]
                for condition in conditions
            ]
        )
        for baseline in baselines:
            baseline_values = np.array(
                [
                    summary[(summary.condition == condition) & (summary.method == baseline)].bit_accuracy.iloc[0]
                    for condition in conditions
                ]
            )
            axis.plot(
                labels,
                (proposed_values - baseline_values) * 100,
                marker="o",
                label=f"vs {baseline}",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(proposed)
        axis.set_ylabel("Gain (percentage points)")
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=22)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_gain_analysis.png", bbox_inches="tight")
    plt.close(fig)


def plot_selector_diagnostics() -> None:
    frame = pd.read_csv(TABLES / "candidate_diagnostics.csv")
    selector_labels = {
        "first": "Coarse first",
        "confidence_mean": "Confidence mean",
        "confidence_l2": "DCSS (L2)",
        "confidence_soft_fusion": "Soft fusion",
        "oracle_index": "Oracle",
    }
    conditions = ["crop70", "crop50", "crop50_blur", "crop50_jpeg"]
    labels = ["70%", "50%", "50% + Blur", "50% + JPEG"]
    selectors = list(selector_labels)
    x = np.arange(len(conditions))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    colors = ["#4C78A8", "#F28E2B", "#E15759", "#59A14F", "#76B7B2"]
    for index, (selector, color) in enumerate(zip(selectors, colors)):
        values = [
            frame[(frame.split == condition) & (frame.selector == selector)].bit_accuracy.iloc[0] * 100
            for condition in conditions
        ]
        ax.bar(x + (index - 2) * width, values, width, label=selector_labels[selector], color=color)
    ax.set_ylabel("Bit Accuracy (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(55, 101)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_selector_ablation.png", bbox_inches="tight")
    plt.close(fig)


def plot_ood_metrics(summary: pd.DataFrame) -> None:
    methods = ["Coarse-Prior", "Multi-View-Average", "PMFA", "DCSS", "R-DCSS", "Oracle-Alignment"]
    conditions = ["crop50", "crop50_blur", "crop50_jpeg"]
    labels = ["Partial", "Partial + Blur", "Partial + JPEG"]
    x = np.arange(len(conditions))
    width = 0.125
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    for index, method in enumerate(methods):
        values = [
            summary[(summary.condition == condition) & (summary.method == method)].bit_accuracy.iloc[0] * 100
            for condition in conditions
        ]
        ax.bar(x + (index - 2.5) * width, values, width, label=method, color=METHOD_COLORS[method])
    ax.set_ylabel("Bit Accuracy (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(55, 101)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_ood_robustness.png", bbox_inches="tight")
    plt.close(fig)


def plot_message_recovery(summary: pd.DataFrame) -> None:
    methods = ["Coarse-Prior", "PMFA", "Decoder-Tail-FT", "DCSS", "R-DCSS", "Oracle-Alignment"]
    conditions = ["crop50", "crop70", "full"]
    labels = ["50% partial", "70% partial", "100% full"]
    x = np.arange(len(conditions))
    width = 0.125
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for index, method in enumerate(methods):
        values = [
            summary[(summary.condition == condition) & (summary.method == method)].message_accuracy.iloc[0] * 100
            for condition in conditions
        ]
        ax.bar(x + (index - 2.5) * width, values, width, label=method, color=METHOD_COLORS[method])
    ax.set_ylabel("Complete 30-bit Recovery (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 103)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_message_recovery.png", bbox_inches="tight")
    plt.close(fig)


def plot_gate_tuning() -> None:
    tuning = pd.read_csv(TABLES / "dcss_gate_tuning.csv")
    fig, ax = plt.subplots(figsize=(7.7, 4.0))
    for condition, label in (
        ("crop50", "50% partial"),
        ("crop50_blur", "50% + Blur"),
        ("crop50_jpeg", "50% + JPEG"),
        ("macro_average", "Macro average"),
    ):
        selected = tuning[(tuning.fallback_mode == "first") & (tuning.condition == condition)]
        ax.plot(selected.threshold, selected.bit_accuracy * 100, label=label)
    ax.axvline(0.174, color="#E15759", linestyle="--", linewidth=1.2, label="Selected 0.174")
    ax.set_xlabel("Binary-consistency Error Threshold")
    ax.set_ylabel("Validation Bit Accuracy (%)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_gate_tuning.png", bbox_inches="tight")
    plt.close(fig)


def plot_costs() -> None:
    runtime = pd.read_csv(TABLES / "decoding_runtime.csv")
    parameters = {
        "PIMoG-Original": 0,
        "Single-View-FT": 7710,
        "PMFA": 24224,
        "Decoder-Tail-FT": 312158,
        "DCSS": 0,
        "R-DCSS": 0,
    }
    selected = runtime[runtime.method.isin(parameters)].copy()
    selected["parameters"] = selected.method.map(parameters)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].barh(
        selected.method,
        selected.decode_ms_per_sample,
        color=[METHOD_COLORS.get(method, "#8C8C8C") for method in selected.method],
    )
    axes[0].set_xlabel("Decode Time (ms/sample)")
    axes[0].grid(axis="x", alpha=0.22)
    trainable = selected[selected.method.isin(["Single-View-FT", "PMFA", "Decoder-Tail-FT", "DCSS", "R-DCSS"])]
    axes[1].barh(
        trainable.method,
        trainable.parameters / 1000,
        color=[METHOD_COLORS.get(method, "#8C8C8C") for method in trainable.method],
    )
    axes[1].set_xlabel("New Trainable Parameters (thousand)")
    axes[1].grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_efficiency.png", bbox_inches="tight")
    plt.close(fig)


def plot_visibility() -> None:
    metadata = pd.read_csv(CACHE / "full_metadata.csv").drop_duplicates(["image_id", "message_id"])
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5))
    axes[0].hist(metadata.psnr, bins=12, color="#4C78A8", edgecolor="white")
    axes[0].axvline(metadata.psnr.mean(), color="#E15759", linestyle="--", label=f"Mean {metadata.psnr.mean():.2f} dB")
    axes[0].set_xlabel("PSNR (dB)")
    axes[0].set_ylabel("Number of Samples")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(metadata.ssim, bins=12, color="#59A14F", edgecolor="white")
    axes[1].axvline(metadata.ssim.mean(), color="#E15759", linestyle="--", label=f"Mean {metadata.ssim.mean():.4f}")
    axes[1].set_xlabel("SSIM")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_visibility_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_examples() -> None:
    sources = [
        ("example_full.png", "Full capture"),
        ("example_crop70.png", "70% partial"),
        ("example_crop50.png", "50% partial"),
        ("example_crop50_blur.png", "50% + Blur"),
        ("example_crop50_jpeg.png", "50% + JPEG"),
    ]
    fig, axes = plt.subplots(len(sources), 1, figsize=(10.0, 6.8))
    for axis, (filename, title) in zip(axes, sources):
        axis.imshow(Image.open(FIGURES / filename))
        axis.set_title(title, fontsize=10, loc="left")
        axis.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(FIGURES / "paper_attack_examples_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_method_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.1)
    ax.axis("off")
    boxes = [
        (0.1, 1.05, 1.55, 0.9, "Watermarked\nImage", "#DCE6F1"),
        (2.0, 1.05, 1.55, 0.9, "Screen-shooting\nSimulation", "#DCE6F1"),
        (3.9, 1.05, 1.55, 0.9, "Partial\nObservation", "#FBE5D6"),
        (5.8, 1.05, 1.55, 0.9, "9 Spatial\nCandidates", "#E2F0D9"),
        (7.7, 1.05, 1.55, 0.9, "Frozen PIMoG\nDecoder", "#E2F0D9"),
        (9.6, 1.05, 1.25, 0.9, "30-bit\nOutput", "#D9EAD3"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=color, edgecolor="#404040", linewidth=1.0
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for left in boxes[:-1]:
        x = left[0] + left[2]
        next_x = boxes[boxes.index(left) + 1][0]
        ax.add_patch(FancyArrowPatch((x + 0.04, 1.5), (next_x - 0.04, 1.5), arrowstyle="->", mutation_scale=12))
    ax.text(
        8.32, 2.55, "DCSS: minimize binary-consistency error",
        ha="center", va="center", fontsize=10, weight="bold", color="#9C0006"
    )
    ax.add_patch(FancyArrowPatch((8.32, 2.35), (8.32, 2.0), arrowstyle="->", mutation_scale=12, color="#9C0006"))
    ax.text(
        8.32, 0.48, "R-DCSS: confidence gate -> fallback when unreliable",
        ha="center", va="center", fontsize=9, color="#5B4B8A"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_method_flow.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure()
    summary = summary_table()
    plot_main_results(summary)
    plot_gain(summary)
    plot_selector_diagnostics()
    plot_ood_metrics(summary)
    plot_message_recovery(summary)
    plot_gate_tuning()
    plot_costs()
    plot_visibility()
    plot_examples()
    plot_method_flow()
    print("\n".join(path.name for path in sorted(FIGURES.glob("paper_*.png"))))


if __name__ == "__main__":
    main()
