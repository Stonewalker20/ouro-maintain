from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = ROOT / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
CACHE_DIR = ROOT / ".cache_local"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "report" / "figures"


def load_metrics(run_name: str, split: str = "test") -> dict[str, float]:
    path = ARTIFACTS / run_name / f"{split}_metrics.json"
    return json.loads(path.read_text())


def load_history(run_name: str) -> dict[str, np.ndarray]:
    path = ARTIFACTS / run_name / "history.csv"
    lines = path.read_text().strip().splitlines()
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:]]
    columns = list(zip(*rows))
    return {
        key: np.asarray([float(value) for value in values])
        for key, values in zip(header, columns)
    }


def ensure_output() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)


def metric_comparison() -> None:
    run_names = [
        ("Baseline", "cmapss_fd001_baseline"),
        ("Fixed loop", "cmapss_fd001_fixed"),
        ("Adaptive loop", "cmapss_fd001_adaptive"),
    ]
    macro_f1 = [load_metrics(run, "test")["macro_f1"] for _, run in run_names]
    avg_depth = [load_metrics(run, "test")["avg_steps"] for _, run in run_names]
    labels = [label for label, _ in run_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, macro_f1, color=["#6C8EBF", "#D79B00", "#5C9E6E"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("C-MAPSS FD001 Test Macro F1")
    for idx, value in enumerate(macro_f1):
        axes[0].text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    axes[1].bar(labels, avg_depth, color=["#6C8EBF", "#D79B00", "#5C9E6E"])
    axes[1].set_ylabel("Average depth")
    axes[1].set_title("C-MAPSS FD001 Test Compute")
    for idx, value in enumerate(avg_depth):
        axes[1].text(idx, value + 0.08, f"{value:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT / "cmapss_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def adaptive_history() -> None:
    history = load_history("cmapss_fd001_adaptive")
    epochs = history["epoch"]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(epochs, history["train_macro_f1"], marker="o", label="Train macro F1", color="#1f77b4")
    ax1.plot(epochs, history["val_macro_f1"], marker="o", label="Validation macro F1", color="#2ca02c")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Macro F1")
    ax1.set_ylim(0.5, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_avg_steps"], marker="s", linestyle="--", label="Validation avg depth", color="#d62728")
    ax2.set_ylabel("Average validation depth")
    ax2.set_ylim(0.0, 6.5)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    ax1.set_title("Adaptive Model Learning Curve and Depth Usage")

    fig.tight_layout()
    fig.savefig(OUTPUT / "adaptive_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix(run_name: str, title: str, output_name: str) -> None:
    matrix = json.loads((ARTIFACTS / run_name / "test_confusion_matrix.json").read_text())
    matrix_arr = np.asarray(matrix)
    labels = ["Normal", "Warning", "Critical"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(matrix_arr.shape[0]):
        for j in range(matrix_arr.shape[1]):
            ax.text(j, i, str(matrix_arr[i, j]), ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output()
    metric_comparison()
    adaptive_history()
    confusion_matrix("cmapss_fd001_adaptive", "Adaptive Model Test Confusion Matrix", "adaptive_confusion.png")


if __name__ == "__main__":
    main()
