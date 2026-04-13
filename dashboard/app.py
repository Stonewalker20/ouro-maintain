from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CACHE_ROOT = Path("/tmp/ouromaintain-dashboard-cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
CACHE_DIR = ROOT / ".cache_local"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ouromaintain.config import DataConfig, ModelConfig  # type: ignore
from ouromaintain.data import (  # type: ignore
    WindowedData,
    apply_standardizer,
    build_windows,
    load_cmapss_train_test,
    load_telemetry_csv,
)
from ouromaintain.models import AdaptiveLoopModel, BaselineClassifier, FixedDepthLoopModel  # type: ignore


ARTIFACTS_DIR = ROOT / "artifacts"
CMAPSS_DIR = ROOT / "CMAPSSData"
APP_TITLE = "OuroMaintain Dashboard"
LABEL_NAMES = ["normal", "warning", "critical"]
PRIMARY_ORDER = ["cmapss_fd001_baseline", "cmapss_fd001_fixed", "cmapss_fd001_adaptive"]


@dataclass(frozen=True)
class ArtifactRecord:
    slug: str
    path: Path
    dataset: str
    run_name: str
    model_name: str
    metrics: dict[str, Any]
    history: pd.DataFrame | None
    test_confusion: list[list[int]] | None
    validation_confusion: list[list[int]] | None
    checkpoint: dict[str, Any] | None

    @property
    def display_name(self) -> str:
        return f"{self.dataset.upper()} / {self.run_name} / {self.model_name}"


def parse_slug(slug: str) -> tuple[str, str, str]:
    dataset = slug.split("_", 1)[0]
    if slug.endswith("_adaptive_v2"):
        return dataset, slug[len(dataset) + 1 : -len("_adaptive_v2")], "adaptive_v2"
    if slug.endswith("_adaptive"):
        return dataset, slug[len(dataset) + 1 : -len("_adaptive")], "adaptive"
    if slug.endswith("_fixed"):
        return dataset, slug[len(dataset) + 1 : -len("_fixed")], "fixed"
    if slug.endswith("_baseline"):
        return dataset, slug[len(dataset) + 1 : -len("_baseline")], "baseline"
    return dataset, "run", "model"


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_text(path: str) -> str:
    return Path(path).read_text()


@st.cache_data(show_spinner=False)
def discover_artifacts() -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    if not ARTIFACTS_DIR.exists():
        return records

    for path in sorted(p for p in ARTIFACTS_DIR.iterdir() if p.is_dir()):
        metrics_path = path / "test_metrics.json"
        validation_metrics_path = path / "validation_metrics.json"
        if not metrics_path.exists() and not validation_metrics_path.exists():
            continue

        dataset, run_name, model_name = parse_slug(path.name)
        metrics: dict[str, Any] = {}
        if validation_metrics_path.exists():
            metrics["validation"] = load_json(str(validation_metrics_path))
        if metrics_path.exists():
            metrics["test"] = load_json(str(metrics_path))

        history = load_csv(str(path / "history.csv")) if (path / "history.csv").exists() else None
        test_confusion = (
            load_json(str(path / "test_confusion_matrix.json")) if (path / "test_confusion_matrix.json").exists() else None
        )
        validation_confusion = (
            load_json(str(path / "validation_confusion_matrix.json"))
            if (path / "validation_confusion_matrix.json").exists()
            else None
        )
        checkpoint = torch.load(path / "best_model.pt", map_location="cpu") if (path / "best_model.pt").exists() else None
        records.append(
            ArtifactRecord(
                slug=path.name,
                path=path,
                dataset=dataset,
                run_name=run_name,
                model_name=model_name,
                metrics=metrics,
                history=history,
                test_confusion=test_confusion,
                validation_confusion=validation_confusion,
                checkpoint=checkpoint,
            )
        )
    return records


def metric_value(record: ArtifactRecord, split: str, name: str, default: float = float("nan")) -> float:
    split_metrics = record.metrics.get(split) or {}
    value = split_metrics.get(name, default)
    return float(value)


def artifact_lookup(records: list[ArtifactRecord]) -> dict[str, ArtifactRecord]:
    return {record.slug: record for record in records}


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def model_kind_from_slug(slug: str) -> str:
    if "adaptive" in slug:
        return "adaptive"
    if "fixed" in slug:
        return "fixed"
    return "baseline"


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[torch.nn.Module, ModelConfig, np.ndarray, np.ndarray]:
    model_config = ModelConfig(**checkpoint["model_config"])
    model_kind = checkpoint.get("model_kind") or "adaptive"
    if model_kind == "baseline":
        model = BaselineClassifier(model_config)
    elif model_kind == "fixed":
        model = FixedDepthLoopModel(model_config)
    else:
        model = AdaptiveLoopModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    return model, model_config, mean, std


def inject_model_kind(record: ArtifactRecord) -> ArtifactRecord:
    if record.checkpoint is None:
        return record
    checkpoint = dict(record.checkpoint)
    checkpoint.setdefault("model_kind", model_kind_from_slug(record.slug))
    return ArtifactRecord(
        slug=record.slug,
        path=record.path,
        dataset=record.dataset,
        run_name=record.run_name,
        model_name=record.model_name,
        metrics=record.metrics,
        history=record.history,
        test_confusion=record.test_confusion,
        validation_confusion=record.validation_confusion,
        checkpoint=checkpoint,
    )


def available_primary_records(records: list[ArtifactRecord]) -> list[ArtifactRecord]:
    lookup = artifact_lookup(records)
    ordered = [lookup[key] for key in PRIMARY_ORDER if key in lookup]
    if ordered:
        return ordered
    return [record for record in records if record.dataset == "cmapss"]


def pretty_metric(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if abs(value) < 10:
        return f"{value:.4f}"
    return f"{value:.2f}"


def render_metric_cards(records: list[ArtifactRecord]) -> None:
    lookup = artifact_lookup(records)
    adaptive = lookup.get("cmapss_fd001_adaptive")
    fixed = lookup.get("cmapss_fd001_fixed")
    baseline = lookup.get("cmapss_fd001_baseline")
    if not adaptive or not fixed or not baseline:
        return

    test_adaptive = adaptive.metrics.get("test", {})
    test_fixed = fixed.metrics.get("test", {})
    test_baseline = baseline.metrics.get("test", {})
    speedup = 1.0 - float(test_adaptive.get("avg_steps", 0.0)) / max(float(test_fixed.get("avg_steps", 1.0)), 1e-8)
    f1_gain = float(test_adaptive.get("macro_f1", 0.0)) - float(test_fixed.get("macro_f1", 0.0))
    accuracy_gain = float(test_adaptive.get("accuracy", 0.0)) - float(test_baseline.get("accuracy", 0.0))

    cols = st.columns(4)
    cols[0].metric("Adaptive test macro F1", pretty_metric(float(test_adaptive.get("macro_f1", float("nan")))))
    cols[1].metric("Depth reduction vs fixed", f"{speedup * 100:.1f}%")
    cols[2].metric("Macro F1 lift vs fixed", f"{f1_gain:+.4f}")
    cols[3].metric("Accuracy lift vs baseline", f"{accuracy_gain:+.4f}")


def benchmark_table(records: list[ArtifactRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        split = "test" if "test" in record.metrics else "validation"
        metrics = record.metrics.get(split, {})
        rows.append(
            {
                "dataset": record.dataset,
                "run": record.run_name,
                "model": record.model_name,
                "split": split,
                "accuracy": metrics.get("accuracy", np.nan),
                "macro_f1": metrics.get("macro_f1", np.nan),
                "weighted_f1": metrics.get("weighted_f1", np.nan),
                "avg_steps": metrics.get("avg_steps", np.nan),
                "max_steps": metrics.get("max_steps", np.nan),
                "examples": metrics.get("num_examples", np.nan),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["dataset", "run", "model"]).reset_index(drop=True)
    return frame


def plot_history(history: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    fig1, ax1 = plt.subplots(figsize=(7.2, 3.6))
    ax1.plot(history["epoch"], history["train_macro_f1"], label="train macro F1", color="#2a6f97", linewidth=2)
    ax1.plot(history["epoch"], history["val_macro_f1"], label="val macro F1", color="#f77f00", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Macro F1")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False)
    ax1.set_title("Training curve")

    fig2, ax2 = plt.subplots(figsize=(7.2, 3.6))
    ax2.plot(history["epoch"], history["val_avg_steps"], label="validation avg steps", color="#8a5cf6", linewidth=2)
    ax2.plot(history["epoch"], history["val_loss"], label="validation loss", color="#6c757d", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False)
    ax2.set_title("Compute and loss")
    return fig1, fig2


def plot_confusion(matrix: list[list[int]], title: str) -> plt.Figure:
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=20)
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="#0b1320", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def load_cmapss_split(subset: str, split: str) -> pd.DataFrame:
    data_config = DataConfig()
    train_df, test_df = load_cmapss_train_test(str(CMAPSS_DIR), subset, data_config)
    return train_df if split == "train" else test_df


def window_preview(df: pd.DataFrame, asset_id: int, window_index: int, window_size: int, stride: int) -> tuple[pd.DataFrame, pd.DataFrame, slice]:
    asset_df = df[df["asset_id"] == asset_id].sort_values("timestamp").reset_index(drop=True)
    start = window_index * stride
    end = start + window_size
    if end > len(asset_df):
        raise ValueError("Selected window index is out of range for the chosen asset.")
    return asset_df, asset_df.iloc[start:end].copy(), slice(start, end)


def model_trace(model: torch.nn.Module, sample: torch.Tensor, model_name: str) -> dict[str, Any]:
    if isinstance(model, BaselineClassifier):
        with torch.no_grad():
            outputs = model(sample)
            probs = torch.softmax(outputs["logits"], dim=-1)
            confidence, pred = probs.max(dim=-1)
        return {
            "predicted_class": int(pred.item()),
            "confidence": float(confidence.item()),
            "steps": 0,
            "step_confidences": [],
            "step_predictions": [],
            "logits": outputs["logits"].detach().cpu().numpy()[0].tolist(),
        }

    if isinstance(model, FixedDepthLoopModel):
        context = model.encoder(sample)
        h = context
        confidences: list[float] = []
        predictions: list[int] = []
        logits_list: list[list[float]] = []
        with torch.no_grad():
            for _ in range(model.max_loops):
                h = model.loop(h, context)
                logits = model.classifier(h)
                probs = torch.softmax(logits, dim=-1)
                confidence, pred = probs.max(dim=-1)
                confidences.append(float(confidence.item()))
                predictions.append(int(pred.item()))
                logits_list.append(logits.detach().cpu().numpy()[0].tolist())
        return {
            "predicted_class": predictions[-1],
            "confidence": confidences[-1],
            "steps": model.max_loops,
            "step_confidences": confidences,
            "step_predictions": predictions,
            "logits": logits_list[-1],
        }

    if isinstance(model, AdaptiveLoopModel):
        context = model.encoder(sample)
        h = context
        confidences: list[float] = []
        predictions: list[int] = []
        logits_list: list[list[float]] = []
        exit_step = model.max_loops
        with torch.no_grad():
            for step_idx in range(1, model.max_loops + 1):
                h = model.loop(h, context)
                logits = model.classifier(h)
                probs = torch.softmax(logits, dim=-1)
                confidence, pred = probs.max(dim=-1)
                confidences.append(float(confidence.item()))
                predictions.append(int(pred.item()))
                logits_list.append(logits.detach().cpu().numpy()[0].tolist())
                if float(confidence.item()) >= model.exit_threshold:
                    exit_step = step_idx
                    break
        return {
            "predicted_class": predictions[-1],
            "confidence": confidences[-1],
            "steps": exit_step,
            "step_confidences": confidences,
            "step_predictions": predictions,
            "logits": logits_list[-1],
            "exit_threshold": model.exit_threshold,
        }

    raise TypeError(f"Unsupported model type: {type(model)!r}")


def prepare_sample_tensor(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    standardized = apply_standardizer(window[None, :, :], mean, std)
    return torch.tensor(standardized, dtype=torch.float32)


def infer_labels(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError("Label column must be numeric for custom CSV uploads.")
    return values.astype(int)


def render_window_plot(window: pd.DataFrame, feature_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    channels = feature_names[: min(6, len(feature_names))]
    for name in channels:
        ax.plot(window["timestamp"], window[name], label=name, linewidth=1.8)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Raw signal")
    ax.set_title("Selected telemetry window")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def render_heatmap(window: np.ndarray, feature_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    im = ax.imshow(window.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Cycle in window")
    ax.set_title("Standardized model input")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="O", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255, 189, 89, 0.14), transparent 28%),
                radial-gradient(circle at left bottom, rgba(42, 111, 151, 0.12), transparent 26%),
                linear-gradient(180deg, #f7f8fb 0%, #eef2f8 100%);
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: rgba(11, 19, 32, 0.94);
            color: #f7f8fb;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 50px rgba(11, 19, 32, 0.15);
        }
        .hero h1, .hero p { margin: 0; }
        .hero p { margin-top: 0.35rem; color: #c7d3e3; }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 10px 30px rgba(11, 19, 32, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    records = [inject_model_kind(record) for record in discover_artifacts()]
    benchmark_df = benchmark_table(records)
    primary_records = available_primary_records(records)
    lookup = artifact_lookup(records)

    st.markdown(
        """
        <div class="hero">
          <h1>OuroMaintain Dashboard</h1>
          <p>Benchmark comparison, CMAPSS sample inspection, and adaptive-depth tracing for the trained looped models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if benchmark_df.empty:
        st.error("No artifact directories with metrics were found under `artifacts/`.")
        st.stop()

    render_metric_cards(records)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Benchmark comparison")
        st.caption("Primary comparison is the C-MAPSS FD001 trio. Secondary runs are included when present.")
        st.dataframe(
            benchmark_df,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Adaptive depth")
        adaptive = lookup.get("cmapss_fd001_adaptive")
        fixed = lookup.get("cmapss_fd001_fixed")
        if adaptive and fixed:
            adaptive_test = adaptive.metrics.get("test", {})
            fixed_test = fixed.metrics.get("test", {})
            reduction = 1.0 - float(adaptive_test.get("avg_steps", 0.0)) / max(float(fixed_test.get("avg_steps", 1.0)), 1e-8)
            st.metric("Average loop reduction", f"{reduction * 100:.1f}%")
            st.write(
                "The adaptive model keeps refining a latent state until its confidence crosses the exit threshold or it reaches the maximum depth. "
                "Easy windows usually exit early; ambiguous windows consume more loop steps."
            )
            st.write(
                f"On FD001 test data, the adaptive run used {adaptive_test.get('avg_steps', 0.0):.2f} average steps versus "
                f"{fixed_test.get('avg_steps', 0.0):.2f} for the fixed-depth loop."
            )
        else:
            st.info("Primary adaptive and fixed CMAPSS artifacts were not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Artifact details", "CMAPSS sample inspector", "Custom CSV", "Architecture notes"])

    with tabs[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        selection = st.selectbox(
            "Choose an artifact bundle",
            options=[record.slug for record in records],
            index=[record.slug for record in records].index("cmapss_fd001_adaptive")
            if "cmapss_fd001_adaptive" in [record.slug for record in records]
            else 0,
        )
        record = lookup[selection]
        split_name = "test" if "test" in record.metrics else "validation"
        metrics = record.metrics.get(split_name, {})
        cols = st.columns(4)
        cols[0].metric("Accuracy", pretty_metric(float(metrics.get("accuracy", np.nan))))
        cols[1].metric("Macro F1", pretty_metric(float(metrics.get("macro_f1", np.nan))))
        cols[2].metric("Weighted F1", pretty_metric(float(metrics.get("weighted_f1", np.nan))))
        cols[3].metric("Avg steps", pretty_metric(float(metrics.get("avg_steps", np.nan))))

        hist = record.history
        if hist is not None and not hist.empty:
            fig1, fig2 = plot_history(hist)
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig1, use_container_width=True)
            with c2:
                st.pyplot(fig2, use_container_width=True)

        matrix = record.test_confusion if split_name == "test" else record.validation_confusion
        if matrix is not None:
            st.pyplot(plot_confusion(matrix, f"{record.slug} {split_name} confusion matrix"), use_container_width=False)
        report_path = record.path / f"{split_name}_classification_report.txt"
        if report_path.exists():
            with st.expander(f"{split_name.title()} classification report"):
                st.code(load_text(str(report_path)), language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("CMAPSS window inspector")
        split_choice = st.radio("Dataset split", ["test", "train"], horizontal=True)
        subset = st.selectbox("Subset", ["FD001", "FD002", "FD003", "FD004"], index=0)
        cmapss_df = load_cmapss_split(subset, split_choice)
        engine_ids = sorted(cmapss_df["asset_id"].unique().tolist())
        engine_id = st.selectbox("Engine", engine_ids, index=0)
        window_size = st.slider("Window size", min_value=16, max_value=64, value=32, step=8)
        stride = st.slider("Stride", min_value=1, max_value=16, value=8, step=1)
        data_config = DataConfig(window_size=window_size, stride=stride)
        windowed = build_windows(cmapss_df, data_config)
        engine_mask = windowed.asset_ids == str(engine_id)
        engine_indices = np.where(engine_mask)[0]
        if len(engine_indices) == 0:
            st.warning("No windows were generated for the selected engine.")
            st.stop()
        window_idx = st.slider("Window index", min_value=0, max_value=len(engine_indices) - 1, value=0, step=1)
        selected_idx = engine_indices[window_idx]
        raw_asset_df, selected_window, window_slice = window_preview(cmapss_df, int(engine_id), window_idx, window_size, stride)
        st.caption(
            f"Engine {engine_id}, cycles {int(selected_window['timestamp'].iloc[0])} to {int(selected_window['timestamp'].iloc[-1])}, "
            f"window label {LABEL_NAMES[int(windowed.labels[selected_idx])]}"
        )

        primary_record = lookup.get("cmapss_fd001_adaptive")
        if not primary_record or primary_record.checkpoint is None:
            st.error("Adaptive CMAPSS checkpoint not found, so sample inference is unavailable.")
        else:
            model, model_config, mean, std = build_model_from_checkpoint(primary_record.checkpoint)
            sample_tensor = prepare_sample_tensor(selected_window[windowed.feature_names].to_numpy(dtype=np.float32), mean, std)
            trace = model_trace(model, sample_tensor, primary_record.slug)
            pred_name = LABEL_NAMES[trace["predicted_class"]]
            actual_name = LABEL_NAMES[int(windowed.labels[selected_idx])]
            delta = trace["steps"] - window_size if trace["steps"] else 0
            cols = st.columns(4)
            cols[0].metric("Predicted class", pred_name)
            cols[1].metric("Actual class", actual_name)
            cols[2].metric("Confidence", f"{trace['confidence']:.3f}")
            cols[3].metric("Loop steps", str(trace["steps"]))

            chart_left, chart_right = st.columns([1.25, 1.0])
            with chart_left:
                st.pyplot(render_window_plot(selected_window, windowed.feature_names), use_container_width=True)
            with chart_right:
                st.pyplot(render_heatmap(sample_tensor.squeeze(0).cpu().numpy(), windowed.feature_names), use_container_width=True)

            if trace["step_confidences"]:
                fig, ax = plt.subplots(figsize=(6.5, 3.4))
                ax.plot(range(1, len(trace["step_confidences"]) + 1), trace["step_confidences"], marker="o", color="#2a6f97")
                threshold = primary_record.checkpoint["model_config"]["exit_threshold"]
                ax.axhline(threshold, color="#d62828", linestyle="--", linewidth=1.6, label="exit threshold")
                ax.set_ylim(0.0, 1.05)
                ax.set_xlabel("Loop step")
                ax.set_ylabel("Confidence")
                ax.set_title("Adaptive depth trace")
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False)
                st.pyplot(fig, use_container_width=True)

            st.write(
                "The trace above explains the adaptive depth decision: the model refines the same latent representation repeatedly and stops "
                "once the max-softmax confidence reaches the learned threshold. If the selected window is ambiguous, the confidence curve stays lower "
                "for longer and the model spends more steps before exiting."
            )
            st.code(
                json.dumps(
                    {
                        "predicted_class": pred_name,
                        "actual_class": actual_name,
                        "loop_steps": trace["steps"],
                        "confidence": round(trace["confidence"], 4),
                        "window_start_cycle": int(selected_window["timestamp"].iloc[0]),
                        "window_end_cycle": int(selected_window["timestamp"].iloc[-1]),
                    },
                    indent=2,
                ),
                language="json",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Custom labeled CSV")
        st.write(
            "Upload a labeled telemetry CSV with `asset_id`, `timestamp`, `label`, and any number of numeric feature columns. "
            "The dashboard will build windows, standardize them, and try to run the adaptive CMAPSS checkpoint if the feature layout matches."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                custom_df = load_telemetry_csv(uploaded, DataConfig())
                st.success(f"Loaded {len(custom_df):,} rows from the uploaded CSV.")
                st.dataframe(custom_df.head(20), use_container_width=True)
                custom_windows = build_windows(custom_df, DataConfig())
                custom_assets = sorted(pd.unique(custom_windows.asset_ids).tolist())
                asset_id = st.selectbox("Uploaded asset", custom_assets)
                asset_indices = np.where(custom_windows.asset_ids == str(asset_id))[0]
                if len(asset_indices):
                    custom_window_idx = st.slider("Uploaded window index", 0, len(asset_indices) - 1, 0)
                    chosen_idx = asset_indices[custom_window_idx]
                    selected = custom_windows.features[chosen_idx]
                    st.write(
                        f"Window label: `{LABEL_NAMES[int(custom_windows.labels[chosen_idx])]}` from asset `{asset_id}`."
                    )
                    st.line_chart(
                        pd.DataFrame(selected[:, : min(6, selected.shape[1])], columns=custom_windows.feature_names[: min(6, selected.shape[1])])
                    )
                    primary_record = lookup.get("cmapss_fd001_adaptive")
                    if primary_record and primary_record.checkpoint is not None and len(custom_windows.feature_names) == len(primary_record.checkpoint["feature_names"]):
                        model, model_config, mean, std = build_model_from_checkpoint(primary_record.checkpoint)
                        sample_tensor = prepare_sample_tensor(selected.astype(np.float32), mean, std)
                        trace = model_trace(model, sample_tensor, primary_record.slug)
                        st.metric("Adaptive predicted class", LABEL_NAMES[trace["predicted_class"]])
                        st.metric("Adaptive loop steps", trace["steps"])
                    else:
                        st.info("The uploaded feature layout does not match the CMAPSS checkpoint, so inference is skipped.")
                else:
                    st.warning("No windows were produced for the chosen uploaded asset.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not process the uploaded CSV: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("How adaptive depth works here")
        st.markdown(
            """
            - The window encoder turns telemetry into a latent state.
            - The loop block reuses the same weights at every step, so the model spends compute only when the window remains uncertain.
            - The exit gate is confidence-based: once the softmax confidence crosses the threshold, the model stops looping.
            - Easy windows should show a short confidence trace and low average depth.
            - Hard or borderline windows should show a slower confidence ramp and more loop steps.
            """
        )
        st.write(
            "This dashboard reports average loop depth as the compute proxy because the trained artifacts do not store wall-clock latency. "
            "That makes the adaptive-vs-fixed comparison interpretable and reproducible across machines."
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
