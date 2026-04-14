from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig, TrainConfig
from .data import (
    TelemetryWindowDataset,
    apply_standardizer,
    build_windows,
    fit_standardizer,
    load_cmapss_train_test,
    load_lbnl_fcu_dataset,
    load_ims_run,
    load_paderborn_dataset,
    load_telemetry_csv,
    split_windowed_by_asset,
)
from .models import AdaptiveLoopModel, BaselineClassifier, FixedDepthLoopModel

LABEL_NAMES = ["normal", "warning", "critical"]
ACTION_NAMES = ["monitor", "schedule_service", "inspect_urgent", "shutdown_now"]
SUBSYSTEM_NAMES = ["general", "thermal", "flow_path", "mechanical"]
HEALTH_LOSS_WEIGHT = 1.0
ACTION_LOSS_WEIGHT = 0.35
SUBSYSTEM_LOSS_WEIGHT = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name: str, config: ModelConfig) -> nn.Module:
    if name == "baseline":
        return BaselineClassifier(config)
    if name == "fixed":
        return FixedDepthLoopModel(config)
    if name == "adaptive":
        return AdaptiveLoopModel(config)
    raise ValueError(f"Unsupported model: {name}")


def metric_summary(preds: np.ndarray, targets: np.ndarray, labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(targets, preds, average="weighted", labels=labels, zero_division=0),
    }


def task_payload(
    preds: np.ndarray,
    targets: np.ndarray,
    labels: list[int],
    target_names: list[str],
) -> dict[str, Any]:
    metrics = metric_summary(preds, targets, labels)
    return {
        "metrics": metrics,
        "report": classification_report(
            targets,
            preds,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),
    }


def maybe_sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 16,
    warmup_batches: int = 3,
) -> dict[str, float]:
    model.eval()
    raw_batch_timings_ms: list[float] = []
    raw_batch_sizes: list[int] = []

    with torch.no_grad():
        for batch in loader:
            features = batch[0].to(device)
            maybe_sync_device(device)
            started = time.perf_counter()
            _ = model(features)
            maybe_sync_device(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            raw_batch_timings_ms.append(elapsed_ms)
            raw_batch_sizes.append(int(features.size(0)))

            if len(raw_batch_timings_ms) >= max_batches:
                break

    if not raw_batch_timings_ms:
        return {
            "batches_measured": 0,
            "samples_measured": 0,
            "avg_batch_latency_ms": 0.0,
            "p50_batch_latency_ms": 0.0,
            "p95_batch_latency_ms": 0.0,
            "avg_sample_latency_ms": 0.0,
            "p95_sample_latency_ms": 0.0,
            "throughput_samples_per_s": 0.0,
        }

    warmup_to_skip = min(warmup_batches, max(len(raw_batch_timings_ms) - 1, 0))
    batch_timings_ms = raw_batch_timings_ms[warmup_to_skip:]
    batch_sizes = raw_batch_sizes[warmup_to_skip:]
    sample_timings_ms = [elapsed / max(size, 1) for elapsed, size in zip(batch_timings_ms, batch_sizes)]
    sample_count = sum(batch_sizes)

    if not batch_timings_ms:
        return {
            "batches_measured": 0,
            "samples_measured": 0,
            "avg_batch_latency_ms": 0.0,
            "p50_batch_latency_ms": 0.0,
            "p95_batch_latency_ms": 0.0,
            "avg_sample_latency_ms": 0.0,
            "p95_sample_latency_ms": 0.0,
            "throughput_samples_per_s": 0.0,
        }

    batch_arr = np.asarray(batch_timings_ms, dtype=np.float64)
    sample_arr = np.asarray(sample_timings_ms, dtype=np.float64)
    throughput = float(sample_count / (batch_arr.sum() / 1000.0)) if batch_arr.sum() > 0 else 0.0
    return {
        "batches_measured": int(len(batch_timings_ms)),
        "samples_measured": int(sample_count),
        "avg_batch_latency_ms": float(batch_arr.mean()),
        "p50_batch_latency_ms": float(np.percentile(batch_arr, 50)),
        "p95_batch_latency_ms": float(np.percentile(batch_arr, 95)),
        "avg_sample_latency_ms": float(sample_arr.mean()),
        "p95_sample_latency_ms": float(np.percentile(sample_arr, 95)),
        "throughput_samples_per_s": throughput,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    health_criterion: nn.Module,
    action_criterion: nn.Module,
    subsystem_criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    health_preds: list[np.ndarray] = []
    health_targets: list[np.ndarray] = []
    action_preds: list[np.ndarray] = []
    action_targets: list[np.ndarray] = []
    subsystem_preds: list[np.ndarray] = []
    subsystem_targets: list[np.ndarray] = []
    step_traces: list[np.ndarray] = []

    for features, labels, action_labels, subsystem_labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        action_labels = action_labels.to(device)
        subsystem_labels = subsystem_labels.to(device)

        outputs = model(features)
        health_logits = outputs["logits"]
        action_logits = outputs["action_logits"]
        subsystem_logits = outputs["subsystem_logits"]

        health_loss = health_criterion(health_logits, labels)
        action_loss = action_criterion(action_logits, action_labels)
        subsystem_loss = subsystem_criterion(subsystem_logits, subsystem_labels)
        loss = (
            HEALTH_LOSS_WEIGHT * health_loss
            + ACTION_LOSS_WEIGHT * action_loss
            + SUBSYSTEM_LOSS_WEIGHT * subsystem_loss
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        health_preds.append(health_logits.argmax(dim=-1).detach().cpu().numpy())
        health_targets.append(labels.detach().cpu().numpy())
        action_preds.append(action_logits.argmax(dim=-1).detach().cpu().numpy())
        action_targets.append(action_labels.detach().cpu().numpy())
        subsystem_preds.append(subsystem_logits.argmax(dim=-1).detach().cpu().numpy())
        subsystem_targets.append(subsystem_labels.detach().cpu().numpy())

        if "steps" in outputs:
            step_traces.append(outputs["steps"].detach().cpu().numpy())

    steps = np.concatenate(step_traces) if step_traces else np.asarray([], dtype=np.int64)
    health_preds_arr = np.concatenate(health_preds)
    health_targets_arr = np.concatenate(health_targets)
    action_preds_arr = np.concatenate(action_preds)
    action_targets_arr = np.concatenate(action_targets)
    subsystem_preds_arr = np.concatenate(subsystem_preds)
    subsystem_targets_arr = np.concatenate(subsystem_targets)

    mean_loss = total_loss / len(loader.dataset)
    return {
        "loss": mean_loss,
        "health_preds": health_preds_arr,
        "health_targets": health_targets_arr,
        "action_preds": action_preds_arr,
        "action_targets": action_targets_arr,
        "subsystem_preds": subsystem_preds_arr,
        "subsystem_targets": subsystem_targets_arr,
        "steps": steps,
    }


def save_metrics(
    output_dir: Path,
    split_name: str,
    epoch_result: dict[str, Any],
    latency: dict[str, float],
) -> dict[str, Any]:
    health_payload = task_payload(epoch_result["health_preds"], epoch_result["health_targets"], [0, 1, 2], LABEL_NAMES)
    action_payload = task_payload(
        epoch_result["action_preds"],
        epoch_result["action_targets"],
        [0, 1, 2, 3],
        ACTION_NAMES,
    )
    subsystem_payload = task_payload(
        epoch_result["subsystem_preds"],
        epoch_result["subsystem_targets"],
        [0, 1, 2, 3],
        SUBSYSTEM_NAMES,
    )

    metrics = dict(health_payload["metrics"])
    metrics["loss"] = epoch_result["loss"]
    metrics["num_examples"] = int(len(epoch_result["health_targets"]))
    metrics["action_accuracy"] = action_payload["metrics"]["accuracy"]
    metrics["action_macro_f1"] = action_payload["metrics"]["macro_f1"]
    metrics["subsystem_accuracy"] = subsystem_payload["metrics"]["accuracy"]
    metrics["subsystem_macro_f1"] = subsystem_payload["metrics"]["macro_f1"]
    metrics.update(latency)

    steps = epoch_result["steps"]
    if len(steps):
        metrics["avg_steps"] = float(steps.mean())
        metrics["max_steps"] = int(steps.max())
    else:
        metrics["avg_steps"] = 0.0
        metrics["max_steps"] = 0

    (output_dir / f"{split_name}_classification_report.txt").write_text(health_payload["report"] + "\n")
    (output_dir / f"{split_name}_confusion_matrix.json").write_text(json.dumps(health_payload["confusion_matrix"], indent=2))
    (output_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / f"{split_name}_action_metrics.json").write_text(json.dumps(action_payload, indent=2))
    (output_dir / f"{split_name}_subsystem_metrics.json").write_text(json.dumps(subsystem_payload, indent=2))
    (output_dir / f"{split_name}_latency.json").write_text(json.dumps(latency, indent=2))
    return metrics


def save_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    lines = [
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,train_macro_f1,val_macro_f1,val_action_accuracy,val_subsystem_accuracy,val_avg_steps"
    ]
    for row in history:
        lines.append(
            ",".join(
                [
                    str(int(row["epoch"])),
                    f"{row['train_loss']:.6f}",
                    f"{row['val_loss']:.6f}",
                    f"{row['train_accuracy']:.6f}",
                    f"{row['val_accuracy']:.6f}",
                    f"{row['train_macro_f1']:.6f}",
                    f"{row['val_macro_f1']:.6f}",
                    f"{row['val_action_accuracy']:.6f}",
                    f"{row['val_subsystem_accuracy']:.6f}",
                    f"{row['val_avg_steps']:.6f}",
                ]
            )
        )
    (output_dir / "history.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predictive-maintenance models.")
    parser.add_argument(
        "--dataset",
        choices=["csv", "cmapss", "ims", "hvac", "paderborn"],
        default="cmapss",
        help="Dataset loader to use.",
    )
    parser.add_argument("--data-path", help="CSV file with telemetry rows.")
    parser.add_argument(
        "--cmapss-root",
        default="CMAPSSData",
        help="Directory containing CMAPSS train/test files.",
    )
    parser.add_argument(
        "--cmapss-subset",
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="CMAPSS subset to train on.",
    )
    parser.add_argument(
        "--ims-root",
        default="IMS_extracted",
        help="Directory containing extracted IMS test folders.",
    )
    parser.add_argument(
        "--ims-run",
        default="1st_test",
        help="IMS run directory to load after extraction.",
    )
    parser.add_argument(
        "--ims-file-step",
        type=int,
        default=1,
        help="Use every Nth IMS snapshot file to reduce preprocessing cost on very long runs.",
    )
    parser.add_argument(
        "--hvac-root",
        default="LBNL_FDD_Dataset_FCU",
        help="Directory containing LBNL HVAC fault CSV files.",
    )
    parser.add_argument(
        "--hvac-pattern",
        default="*.csv",
        help="Glob pattern for HVAC CSV files inside --hvac-root.",
    )
    parser.add_argument(
        "--hvac-row-step",
        type=int,
        default=60,
        help="Use every Nth HVAC row to reduce minute-level sequence volume.",
    )
    parser.add_argument(
        "--hvac-max-files",
        type=int,
        default=0,
        help="Optional limit on the number of HVAC CSV files to load. Use 0 for all files.",
    )
    parser.add_argument(
        "--paderborn-zip",
        default="data_downloads/paderborn/paderborn-db.zip",
        help="Path to the zipped Paderborn bearing dataset.",
    )
    parser.add_argument(
        "--paderborn-sample-stride",
        type=int,
        default=256,
        help="Use every Nth raw sample within each Paderborn measurement.",
    )
    parser.add_argument(
        "--paderborn-max-measurements-per-bearing",
        type=int,
        default=4,
        help="Optional limit on measurements loaded per Paderborn bearing. Use 0 for all measurements.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "fixed", "adaptive"],
        default="adaptive",
        help="Model family to train.",
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-loops", type=int, default=6)
    parser.add_argument("--exit-threshold", type=float, default=0.8)
    parser.add_argument("--warning-rul", type=int, default=50)
    parser.add_argument("--critical-rul", type=int, default=15)
    parser.add_argument("--shutdown-rul", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/latest", help="Directory for metrics and checkpoints.")
    parser.add_argument(
        "--single-asset-split",
        choices=["temporal", "stratified", "stage_temporal", "window_stratified", "asset_label_stratified"],
        default="temporal",
        help="Validation split mode when a dataset contains only one asset/run.",
    )
    parser.add_argument(
        "--latency-batches",
        type=int,
        default=16,
        help="Maximum batches to time for each latency benchmark.",
    )
    args = parser.parse_args()

    data_config = DataConfig(
        window_size=args.window_size,
        stride=args.stride,
        warning_rul=args.warning_rul,
        critical_rul=args.critical_rul,
        shutdown_rul=args.shutdown_rul,
    )
    train_config = TrainConfig(batch_size=args.batch_size, epochs=args.epochs)
    set_seed(train_config.seed)

    split_mode = args.single_asset_split
    if args.dataset == "ims" and split_mode == "temporal":
        split_mode = "stage_temporal"
        print("ims_single_asset_split=stage_temporal")
    if args.dataset == "hvac" and split_mode == "temporal":
        split_mode = "asset_label_stratified"
        print("hvac_split_mode=asset_label_stratified")
    if args.dataset == "paderborn" and split_mode == "temporal":
        split_mode = "asset_label_stratified"
        print("paderborn_split_mode=asset_label_stratified")

    if args.dataset == "csv":
        if not args.data_path:
            raise ValueError("--data-path is required when --dataset csv is used.")
        df = load_telemetry_csv(args.data_path, data_config)
        windowed = build_windows(df, data_config)
        train_data, val_data = split_windowed_by_asset(windowed, train_config.val_ratio, train_config.seed, split_mode)
        test_data = None
    elif args.dataset == "hvac":
        df = load_lbnl_fcu_dataset(
            args.hvac_root,
            data_config,
            pattern=args.hvac_pattern,
            row_step=args.hvac_row_step,
            max_files=None if args.hvac_max_files <= 0 else args.hvac_max_files,
        )
        windowed = build_windows(df, data_config)
        train_data, val_data = split_windowed_by_asset(windowed, train_config.val_ratio, train_config.seed, split_mode)
        test_data = None
    elif args.dataset == "paderborn":
        df = load_paderborn_dataset(
            args.paderborn_zip,
            data_config,
            sample_stride=args.paderborn_sample_stride,
            max_measurements_per_bearing=(
                None
                if args.paderborn_max_measurements_per_bearing <= 0
                else args.paderborn_max_measurements_per_bearing
            ),
        )
        windowed = build_windows(df, data_config)
        train_data, val_data = split_windowed_by_asset(windowed, train_config.val_ratio, train_config.seed, split_mode)
        test_data = None
    elif args.dataset == "ims":
        df = load_ims_run(args.ims_root, args.ims_run, data_config, file_step=args.ims_file_step)
        windowed = build_windows(df, data_config)
        train_data, val_data = split_windowed_by_asset(windowed, train_config.val_ratio, train_config.seed, split_mode)
        test_data = None
    else:
        train_df, test_df = load_cmapss_train_test(args.cmapss_root, args.cmapss_subset, data_config)
        train_windowed = build_windows(train_df, data_config)
        test_data = build_windows(test_df, data_config)
        train_data, val_data = split_windowed_by_asset(
            train_windowed, train_config.val_ratio, train_config.seed, split_mode
        )

    mean, std = fit_standardizer(train_data.features)
    train_data.features = apply_standardizer(train_data.features, mean, std)
    val_data.features = apply_standardizer(val_data.features, mean, std)
    if test_data is not None:
        test_data.features = apply_standardizer(test_data.features, mean, std)

    train_loader = DataLoader(TelemetryWindowDataset(train_data), batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(TelemetryWindowDataset(val_data), batch_size=train_config.batch_size, shuffle=False)
    test_loader = (
        DataLoader(TelemetryWindowDataset(test_data), batch_size=train_config.batch_size, shuffle=False)
        if test_data is not None
        else None
    )

    model_config = ModelConfig(
        input_dim=train_data.features.shape[-1],
        hidden_dim=args.hidden_dim,
        max_loops=args.max_loops,
        exit_threshold=args.exit_threshold,
    )

    device = resolve_device()
    model = build_model(args.model, model_config).to(device)
    health_criterion = nn.CrossEntropyLoss()
    action_criterion = nn.CrossEntropyLoss()
    subsystem_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0

    for epoch in range(1, train_config.epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            health_criterion,
            action_criterion,
            subsystem_criterion,
            optimizer,
            device,
        )
        val_result = run_epoch(
            model,
            val_loader,
            health_criterion,
            action_criterion,
            subsystem_criterion,
            None,
            device,
        )

        train_metrics = metric_summary(train_result["health_preds"], train_result["health_targets"], [0, 1, 2])
        val_metrics = metric_summary(val_result["health_preds"], val_result["health_targets"], [0, 1, 2])
        val_action_metrics = metric_summary(val_result["action_preds"], val_result["action_targets"], [0, 1, 2, 3])
        val_subsystem_metrics = metric_summary(
            val_result["subsystem_preds"], val_result["subsystem_targets"], [0, 1, 2, 3]
        )
        val_avg_steps = float(val_result["steps"].mean()) if len(val_result["steps"]) else 0.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_action_accuracy": val_action_metrics["accuracy"],
                "val_subsystem_accuracy": val_subsystem_metrics["accuracy"],
                "val_avg_steps": val_avg_steps,
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_action_acc={val_action_metrics['accuracy']:.4f} "
            f"val_subsystem_acc={val_subsystem_metrics['accuracy']:.4f}"
        )
        if len(val_result["steps"]):
            print(
                f"avg_validation_steps={val_result['steps'].mean():.2f} "
                f"max_validation_steps={val_result['steps'].max()}"
            )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config.__dict__,
                    "model_kind": args.model,
                    "data_config": data_config.__dict__,
                    "feature_names": train_data.feature_names,
                    "normalization_mean": mean.tolist(),
                    "normalization_std": std.tolist(),
                },
                output_dir / "best_model.pt",
            )

    save_history(output_dir, history)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_result = run_epoch(
        model,
        val_loader,
        health_criterion,
        action_criterion,
        subsystem_criterion,
        None,
        device,
    )
    val_latency = benchmark_latency(model, val_loader, device, max_batches=args.latency_batches)
    val_metrics = save_metrics(output_dir, "validation", val_result, val_latency)

    print("\nValidation health report:")
    print(
        classification_report(
            val_result["health_targets"],
            val_result["health_preds"],
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print(json.dumps(val_metrics, indent=2))

    if test_loader is not None:
        test_result = run_epoch(
            model,
            test_loader,
            health_criterion,
            action_criterion,
            subsystem_criterion,
            None,
            device,
        )
        test_latency = benchmark_latency(model, test_loader, device, max_batches=args.latency_batches)
        test_metrics = save_metrics(output_dir, "test", test_result, test_latency)
        print("\nTest health report:")
        print(
            classification_report(
                test_result["health_targets"],
                test_result["health_preds"],
                labels=[0, 1, 2],
                target_names=LABEL_NAMES,
                digits=4,
                zero_division=0,
            )
        )
        print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
