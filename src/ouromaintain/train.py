from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
    load_cmapss_subset,
    load_ims_run,
    load_telemetry_csv,
    split_windowed_by_asset,
)
from .models import AdaptiveLoopModel, BaselineClassifier, FixedDepthLoopModel

LABEL_NAMES = ["normal", "warning", "critical"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name: str, config: ModelConfig) -> nn.Module:
    if name == "baseline":
        return BaselineClassifier(config)
    if name == "fixed":
        return FixedDepthLoopModel(config)
    if name == "adaptive":
        return AdaptiveLoopModel(config)
    raise ValueError(f"Unsupported model: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    step_traces: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        logits = outputs["logits"]
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

        if "steps" in outputs:
            step_traces.append(outputs["steps"].detach().cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    if step_traces:
        step_array = np.concatenate(step_traces)
    else:
        step_array = np.asarray([], dtype=np.int64)

    return (mean_loss, np.concatenate(all_preds), np.concatenate(all_targets), step_array)


def summarize_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro", labels=[0, 1, 2], zero_division=0),
        "weighted_f1": f1_score(
            targets,
            preds,
            average="weighted",
            labels=[0, 1, 2],
            zero_division=0,
        ),
    }


def save_metrics(
    output_dir: Path,
    split_name: str,
    loss: float,
    preds: np.ndarray,
    targets: np.ndarray,
    steps: np.ndarray,
) -> dict[str, float]:
    metrics = summarize_metrics(preds, targets)
    metrics["loss"] = loss
    metrics["num_examples"] = int(len(targets))

    if len(steps):
        metrics["avg_steps"] = float(steps.mean())
        metrics["max_steps"] = int(steps.max())
    else:
        metrics["avg_steps"] = 0.0
        metrics["max_steps"] = 0

    report = classification_report(
        targets,
        preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, preds, labels=[0, 1, 2]).tolist()

    (output_dir / f"{split_name}_classification_report.txt").write_text(report + "\n")
    (output_dir / f"{split_name}_confusion_matrix.json").write_text(json.dumps(matrix, indent=2))
    (output_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def save_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    lines = [
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,train_macro_f1,val_macro_f1,val_avg_steps"
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
                    f"{row['val_avg_steps']:.6f}",
                ]
            )
        )
    (output_dir / "history.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predictive-maintenance models.")
    parser.add_argument(
        "--dataset",
        choices=["csv", "cmapss", "ims"],
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
    parser.add_argument("--output-dir", default="artifacts/latest", help="Directory for metrics and checkpoints.")
    parser.add_argument(
        "--single-asset-split",
        choices=["temporal", "stratified"],
        default="temporal",
        help="Validation split mode when a dataset contains only one asset/run.",
    )
    args = parser.parse_args()

    data_config = DataConfig(
        window_size=args.window_size,
        stride=args.stride,
        warning_rul=args.warning_rul,
        critical_rul=args.critical_rul,
    )
    train_config = TrainConfig(batch_size=args.batch_size, epochs=args.epochs)
    set_seed(train_config.seed)

    if args.dataset == "csv":
        if not args.data_path:
            raise ValueError("--data-path is required when --dataset csv is used.")
        df = load_telemetry_csv(args.data_path, data_config)
        windowed = build_windows(df, data_config)
        train_data, val_data = split_windowed_by_asset(
            windowed,
            train_config.val_ratio,
            train_config.seed,
            single_asset_mode=args.single_asset_split,
        )
        test_data = None
    elif args.dataset == "ims":
        df = load_ims_run(args.ims_root, args.ims_run, data_config)
        windowed = build_windows(df, data_config)
        train_data, val_data = split_windowed_by_asset(
            windowed,
            train_config.val_ratio,
            train_config.seed,
            single_asset_mode=args.single_asset_split,
        )
        test_data = None
    else:
        train_df, test_df = load_cmapss_train_test(args.cmapss_root, args.cmapss_subset, data_config)
        train_windowed = build_windows(train_df, data_config)
        test_data = build_windows(test_df, data_config)
        train_data, val_data = split_windowed_by_asset(
            train_windowed, train_config.val_ratio, train_config.seed, single_asset_mode=args.single_asset_split
        )

    mean, std = fit_standardizer(train_data.features)
    train_data.features = apply_standardizer(train_data.features, mean, std)
    val_data.features = apply_standardizer(val_data.features, mean, std)
    if test_data is not None:
        test_data.features = apply_standardizer(test_data.features, mean, std)

    train_loader = DataLoader(
        TelemetryWindowDataset(train_data),
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TelemetryWindowDataset(val_data),
        batch_size=train_config.batch_size,
        shuffle=False,
    )
    test_loader = None
    if test_data is not None:
        test_loader = DataLoader(
            TelemetryWindowDataset(test_data),
            batch_size=train_config.batch_size,
            shuffle=False,
        )

    input_dim = train_data.features.shape[-1]
    model_config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        max_loops=args.max_loops,
        exit_threshold=args.exit_threshold,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, model_config).to(device)
    criterion = nn.CrossEntropyLoss()
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
        train_loss, train_preds, train_targets, train_steps = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_preds, val_targets, val_steps = run_epoch(
            model, val_loader, criterion, None, device
        )
        train_metrics = summarize_metrics(train_preds, train_targets)
        val_metrics = summarize_metrics(val_preds, val_targets)
        val_avg_steps = float(val_steps.mean()) if len(val_steps) else 0.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_avg_steps": val_avg_steps,
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if len(val_steps):
            print(f"avg_validation_steps={val_steps.mean():.2f} max_validation_steps={val_steps.max()}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config.__dict__,
                    "data_config": data_config.__dict__,
                    "feature_names": train_data.feature_names,
                    "normalization_mean": mean.tolist(),
                    "normalization_std": std.tolist(),
                },
                output_dir / "best_model.pt",
            )

    save_history(output_dir, history)
    val_metrics = save_metrics(output_dir, "validation", val_loss, val_preds, val_targets, val_steps)

    print("\nValidation metrics:")
    print(
        classification_report(
            val_targets,
            val_preds,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print(json.dumps(val_metrics, indent=2))

    if test_loader is not None:
        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_preds, test_targets, test_steps = run_epoch(
            model, test_loader, criterion, None, device
        )
        test_metrics = save_metrics(output_dir, "test", test_loss, test_preds, test_targets, test_steps)
        print("\nTest metrics:")
        print(
            classification_report(
                test_targets,
                test_preds,
                labels=[0, 1, 2],
                target_names=LABEL_NAMES,
                digits=4,
                zero_division=0,
            )
        )
        print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
