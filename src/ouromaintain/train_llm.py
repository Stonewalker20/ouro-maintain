from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .config import DataConfig, TrainConfig
from .data import build_windows, load_cmapss_train_test, split_windowed_by_asset

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


def task_payload(
    preds: np.ndarray,
    targets: np.ndarray,
    labels: list[int],
    target_names: list[str],
) -> dict[str, Any]:
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(targets, preds, average="weighted", labels=labels, zero_division=0),
    }
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


def summarize_metrics(preds: np.ndarray, targets: np.ndarray, labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(targets, preds, average="weighted", labels=labels, zero_division=0),
    }


def maybe_sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def feature_alias(name: str) -> str:
    if name.startswith("op_setting_"):
        return f"op{name.split('_')[-1]}"
    if name.startswith("sensor_"):
        return f"s{name.split('_')[-1]}"
    return name.replace("_", "")


def selected_feature_indices(feature_names: list[str]) -> list[int]:
    preferred = {
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_8",
        "sensor_11",
        "sensor_12",
        "sensor_17",
        "sensor_20",
        "sensor_21",
    }
    selected = [idx for idx, name in enumerate(feature_names) if name in preferred]
    return selected or list(range(min(len(feature_names), 12)))


def serialize_window(window: np.ndarray, feature_names: list[str]) -> str:
    start = window[0]
    end = window[-1]
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    parts = ["predictive maintenance telemetry summary."]
    for idx in selected_feature_indices(feature_names):
        name = feature_names[idx]
        alias = feature_alias(name)
        parts.append(
            f"{alias} end {end[idx]:.2f} delta {(end[idx] - start[idx]):.2f} mean {mean[idx]:.2f} std {std[idx]:.2f}."
        )
    return " ".join(parts)


@dataclass
class TextWindowData:
    texts: list[str]
    labels: np.ndarray
    action_labels: np.ndarray
    subsystem_labels: np.ndarray


class TokenizedTelemetryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        action_labels: np.ndarray,
        subsystem_labels: np.ndarray,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.action_labels = torch.tensor(action_labels, dtype=torch.long)
        self.subsystem_labels = torch.tensor(subsystem_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["action_labels"] = self.action_labels[idx]
        item["subsystem_labels"] = self.subsystem_labels[idx]
        return item


class EmbeddedTelemetryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        action_labels: torch.Tensor,
        subsystem_labels: torch.Tensor,
    ) -> None:
        self.embeddings = embeddings
        self.labels = labels
        self.action_labels = action_labels
        self.subsystem_labels = subsystem_labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "embeddings": self.embeddings[idx],
            "labels": self.labels[idx],
            "action_labels": self.action_labels[idx],
            "subsystem_labels": self.subsystem_labels[idx],
        }


class MultitaskLLMBaseline(nn.Module):
    def __init__(self, backbone_name: str, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name, local_files_only=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        hidden = getattr(self.backbone.config, "hidden_size", None) or getattr(self.backbone.config, "dim")
        self.dropout = nn.Dropout(0.1)
        self.health_head = nn.Linear(hidden, 3)
        self.action_head = nn.Linear(hidden, 4)
        self.subsystem_head = nn.Linear(hidden, 4)

    def pooled(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def classify_embeddings(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.dropout(pooled)
        return {
            "logits": self.health_head(pooled),
            "action_logits": self.action_head(pooled),
            "subsystem_logits": self.subsystem_head(pooled),
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classify_embeddings(self.pooled(outputs))


def build_text_data(windowed: Any) -> TextWindowData:
    texts = [serialize_window(window, windowed.feature_names) for window in windowed.features]
    return TextWindowData(
        texts=texts,
        labels=windowed.labels,
        action_labels=windowed.action_labels,
        subsystem_labels=windowed.subsystem_labels,
    )


def benchmark_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 16,
    warmup_batches: int = 3,
) -> dict[str, float]:
    model.eval()
    raw_ms: list[float] = []
    raw_sizes: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            maybe_sync_device(device)
            started = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            maybe_sync_device(device)
            raw_ms.append((time.perf_counter() - started) * 1000.0)
            raw_sizes.append(int(input_ids.size(0)))
            if len(raw_ms) >= max_batches:
                break

    if not raw_ms:
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

    warmup_to_skip = min(warmup_batches, max(len(raw_ms) - 1, 0))
    batch_ms = raw_ms[warmup_to_skip:]
    batch_sizes = raw_sizes[warmup_to_skip:]
    sample_ms = [elapsed / max(size, 1) for elapsed, size in zip(batch_ms, batch_sizes)]
    total_samples = sum(batch_sizes)
    arr = np.asarray(batch_ms, dtype=np.float64)
    sample_arr = np.asarray(sample_ms, dtype=np.float64)
    throughput = float(total_samples / (arr.sum() / 1000.0)) if arr.sum() > 0 else 0.0
    return {
        "batches_measured": int(len(batch_ms)),
        "samples_measured": int(total_samples),
        "avg_batch_latency_ms": float(arr.mean()),
        "p50_batch_latency_ms": float(np.percentile(arr, 50)),
        "p95_batch_latency_ms": float(np.percentile(arr, 95)),
        "avg_sample_latency_ms": float(sample_arr.mean()),
        "p95_sample_latency_ms": float(np.percentile(sample_arr, 95)),
        "throughput_samples_per_s": throughput,
    }


def precompute_embeddings(model: MultitaskLLMBaseline, loader: DataLoader, device: torch.device) -> EmbeddedTelemetryDataset:
    model.backbone.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    action_labels: list[torch.Tensor] = []
    subsystem_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(model.pooled(outputs).detach().cpu())
            labels.append(batch["labels"].detach().cpu())
            action_labels.append(batch["action_labels"].detach().cpu())
            subsystem_labels.append(batch["subsystem_labels"].detach().cpu())
    return EmbeddedTelemetryDataset(
        embeddings=torch.cat(embeddings, dim=0),
        labels=torch.cat(labels, dim=0),
        action_labels=torch.cat(action_labels, dim=0),
        subsystem_labels=torch.cat(subsystem_labels, dim=0),
    )


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

    for batch in loader:
        labels = batch["labels"].to(device)
        action_labels = batch["action_labels"].to(device)
        subsystem_labels = batch["subsystem_labels"].to(device)
        if "embeddings" in batch:
            outputs = model.classify_embeddings(batch["embeddings"].to(device))
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

    return {
        "loss": total_loss / len(loader.dataset),
        "health_preds": np.concatenate(health_preds),
        "health_targets": np.concatenate(health_targets),
        "action_preds": np.concatenate(action_preds),
        "action_targets": np.concatenate(action_targets),
        "subsystem_preds": np.concatenate(subsystem_preds),
        "subsystem_targets": np.concatenate(subsystem_targets),
        "steps": np.zeros(len(loader.dataset), dtype=np.int64),
    }


def save_metrics(output_dir: Path, split_name: str, result: dict[str, Any], latency: dict[str, float]) -> dict[str, Any]:
    health_payload = task_payload(result["health_preds"], result["health_targets"], [0, 1, 2], LABEL_NAMES)
    action_payload = task_payload(result["action_preds"], result["action_targets"], [0, 1, 2, 3], ACTION_NAMES)
    subsystem_payload = task_payload(
        result["subsystem_preds"], result["subsystem_targets"], [0, 1, 2, 3], SUBSYSTEM_NAMES
    )
    metrics = dict(health_payload["metrics"])
    metrics["loss"] = result["loss"]
    metrics["num_examples"] = int(len(result["health_targets"]))
    metrics["action_accuracy"] = action_payload["metrics"]["accuracy"]
    metrics["action_macro_f1"] = action_payload["metrics"]["macro_f1"]
    metrics["subsystem_accuracy"] = subsystem_payload["metrics"]["accuracy"]
    metrics["subsystem_macro_f1"] = subsystem_payload["metrics"]["macro_f1"]
    metrics["avg_steps"] = 0.0
    metrics["max_steps"] = 0
    metrics.update(latency)
    (output_dir / f"{split_name}_classification_report.txt").write_text(health_payload["report"] + "\n")
    (output_dir / f"{split_name}_confusion_matrix.json").write_text(json.dumps(health_payload["confusion_matrix"], indent=2))
    (output_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / f"{split_name}_action_metrics.json").write_text(json.dumps(action_payload, indent=2))
    (output_dir / f"{split_name}_subsystem_metrics.json").write_text(json.dumps(subsystem_payload, indent=2))
    (output_dir / f"{split_name}_latency.json").write_text(json.dumps(latency, indent=2))
    return metrics


def save_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    lines = [
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,train_macro_f1,val_macro_f1,val_action_accuracy,val_subsystem_accuracy,val_latency_ms_per_example"
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
                    f"{row['val_latency_ms_per_example']:.6f}",
                ]
            )
        )
    (output_dir / "history.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LLM baseline on text-serialized telemetry windows.")
    parser.add_argument("--cmapss-root", default="CMAPSSData")
    parser.add_argument("--cmapss-subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--backbone", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default="artifacts/cmapss_fd001_llm")
    parser.add_argument("--latency-batches", type=int, default=16)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    args = parser.parse_args()

    data_config = DataConfig()
    train_config = TrainConfig(batch_size=args.batch_size, epochs=args.epochs)
    set_seed(train_config.seed)

    train_df, test_df = load_cmapss_train_test(args.cmapss_root, args.cmapss_subset, data_config)
    train_windowed = build_windows(train_df, data_config)
    train_split, val_split = split_windowed_by_asset(
        train_windowed, train_config.val_ratio, train_config.seed, "temporal"
    )
    train_text = build_text_data(train_split)
    val_text = build_text_data(val_split)
    test_text = build_text_data(build_windows(test_df, data_config))
    print(
        f"text_windows train={len(train_text.texts)} val={len(val_text.texts)} test={len(test_text.texts)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    print(f"tokenizer_ready backbone={args.backbone}", flush=True)
    train_dataset = TokenizedTelemetryDataset(
        train_text.texts,
        train_text.labels,
        train_text.action_labels,
        train_text.subsystem_labels,
        tokenizer,
        args.max_length,
    )
    val_dataset = TokenizedTelemetryDataset(
        val_text.texts,
        val_text.labels,
        val_text.action_labels,
        val_text.subsystem_labels,
        tokenizer,
        args.max_length,
    )
    test_dataset = TokenizedTelemetryDataset(
        test_text.texts,
        test_text.labels,
        test_text.action_labels,
        test_text.subsystem_labels,
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskLLMBaseline(args.backbone, freeze_backbone=args.freeze_backbone).to(device)
    print(f"model_ready device={device} freeze_backbone={args.freeze_backbone}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    health_criterion = nn.CrossEntropyLoss()
    action_criterion = nn.CrossEntropyLoss()
    subsystem_criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    train_head_loader = train_loader
    val_head_loader = val_loader
    test_head_loader = test_loader
    if args.freeze_backbone:
        train_embedded = precompute_embeddings(model, train_loader, device)
        val_embedded = precompute_embeddings(model, val_loader, device)
        test_embedded = precompute_embeddings(model, test_loader, device)
        train_head_loader = DataLoader(train_embedded, batch_size=args.batch_size, shuffle=True)
        val_head_loader = DataLoader(val_embedded, batch_size=args.batch_size, shuffle=False)
        test_head_loader = DataLoader(test_embedded, batch_size=args.batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(
            list(model.health_head.parameters())
            + list(model.action_head.parameters())
            + list(model.subsystem_head.parameters()),
            lr=5e-4,
        )
        print("embedding_cache_ready train/val/test", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model, train_head_loader, health_criterion, action_criterion, subsystem_criterion, optimizer, device
        )
        val_result = run_epoch(
            model, val_head_loader, health_criterion, action_criterion, subsystem_criterion, None, device
        )
        train_metrics = summarize_metrics(train_result["health_preds"], train_result["health_targets"], [0, 1, 2])
        val_metrics = summarize_metrics(val_result["health_preds"], val_result["health_targets"], [0, 1, 2])
        val_action_metrics = summarize_metrics(val_result["action_preds"], val_result["action_targets"], [0, 1, 2, 3])
        val_subsystem_metrics = summarize_metrics(
            val_result["subsystem_preds"], val_result["subsystem_targets"], [0, 1, 2, 3]
        )
        val_latency = benchmark_latency(model, val_loader, device, max_batches=min(args.latency_batches, 4))
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
                "val_latency_ms_per_example": val_latency["avg_sample_latency_ms"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} val_loss={val_result['loss']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    save_history(output_dir, history)
    if best_state is not None:
        model.load_state_dict(best_state)

    val_result = run_epoch(model, val_head_loader, health_criterion, action_criterion, subsystem_criterion, None, device)
    val_latency = benchmark_latency(model, val_loader, device, max_batches=args.latency_batches)
    val_metrics = save_metrics(output_dir, "validation", val_result, val_latency)
    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))

    test_result = run_epoch(model, test_head_loader, health_criterion, action_criterion, subsystem_criterion, None, device)
    test_latency = benchmark_latency(model, test_loader, device, max_batches=args.latency_batches)
    test_metrics = save_metrics(output_dir, "test", test_result, test_latency)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
