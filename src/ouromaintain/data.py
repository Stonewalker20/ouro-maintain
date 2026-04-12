from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig


@dataclass
class WindowedData:
    features: np.ndarray
    labels: np.ndarray
    asset_ids: np.ndarray
    feature_names: list[str]


CMAPSS_COLUMNS = (
    ["asset_id", "timestamp"]
    + [f"op_setting_{idx}" for idx in range(1, 4)]
    + [f"sensor_{idx}" for idx in range(1, 22)]
)


def load_telemetry_csv(path: str, config: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {config.asset_id_col, config.timestamp_col, config.label_col}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    return df.sort_values([config.asset_id_col, config.timestamp_col]).reset_index(drop=True)


def rul_to_class(rul: pd.Series, config: DataConfig) -> pd.Series:
    labels = np.zeros(len(rul), dtype=np.int64)
    labels[rul <= config.warning_rul] = 1
    labels[rul <= config.critical_rul] = 2
    return pd.Series(labels, index=rul.index, name=config.label_col)


def load_cmapss_subset(root_dir: str, subset: str, config: DataConfig) -> pd.DataFrame:
    subset = subset.upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported CMAPSS subset: {subset}")

    path = f"{root_dir.rstrip('/')}/train_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="python")
    df["max_cycle"] = df.groupby("asset_id")["timestamp"].transform("max")
    df["rul"] = df["max_cycle"] - df["timestamp"]
    df[config.label_col] = rul_to_class(df["rul"], config)
    return df.drop(columns=["max_cycle"]).reset_index(drop=True)


def load_cmapss_train_test(
    root_dir: str, subset: str, config: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = subset.upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported CMAPSS subset: {subset}")

    root = Path(root_dir)
    train_path = root / f"train_{subset}.txt"
    test_path = root / f"test_{subset}.txt"
    rul_path = root / f"RUL_{subset}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="python")
    train_df["max_cycle"] = train_df.groupby("asset_id")["timestamp"].transform("max")
    train_df["rul"] = train_df["max_cycle"] - train_df["timestamp"]
    train_df[config.label_col] = rul_to_class(train_df["rul"], config)
    train_df = train_df.drop(columns=["max_cycle"]).reset_index(drop=True)

    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="python")
    test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"], engine="python")
    test_rul["asset_id"] = np.arange(1, len(test_rul) + 1)

    test_df["observed_max_cycle"] = test_df.groupby("asset_id")["timestamp"].transform("max")
    test_df = test_df.merge(test_rul, on="asset_id", how="left", suffixes=("", "_final"))
    test_df["failure_cycle"] = test_df["observed_max_cycle"] + test_df["rul"]
    test_df["rul"] = test_df["failure_cycle"] - test_df["timestamp"]
    test_df[config.label_col] = rul_to_class(test_df["rul"], config)
    test_df = test_df.drop(columns=["observed_max_cycle", "failure_cycle"]).reset_index(drop=True)

    return train_df, test_df


def _stat_features(signal: np.ndarray, prefix: str) -> dict[str, float]:
    centered = signal - signal.mean()
    variance = float(np.mean(centered**2))
    std = float(np.sqrt(max(variance, 1e-12)))
    normalized = centered / std
    kurtosis = float(np.mean(normalized**4))

    return {
        f"{prefix}_mean": float(signal.mean()),
        f"{prefix}_std": std,
        f"{prefix}_rms": float(np.sqrt(np.mean(signal**2))),
        f"{prefix}_max_abs": float(np.max(np.abs(signal))),
        f"{prefix}_kurtosis": kurtosis,
    }


def load_ims_run(extracted_root: str, run_name: str, config: DataConfig) -> pd.DataFrame:
    run_dir = Path(extracted_root) / run_name
    if not run_dir.exists():
        raise ValueError(f"IMS run directory not found: {run_dir}")

    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and not path.name.startswith("."))
    if not files:
        raise ValueError(f"No IMS snapshot files found under {run_dir}")

    rows: list[dict[str, float | int | str]] = []
    total = len(files)
    critical_cutoff = max(int(total * 0.15), 1)
    warning_cutoff = max(int(total * 0.5), 1)

    for idx, path in enumerate(files):
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        row: dict[str, float | int | str] = {
            config.asset_id_col: run_name,
            config.timestamp_col: idx,
        }
        for channel_idx in range(frame.shape[1]):
            row.update(_stat_features(frame.iloc[:, channel_idx].to_numpy(dtype=np.float32), f"ch{channel_idx + 1}"))

        remaining = total - idx - 1
        row["rul"] = remaining
        if remaining <= critical_cutoff:
            label = 2
        elif remaining <= warning_cutoff:
            label = 1
        else:
            label = 0
        row[config.label_col] = label
        rows.append(row)

    return pd.DataFrame(rows)


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=(0, 1), keepdims=True)
    std = features.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def split_windowed_by_asset(
    data: WindowedData, val_ratio: float, seed: int
) -> tuple[WindowedData, WindowedData]:
    asset_ids = np.unique(data.asset_ids)
    rng = np.random.default_rng(seed)
    shuffled = asset_ids.copy()
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    if split_idx >= len(shuffled):
        split_idx = len(shuffled) - 1

    train_assets = set(shuffled[:split_idx])
    train_mask = np.array([asset_id in train_assets for asset_id in data.asset_ids])
    val_mask = ~train_mask

    return (
        WindowedData(
            features=data.features[train_mask],
            labels=data.labels[train_mask],
            asset_ids=data.asset_ids[train_mask],
            feature_names=data.feature_names,
        ),
        WindowedData(
            features=data.features[val_mask],
            labels=data.labels[val_mask],
            asset_ids=data.asset_ids[val_mask],
            feature_names=data.feature_names,
        ),
    )


def build_windows(df: pd.DataFrame, config: DataConfig) -> WindowedData:
    feature_cols = [
        col
        for col in df.columns
        if col not in {config.asset_id_col, config.timestamp_col, config.label_col}
    ]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding id/timestamp/label.")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    asset_ids: list[str] = []

    for asset_id, asset_df in df.groupby(config.asset_id_col, sort=False):
        values = asset_df[feature_cols].to_numpy(dtype=np.float32)
        target = asset_df[config.label_col].to_numpy()
        total = len(asset_df)

        for start in range(0, total - config.window_size + 1, config.stride):
            end = start + config.window_size
            windows.append(values[start:end])
            labels.append(int(target[end - 1]))
            asset_ids.append(str(asset_id))

    if not windows:
        raise ValueError("No windows were generated. Check window size and dataset length.")

    return WindowedData(
        features=np.stack(windows),
        labels=np.asarray(labels, dtype=np.int64),
        asset_ids=np.asarray(asset_ids),
        feature_names=feature_cols,
    )


class TelemetryWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: WindowedData) -> None:
        self.features = torch.tensor(data.features, dtype=torch.float32)
        self.labels = torch.tensor(data.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
