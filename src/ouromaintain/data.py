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
    action_labels: np.ndarray
    subsystem_labels: np.ndarray
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


def rul_to_action(rul: pd.Series, config: DataConfig) -> pd.Series:
    actions = np.zeros(len(rul), dtype=np.int64)
    actions[rul <= config.warning_rul] = 1
    actions[rul <= config.critical_rul] = 2
    actions[rul <= config.shutdown_rul] = 3
    return pd.Series(actions, index=rul.index, name=config.action_col)


def _default_action_from_label(labels: pd.Series, config: DataConfig) -> pd.Series:
    actions = labels.to_numpy(dtype=np.int64).copy()
    return pd.Series(actions, index=labels.index, name=config.action_col)


def _safe_fractional_change(current: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    denom = baseline.abs().replace(0.0, 1.0)
    return ((current - baseline).abs() / denom).fillna(0.0)


def assign_cmapss_subsystems(df: pd.DataFrame, config: DataConfig) -> pd.Series:
    thermal_cols = ["sensor_2", "sensor_3", "sensor_4", "sensor_11", "sensor_12", "sensor_15"]
    flow_cols = ["sensor_7", "sensor_8", "sensor_9", "sensor_13", "sensor_14"]
    mechanical_cols = ["sensor_17", "sensor_20", "sensor_21"]
    tracked_cols = thermal_cols + flow_cols + mechanical_cols

    baseline = df.groupby(config.asset_id_col)[tracked_cols].transform("first")
    delta = _safe_fractional_change(df[tracked_cols], baseline)

    group_scores = pd.DataFrame(
        {
            "thermal": delta[thermal_cols].mean(axis=1),
            "flow_path": delta[flow_cols].mean(axis=1),
            "mechanical": delta[mechanical_cols].mean(axis=1),
        }
    )
    dominant = group_scores.idxmax(axis=1)
    overall = group_scores.max(axis=1)
    subsystem = np.zeros(len(df), dtype=np.int64)
    subsystem[(dominant == "thermal") & (overall >= 0.05)] = 1
    subsystem[(dominant == "flow_path") & (overall >= 0.05)] = 2
    subsystem[(dominant == "mechanical") & (overall >= 0.05)] = 3
    return pd.Series(subsystem, index=df.index, name=config.subsystem_col)


def _ims_health_label(progress: float, config: DataConfig) -> int:
    if progress >= config.ims_warning_ratio:
        return 2
    if progress >= config.ims_normal_ratio:
        return 1
    return 0


def _ims_action_label(progress: float, config: DataConfig) -> int:
    if progress >= config.ims_shutdown_ratio:
        return 3
    if progress >= config.ims_warning_ratio:
        return 2
    if progress >= config.ims_normal_ratio:
        return 1
    return 0


def load_cmapss_subset(root_dir: str, subset: str, config: DataConfig) -> pd.DataFrame:
    subset = subset.upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported CMAPSS subset: {subset}")

    path = f"{root_dir.rstrip('/')}/train_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="python")
    df["max_cycle"] = df.groupby("asset_id")["timestamp"].transform("max")
    df["rul"] = df["max_cycle"] - df["timestamp"]
    df[config.label_col] = rul_to_class(df["rul"], config)
    df[config.action_col] = rul_to_action(df["rul"], config)
    df[config.subsystem_col] = assign_cmapss_subsystems(df, config)
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
    train_df[config.action_col] = rul_to_action(train_df["rul"], config)
    train_df[config.subsystem_col] = assign_cmapss_subsystems(train_df, config)
    train_df = train_df.drop(columns=["max_cycle"]).reset_index(drop=True)

    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="python")
    test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"], engine="python")
    test_rul["asset_id"] = np.arange(1, len(test_rul) + 1)

    test_df["observed_max_cycle"] = test_df.groupby("asset_id")["timestamp"].transform("max")
    test_df = test_df.merge(test_rul, on="asset_id", how="left", suffixes=("", "_final"))
    test_df["failure_cycle"] = test_df["observed_max_cycle"] + test_df["rul"]
    test_df["rul"] = test_df["failure_cycle"] - test_df["timestamp"]
    test_df[config.label_col] = rul_to_class(test_df["rul"], config)
    test_df[config.action_col] = rul_to_action(test_df["rul"], config)
    test_df[config.subsystem_col] = assign_cmapss_subsystems(test_df, config)
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


def load_ims_run(extracted_root: str, run_name: str, config: DataConfig, file_step: int = 1) -> pd.DataFrame:
    run_dir = Path(extracted_root) / run_name
    if not run_dir.exists():
        raise ValueError(f"IMS run directory not found: {run_dir}")
    if file_step < 1:
        raise ValueError("IMS file_step must be at least 1.")

    cache_dir = Path(extracted_root) / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = run_name.replace("/", "__")
    cache_path = cache_dir / f"{cache_key}_step{file_step}_features.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and not path.name.startswith("."))
    files = files[::file_step]
    if not files:
        raise ValueError(f"No IMS snapshot files found under {run_dir}")

    rows: list[dict[str, float | int | str]] = []
    total = len(files)
    for idx, path in enumerate(files):
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        row: dict[str, float | int | str] = {
            config.asset_id_col: run_name,
            config.timestamp_col: idx,
        }
        for channel_idx in range(frame.shape[1]):
            row.update(_stat_features(frame.iloc[:, channel_idx].to_numpy(dtype=np.float32), f"ch{channel_idx + 1}"))

        remaining = total - idx - 1
        progress = idx / max(total - 1, 1)
        row["rul"] = remaining
        row["progress"] = progress
        row[config.label_col] = _ims_health_label(progress, config)
        row[config.action_col] = _ims_action_label(progress, config)
        row[config.subsystem_col] = 3
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=(0, 1), keepdims=True)
    std = features.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def split_windowed_by_asset(
    data: WindowedData, val_ratio: float, seed: int, single_asset_mode: str = "temporal"
) -> tuple[WindowedData, WindowedData]:
    asset_ids = np.unique(data.asset_ids)
    if len(asset_ids) < 2:
        if single_asset_mode == "stratified":
            rng = np.random.default_rng(seed)
            train_indices: list[int] = []
            val_indices: list[int] = []
            unique_labels = np.unique(data.labels)
            for label in unique_labels:
                label_indices = np.where(data.labels == label)[0]
                shuffled = label_indices.copy()
                rng.shuffle(shuffled)
                split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
                if split_idx >= len(shuffled):
                    split_idx = len(shuffled) - 1
                train_indices.extend(shuffled[:split_idx].tolist())
                val_indices.extend(shuffled[split_idx:].tolist())

            train_indices_arr = np.asarray(sorted(train_indices))
            val_indices_arr = np.asarray(sorted(val_indices))
            return (
                WindowedData(
                    features=data.features[train_indices_arr],
                    labels=data.labels[train_indices_arr],
                    action_labels=data.action_labels[train_indices_arr],
                    subsystem_labels=data.subsystem_labels[train_indices_arr],
                    asset_ids=data.asset_ids[train_indices_arr],
                    feature_names=data.feature_names,
                ),
                WindowedData(
                    features=data.features[val_indices_arr],
                    labels=data.labels[val_indices_arr],
                    action_labels=data.action_labels[val_indices_arr],
                    subsystem_labels=data.subsystem_labels[val_indices_arr],
                    asset_ids=data.asset_ids[val_indices_arr],
                    feature_names=data.feature_names,
                ),
            )

        if single_asset_mode == "stage_temporal":
            train_indices: list[int] = []
            val_indices: list[int] = []
            unique_labels = np.unique(data.labels)
            for label in unique_labels:
                label_indices = np.where(data.labels == label)[0]
                split_idx = max(1, int(len(label_indices) * (1 - val_ratio)))
                if split_idx >= len(label_indices):
                    split_idx = len(label_indices) - 1
                train_indices.extend(label_indices[:split_idx].tolist())
                val_indices.extend(label_indices[split_idx:].tolist())

            train_indices_arr = np.asarray(sorted(train_indices))
            val_indices_arr = np.asarray(sorted(val_indices))
            return (
                WindowedData(
                    features=data.features[train_indices_arr],
                    labels=data.labels[train_indices_arr],
                    action_labels=data.action_labels[train_indices_arr],
                    subsystem_labels=data.subsystem_labels[train_indices_arr],
                    asset_ids=data.asset_ids[train_indices_arr],
                    feature_names=data.feature_names,
                ),
                WindowedData(
                    features=data.features[val_indices_arr],
                    labels=data.labels[val_indices_arr],
                    action_labels=data.action_labels[val_indices_arr],
                    subsystem_labels=data.subsystem_labels[val_indices_arr],
                    asset_ids=data.asset_ids[val_indices_arr],
                    feature_names=data.feature_names,
                ),
            )

        split_idx = max(1, int(len(data.labels) * (1 - val_ratio)))
        if split_idx >= len(data.labels):
            split_idx = len(data.labels) - 1
        train_slice = slice(0, split_idx)
        val_slice = slice(split_idx, len(data.labels))
        return (
            WindowedData(
                features=data.features[train_slice],
                labels=data.labels[train_slice],
                action_labels=data.action_labels[train_slice],
                subsystem_labels=data.subsystem_labels[train_slice],
                asset_ids=data.asset_ids[train_slice],
                feature_names=data.feature_names,
            ),
            WindowedData(
                features=data.features[val_slice],
                labels=data.labels[val_slice],
                action_labels=data.action_labels[val_slice],
                subsystem_labels=data.subsystem_labels[val_slice],
                asset_ids=data.asset_ids[val_slice],
                feature_names=data.feature_names,
            ),
        )

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
            action_labels=data.action_labels[train_mask],
            subsystem_labels=data.subsystem_labels[train_mask],
            asset_ids=data.asset_ids[train_mask],
            feature_names=data.feature_names,
        ),
        WindowedData(
            features=data.features[val_mask],
            labels=data.labels[val_mask],
            action_labels=data.action_labels[val_mask],
            subsystem_labels=data.subsystem_labels[val_mask],
            asset_ids=data.asset_ids[val_mask],
            feature_names=data.feature_names,
        ),
    )


def build_windows(df: pd.DataFrame, config: DataConfig) -> WindowedData:
    frame = df.copy()
    if config.action_col not in frame.columns:
        frame[config.action_col] = _default_action_from_label(frame[config.label_col], config)
    if config.subsystem_col not in frame.columns:
        frame[config.subsystem_col] = 0

    feature_cols = [
        col
        for col in frame.columns
        if col not in {config.asset_id_col, config.timestamp_col, config.label_col, config.action_col, config.subsystem_col}
    ]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding id/timestamp/label.")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    action_labels: list[int] = []
    subsystem_labels: list[int] = []
    asset_ids: list[str] = []

    for asset_id, asset_df in frame.groupby(config.asset_id_col, sort=False):
        values = asset_df[feature_cols].to_numpy(dtype=np.float32)
        target = asset_df[config.label_col].to_numpy(dtype=np.int64)
        action_target = asset_df[config.action_col].to_numpy(dtype=np.int64)
        subsystem_target = asset_df[config.subsystem_col].to_numpy(dtype=np.int64)
        total = len(asset_df)

        for start in range(0, total - config.window_size + 1, config.stride):
            end = start + config.window_size
            windows.append(values[start:end])
            labels.append(int(target[end - 1]))
            action_labels.append(int(action_target[end - 1]))
            subsystem_labels.append(int(subsystem_target[end - 1]))
            asset_ids.append(str(asset_id))

    if not windows:
        raise ValueError("No windows were generated. Check window size and dataset length.")

    return WindowedData(
        features=np.stack(windows),
        labels=np.asarray(labels, dtype=np.int64),
        action_labels=np.asarray(action_labels, dtype=np.int64),
        subsystem_labels=np.asarray(subsystem_labels, dtype=np.int64),
        asset_ids=np.asarray(asset_ids),
        feature_names=feature_cols,
    )


class TelemetryWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, data: WindowedData) -> None:
        self.features = torch.tensor(data.features, dtype=torch.float32)
        self.labels = torch.tensor(data.labels, dtype=torch.long)
        self.action_labels = torch.tensor(data.action_labels, dtype=torch.long)
        self.subsystem_labels = torch.tensor(data.subsystem_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.action_labels[idx], self.subsystem_labels[idx]
