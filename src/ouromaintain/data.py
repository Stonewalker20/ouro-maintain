from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
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


def _hvac_health_label_from_name(name: str) -> int:
    lowered = name.lower()
    if "faultfree" in lowered:
        return 0

    critical_tokens = [
        "severe",
        "blockage",
        "reverse",
        "unstable",
        "_80",
        "_100",
        "+4c",
        "-4c",
    ]
    warning_tokens = [
        "minor",
        "moderate",
        "10%",
        "20%",
        "_20",
        "_30",
        "_50",
        "+2c",
        "-2c",
    ]
    if any(token in lowered for token in critical_tokens):
        return 2
    if any(token in lowered for token in warning_tokens):
        return 1
    return 2


def _hvac_action_label_from_name(name: str) -> int:
    lowered = name.lower()
    if "faultfree" in lowered:
        return 0
    shutdown_tokens = ["blockage", "reverse", "unstable", "_100", "_80", "severe"]
    if any(token in lowered for token in shutdown_tokens):
        return 3
    if any(token in lowered for token in ["moderate", "_50", "_30", "+4c", "-4c"]):
        return 2
    return 1


def _hvac_subsystem_label_from_name(name: str) -> int:
    lowered = name.lower()
    if any(token in lowered for token in ["fouling", "vlv", "cvlv", "hvlv", "cooling", "heating", "waterside"]):
        return 1
    if any(token in lowered for token in ["oadmpr", "oablockage", "filterrestriction"]):
        return 2
    if "fan" in lowered:
        return 3
    return 0


def load_lbnl_fcu_dataset(
    root_dir: str,
    config: DataConfig,
    pattern: str = "*.csv",
    row_step: int = 60,
    max_files: int | None = None,
) -> pd.DataFrame:
    root = Path(root_dir)
    if not root.exists():
        raise ValueError(f"LBNL HVAC dataset directory not found: {root}")
    if row_step < 1:
        raise ValueError("LBNL HVAC row_step must be at least 1.")

    files = sorted(root.glob(pattern))
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No HVAC CSV files found under {root} with pattern {pattern!r}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        frame = frame.iloc[::row_step].copy()
        if "Datetime" not in frame.columns:
            raise ValueError(f"Expected Datetime column in HVAC file: {path}")
        frame[config.asset_id_col] = path.stem
        frame[config.timestamp_col] = np.arange(len(frame), dtype=np.int64)
        frame[config.label_col] = _hvac_health_label_from_name(path.stem)
        frame[config.action_col] = _hvac_action_label_from_name(path.stem)
        frame[config.subsystem_col] = _hvac_subsystem_label_from_name(path.stem)
        frame = frame.drop(columns=["Datetime"])
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _paderborn_health_label(folder_name: str) -> int:
    if folder_name.startswith("K") and not folder_name.startswith(("KA", "KB", "KI")):
        return 0
    if folder_name.startswith("KA"):
        return 1
    return 2


def _paderborn_action_label(folder_name: str) -> int:
    if folder_name.startswith("K") and not folder_name.startswith(("KA", "KB", "KI")):
        return 0
    if folder_name.startswith("KA"):
        return 1
    if folder_name.startswith("KI"):
        return 2
    return 3


def _paderborn_subsystem_label(folder_name: str) -> int:
    if folder_name.startswith("KA"):
        return 1
    if folder_name.startswith("KI"):
        return 3
    if folder_name.startswith("KB"):
        return 2
    return 0


def _flatten_numeric_array(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _resample_series(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) == target_length:
        return values.astype(np.float32)
    if len(values) == 0:
        return np.zeros(target_length, dtype=np.float32)
    source = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    target = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(target, source, values).astype(np.float32)


def _extract_paderborn_channels(mat_payload: bytes) -> dict[str, np.ndarray]:
    mat = loadmat(BytesIO(mat_payload))
    key = next(name for name in mat.keys() if not name.startswith("__"))
    record = mat[key][0, 0]
    channels: dict[str, np.ndarray] = {}
    for idx in range(record["Y"].shape[1]):
        item = record["Y"][0, idx]
        raw_name = item["Name"]
        if raw_name.size == 0:
            continue
        name = str(raw_name[0])
        values = _flatten_numeric_array(item["Data"])
        channels[name] = values
    return channels


def _paderborn_row_from_measurement(
    folder_name: str,
    measurement_name: str,
    channels: dict[str, np.ndarray],
    config: DataConfig,
    sample_stride: int,
) -> pd.DataFrame:
    vibration = channels.get("vibration_1")
    if vibration is None or len(vibration) == 0:
        raise ValueError(f"Measurement {measurement_name} does not contain vibration_1 data.")

    target_length = len(vibration)
    aligned: dict[str, np.ndarray] = {
        "vibration_1": vibration.astype(np.float32),
        "phase_current_1": _resample_series(channels.get("phase_current_1", np.zeros(1, dtype=np.float32)), target_length),
        "phase_current_2": _resample_series(channels.get("phase_current_2", np.zeros(1, dtype=np.float32)), target_length),
        "force": _resample_series(channels.get("force", np.zeros(1, dtype=np.float32)), target_length),
        "speed": _resample_series(channels.get("speed", np.zeros(1, dtype=np.float32)), target_length),
        "torque": _resample_series(channels.get("torque", np.zeros(1, dtype=np.float32)), target_length),
    }
    if sample_stride > 1:
        aligned = {name: values[::sample_stride] for name, values in aligned.items()}

    sample_count = len(next(iter(aligned.values())))
    frame = pd.DataFrame(aligned)
    frame[config.asset_id_col] = measurement_name
    frame[config.timestamp_col] = np.arange(sample_count, dtype=np.int64)
    frame[config.label_col] = _paderborn_health_label(folder_name)
    frame[config.action_col] = _paderborn_action_label(folder_name)
    frame[config.subsystem_col] = _paderborn_subsystem_label(folder_name)
    return frame


def load_paderborn_dataset(
    zip_path: str,
    config: DataConfig,
    sample_stride: int = 256,
    max_measurements_per_bearing: int | None = 4,
    include_prefixes: tuple[str, ...] = ("K", "KA", "KB", "KI"),
) -> pd.DataFrame:
    archive_path = Path(zip_path)
    if not archive_path.exists():
        raise ValueError(f"Paderborn archive not found: {archive_path}")
    if sample_stride < 1:
        raise ValueError("Paderborn sample_stride must be at least 1.")

    frames: list[pd.DataFrame] = []
    per_bearing_counts: dict[str, int] = {}
    with ZipFile(archive_path) as archive:
        mat_files = sorted(name for name in archive.namelist() if name.endswith(".mat"))
        for member_name in mat_files:
            folder_name = member_name.split("/", 1)[0]
            if not folder_name.startswith(include_prefixes):
                continue

            taken = per_bearing_counts.get(folder_name, 0)
            if max_measurements_per_bearing is not None and taken >= max_measurements_per_bearing:
                continue

            measurement_name = Path(member_name).stem
            channels = _extract_paderborn_channels(archive.read(member_name))
            frames.append(
                _paderborn_row_from_measurement(
                    folder_name=folder_name,
                    measurement_name=measurement_name,
                    channels=channels,
                    config=config,
                    sample_stride=sample_stride,
                )
            )
            per_bearing_counts[folder_name] = taken + 1

    if not frames:
        raise ValueError(f"No Paderborn measurements matched in archive: {archive_path}")

    return pd.concat(frames, ignore_index=True)


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
    def _slice(indices: np.ndarray) -> WindowedData:
        return WindowedData(
            features=data.features[indices],
            labels=data.labels[indices],
            action_labels=data.action_labels[indices],
            subsystem_labels=data.subsystem_labels[indices],
            asset_ids=data.asset_ids[indices],
            feature_names=data.feature_names,
        )

    if single_asset_mode == "window_stratified":
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
        return (_slice(train_indices_arr), _slice(val_indices_arr))

    if single_asset_mode == "asset_label_stratified":
        rng = np.random.default_rng(seed)
        asset_ids = np.unique(data.asset_ids)
        if len(asset_ids) < 2:
            return split_windowed_by_asset(data, val_ratio, seed, "window_stratified")

        asset_to_label = {
            asset_id: int(data.labels[np.where(data.asset_ids == asset_id)[0][-1]]) for asset_id in asset_ids
        }
        train_assets: set[str] = set()
        val_assets: set[str] = set()
        for label in sorted(set(asset_to_label.values())):
            label_assets = [asset_id for asset_id, asset_label in asset_to_label.items() if asset_label == label]
            shuffled = np.asarray(label_assets, dtype=object)
            rng.shuffle(shuffled)
            split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
            if split_idx >= len(shuffled):
                split_idx = len(shuffled) - 1
            train_assets.update(str(asset) for asset in shuffled[:split_idx])
            val_assets.update(str(asset) for asset in shuffled[split_idx:])

        train_mask = np.array([asset_id in train_assets for asset_id in data.asset_ids])
        val_mask = np.array([asset_id in val_assets for asset_id in data.asset_ids])
        return (_slice(np.where(train_mask)[0]), _slice(np.where(val_mask)[0]))

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
            return (_slice(train_indices_arr), _slice(val_indices_arr))

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
            return (_slice(train_indices_arr), _slice(val_indices_arr))

        split_idx = max(1, int(len(data.labels) * (1 - val_ratio)))
        if split_idx >= len(data.labels):
            split_idx = len(data.labels) - 1
        train_slice = slice(0, split_idx)
        val_slice = slice(split_idx, len(data.labels))
        return (
            _slice(np.arange(len(data.labels))[train_slice]),
            _slice(np.arange(len(data.labels))[val_slice]),
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

    return (_slice(np.where(train_mask)[0]), _slice(np.where(val_mask)[0]))


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
