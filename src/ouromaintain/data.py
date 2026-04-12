from __future__ import annotations

from dataclasses import dataclass

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
    )


class TelemetryWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: WindowedData) -> None:
        self.features = torch.tensor(data.features, dtype=torch.float32)
        self.labels = torch.tensor(data.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
