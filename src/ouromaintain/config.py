from dataclasses import dataclass


@dataclass
class DataConfig:
    window_size: int = 32
    stride: int = 8
    label_col: str = "label"
    asset_id_col: str = "asset_id"
    timestamp_col: str = "timestamp"
    warning_rul: int = 50
    critical_rul: int = 15


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 128
    num_classes: int = 3
    max_loops: int = 6
    exit_threshold: float = 0.8


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
