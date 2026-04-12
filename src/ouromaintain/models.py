from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class WindowEncoder(nn.Module):
    """Simple GRU encoder for windowed telemetry."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return hidden[-1]


class BaselineClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = WindowEncoder(config.input_dim, config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return {"logits": logits}


class SharedLoopBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        update = self.block(torch.cat([h, context], dim=-1))
        return self.norm(h + update)


class FixedDepthLoopModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_loops = config.max_loops
        self.encoder = WindowEncoder(config.input_dim, config.hidden_dim)
        self.loop = SharedLoopBlock(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encoder(x)
        h = context
        for _ in range(self.max_loops):
            h = self.loop(h, context)

        logits = self.classifier(h)
        steps = torch.full((x.size(0),), self.max_loops, dtype=torch.long, device=x.device)
        return {"logits": logits, "steps": steps}


class AdaptiveLoopModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_loops = config.max_loops
        self.exit_threshold = config.exit_threshold
        self.encoder = WindowEncoder(config.input_dim, config.hidden_dim)
        self.loop = SharedLoopBlock(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.exit_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encoder(x)
        h = context
        batch_size = x.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        last_exit_prob = torch.zeros(batch_size, device=x.device)

        for step_idx in range(1, self.max_loops + 1):
            h = self.loop(h, context)
            exit_prob = torch.sigmoid(self.exit_head(h)).squeeze(-1)
            last_exit_prob = exit_prob

            should_exit = exit_prob >= self.exit_threshold
            new_finished = should_exit & ~finished
            steps = torch.where(new_finished, torch.full_like(steps, step_idx), steps)
            finished = finished | should_exit

            if bool(finished.all()):
                break

        steps = torch.where(steps == 0, torch.full_like(steps, self.max_loops), steps)
        logits = self.classifier(h)
        return {
            "logits": logits,
            "steps": steps,
            "exit_probability": last_exit_prob,
        }
