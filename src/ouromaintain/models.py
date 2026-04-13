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
        self.health_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_action_classes),
        )
        self.subsystem_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_subsystem_classes),
        )
        self.classifier = self.health_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(x)
        logits = self.health_head(latent)
        return {
            "logits": logits,
            "action_logits": self.action_head(latent),
            "subsystem_logits": self.subsystem_head(latent),
        }


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
        self.action_classifier = nn.Linear(config.hidden_dim, config.num_action_classes)
        self.subsystem_classifier = nn.Linear(config.hidden_dim, config.num_subsystem_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encoder(x)
        h = context
        for _ in range(self.max_loops):
            h = self.loop(h, context)

        logits = self.classifier(h)
        steps = torch.full((x.size(0),), self.max_loops, dtype=torch.long, device=x.device)
        return {
            "logits": logits,
            "action_logits": self.action_classifier(h),
            "subsystem_logits": self.subsystem_classifier(h),
            "steps": steps,
        }


class AdaptiveLoopModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_loops = config.max_loops
        self.exit_threshold = config.exit_threshold
        self.encoder = WindowEncoder(config.input_dim, config.hidden_dim)
        self.loop = SharedLoopBlock(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.action_classifier = nn.Linear(config.hidden_dim, config.num_action_classes)
        self.subsystem_classifier = nn.Linear(config.hidden_dim, config.num_subsystem_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encoder(x)
        h = context
        batch_size = x.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        final_logits = torch.zeros(batch_size, self.classifier.out_features, device=x.device)
        final_action_logits = torch.zeros(batch_size, self.action_classifier.out_features, device=x.device)
        final_subsystem_logits = torch.zeros(batch_size, self.subsystem_classifier.out_features, device=x.device)
        last_exit_prob = torch.zeros(batch_size, device=x.device)

        for step_idx in range(1, self.max_loops + 1):
            h = self.loop(h, context)
            logits = self.classifier(h)
            action_logits = self.action_classifier(h)
            subsystem_logits = self.subsystem_classifier(h)
            confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
            last_exit_prob = confidence

            should_exit = confidence >= self.exit_threshold
            new_finished = should_exit & ~finished
            steps = torch.where(new_finished, torch.full_like(steps, step_idx), steps)
            final_logits = torch.where(new_finished.unsqueeze(-1), logits, final_logits)
            final_action_logits = torch.where(new_finished.unsqueeze(-1), action_logits, final_action_logits)
            final_subsystem_logits = torch.where(
                new_finished.unsqueeze(-1), subsystem_logits, final_subsystem_logits
            )
            finished = finished | should_exit

            if bool(finished.all()):
                break

        logits = self.classifier(h)
        action_logits = self.action_classifier(h)
        subsystem_logits = self.subsystem_classifier(h)
        steps = torch.where(steps == 0, torch.full_like(steps, self.max_loops), steps)
        final_logits = torch.where(finished.unsqueeze(-1), final_logits, logits)
        final_action_logits = torch.where(finished.unsqueeze(-1), final_action_logits, action_logits)
        final_subsystem_logits = torch.where(finished.unsqueeze(-1), final_subsystem_logits, subsystem_logits)
        return {
            "logits": final_logits,
            "action_logits": final_action_logits,
            "subsystem_logits": final_subsystem_logits,
            "steps": steps,
            "exit_probability": last_exit_prob,
        }
