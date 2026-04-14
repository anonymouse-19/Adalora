from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import TrainerCallback


@dataclass
class _LayerState:
    score: float = 0.0
    rank: int = 0


@dataclass
class TarsRankScheduler:
    r_max: int
    schedule_steps: set[int]
    momentum: float = 0.9
    epsilon: float = 1e-8
    layer_state: dict[str, _LayerState] = field(default_factory=dict)

    def update_scores(self, model: Any) -> None:
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue

            grad_norm = 0.0
            adapter_keys = list(module.lora_A.keys())
            if not adapter_keys:
                continue
            key = adapter_keys[0]

            grad_a = module.lora_A[key].weight.grad
            grad_b = module.lora_B[key].weight.grad
            if grad_a is not None:
                grad_norm += float(torch.norm(grad_a, p="fro").item())
            if grad_b is not None:
                grad_norm += float(torch.norm(grad_b, p="fro").item())

            if name not in self.layer_state:
                self.layer_state[name] = _LayerState(score=grad_norm, rank=self.r_max)
            else:
                prev = self.layer_state[name].score
                self.layer_state[name].score = self.momentum * prev + (1.0 - self.momentum) * grad_norm

    def maybe_apply_schedule(self, model: Any, step: int) -> dict[str, int]:
        if step not in self.schedule_steps:
            return {}

        if not self.layer_state:
            return {}

        scores = torch.tensor([s.score for s in self.layer_state.values()], dtype=torch.float32)
        mean = scores.mean().item()
        std = scores.std(unbiased=False).item()

        chosen_ranks: dict[str, int] = {}

        for name, st in self.layer_state.items():
            z = (st.score - mean) / (std + self.epsilon)
            rank = int(round(self.r_max * torch.sigmoid(torch.tensor(z)).item()))
            rank = max(1, min(self.r_max, rank))
            st.rank = rank
            chosen_ranks[name] = rank

        self._apply_masks(model, chosen_ranks)
        return chosen_ranks

    def _apply_masks(self, model: Any, chosen_ranks: dict[str, int]) -> None:
        for name, module in model.named_modules():
            if name not in chosen_ranks:
                continue
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue

            keys = list(module.lora_A.keys())
            if not keys:
                continue
            key = keys[0]
            rank = chosen_ranks[name]

            weight_a = module.lora_A[key].weight.data
            weight_b = module.lora_B[key].weight.data

            if weight_a.size(0) > rank:
                weight_a[rank:, :] = 0.0
            if weight_b.size(1) > rank:
                weight_b[:, rank:] = 0.0


class TarsTrainerCallback(TrainerCallback):
    def __init__(self, scheduler: TarsRankScheduler):
        self.scheduler = scheduler

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        self.scheduler.update_scores(model)
        self.scheduler.maybe_apply_schedule(model, int(state.global_step))
        return control
