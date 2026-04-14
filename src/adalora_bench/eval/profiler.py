from __future__ import annotations

from typing import Any


def count_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": (trainable / total * 100.0) if total else 0.0,
    }
