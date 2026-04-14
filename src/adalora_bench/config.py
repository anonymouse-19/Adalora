from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    output_dir: str = "outputs"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    warmup_ratio: float = 0.06
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    max_steps: int = -1
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_accumulation_steps: int = 1


@dataclass
class TarsConfig:
    enabled: bool = False
    r_max: int = 16
    schedule_steps: list[int] = field(default_factory=lambda: [500, 1000, 2000])
    momentum: float = 0.9
    epsilon: float = 1e-8


@dataclass
class ExperimentConfig:
    project_name: str = "adalora-bench"
    task_name: str = "sst2"
    model_name: str = "roberta-base"
    method_name: str = "lora"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_seq_length: int = 128
    generation_max_length: int = 128
    report_to: list[str] = field(default_factory=lambda: ["none"])
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "query", "value"])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tars: TarsConfig = field(default_factory=TarsConfig)



def _to_dataclass(config: dict[str, Any]) -> ExperimentConfig:
    training_raw = dict(config.get("training", {}))

    for key in ("eval_strategy", "save_strategy"):
        value = training_raw.get(key)
        if isinstance(value, bool):
            training_raw[key] = "epoch" if value else "no"

    training_cfg = TrainingConfig(**training_raw)
    tars_cfg = TarsConfig(**config.get("tars", {}))

    top_level = {k: v for k, v in config.items() if k not in {"training", "tars"}}
    return ExperimentConfig(**top_level, training=training_cfg, tars=tars_cfg)



def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config file must define a YAML mapping at top level.")
    return _to_dataclass(raw)
