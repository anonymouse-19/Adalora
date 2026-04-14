from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import inspect

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adalora_bench import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run AdaLoRA-Bench experiment")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import DatasetDict
    from transformers import Trainer, TrainingArguments, set_seed
    from adalora_bench.data.tasks import maybe_truncate, load_task_dataset
    from adalora_bench.eval.metrics import build_compute_metrics
    from adalora_bench.eval.profiler import count_parameters
    from adalora_bench.models.factory import build_model_and_tokenizer
    from adalora_bench.peft.registry import build_peft_method
    from adalora_bench.runner import build_data_collator, preprocess_datasets
    from adalora_bench.tars.scheduler import TarsRankScheduler, TarsTrainerCallback

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    task_spec, dataset = load_task_dataset(cfg.task_name)

    train_split = maybe_truncate(dataset["train"], cfg.max_train_samples)
    eval_key = "validation" if "validation" in dataset else "test"
    eval_split = maybe_truncate(dataset[eval_key], cfg.max_eval_samples)

    if task_spec.task_type in {"classification", "multiple_choice", "token_classification"}:
        label_feature = train_split.features[task_spec.label_column]
        num_labels = getattr(label_feature, "num_classes", None)
    else:
        num_labels = None

    model, tokenizer = build_model_and_tokenizer(cfg.model_name, task_spec.task_type, num_labels=num_labels)

    method_kwargs = {}
    if cfg.method_name in {"lora", "adalora"}:
        method_kwargs = {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
        }
    method = build_peft_method(cfg.method_name, **method_kwargs)

    if cfg.method_name != "full" and cfg.method_name != "zero_shot":
        model = method.apply(
            model,
            task_type=task_spec.task_type,
            target_modules=cfg.target_modules,
            tokenizer_name=cfg.model_name,
        )

    processed_train = preprocess_datasets(DatasetDict({"train": train_split}), tokenizer, task_spec, cfg)["train"]
    processed_eval = preprocess_datasets(DatasetDict({"validation": eval_split}), tokenizer, task_spec, cfg)["validation"]

    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels", "start_positions", "end_positions"}

    def _cleanup(ds):
        cols = [c for c in ds.column_names if c in keep_cols]
        return ds.remove_columns([c for c in ds.column_names if c not in cols])

    processed_train = _cleanup(processed_train)
    processed_eval = _cleanup(processed_eval)

    training_args_kwargs = {
        "output_dir": cfg.training.output_dir,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "num_train_epochs": cfg.training.num_train_epochs,
        "warmup_ratio": cfg.training.warmup_ratio,
        "save_strategy": cfg.training.save_strategy,
        "logging_steps": cfg.training.logging_steps,
        "max_steps": cfg.training.max_steps,
        "fp16": cfg.training.fp16,
        "bf16": cfg.training.bf16,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "report_to": cfg.report_to,
        "remove_unused_columns": False,
    }

    strategy_param = "eval_strategy"
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        strategy_param = "evaluation_strategy"
    training_args_kwargs[strategy_param] = cfg.training.eval_strategy

    training_args = TrainingArguments(**training_args_kwargs)

    compute_metrics = build_compute_metrics(cfg.task_name, task_spec.task_type)
    data_collator = build_data_collator(task_spec.task_type, tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": processed_train,
        "eval_dataset": processed_eval,
        "compute_metrics": compute_metrics,
        "data_collator": data_collator,
    }

    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    if cfg.tars.enabled:
        scheduler = TarsRankScheduler(
            r_max=cfg.tars.r_max,
            schedule_steps=set(cfg.tars.schedule_steps),
            momentum=cfg.tars.momentum,
            epsilon=cfg.tars.epsilon,
        )
        trainer.add_callback(TarsTrainerCallback(scheduler))

    if cfg.method_name != "zero_shot":
        trainer.train()

    eval_metrics = trainer.evaluate()
    param_stats = count_parameters(model)

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    output = {
        "task": cfg.task_name,
        "model": cfg.model_name,
        "method": cfg.method_name,
        "metrics": eval_metrics,
        "params": param_stats,
    }

    out_file = Path(cfg.training.output_dir) / "result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
