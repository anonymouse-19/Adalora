# AdaLoRA-Bench

This repository contains a runnable implementation scaffold for the project described in the paper **"AdaLoRA-Bench: A Unified Benchmarking Framework for Parameter-Efficient Fine-Tuning Methods on Heterogeneous NLP Tasks"**.

## What is implemented

- Modular 4-layer benchmark design:
  - Data layer: standardized task loading via Hugging Face datasets
  - Model layer: unified model factory for classification, token classification, QA, and seq2seq
  - PEFT layer: plugin-style methods (`full`, `zero_shot`, `lora`, `adalora`, `prefix`, `prompt`, `ia3`)
  - Evaluation layer: metrics + parameter-efficiency profiling
- TARS implementation:
  - Gradient-signal-based per-layer score tracking
  - Scheduled dynamic rank masking callback for LoRA layers
- Config-driven experiment execution with YAML
- JSON output logs for reproducible runs

## Structure

- `src/adalora_bench/config.py`: experiment config dataclasses and loader
- `src/adalora_bench/data/tasks.py`: benchmark task registry and dataset loader
- `src/adalora_bench/models/factory.py`: model and tokenizer creation
- `src/adalora_bench/peft/methods/impl.py`: PEFT method implementations
- `src/adalora_bench/tars/scheduler.py`: TARS scheduler + trainer callback
- `src/adalora_bench/eval/metrics.py`: task-aware metric functions
- `scripts/run_benchmark.py`: command-line runner
- `configs/default.yaml`: sample experiment config

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python scripts/run_benchmark.py --config configs/default.yaml
```

Or via module entrypoint:

```bash
python -m adalora_bench --config configs/default.yaml
```

## Notes

- `adapters` (Houlsby adapters) is included as a named method placeholder but requires integrating `adapter-transformers`, since it is not provided directly by the core `peft` package.
- QA training labels are currently a simplified scaffold. For full SQuAD-v2 fidelity, add span alignment logic using offset mappings.
- This scaffold is intended to be reproducible and extensible, and can be expanded with multi-seed loops, profiling hooks, and W&B logging.
