from __future__ import annotations

from collections.abc import Callable
from typing import Any

import evaluate
import numpy as np


def _classification_metrics(task_name: str) -> Callable[[tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    metric_name = "accuracy" if task_name in {"sst2", "mnli", "qqp", "hellaswag"} else "f1"
    metric = evaluate.load(metric_name)

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    return compute


def _token_cls_metrics() -> Callable[[tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    seqeval = evaluate.load("seqeval")

    def compute(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)

        true_predictions = []
        true_labels = []
        for pred, lab in zip(predictions, labels):
            pred_filtered = []
            lab_filtered = []
            for p, l in zip(pred, lab):
                if l == -100:
                    continue
                pred_filtered.append(str(int(p)))
                lab_filtered.append(str(int(l)))
            true_predictions.append(pred_filtered)
            true_labels.append(lab_filtered)

        result = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"f1": float(result.get("overall_f1", 0.0))}

    return compute


def build_compute_metrics(task_name: str, task_type: str) -> Callable[[Any], dict[str, float]] | None:
    if task_type in {"classification", "multiple_choice"}:
        return _classification_metrics(task_name)
    if task_type == "token_classification":
        return _token_cls_metrics()
    return None
