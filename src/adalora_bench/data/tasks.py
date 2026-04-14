from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import DatasetDict, load_dataset


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    subset: str | None
    task_type: str
    text_column: str | None = None
    text_pair_column: str | None = None
    label_column: str | None = "label"
    context_column: str | None = None
    question_column: str | None = None
    answers_column: str | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    "sst2": TaskSpec("glue", "sst2", "classification", text_column="sentence"),
    "mnli": TaskSpec("glue", "mnli", "classification", text_column="premise", text_pair_column="hypothesis"),
    "qqp": TaskSpec("glue", "qqp", "classification", text_column="question1", text_pair_column="question2"),
    "conll2003": TaskSpec("conll2003", None, "token_classification", text_column="tokens", label_column="ner_tags"),
    "squad_v2": TaskSpec(
        "squad_v2",
        None,
        "question_answering",
        question_column="question",
        context_column="context",
        answers_column="answers",
    ),
    "xsum": TaskSpec("xsum", None, "seq2seq", text_column="document", label_column="summary"),
    "hellaswag": TaskSpec("hellaswag", None, "multiple_choice", text_column="ctx", label_column="label"),
}


def load_task_dataset(task_name: str) -> tuple[TaskSpec, DatasetDict]:
    key = task_name.lower().strip()
    if key not in TASK_SPECS:
        raise ValueError(f"Unsupported task '{task_name}'. Available: {sorted(TASK_SPECS)}")

    spec = TASK_SPECS[key]
    if spec.subset:
        ds = load_dataset(spec.dataset, spec.subset)
    else:
        ds = load_dataset(spec.dataset)

    if "validation" not in ds and "validation_matched" in ds:
        ds = DatasetDict(
            {
                "train": ds["train"],
                "validation": ds["validation_matched"],
                "test": ds.get("test_matched", ds.get("test")),
            }
        )

    return spec, ds


def maybe_truncate(dataset_split: Any, max_samples: int | None):
    if max_samples is None:
        return dataset_split
    return dataset_split.select(range(min(max_samples, len(dataset_split))))
