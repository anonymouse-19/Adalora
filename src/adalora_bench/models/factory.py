from __future__ import annotations

from typing import Any

from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


MODEL_TASK_MAPPING = {
    "classification": AutoModelForSequenceClassification,
    "multiple_choice": AutoModelForMultipleChoice,
    "token_classification": AutoModelForTokenClassification,
    "question_answering": AutoModelForQuestionAnswering,
    "seq2seq": AutoModelForSeq2SeqLM,
}


def build_model_and_tokenizer(
    model_name: str,
    task_type: str,
    num_labels: int | None = None,
) -> tuple[Any, Any]:
    if task_type not in MODEL_TASK_MAPPING:
        raise ValueError(f"No model mapping for task type '{task_type}'.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_cls = MODEL_TASK_MAPPING[task_type]

    kwargs: dict[str, Any] = {}
    if num_labels is not None and task_type in {"classification", "multiple_choice", "token_classification"}:
        kwargs["num_labels"] = num_labels

    model = model_cls.from_pretrained(model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
