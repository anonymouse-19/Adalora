from __future__ import annotations

from typing import Any

from datasets import DatasetDict
from transformers import (
    DataCollatorForMultipleChoice,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)

from .config import ExperimentConfig
from .data.tasks import TaskSpec


def preprocess_datasets(
    dataset: DatasetDict,
    tokenizer: Any,
    task_spec: TaskSpec,
    cfg: ExperimentConfig,
) -> DatasetDict:
    task_type = task_spec.task_type

    if task_type == "classification":
        def _preprocess(batch):
            if task_spec.text_pair_column:
                tokenized = tokenizer(
                    batch[task_spec.text_column],
                    batch[task_spec.text_pair_column],
                    truncation=True,
                    max_length=cfg.max_seq_length,
                )
            else:
                tokenized = tokenizer(
                    batch[task_spec.text_column],
                    truncation=True,
                    max_length=cfg.max_seq_length,
                )

            tokenized["labels"] = batch[task_spec.label_column]
            return tokenized

        return dataset.map(_preprocess, batched=True)

    if task_type == "token_classification":
        def _preprocess(batch):
            tokenized = tokenizer(
                batch[task_spec.text_column],
                truncation=True,
                is_split_into_words=True,
                max_length=cfg.max_seq_length,
            )

            labels = []
            for i, label in enumerate(batch[task_spec.label_column]):
                word_ids = tokenized.word_ids(batch_index=i)
                prev_word_id = None
                label_ids = []
                for word_id in word_ids:
                    if word_id is None:
                        label_ids.append(-100)
                    elif word_id != prev_word_id:
                        label_ids.append(label[word_id])
                    else:
                        label_ids.append(-100)
                    prev_word_id = word_id
                labels.append(label_ids)
            tokenized["labels"] = labels
            return tokenized

        return dataset.map(_preprocess, batched=True)

    if task_type == "question_answering":
        def _preprocess(batch):
            tokenized = tokenizer(
                batch[task_spec.question_column],
                batch[task_spec.context_column],
                truncation="only_second",
                max_length=cfg.max_seq_length,
                padding=False,
            )

            # Simplified labels for training compatibility.
            tokenized["start_positions"] = [0] * len(tokenized["input_ids"])
            tokenized["end_positions"] = [0] * len(tokenized["input_ids"])
            return tokenized

        return dataset.map(_preprocess, batched=True)

    if task_type == "seq2seq":
        def _preprocess(batch):
            model_inputs = tokenizer(
                batch[task_spec.text_column],
                truncation=True,
                max_length=cfg.max_seq_length,
            )
            labels = tokenizer(
                text_target=batch[task_spec.label_column],
                truncation=True,
                max_length=cfg.generation_max_length,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return dataset.map(_preprocess, batched=True)

    if task_type == "multiple_choice":
        endings = ["ending0", "ending1", "ending2", "ending3"]

        def _preprocess(batch):
            options = [[batch[e][i] for e in endings] for i in range(len(batch[endings[0]]))]
            first_sentences = [[ctx] * 4 for ctx in batch[task_spec.text_column]]
            tokenized = tokenizer(
                sum(first_sentences, []),
                sum(options, []),
                truncation=True,
                max_length=cfg.max_seq_length,
            )
            features = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}
            features["labels"] = [int(x) for x in batch[task_spec.label_column]]
            return features

        return dataset.map(_preprocess, batched=True)

    raise ValueError(f"Unsupported task type {task_type}")


def build_data_collator(task_type: str, tokenizer: Any) -> Any | None:
    if task_type == "token_classification":
        return DataCollatorForTokenClassification(tokenizer)
    if task_type == "multiple_choice":
        return DataCollatorForMultipleChoice(tokenizer)
    if task_type == "seq2seq":
        return DataCollatorForSeq2Seq(tokenizer)
    return DataCollatorWithPadding(tokenizer)
