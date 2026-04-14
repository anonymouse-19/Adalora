from __future__ import annotations

from typing import Any

from peft import (
    AdaLoraConfig,
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)

from ..registry import register
from .base import PEFTMethod


TASK_TYPE_MAP = {
    "classification": TaskType.SEQ_CLS,
    "multiple_choice": TaskType.SEQ_CLS,
    "token_classification": TaskType.TOKEN_CLS,
    "question_answering": TaskType.QUESTION_ANS,
    "seq2seq": TaskType.SEQ_2_SEQ_LM,
}


def _task_type(task_type: str) -> TaskType:
    if task_type not in TASK_TYPE_MAP:
        raise ValueError(f"Unsupported task type for PEFT: {task_type}")
    return TASK_TYPE_MAP[task_type]


@register("full")
class FullFineTuneMethod(PEFTMethod):
    name = "full"

    def apply(self, model: Any, **kwargs) -> Any:
        return model


@register("zero_shot")
class ZeroShotMethod(PEFTMethod):
    name = "zero_shot"

    def apply(self, model: Any, **kwargs) -> Any:
        return model


@register("lora")
class LoRAMethod(PEFTMethod):
    name = "lora"

    def __init__(self, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout

    def apply(self, model: Any, **kwargs) -> Any:
        task_type = _task_type(kwargs["task_type"])
        target_modules = kwargs.get("target_modules") or ["q_proj", "v_proj", "query", "value"]
        config = LoraConfig(
            task_type=task_type,
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_modules,
            bias="none",
        )
        return get_peft_model(model, config)


@register("adalora")
class AdaLoRAMethod(PEFTMethod):
    name = "adalora"

    def __init__(self, init_r: int = 12, target_r: int = 8, alpha: int = 16, dropout: float = 0.05):
        self.init_r = init_r
        self.target_r = target_r
        self.alpha = alpha
        self.dropout = dropout

    def apply(self, model: Any, **kwargs) -> Any:
        task_type = _task_type(kwargs["task_type"])
        target_modules = kwargs.get("target_modules") or ["q_proj", "v_proj", "query", "value"]
        config = AdaLoraConfig(
            task_type=task_type,
            init_r=self.init_r,
            target_r=self.target_r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_modules,
            bias="none",
        )
        return get_peft_model(model, config)


@register("prefix")
class PrefixMethod(PEFTMethod):
    name = "prefix"

    def __init__(self, num_virtual_tokens: int = 10):
        self.num_virtual_tokens = num_virtual_tokens

    def apply(self, model: Any, **kwargs) -> Any:
        task_type = _task_type(kwargs["task_type"])
        config = PrefixTuningConfig(task_type=task_type, num_virtual_tokens=self.num_virtual_tokens)
        return get_peft_model(model, config)


@register("prompt")
class PromptMethod(PEFTMethod):
    name = "prompt"

    def __init__(self, num_virtual_tokens: int = 20):
        self.num_virtual_tokens = num_virtual_tokens

    def apply(self, model: Any, **kwargs) -> Any:
        task_type = _task_type(kwargs["task_type"])
        tokenizer_name = kwargs.get("tokenizer_name")
        config = PromptTuningConfig(
            task_type=task_type,
            num_virtual_tokens=self.num_virtual_tokens,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            tokenizer_name_or_path=tokenizer_name,
        )
        return get_peft_model(model, config)


@register("ia3")
class IA3Method(PEFTMethod):
    name = "ia3"

    def apply(self, model: Any, **kwargs) -> Any:
        task_type = _task_type(kwargs["task_type"])
        target_modules = kwargs.get("target_modules") or ["q_proj", "v_proj", "k_proj", "query", "value", "key"]
        config = IA3Config(task_type=task_type, target_modules=target_modules)
        return get_peft_model(model, config)


@register("adapters")
class AdaptersMethod(PEFTMethod):
    name = "adapters"

    def __init__(self, bottleneck_dim: int = 64):
        self.bottleneck_dim = bottleneck_dim

    def apply(self, model: Any, **kwargs) -> Any:
        raise NotImplementedError(
            "Houlsby adapters are not directly available through Hugging Face PEFT. "
            "Install adapter-transformers and integrate an adapter backend for this method."
        )
