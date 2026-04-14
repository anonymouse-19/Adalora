from __future__ import annotations

from typing import Any, Callable

from .methods.base import PEFTMethod

_METHODS: dict[str, type[PEFTMethod]] = {}


def register(name: str) -> Callable[[type[PEFTMethod]], type[PEFTMethod]]:
    def decorator(cls: type[PEFTMethod]) -> type[PEFTMethod]:
        _METHODS[name] = cls
        return cls

    return decorator


def build_peft_method(method_name: str, **kwargs) -> PEFTMethod:
    from .methods import impl  # noqa: F401

    key = method_name.lower().strip()
    if key not in _METHODS:
        raise ValueError(f"Unknown method '{method_name}'. Available: {sorted(_METHODS)}")
    return _METHODS[key](**kwargs)


def available_methods() -> list[str]:
    from .methods import impl  # noqa: F401

    return sorted(_METHODS)
