from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PEFTMethod(ABC):
    name: str

    @abstractmethod
    def apply(self, model: Any, **kwargs) -> Any:
        raise NotImplementedError
