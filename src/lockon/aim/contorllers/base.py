from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AimMetrics(ABC):
    @abstractmethod
    def as_dict(self) -> dict[str, float]:
        raise NotImplementedError


class AimController(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        info: dict[str, Any],
        frame_shape: tuple[int, int, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, AimMetrics] | None:
        raise NotImplementedError
