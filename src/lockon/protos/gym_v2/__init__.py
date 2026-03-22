from __future__ import annotations

from importlib import import_module
from typing import Any

from .validation import (
    validate_reset,
    validate_reset_reply,
    validate_step,
    validate_step_reply,
    validate_tensor_value,
)

__all__ = [
    "gym_env_pb2",
    "gym_env_pb2_grpc",
    "validate_reset",
    "validate_reset_reply",
    "validate_step",
    "validate_step_reply",
    "validate_tensor_value",
]


def __getattr__(name: str) -> Any:
    if name == "gym_env_pb2":
        return import_module(".gym_env_pb2", __name__)
    if name == "gym_env_pb2_grpc":
        return import_module(".gym_env_pb2_grpc", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
