from __future__ import annotations

import numpy as np

from lockon.protos.gym_env import gym_env_pb2


def tensor_from_array(array: np.ndarray) -> gym_env_pb2.Tensor:
    arr = np.asarray(array)
    return gym_env_pb2.Tensor(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def array_from_tensor(tensor: gym_env_pb2.Tensor) -> np.ndarray:
    try:
        dtype = np.dtype(tensor.dtype)
    except TypeError as exc:
        raise ValueError(f"unsupported tensor dtype: {tensor.dtype}") from exc

    item_count = int(np.prod(tensor.shape, dtype=np.int64)) if tensor.shape else 1
    expected_size = item_count * dtype.itemsize
    if len(tensor.data) != expected_size:
        raise ValueError(
            f"tensor byte size mismatch: expected {expected_size}, got {len(tensor.data)}"
        )

    return np.frombuffer(tensor.data, dtype=dtype).reshape(tuple(tensor.shape))
