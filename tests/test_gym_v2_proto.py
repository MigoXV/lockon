from __future__ import annotations

import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct

from lockon.protos.gym_env import gym_env_pb2 as gym_env_v1_pb2
from lockon.protos.gym_v2 import (
    gym_env_pb2,
    gym_env_pb2_grpc,
    validate_reset,
    validate_reset_reply,
    validate_step,
    validate_step_reply,
    validate_tensor_value,
)


def _tensor(array: object, *, dtype: str | None = None) -> gym_env_pb2.Tensor:
    arr = np.asarray(array, dtype=dtype)
    return gym_env_pb2.Tensor(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def _struct(**kwargs: object) -> Struct:
    msg = Struct()
    msg.update(kwargs)
    return msg


def test_reset_reply_tensor_roundtrip_with_info() -> None:
    reply = gym_env_pb2.ResetReply(
        observation=gym_env_pb2.TensorValue(tensor=_tensor([[1.0, 2.0]], dtype="float32")),
        info=_struct(codec="rgb"),
    )

    validate_reset_reply(reply)

    clone = gym_env_pb2.ResetReply()
    clone.ParseFromString(reply.SerializeToString())

    assert clone.observation.WhichOneof("kind") == "tensor"
    assert list(clone.observation.tensor.shape) == [1, 2]
    assert clone.info["codec"] == "rgb"


def test_step_accepts_tensor_list_action() -> None:
    step = gym_env_pb2.Step(
        action=gym_env_pb2.TensorValue(
            list=gym_env_pb2.TensorList(
                items=[
                    _tensor([1.0, 2.0], dtype="float32"),
                    _tensor([3.0, 4.0], dtype="float32"),
                ]
            )
        )
    )

    validate_step(step)

    clone = gym_env_pb2.Step()
    clone.ParseFromString(step.SerializeToString())

    assert clone.action.WhichOneof("kind") == "list"
    assert len(clone.action.list.items) == 2


def test_step_reply_accepts_tensor_dict_observation() -> None:
    reply = gym_env_pb2.StepReply(
        observation=gym_env_pb2.TensorValue(
            dict=gym_env_pb2.TensorDict(
                fields={
                    "obs": _tensor([1.0, 2.0], dtype="float32"),
                    "action_mask": _tensor([1, 0, 1], dtype="uint8"),
                }
            )
        ),
        reward=[0.5],
        terminated=[False],
        truncated=[True],
        info=_struct(source="dict"),
    )

    validate_step_reply(reply)

    clone = gym_env_pb2.StepReply()
    clone.ParseFromString(reply.SerializeToString())

    assert clone.observation.WhichOneof("kind") == "dict"
    assert set(clone.observation.dict.fields) == {"obs", "action_mask"}
    assert list(clone.reward) == [pytest.approx(0.5)]
    assert list(clone.terminated) == [False]
    assert list(clone.truncated) == [True]
    assert clone.info["source"] == "dict"


def test_batch_reset_accepts_matching_seed_and_options_lengths() -> None:
    reset = gym_env_pb2.Reset(
        seed=[11, 12, 13],
        options=[
            _struct(mode="a"),
            _struct(mode="b"),
            _struct(mode="c"),
        ],
    )

    validate_reset(reset)

    clone = gym_env_pb2.Reset()
    clone.ParseFromString(reset.SerializeToString())

    assert list(clone.seed) == [11, 12, 13]
    assert [item["mode"] for item in clone.options] == ["a", "b", "c"]


def test_step_reply_repeated_fields_can_encode_batch_values() -> None:
    reply = gym_env_pb2.StepReply(
        observation=gym_env_pb2.TensorValue(tensor=_tensor(np.zeros((3, 4), dtype=np.float32))),
        reward=[1.0, 2.0, 3.0],
        terminated=[False, True, False],
        truncated=[False, False, True],
        info=_struct(batch=True),
    )

    validate_step_reply(reply)

    assert len(reply.reward) == 3
    assert len(reply.terminated) == 3
    assert len(reply.truncated) == 3


def test_validate_tensor_value_rejects_unset_kind() -> None:
    with pytest.raises(ValueError, match="must set one of tensor/list/dict"):
        validate_tensor_value(gym_env_pb2.TensorValue(), field_name="action")


def test_validate_tensor_value_rejects_empty_dict_key() -> None:
    value = gym_env_pb2.TensorValue(
        dict=gym_env_pb2.TensorDict(fields={"": _tensor([1], dtype="int64")})
    )

    with pytest.raises(ValueError, match="must not contain an empty key"):
        validate_tensor_value(value, field_name="action")


def test_validate_reset_rejects_mismatched_options_length() -> None:
    reset = gym_env_pb2.Reset(
        seed=[1, 2],
        options=[_struct(mode="only-one")],
    )

    with pytest.raises(ValueError, match="must match reset.seed length"):
        validate_reset(reset)


def test_validate_step_reply_rejects_mismatched_repeated_lengths() -> None:
    reply = gym_env_pb2.StepReply(
        observation=gym_env_pb2.TensorValue(tensor=_tensor([1.0], dtype="float32")),
        reward=[1.0, 2.0],
        terminated=[False],
        truncated=[False, False],
    )

    with pytest.raises(ValueError, match="lengths must match"):
        validate_step_reply(reply)


def test_v1_and_v2_proto_coexist() -> None:
    assert gym_env_pb2.ResetReply.DESCRIPTOR.fields_by_name["observation"].message_type.full_name == "gym_v2.TensorValue"
    assert (
        gym_env_v1_pb2.ResetReply.DESCRIPTOR.fields_by_name["observation"].message_type.full_name
        == "arm_stream.Tensor"
    )
    assert hasattr(gym_env_pb2_grpc, "GymEnvStub")
