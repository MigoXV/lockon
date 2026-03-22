from __future__ import annotations

from google.protobuf.struct_pb2 import Struct

from . import gym_env_pb2


def validate_tensor_value(value: gym_env_pb2.TensorValue, *, field_name: str) -> None:
    kind = value.WhichOneof("kind")
    if kind is None:
        raise ValueError(f"{field_name} must set one of tensor/list/dict")

    if kind != "dict":
        return

    for key in value.dict.fields:
        if key == "":
            raise ValueError(f"{field_name}.dict fields must not contain an empty key")


def validate_reset(reset: gym_env_pb2.Reset) -> None:
    if reset.options and reset.seed and len(reset.options) != len(reset.seed):
        raise ValueError("reset.options length must match reset.seed length when both are provided")

    for index, options in enumerate(reset.options):
        if not isinstance(options, Struct):
            raise ValueError(f"reset.options[{index}] must be a Struct")


def validate_reset_reply(reply: gym_env_pb2.ResetReply) -> None:
    validate_tensor_value(reply.observation, field_name="reset_reply.observation")


def validate_step(step: gym_env_pb2.Step) -> None:
    validate_tensor_value(step.action, field_name="step.action")


def validate_step_reply(reply: gym_env_pb2.StepReply) -> None:
    validate_tensor_value(reply.observation, field_name="step_reply.observation")

    reward_len = len(reply.reward)
    terminated_len = len(reply.terminated)
    truncated_len = len(reply.truncated)
    lengths = {reward_len, terminated_len, truncated_len}
    if len(lengths) != 1:
        raise ValueError("step_reply reward/terminated/truncated lengths must match")

    if reward_len == 0:
        raise ValueError("step_reply reward/terminated/truncated must contain at least one element")
