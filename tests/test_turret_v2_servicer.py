from __future__ import annotations

import unittest

import numpy as np
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from lockon.protos.gym_env import gym_env_pb2 as gym_env_v1_pb2
from lockon.servicers.turret import create_servicer as create_v1_servicer
from lockon.protos.gym_v2 import gym_env_pb2
from lockon.servicers.turret_v2 import create_servicer


class _FakeContext:
    def abort(self, code: object, details: str) -> None:
        raise RuntimeError(f"{code}: {details}")


def _tensor(array: np.ndarray) -> gym_env_pb2.Tensor:
    arr = np.asarray(array)
    return gym_env_pb2.Tensor(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def _struct(**kwargs: object) -> Struct:
    msg = Struct()
    msg.update(kwargs)
    return msg


class TurretGymV2ServicerTests(unittest.TestCase):
    def test_v1_info_shape_is_unchanged(self) -> None:
        servicer = create_v1_servicer(camera_width=64, camera_height=48)
        context = _FakeContext()
        requests = iter(
            [
                gym_env_v1_pb2.EnvRequest(reset=gym_env_v1_pb2.Reset()),
                gym_env_v1_pb2.EnvRequest(
                    step=gym_env_v1_pb2.Step(
                        action=gym_env_v1_pb2.Tensor(
                            data=np.zeros(6, dtype=np.float32).tobytes(),
                            shape=[6],
                            dtype="float32",
                        )
                    )
                ),
                gym_env_v1_pb2.EnvRequest(close=gym_env_v1_pb2.Close()),
            ]
        )

        stream = servicer.StreamEnv(requests, context)
        try:
            next(stream)
            step_reply = next(stream)
            info = MessageToDict(step_reply.step.info, preserving_proto_field_name=True)
            self.assertNotIn("turret_motor_angles", info)
            self.assertNotIn("turret_motor_velocities", info)
        finally:
            servicer.close()

    def test_reset_and_step_include_turret_motor_vectors(self) -> None:
        servicer = create_servicer(camera_width=64, camera_height=48)
        context = _FakeContext()
        action = gym_env_pb2.TensorValue(tensor=_tensor(np.zeros(6, dtype=np.float32)))
        requests = iter(
            [
                gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()),
                gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=action)),
                gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()),
            ]
        )

        stream = servicer.StreamEnv(requests, context)
        try:
            reset_reply = next(stream)
            self.assertEqual(reset_reply.WhichOneof("result"), "reset")
            self.assertEqual(reset_reply.reset.observation.WhichOneof("kind"), "tensor")
            reset_info = MessageToDict(reset_reply.reset.info, preserving_proto_field_name=True)
            self.assertEqual(len(reset_info["turret_motor_angles"]), 2)
            self.assertEqual(len(reset_info["turret_motor_velocities"]), 2)

            step_reply = next(stream)
            self.assertEqual(step_reply.WhichOneof("result"), "step")
            self.assertEqual(step_reply.step.observation.WhichOneof("kind"), "tensor")
            self.assertEqual(len(step_reply.step.reward), 1)
            self.assertEqual(len(step_reply.step.terminated), 1)
            self.assertEqual(len(step_reply.step.truncated), 1)
            step_info = MessageToDict(step_reply.step.info, preserving_proto_field_name=True)
            self.assertEqual(len(step_info["turret_motor_angles"]), 2)
            self.assertEqual(len(step_info["turret_motor_velocities"]), 2)

            next(stream)
            with self.assertRaises(StopIteration):
                next(stream)
        finally:
            servicer.close()

    def test_batch_reset_options_and_step_return_batched_motor_vectors(self) -> None:
        servicer = create_servicer(camera_width=64, camera_height=48, observation_format="rgb")
        context = _FakeContext()
        action = gym_env_pb2.TensorValue(tensor=_tensor(np.zeros((2, 6), dtype=np.float32)))
        requests = iter(
            [
                gym_env_pb2.EnvRequest(
                    reset=gym_env_pb2.Reset(
                        seed=[1, 2],
                        options=[
                            _struct(qpos_noise_scale=0.0),
                            _struct(qpos_noise_scale=0.0),
                        ],
                    )
                ),
                gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=action)),
                gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()),
            ]
        )

        stream = servicer.StreamEnv(requests, context)
        try:
            reset_reply = next(stream)
            reset_info = MessageToDict(reset_reply.reset.info, preserving_proto_field_name=True)
            self.assertEqual(len(reset_info["turret_motor_angles"]), 2)
            self.assertEqual(len(reset_info["turret_motor_angles"][0]), 2)
            self.assertEqual(len(reset_info["turret_motor_velocities"]), 2)
            self.assertEqual(reset_reply.reset.observation.WhichOneof("kind"), "tensor")

            step_reply = next(stream)
            step_info = MessageToDict(step_reply.step.info, preserving_proto_field_name=True)
            self.assertEqual(len(step_reply.step.reward), 2)
            self.assertEqual(len(step_reply.step.terminated), 2)
            self.assertEqual(len(step_reply.step.truncated), 2)
            self.assertEqual(len(step_info["turret_motor_angles"]), 2)
            self.assertEqual(len(step_info["turret_motor_angles"][0]), 2)
            self.assertEqual(len(step_info["turret_motor_velocities"]), 2)

            next(stream)
            with self.assertRaises(StopIteration):
                next(stream)
        finally:
            servicer.close()
