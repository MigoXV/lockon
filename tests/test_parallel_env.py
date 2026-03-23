from __future__ import annotations

import unittest

import numpy as np
from google.protobuf.json_format import MessageToDict

from lockon.envs.turret import TurretEnv, TurretEnvConfig, make_vector_env
from lockon.protos.gym_env import gym_env_pb2
from lockon.servicers.turret import create_servicer
from lockon.utils import array_from_tensor, tensor_from_array


class _FakeContext:
    def abort(self, code: object, details: str) -> None:
        raise RuntimeError(f"{code}: {details}")


class TurretEnvBehaviorTests(unittest.TestCase):
    def test_reset_seed_render_and_truncation(self) -> None:
        env = TurretEnv(
            render_mode=None,
            max_episode_steps=3,
            qpos_reset_noise_scale=0.02,
            target_reset_noise_scale=0.02,
        )

        try:
            obs1, info1 = env.reset(seed=123)
            obs2, info2 = env.reset(seed=123)
            obs3, info3 = env.reset(seed=456)

            self.assertTrue(np.allclose(obs1, obs2))
            self.assertTrue(np.allclose(info1["qpos"], info2["qpos"]))
            self.assertFalse(np.allclose(obs1, obs3))
            self.assertIsNone(env.render())
            self.assertIsNone(env.renderer)

            _, _, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["elapsed_steps"], 1)

            _, _, terminated, truncated, _ = env.step(np.zeros(5, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)

            _, _, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["elapsed_steps"], 3)
        finally:
            env.close()

    def test_multiple_env_instances_do_not_share_state(self) -> None:
        env_a = TurretEnv(render_mode=None, qpos_reset_noise_scale=0.01)
        env_b = TurretEnv(render_mode=None, qpos_reset_noise_scale=0.01)

        try:
            obs_a, _ = env_a.reset(seed=7)
            obs_b, _ = env_b.reset(seed=8)
            self.assertFalse(np.allclose(obs_a, obs_b))

            env_a.step(np.ones(5, dtype=np.float32))
            env_b.step(-np.ones(5, dtype=np.float32))
            self.assertFalse(np.allclose(env_a.data.qpos[: env_a.qpos_size], env_b.data.qpos[: env_b.qpos_size]))
        finally:
            env_a.close()
            env_b.close()


class TurretVectorEnvTests(unittest.TestCase):
    def test_async_vector_env_sampling(self) -> None:
        config = TurretEnvConfig(
            render_mode=None,
            max_episode_steps=4,
            qpos_reset_noise_scale=0.01,
            target_reset_noise_scale=0.01,
        )
        vector_env = make_vector_env(4, base_seed=100, config=config, shared_memory=False, context="spawn")

        try:
            observations, infos = vector_env.reset()
            self.assertEqual(observations.shape, (4, 15))
            self.assertEqual(np.asarray(infos["qpos"]).shape, (4, 5))

            actions = np.zeros((4, 5), dtype=np.float32)
            truncated_seen = False

            for _ in range(5):
                observations, rewards, terminated, truncated, infos = vector_env.step(actions)
                self.assertEqual(observations.shape, (4, 15))
                self.assertEqual(rewards.shape, (4,))
                self.assertEqual(terminated.shape, (4,))
                self.assertEqual(truncated.shape, (4,))
                if np.any(truncated):
                    truncated_seen = True

            self.assertTrue(truncated_seen)
            self.assertEqual(np.asarray(infos["qpos"]).shape, (4, 5))
        finally:
            vector_env.close()


class TurretGrpcSessionTests(unittest.TestCase):
    def test_stream_env_sessions_are_isolated(self) -> None:
        servicer = create_servicer(camera_width=64, camera_height=48)
        context_a = _FakeContext()
        context_b = _FakeContext()

        zero_action = tensor_from_array(np.zeros(6, dtype=np.float32))
        requests_a = iter(
            [
                gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()),
                gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=zero_action)),
                gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()),
            ]
        )
        requests_b = iter(
            [
                gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()),
                gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=zero_action)),
                gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()),
            ]
        )

        stream_a = servicer.StreamEnv(requests_a, context_a)
        stream_b = servicer.StreamEnv(requests_b, context_b)

        try:
            next(stream_a)
            next(stream_b)

            reply_a = next(stream_a)
            reply_b = next(stream_b)
            info_a = MessageToDict(reply_a.step.info)
            info_b = MessageToDict(reply_b.step.info)

            self.assertEqual(info_a["qpos"], info_b["qpos"])
            self.assertEqual(info_a["targets"], info_b["targets"])

            next(stream_a)
            next(stream_b)
            with self.assertRaises(StopIteration):
                next(stream_a)
            with self.assertRaises(StopIteration):
                next(stream_b)
        finally:
            servicer.close()

    def test_reset_seed_dimension_controls_batch_size(self) -> None:
        servicer = create_servicer(camera_width=64, camera_height=48)
        context = _FakeContext()
        batch_action = tensor_from_array(np.zeros((4, 6), dtype=np.float32))
        requests = iter(
            [
                gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset(seed=[11, 12, 13, 14])),
                gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=batch_action)),
                gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()),
            ]
        )

        stream = servicer.StreamEnv(requests, context)
        try:
            reset_reply = next(stream)
            self.assertEqual(reset_reply.WhichOneof("result"), "reset")
            self.assertEqual(tuple(reset_reply.reset.observation.shape[:1]), (4,))

            step_reply = next(stream)
            self.assertEqual(step_reply.WhichOneof("result"), "step")
            self.assertEqual(array_from_tensor(step_reply.step.reward).shape, (4,))
            self.assertEqual(array_from_tensor(step_reply.step.terminated).shape, (4,))
            self.assertEqual(array_from_tensor(step_reply.step.truncated).shape, (4,))

            info = MessageToDict(step_reply.step.info)
            self.assertEqual(len(info["qpos"]), 4)
            self.assertEqual(len(info["targets"]), 4)
            elapsed_steps = info.get("elapsed_steps", info.get("elapsedSteps"))
            self.assertIsNotNone(elapsed_steps)
            self.assertEqual(len(elapsed_steps), 4)

            next(stream)
            with self.assertRaises(StopIteration):
                next(stream)
        finally:
            servicer.close()


if __name__ == "__main__":
    unittest.main()
