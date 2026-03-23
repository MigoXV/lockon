from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import grpc
import numpy as np

from lockon.envs.turret import TurretEnv, TurretEnvConfig
from lockon.protos.gym_env import gym_env_pb2, gym_env_pb2_grpc
from lockon.servicers.turret.utils import (
    array_from_tensor,
    build_batch_state_info,
    build_state_info,
    info_to_struct,
    scalar_tensor,
)
from lockon.utils import create_observation_encoder, tensor_from_array


@dataclass(slots=True)
class _SessionState:
    session_id: int
    envs: list[TurretEnv]
    encoder: object
    lock: RLock
    has_reset: bool = False
    closed: bool = False

    @property
    def env(self) -> TurretEnv | None:
        return self.envs[0] if self.envs else None

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.encoder.close()
            for env in self.envs:
                env.close()
            self.envs.clear()


class TurretGymServicer(gym_env_pb2_grpc.ArmEnvServicer):
    def __init__(
        self,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fovy_deg: float | None = None,
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fovy_deg = camera_fovy_deg
        self.env_config = TurretEnvConfig(
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            camera_fovy_deg=self.camera_fovy_deg,
            render_mode="rgb_array",
        )
        self._sessions_lock = RLock()
        self._sessions: dict[int, _SessionState] = {}
        self._next_session_id = 0
        self._viewer_session_id: int | None = None

    def _create_session(self) -> _SessionState:
        session = _SessionState(
            session_id=self._next_session_id,
            envs=[],
            encoder=create_observation_encoder(),
            lock=RLock(),
        )
        with self._sessions_lock:
            self._next_session_id += 1
            self._sessions[session.session_id] = session
            if self._viewer_session_id is None:
                self._viewer_session_id = session.session_id
        return session

    def _build_env(self) -> TurretEnv:
        return TurretEnv(**self.env_config.to_kwargs())

    def _configure_session_envs(self, session: _SessionState, env_count: int) -> None:
        if env_count <= 0:
            raise ValueError(f"env_count must be positive, got {env_count}")
        if len(session.envs) == env_count:
            return

        for env in session.envs:
            env.close()
        session.envs = [self._build_env() for _ in range(env_count)]
        session.has_reset = False
        session.encoder.reset()

    def _encode_observation(self, session: _SessionState) -> tuple[gym_env_pb2.Tensor, dict[str, object]]:
        frames = [env.render() for env in session.envs]
        if len(frames) == 1:
            return session.encoder.encode(frames[0])
        return session.encoder.encode(np.stack(frames, axis=0))

    def _release_session(self, session: _SessionState) -> None:
        with self._sessions_lock:
            self._sessions.pop(session.session_id, None)
            if self._viewer_session_id == session.session_id:
                self._viewer_session_id = next(iter(self._sessions), None)

    def get_viewer_session(self) -> _SessionState | None:
        with self._sessions_lock:
            if self._viewer_session_id is None:
                return None
            return self._sessions.get(self._viewer_session_id)

    def StreamEnv(
        self,
        request_iterator: Iterator[gym_env_pb2.EnvRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[gym_env_pb2.EnvReply]:
        session = self._create_session()
        try:
            for request in request_iterator:
                cmd = request.WhichOneof("cmd")
                if cmd is None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request command is required")

                if cmd == "reset":
                    seed_values = list(request.reset.seed)
                    env_count = len(seed_values) if seed_values else 1

                    seeds: list[int | None] = [int(seed) for seed in seed_values] if seed_values else [None]
                    with session.lock:
                        self._configure_session_envs(session, env_count)
                        for env, seed in zip(session.envs, seeds, strict=True):
                            env.reset(seed=seed)
                        session.encoder.reset()
                        observation, _ = self._encode_observation(session)
                        session.has_reset = True
                    yield gym_env_pb2.EnvReply(reset=gym_env_pb2.ResetReply(observation=observation))
                    continue

                if cmd == "step":
                    if not session.has_reset:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "reset must be called before step")

                    try:
                        action = array_from_tensor(request.step.action)
                    except ValueError as exc:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

                    env_count = len(session.envs)
                    if env_count == 1:
                        if action.shape != (6,):
                            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "step.action must have shape [6]")
                        action_batch = action.reshape(1, 6)
                    else:
                        if action.shape != (env_count, 6):
                            context.abort(
                                grpc.StatusCode.INVALID_ARGUMENT,
                                f"step.action must have shape [{env_count}, 6]",
                            )
                        action_batch = action.astype(np.float32, copy=False)

                    with session.lock:
                        rewards: list[float] = []
                        terminated_flags: list[bool] = []
                        truncated_flags: list[bool] = []
                        infos: list[dict[str, object]] = []
                        fire_infos: list[dict[str, object]] = []

                        for env, env_action in zip(session.envs, action_batch, strict=True):
                            reward, terminated, truncated, info = env.step_control(env_action[:5].astype(np.float32, copy=False))
                            rewards.append(reward)
                            terminated_flags.append(terminated)
                            truncated_flags.append(truncated)
                            infos.append(info)

                            fire_triggered = bool(float(env_action[5]) > 0.5)
                            fire_payload: dict[str, object] = {"triggered": False}
                            if fire_triggered:
                                fire_payload = {"triggered": True, **env.fire()}
                                fire_payload.pop("hit_world", None)
                            fire_infos.append(fire_payload)

                        observation, frame_info = self._encode_observation(session)

                        if env_count == 1:
                            state_info = build_state_info(session.envs[0], infos[0])
                            state_info["fire"] = fire_infos[0]
                            reward_tensor = scalar_tensor(rewards[0], "float32")
                            terminated_tensor = scalar_tensor(terminated_flags[0], "bool")
                            truncated_tensor = scalar_tensor(truncated_flags[0], "bool")
                        else:
                            state_info = build_batch_state_info(session.envs, infos)
                            state_info["fire"] = fire_infos
                            reward_tensor = tensor_from_array(np.asarray(rewards, dtype=np.float32))
                            terminated_tensor = tensor_from_array(np.asarray(terminated_flags, dtype=np.bool_))
                            truncated_tensor = tensor_from_array(np.asarray(truncated_flags, dtype=np.bool_))

                        state_info.update(frame_info)

                    yield gym_env_pb2.EnvReply(
                        step=gym_env_pb2.StepReply(
                            observation=observation,
                            reward=reward_tensor,
                            terminated=terminated_tensor,
                            truncated=truncated_tensor,
                            info=info_to_struct(state_info),
                        )
                    )
                    continue

                if cmd == "close":
                    yield gym_env_pb2.EnvReply(close=gym_env_pb2.CloseReply())
                    break
        finally:
            self._release_session(session)
            session.close()

    def close(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._viewer_session_id = None

        for session in sessions:
            session.close()


def create_servicer(
    *,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_fovy_deg: float | None = None,
) -> TurretGymServicer:
    return TurretGymServicer(
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy_deg,
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 50051,
    *,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_fovy_deg: float | None = None,
    servicer: TurretGymServicer | None = None,
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    turret_servicer = servicer or create_servicer(
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy_deg,
    )
    gym_env_pb2_grpc.add_ArmEnvServicer_to_server(
        turret_servicer,
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server
