from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterator

import grpc
import numpy as np
from google.protobuf.json_format import MessageToDict

from lockon.envs.turret import TurretEnv, TurretEnvConfig
from lockon.protos.gym_v2 import gym_env_pb2, gym_env_pb2_grpc
from lockon.protos.gym_v2.validation import validate_reset, validate_step
from lockon.servicers.turret.utils import build_batch_state_info, build_state_info, info_to_struct
from lockon.vcodec import ObservationCodecConfig, ObservationFormat, create_observation_encoder


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
        raise ValueError(f"tensor byte size mismatch: expected {expected_size}, got {len(tensor.data)}")

    return np.frombuffer(tensor.data, dtype=dtype).reshape(tuple(tensor.shape))


def tensor_value_from_array(array: np.ndarray) -> gym_env_pb2.TensorValue:
    return gym_env_pb2.TensorValue(tensor=tensor_from_array(array))


def _append_v2_motor_info(state_info: dict[str, object], infos: list[dict[str, object]], env_count: int) -> None:
    if env_count == 1:
        state_info["turret_motor_angles"] = infos[0]["turret_motor_angles"].tolist()
        state_info["turret_motor_velocities"] = infos[0]["turret_motor_velocities"].tolist()
        return

    state_info["turret_motor_angles"] = [info["turret_motor_angles"].tolist() for info in infos]
    state_info["turret_motor_velocities"] = [info["turret_motor_velocities"].tolist() for info in infos]


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


class TurretGymV2Servicer(gym_env_pb2_grpc.GymEnvServicer):
    def __init__(
        self,
        observation_format: str = "rgb",
        jpeg_quality: int = 80,
        h264_bitrate_kbps: int = 4000,
        h264_gop: int = 30,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fovy_deg: float | None = None,
    ) -> None:
        self.codec_config = ObservationCodecConfig(
            observation_format=ObservationFormat.from_value(observation_format),
            jpeg_quality=jpeg_quality,
            h264_bitrate_kbps=h264_bitrate_kbps,
            h264_gop=h264_gop,
        )
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
            encoder=create_observation_encoder(
                self.codec_config.observation_format,
                jpeg_quality=self.codec_config.jpeg_quality,
                h264_bitrate_kbps=self.codec_config.h264_bitrate_kbps,
                h264_gop=self.codec_config.h264_gop,
            ),
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

    def _encode_observation(self, session: _SessionState) -> tuple[gym_env_pb2.TensorValue, dict[str, object]]:
        frames = [env.render() for env in session.envs]
        if len(frames) == 1:
            observation, frame_info = session.encoder.encode(frames[0])
            return gym_env_pb2.TensorValue(tensor=tensor_from_array(array_from_tensor(observation))), frame_info
        if self.codec_config.observation_format is not ObservationFormat.RGB:
            raise RuntimeError("batched sessions require rgb observation format")
        observation, frame_info = session.encoder.encode(np.stack(frames, axis=0))
        return gym_env_pb2.TensorValue(tensor=tensor_from_array(array_from_tensor(observation))), frame_info

    def _build_reset_info(
        self,
        session: _SessionState,
        env_count: int,
        infos: list[dict[str, object]],
        frame_info: dict[str, object],
    ) -> dict[str, object]:
        if env_count == 1:
            state_info = build_state_info(session.envs[0], infos[0])
        else:
            state_info = build_batch_state_info(session.envs, infos)
        _append_v2_motor_info(state_info, infos, env_count)
        state_info.update(frame_info)
        return state_info

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

    def _coerce_reset_options(self, options: list[Any], env_count: int) -> list[dict[str, Any] | None]:
        if not options:
            return [None] * env_count
        payloads = [MessageToDict(item, preserving_proto_field_name=True) for item in options]
        if len(payloads) != env_count:
            raise ValueError("reset.options length must match resolved batch size")
        return payloads

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
                    try:
                        validate_reset(request.reset)
                    except ValueError as exc:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

                    seed_values = list(request.reset.seed)
                    env_count = max(len(seed_values), len(request.reset.options), 1)
                    if env_count > 1 and self.codec_config.observation_format is not ObservationFormat.RGB:
                        context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "batched sessions require rgb observation format",
                        )

                    seeds: list[int | None]
                    if seed_values:
                        seeds = [int(seed) for seed in seed_values]
                    else:
                        seeds = [None] * env_count

                    try:
                        options = self._coerce_reset_options(list(request.reset.options), env_count)
                    except ValueError as exc:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

                    with session.lock:
                        self._configure_session_envs(session, env_count)
                        infos: list[dict[str, object]] = []
                        for env, seed, options_payload in zip(session.envs, seeds, options, strict=True):
                            _, info = env.reset(seed=seed, options=options_payload)
                            infos.append(info)
                        session.encoder.reset()
                        observation, frame_info = self._encode_observation(session)
                        reset_info = self._build_reset_info(session, env_count, infos, frame_info)
                        session.has_reset = True
                    yield gym_env_pb2.EnvReply(
                        reset=gym_env_pb2.ResetReply(
                            observation=observation,
                            info=info_to_struct(reset_info),
                        )
                    )
                    continue

                if cmd == "step":
                    if not session.has_reset:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "reset must be called before step")

                    try:
                        validate_step(request.step)
                    except ValueError as exc:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

                    if request.step.action.WhichOneof("kind") != "tensor":
                        context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "turret_v2 currently only accepts tensor-valued actions",
                        )

                    try:
                        action = array_from_tensor(request.step.action.tensor)
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
                            reward, terminated, truncated, info = env.step_control(
                                env_action[:5].astype(np.float32, copy=False)
                            )
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
                        else:
                            state_info = build_batch_state_info(session.envs, infos)
                            state_info["fire"] = fire_infos

                        _append_v2_motor_info(state_info, infos, env_count)
                        state_info.update(frame_info)

                    yield gym_env_pb2.EnvReply(
                        step=gym_env_pb2.StepReply(
                            observation=observation,
                            reward=rewards,
                            terminated=terminated_flags,
                            truncated=truncated_flags,
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
    observation_format: str = "rgb",
    jpeg_quality: int = 80,
    h264_bitrate_kbps: int = 4000,
    h264_gop: int = 30,
) -> TurretGymV2Servicer:
    return TurretGymV2Servicer(
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy_deg,
        observation_format=observation_format,
        jpeg_quality=jpeg_quality,
        h264_bitrate_kbps=h264_bitrate_kbps,
        h264_gop=h264_gop,
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 50051,
    *,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_fovy_deg: float | None = None,
    observation_format: str = "rgb",
    jpeg_quality: int = 80,
    h264_bitrate_kbps: int = 4000,
    h264_gop: int = 30,
    servicer: TurretGymV2Servicer | None = None,
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    turret_servicer = servicer or create_servicer(
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fovy_deg=camera_fovy_deg,
        observation_format=observation_format,
        jpeg_quality=jpeg_quality,
        h264_bitrate_kbps=h264_bitrate_kbps,
        h264_gop=h264_gop,
    )
    gym_env_pb2_grpc.add_GymEnvServicer_to_server(
        turret_servicer,
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server
