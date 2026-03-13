from __future__ import annotations

from concurrent import futures
from typing import Iterator

import grpc
import numpy as np

from lockon.envs.turret import TurretEnv
from lockon.protos.gym_env import gym_env_pb2, gym_env_pb2_grpc
from lockon.servicers.turret.utils import (
    array_from_tensor,
    build_state_info,
    info_to_struct,
    scalar_tensor,
)
from lockon.vcodec import ObservationCodecConfig, ObservationFormat, create_observation_encoder


class TurretGymServicer(gym_env_pb2_grpc.ArmEnvServicer):
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

    def StreamEnv(
        self,
        request_iterator: Iterator[gym_env_pb2.EnvRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[gym_env_pb2.EnvReply]:
        env = TurretEnv(
            render_mode="rgb_array",
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            camera_fovy_deg=self.camera_fovy_deg,
        )
        encoder = create_observation_encoder(
            self.codec_config.observation_format,
            jpeg_quality=self.codec_config.jpeg_quality,
            h264_bitrate_kbps=self.codec_config.h264_bitrate_kbps,
            h264_gop=self.codec_config.h264_gop,
        )
        has_reset = False

        try:
            for request in request_iterator:
                cmd = request.WhichOneof("cmd")
                if cmd is None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request command is required")

                if cmd == "reset":
                    seed_values = list(request.reset.seed)
                    if len(seed_values) > 1:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "reset.seed accepts at most one value")

                    seed = seed_values[0] if seed_values else None
                    env.reset(seed=seed)
                    encoder.reset()
                    observation, _ = encoder.encode(env.render())
                    has_reset = True
                    yield gym_env_pb2.EnvReply(reset=gym_env_pb2.ResetReply(observation=observation))
                    continue

                if cmd == "step":
                    if not has_reset:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "reset must be called before step")

                    try:
                        action = array_from_tensor(request.step.action)
                    except ValueError as exc:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

                    if action.shape != (5,):
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "step.action must have shape [5]")

                    control_action = action[:4].astype(np.float32, copy=False)
                    fire_triggered = bool(float(action[4]) > 0.5)

                    reward, terminated, truncated, info = env.step_control(control_action)
                    observation, frame_info = encoder.encode(env.render())
                    state_info = build_state_info(env, info)
                    state_info.update(frame_info)

                    if fire_triggered:
                        fire_info = env.fire()
                        state_info["fire"] = {"triggered": True, **fire_info}
                        state_info["fire"].pop("hit_world", None)

                    yield gym_env_pb2.EnvReply(
                        step=gym_env_pb2.StepReply(
                            observation=observation,
                            reward=scalar_tensor(reward, "float32"),
                            terminated=scalar_tensor(terminated, "bool"),
                            truncated=scalar_tensor(truncated, "bool"),
                            info=info_to_struct(state_info),
                        )
                    )
                    continue

                if cmd == "close":
                    yield gym_env_pb2.EnvReply(close=gym_env_pb2.CloseReply())
                    break
        finally:
            encoder.close()
            env.close()


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
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    gym_env_pb2_grpc.add_ArmEnvServicer_to_server(
        TurretGymServicer(
            camera_width=camera_width,
            camera_height=camera_height,
            camera_fovy_deg=camera_fovy_deg,
            observation_format=observation_format,
            jpeg_quality=jpeg_quality,
            h264_bitrate_kbps=h264_bitrate_kbps,
            h264_gop=h264_gop,
        ),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server
