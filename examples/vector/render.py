from __future__ import annotations

import argparse
import os
import queue
import time
from pathlib import Path
from typing import Iterator

import cv2
import grpc
import numpy as np
from google.protobuf.json_format import MessageToDict

from lockon.protos.gym_env import gym_env_pb2, gym_env_pb2_grpc
from lockon.utils import array_from_tensor, tensor_from_array
from lockon.vcodec import create_observation_decoder


DEFAULT_SERVER_ADDR = os.getenv("LOCKON_SERVER_ADDR", "127.0.0.1:50051")
WINDOW_NAME = "Lockon Vector Client Render"
ENV_COUNT = 4
_STREAM_END = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render four remote turret environments in a single tiled window.")
    parser.add_argument("--server-addr", default=DEFAULT_SERVER_ADDR, help="gRPC server address.")
    parser.add_argument("--steps", type=int, default=0, help="Number of batched steps to run. 0 means run until quit.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reset. The session uses 4 consecutive seeds.")
    parser.add_argument("--fps", type=float, default=30.0, help="Viewer refresh rate.")
    parser.add_argument("--headless", action="store_true", help="Run without opening an OpenCV window.")
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=None,
        help="Optional path used to save the final tiled frame as a PNG or JPEG image.",
    )
    return parser.parse_args()


def _request_iterator(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
) -> Iterator[gym_env_pb2.EnvRequest]:
    while True:
        item = request_queue.get()
        if item is _STREAM_END:
            return
        yield item


def build_actions(step_idx: int, num_envs: int) -> np.ndarray:
    phases = np.linspace(0.0, np.pi, num_envs, endpoint=False, dtype=np.float32)
    t = step_idx * 0.06
    actions = np.zeros((num_envs, 6), dtype=np.float32)
    actions[:, 0] = 0.35 * np.sin(t + phases)
    actions[:, 1] = 0.20 * np.cos(t * 0.7 + phases * 0.5)
    actions[:, 2] = 0.45 * np.sin(t * 0.5 + phases * 1.3)
    actions[:, 3] = 0.85 * np.sin(t * 1.1 + phases)
    actions[:, 4] = 0.65 * np.cos(t * 0.9 + phases * 0.8)
    actions[:, 5] = 0.0
    return np.clip(actions, -1.0, 1.0)


def annotate_frame(
    frame_rgb: np.ndarray,
    env_idx: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    elapsed_steps: int,
    max_episode_steps: int,
    bullseye_pixel: object,
) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    status = "running"
    if terminated:
        status = "terminated"
    elif truncated:
        status = "truncated"

    if isinstance(bullseye_pixel, list) and len(bullseye_pixel) == 2:
        px = int(float(bullseye_pixel[0]))
        py = int(float(bullseye_pixel[1]))
        cv2.circle(frame_bgr, (px, py), 5, (255, 0, 0), -1)

    cv2.putText(frame_bgr, f"env {env_idx}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        f"reward={reward:.3f}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 230, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"step={elapsed_steps}/{max_episode_steps}",
        (12, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 200, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"status={status}",
        (12, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 180, 255) if status == "running" else (40, 120, 255),
        1,
        cv2.LINE_AA,
    )
    return frame_bgr


def tile_frames(frames_bgr: list[np.ndarray], *, gap: int = 8) -> np.ndarray:
    if len(frames_bgr) != ENV_COUNT:
        raise ValueError(f"expected exactly {ENV_COUNT} frames, got {len(frames_bgr)}")

    height, width = frames_bgr[0].shape[:2]
    canvas = np.full((height * 2 + gap * 3, width * 2 + gap * 3, 3), 18, dtype=np.uint8)
    positions = (
        (gap, gap),
        (gap * 2 + width, gap),
        (gap, gap * 2 + height),
        (gap * 2 + width, gap * 2 + height),
    )

    for frame, (x, y) in zip(frames_bgr, positions, strict=True):
        canvas[y : y + height, x : x + width] = frame

    return canvas


def save_frame(path: Path, frame_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"failed to save frame to {path}")


def _send_reset(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
    responses: Iterator[gym_env_pb2.EnvReply],
    *,
    base_seed: int,
) -> gym_env_pb2.Tensor:
    reset_request = gym_env_pb2.Reset(seed=[base_seed + idx for idx in range(ENV_COUNT)])
    request_queue.put(gym_env_pb2.EnvRequest(reset=reset_request))
    reply = next(responses)
    if reply.WhichOneof("result") != "reset":
        raise RuntimeError("expected ResetReply")
    return reply.reset.observation


def _send_step(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
    responses: Iterator[gym_env_pb2.EnvReply],
    action: np.ndarray,
) -> dict[str, object]:
    request_queue.put(gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=tensor_from_array(action.astype(np.float32)))))
    reply = next(responses)
    if reply.WhichOneof("result") != "step":
        raise RuntimeError("expected StepReply")

    info = MessageToDict(reply.step.info, preserving_proto_field_name=True)
    return {
        "observation": reply.step.observation,
        "info": info,
        "reward": np.asarray(array_from_tensor(reply.step.reward), dtype=np.float32),
        "terminated": np.asarray(array_from_tensor(reply.step.terminated), dtype=bool),
        "truncated": np.asarray(array_from_tensor(reply.step.truncated), dtype=bool),
    }


def _decode_frames(
    observation: gym_env_pb2.Tensor,
    info: dict[str, object],
    decoder,
) -> tuple[np.ndarray, object]:
    if decoder is None or getattr(decoder, "tensor_dtype", None) != observation.dtype:
        if decoder is not None:
            decoder.close()
        decoder = create_observation_decoder(observation.dtype)
        decoder.reset()

    frames = np.asarray(decoder.decode(observation, info), dtype=np.uint8)
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.shape[0] != ENV_COUNT:
        raise RuntimeError(f"expected {ENV_COUNT} decoded frames, got shape {frames.shape}")
    return frames, decoder


def main() -> None:
    args = parse_args()
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]" = queue.Queue()
    decoder = None
    last_tiled_frame: np.ndarray | None = None
    rewards = np.zeros(ENV_COUNT, dtype=np.float32)
    terminated = np.zeros(ENV_COUNT, dtype=bool)
    truncated = np.zeros(ENV_COUNT, dtype=bool)
    infos: dict[str, object] = {}

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with grpc.insecure_channel(args.server_addr) as channel:
        stub = gym_env_pb2_grpc.ArmEnvStub(channel)
        responses = stub.StreamEnv(_request_iterator(request_queue))

        try:
            observation = _send_reset(request_queue, responses, base_seed=args.seed)
            frames_rgb, decoder = _decode_frames(observation, infos, decoder)

            start_time = time.perf_counter()
            step_idx = 0

            while args.steps <= 0 or step_idx < args.steps:
                step_result = _send_step(request_queue, responses, build_actions(step_idx, ENV_COUNT))
                infos = step_result["info"]
                rewards = step_result["reward"]
                terminated = step_result["terminated"]
                truncated = step_result["truncated"]
                frames_rgb, decoder = _decode_frames(step_result["observation"], infos, decoder)

                elapsed_steps = np.asarray(infos.get("elapsed_steps", [0] * ENV_COUNT), dtype=np.int32)
                max_episode_steps = np.asarray(infos.get("max_episode_steps", [0] * ENV_COUNT), dtype=np.int32)
                bullseye_pixels = infos.get("bullseye_pixel", [None] * ENV_COUNT)
                annotated_frames = [
                    annotate_frame(
                        frames_rgb[idx],
                        env_idx=idx,
                        reward=float(rewards[idx]),
                        terminated=bool(terminated[idx]),
                        truncated=bool(truncated[idx]),
                        elapsed_steps=int(elapsed_steps[idx]),
                        max_episode_steps=int(max_episode_steps[idx]),
                        bullseye_pixel=bullseye_pixels[idx] if idx < len(bullseye_pixels) else None,
                    )
                    for idx in range(ENV_COUNT)
                ]
                tiled_frame = tile_frames(annotated_frames)
                last_tiled_frame = tiled_frame

                if not args.headless:
                    cv2.imshow(WINDOW_NAME, tiled_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("r"):
                        observation = _send_reset(request_queue, responses, base_seed=args.seed)
                        infos = {}
                        rewards.fill(0.0)
                        terminated.fill(False)
                        truncated.fill(False)
                        frames_rgb, decoder = _decode_frames(observation, infos, decoder)

                step_idx += 1
                frame_time = 1.0 / max(args.fps, 1e-6)
                elapsed = time.perf_counter() - start_time
                target_elapsed = step_idx * frame_time
                if target_elapsed > elapsed:
                    time.sleep(target_elapsed - elapsed)

            if args.save_frame is not None and last_tiled_frame is not None:
                save_frame(args.save_frame, last_tiled_frame)
        finally:
            try:
                request_queue.put(gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()))
                close_reply = next(responses)
                if close_reply.WhichOneof("result") != "close":
                    raise RuntimeError("expected CloseReply")
            except Exception:
                pass
            if decoder is not None:
                decoder.close()
            request_queue.put(_STREAM_END)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
