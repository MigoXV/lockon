from __future__ import annotations

import argparse
import os
import queue
import time
from typing import Iterator

import cv2
import grpc
import numpy as np
from google.protobuf.json_format import MessageToDict

from lockon.aim.contorllers import OpenLoopAimController
from lockon.protos.gym_env import gym_env_pb2, gym_env_pb2_grpc
from lockon.utils import array_from_tensor, tensor_from_array
from lockon.utils import create_observation_decoder

DEFAULT_SERVER_ADDR = os.getenv("LOCKON_SERVER_ADDR", "127.0.0.1:50051")
FRAME_SKIP = 5
IDLE_ACTION = np.zeros(6, dtype=np.float32)
STREAM_END = object()

def _request_iterator(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
) -> Iterator[gym_env_pb2.EnvRequest]:
    while True:
        item = request_queue.get()
        if item is STREAM_END:
            return
        yield item


def _send_step(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
    responses: Iterator[gym_env_pb2.EnvReply],
    action: np.ndarray,
) -> dict[str, object]:
    request_queue.put(
        gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=tensor_from_array(action.astype(np.float32))))
    )
    reply = next(responses)
    if reply.WhichOneof("result") != "step":
        raise RuntimeError("expected StepReply")

    info = MessageToDict(reply.step.info, preserving_proto_field_name=True)
    reward = float(array_from_tensor(reply.step.reward))
    terminated = bool(array_from_tensor(reply.step.terminated))
    truncated = bool(array_from_tensor(reply.step.truncated))
    return {
        "observation": reply.step.observation,
        "info": info,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def _decode_frame(
    observation: gym_env_pb2.Tensor,
    info: dict[str, object],
    decoder,
) -> tuple[np.ndarray, object]:
    if decoder is None or getattr(decoder, "tensor_dtype", None) != observation.dtype:
        if decoder is not None:
            decoder.close()
        decoder = create_observation_decoder(observation.dtype)
        decoder.reset()

    return decoder.decode(observation, info), decoder


def _format_aim_error(info: dict[str, object]) -> str:
    value = info.get("aim_error")
    if value is None:
        return "n/a"
    return f"{float(value):.5f}"


def _draw_overlay(frame: np.ndarray, info: dict[str, object], metrics: dict[str, float] | None) -> None:
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    cross_size = 10
    cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 0), 1)

    bullseye_pixel = info.get("bullseye_pixel")
    if isinstance(bullseye_pixel, list) and len(bullseye_pixel) == 2:
        px = int(float(bullseye_pixel[0]))
        py = int(float(bullseye_pixel[1]))
        cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)

    if metrics is None:
        return

    lines = [
        f"azimuth={metrics['azimuth_deg']:.2f} deg",
        f"elevation={metrics['elevation_deg']:.2f} deg",
        f"plane=({metrics['plane_x']:.3f}, {metrics['plane_y']:.3f})",
        f"aim_error={_format_aim_error(info)}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-addr", default=DEFAULT_SERVER_ADDR)
    parser.add_argument("--align-threshold-deg", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fire-when-aligned", action="store_true", default=True)
    parser.add_argument("--no-fire", action="store_false", dest="fire_when_aligned")
    args = parser.parse_args()

    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]" = queue.Queue()
    decoder = None
    last_info: dict[str, object] = {}
    last_metrics: dict[str, float] | None = None
    aligned_last_step = False
    controller = OpenLoopAimController()

    with grpc.insecure_channel(args.server_addr) as channel:
        stub = gym_env_pb2_grpc.ArmEnvStub(channel)
        responses = stub.StreamEnv(_request_iterator(request_queue))

        request_queue.put(gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()))
        reset_reply = next(responses)
        if reset_reply.WhichOneof("result") != "reset":
            raise RuntimeError("expected ResetReply")

        frame_rgb, decoder = _decode_frame(reset_reply.reset.observation, {}, decoder)
        cv2.namedWindow("Oracle Auto Aim", cv2.WINDOW_NORMAL)

        try:
            for step_idx in range(args.max_steps):
                step_result = _send_step(request_queue, responses, IDLE_ACTION if step_idx == 0 else action)
                last_info = step_result["info"]
                frame_rgb, decoder = _decode_frame(step_result["observation"], last_info, decoder)

                computed = controller.update(last_info, frame_rgb.shape)
                action = IDLE_ACTION.copy()
                last_metrics = None
                if computed is not None:
                    action, metrics = computed
                    last_metrics = metrics.as_dict()

                should_fire = False
                if last_metrics is not None:
                    aligned = (
                        abs(last_metrics["azimuth_deg"]) <= args.align_threshold_deg
                        and abs(last_metrics["elevation_deg"]) <= args.align_threshold_deg
                    )
                    should_fire = args.fire_when_aligned and aligned and not aligned_last_step
                    aligned_last_step = aligned
                else:
                    aligned_last_step = False

                if should_fire:
                    fire_action = action.copy()
                    fire_action[5] = 1.0
                    step_result = _send_step(request_queue, responses, fire_action)
                    last_info = step_result["info"]
                    frame_rgb, decoder = _decode_frame(step_result["observation"], last_info, decoder)
                    fire_info = last_info.get("fire", {})
                    if isinstance(fire_info, dict):
                        if fire_info.get("hit"):
                            print(
                                "[FIRE] 命中! "
                                f"r={float(fire_info['radius']):.4f}m "
                                f"theta={float(fire_info['theta_deg']):.1f}deg "
                                f"dy={float(fire_info['dy']):.4f} "
                                f"dz={float(fire_info['dz']):.4f} "
                                f"| {fire_info['zone']}"
                            )
                        else:
                            print(f"[FIRE] {fire_info.get('reason', 'unknown')}")

                if step_idx % FRAME_SKIP == 0:
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    _draw_overlay(frame, last_info, last_metrics)
                    cv2.imshow("Oracle Auto Aim", frame)
                    key = cv2.waitKey(1)
                    if (key & 0xFF) == 27:
                        break

                time.sleep(0.01)
            else:
                print(f"[AUTO_AIM] reached max steps without firing: {args.max_steps}")
        finally:
            if decoder is not None:
                decoder.close()
            request_queue.put(gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()))
            try:
                next(responses)
            except StopIteration:
                pass
            request_queue.put(STREAM_END)
            cv2.destroyAllWindows()

    qpos = last_info.get("qpos", [])
    if isinstance(qpos, list) and len(qpos) == 5:
        print(
            f"x={float(qpos[0]):.3f} y={float(qpos[1]):.3f} "
            f"base_yaw={float(qpos[2]):.3f} turret_yaw={float(qpos[3]):.3f} "
            f"pitch={float(qpos[4]):.3f}"
        )


if __name__ == "__main__":
    main()
