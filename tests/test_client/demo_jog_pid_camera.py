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

from lockon.protos.gym_env import gym_env_pb2, gym_env_pb2_grpc
from lockon.utils import array_from_tensor, tensor_from_array
from lockon.vcodec import create_observation_decoder

DEFAULT_SERVER_ADDR = os.getenv("LOCKON_SERVER_ADDR", "127.0.0.1:50051")
FRAME_SKIP = 5
ACTION_DIM = 6
WINDOW_NAME = "Turret Camera"
FIRE_INDEX = 5
STEP_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
_STREAM_END = object()


def _request_iterator(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
) -> Iterator[gym_env_pb2.EnvRequest]:
    while True:
        item = request_queue.get()
        if item is _STREAM_END:
            return
        yield item


def _process_key(key_code: int) -> np.ndarray:
    ch = chr(key_code & 0xFF)
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    if ch == "j":
        action[3] = 1.0
    elif ch == "l":
        action[3] = -1.0
    elif ch == "i":
        action[4] = 1.0
    elif ch == "k":
        action[4] = -1.0
    elif ch == "w":
        action[0] = 1.0
    elif ch == "s":
        action[0] = -1.0
    elif ch == "a":
        action[1] = 1.0
    elif ch == "d":
        action[1] = -1.0
    elif ch == "f":
        action[FIRE_INDEX] = 1.0
    elif ch == " ":
        action[FIRE_INDEX] = 1.0

    return action


def _close_stream(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
    responses: Iterator[gym_env_pb2.EnvReply],
) -> None:
    request_queue.put(gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()))
    close_reply = next(responses)
    if close_reply.WhichOneof("result") != "close":
        raise RuntimeError("expected CloseReply")


def _draw_crosshair(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    cross_size = 10
    cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 0), 1)


def _draw_bullseye(frame: np.ndarray, info: dict[str, object]) -> None:
    bullseye_pixel = info.get("bullseye_pixel")
    if not isinstance(bullseye_pixel, list) or len(bullseye_pixel) != 2:
        return

    px = int(float(bullseye_pixel[0]))
    py = int(float(bullseye_pixel[1]))
    cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-addr", default=DEFAULT_SERVER_ADDR)
    args = parser.parse_args()

    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]" = queue.Queue()
    decoder = None

    with grpc.insecure_channel(args.server_addr) as channel:
        stub = gym_env_pb2_grpc.ArmEnvStub(channel)
        responses = stub.StreamEnv(_request_iterator(request_queue))

        request_queue.put(gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()))
        reset_reply = next(responses)
        if reset_reply.WhichOneof("result") != "reset":
            raise RuntimeError("expected ResetReply")

        frame_rgb, decoder = _decode_frame(reset_reply.reset.observation, {}, decoder)
        last_info: dict[str, object] = {}
        step_count = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                step_result = _send_step(request_queue, responses, STEP_ACTION)
                last_info = step_result["info"]
                frame_rgb, decoder = _decode_frame(
                    step_result["observation"],
                    last_info,
                    decoder,
                )
                step_count += 1

                if step_count % FRAME_SKIP == 0:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        _close_stream(request_queue, responses)
                        break

                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    _draw_bullseye(frame, last_info)
                    _draw_crosshair(frame)
                    cv2.imshow(WINDOW_NAME, frame)

                    key = cv2.waitKey(1)
                    if key != -1:
                        if (key & 0xFF) == 27:
                            _close_stream(request_queue, responses)
                            break

                        action = _process_key(key)
                        step_result = _send_step(request_queue, responses, action)
                        last_info = step_result["info"]
                        frame_rgb, decoder = _decode_frame(
                            step_result["observation"],
                            last_info,
                            decoder,
                        )

                        fire_info = last_info.get("fire", {})
                        if isinstance(fire_info, dict) and fire_info.get("triggered"):
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
                                print(f"[FIRE] {fire_info['reason']}")

                time.sleep(0.01)
        finally:
            if decoder is not None:
                decoder.close()
            request_queue.put(_STREAM_END)
            cv2.destroyAllWindows()

    qpos = last_info.get("qpos", [])
    if isinstance(qpos, list) and len(qpos) >= 4:
        print(f"x={float(qpos[0]):.3f} y={float(qpos[1]):.3f} yaw={float(qpos[2]):.3f} pitch={float(qpos[3]):.3f}")


if __name__ == "__main__":
    main()
