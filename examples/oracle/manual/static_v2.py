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

from lockon.protos.gym_v2 import gym_env_pb2, gym_env_pb2_grpc
from lockon.utils import create_observation_decoder

DEFAULT_SERVER_ADDR = os.getenv("LOCKON_SERVER_ADDR", "127.0.0.1:50051")
FRAME_SKIP = 5
STEP_ACTION = np.zeros(6, dtype=np.float32)
_STREAM_END = object()


def _tensor_from_array(array: np.ndarray) -> gym_env_pb2.Tensor:
    arr = np.asarray(array)
    return gym_env_pb2.Tensor(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


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
    action = np.zeros(6, dtype=np.float32)

    if ch == "j":
        action[2] = 1.0
    elif ch == "l":
        action[2] = -1.0
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
        action[5] = 1.0

    return action


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


def _extract_observation_tensor(observation: gym_env_pb2.TensorValue) -> gym_env_pb2.Tensor:
    if observation.WhichOneof("kind") != "tensor":
        raise RuntimeError("manual v2 demo expects tensor observation")
    return observation.tensor


def _send_step(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
    responses: Iterator[gym_env_pb2.EnvReply],
    action: np.ndarray,
) -> dict[str, object]:
    request_queue.put(
        gym_env_pb2.EnvRequest(
            step=gym_env_pb2.Step(
                action=gym_env_pb2.TensorValue(tensor=_tensor_from_array(action.astype(np.float32)))
            )
        )
    )
    reply = next(responses)
    if reply.WhichOneof("result") != "step":
        raise RuntimeError("expected StepReply")

    info = MessageToDict(reply.step.info, preserving_proto_field_name=True)
    if len(reply.step.reward) != 1 or len(reply.step.terminated) != 1 or len(reply.step.truncated) != 1:
        raise RuntimeError("manual v2 demo expects single-env repeated scalars")
    return {
        "observation": _extract_observation_tensor(reply.step.observation),
        "info": info,
        "reward": float(reply.step.reward[0]),
        "terminated": bool(reply.step.terminated[0]),
        "truncated": bool(reply.step.truncated[0]),
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
        stub = gym_env_pb2_grpc.GymEnvStub(channel)
        responses = stub.StreamEnv(_request_iterator(request_queue))

        request_queue.put(gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()))
        reset_reply = next(responses)
        if reset_reply.WhichOneof("result") != "reset":
            raise RuntimeError("expected ResetReply")

        reset_info = MessageToDict(reset_reply.reset.info, preserving_proto_field_name=True)
        frame_rgb, decoder = _decode_frame(_extract_observation_tensor(reset_reply.reset.observation), reset_info, decoder)
        last_info: dict[str, object] = reset_info
        step_count = 0

        cv2.namedWindow("Turret Camera V2", cv2.WINDOW_NORMAL)

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
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    _draw_bullseye(frame, last_info)
                    _draw_crosshair(frame)
                    cv2.imshow("Turret Camera V2", frame)

                    key = cv2.waitKey(1)
                    if key != -1:
                        if (key & 0xFF) == 27:
                            request_queue.put(gym_env_pb2.EnvRequest(close=gym_env_pb2.Close()))
                            close_reply = next(responses)
                            if close_reply.WhichOneof("result") != "close":
                                raise RuntimeError("expected CloseReply")
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
    if isinstance(qpos, list) and len(qpos) == 5:
        print(
            f"x={float(qpos[0]):.3f} y={float(qpos[1]):.3f} "
            f"base_yaw={float(qpos[2]):.3f} turret_yaw={float(qpos[3]):.3f} "
            f"pitch={float(qpos[4]):.3f}"
        )


if __name__ == "__main__":
    main()
