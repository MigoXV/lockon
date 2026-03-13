from __future__ import annotations

import queue
import time
from typing import Iterator

import cv2
import grpc
import numpy as np
from google.protobuf.json_format import MessageToDict

from lockon.protos.gym_env import gym_env_pb2, gym_env_pb2_grpc

SERVER_ADDR = "127.0.0.1:50051"
FRAME_SKIP = 5
CAM_WIDTH = 320
CAM_HEIGHT = 240
STEP_ACTION = np.zeros(5, dtype=np.float32)
_STREAM_END = object()


def _tensor_from_array(array: np.ndarray) -> gym_env_pb2.Tensor:
    arr = np.asarray(array)
    return gym_env_pb2.Tensor(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def _array_from_tensor(tensor: gym_env_pb2.Tensor) -> np.ndarray:
    dtype = np.dtype(tensor.dtype)
    return np.frombuffer(tensor.data, dtype=dtype).reshape(tuple(tensor.shape))


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
    action = np.zeros(5, dtype=np.float32)

    if ch == "j":
        action[2] = 1.0
    elif ch == "l":
        action[2] = -1.0
    elif ch == "i":
        action[3] = 1.0
    elif ch == "k":
        action[3] = -1.0
    elif ch == "w":
        action[0] = 1.0
    elif ch == "s":
        action[0] = -1.0
    elif ch == "a":
        action[1] = 1.0
    elif ch == "d":
        action[1] = -1.0
    elif ch == "f":
        action[4] = 1.0

    return action


def _draw_crosshair(frame: np.ndarray) -> None:
    cx, cy = CAM_WIDTH // 2, CAM_HEIGHT // 2
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
        gym_env_pb2.EnvRequest(step=gym_env_pb2.Step(action=_tensor_from_array(action.astype(np.float32))))
    )
    reply = next(responses)
    if reply.WhichOneof("result") != "step":
        raise RuntimeError("expected StepReply")

    observation = _array_from_tensor(reply.step.observation)
    info = MessageToDict(reply.step.info, preserving_proto_field_name=True)
    reward = float(_array_from_tensor(reply.step.reward))
    terminated = bool(_array_from_tensor(reply.step.terminated))
    truncated = bool(_array_from_tensor(reply.step.truncated))
    return {
        "observation": observation,
        "info": info,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> None:
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]" = queue.Queue()

    with grpc.insecure_channel(SERVER_ADDR) as channel:
        stub = gym_env_pb2_grpc.ArmEnvStub(channel)
        responses = stub.StreamEnv(_request_iterator(request_queue))

        request_queue.put(gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()))
        reset_reply = next(responses)
        if reset_reply.WhichOneof("result") != "reset":
            raise RuntimeError("expected ResetReply")

        frame_rgb = _array_from_tensor(reset_reply.reset.observation)
        last_info: dict[str, object] = {}
        step_count = 0

        cv2.namedWindow("Turret Camera", cv2.WINDOW_NORMAL)

        try:
            while True:
                step_result = _send_step(request_queue, responses, STEP_ACTION)
                frame_rgb = step_result["observation"]
                last_info = step_result["info"]
                step_count += 1

                if step_count % FRAME_SKIP == 0:
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    _draw_bullseye(frame, last_info)
                    _draw_crosshair(frame)
                    cv2.imshow("Turret Camera", frame)

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
                        frame_rgb = step_result["observation"]
                        last_info = step_result["info"]

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
            request_queue.put(_STREAM_END)
            cv2.destroyAllWindows()

    qpos = last_info.get("qpos", [])
    if isinstance(qpos, list) and len(qpos) == 4:
        print(f"x={float(qpos[0]):.3f} y={float(qpos[1]):.3f} yaw={float(qpos[2]):.3f} pitch={float(qpos[3]):.3f}")


if __name__ == "__main__":
    main()
