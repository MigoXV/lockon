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

import dotenv

dotenv.load_dotenv()

from lockon.aim.contorllers import PidAimConfig, PidAimController
from lockon.protos.gym_v2 import gym_env_pb2, gym_env_pb2_grpc
from lockon.vcodec import create_observation_decoder

DEFAULT_SERVER_ADDR = os.getenv("LOCKON_SERVER_ADDR", "127.0.0.1:50051")
FRAME_SKIP = 5
CONTROL_DT = 0.01
IDLE_ACTION = np.zeros(6, dtype=np.float32)
STREAM_END = object()
SCHEMATIC_TARGET_X = 0.5
SCHEMATIC_TARGET_Y = 0.0
SCHEMATIC_WORLD_X = (-1.0, 0.7)
SCHEMATIC_WORLD_Y = (-1.0, 1.0)


def _tensor_from_array(array: np.ndarray) -> gym_env_pb2.Tensor:
    arr = np.asarray(array)
    return gym_env_pb2.Tensor(
        data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype)
    )


def _request_iterator(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
) -> Iterator[gym_env_pb2.EnvRequest]:
    while True:
        item = request_queue.get()
        if item is STREAM_END:
            return
        yield item


def _extract_observation_tensor(
    observation: gym_env_pb2.TensorValue,
) -> gym_env_pb2.Tensor:
    if observation.WhichOneof("kind") != "tensor":
        raise RuntimeError("pid move v2 demo expects tensor observation")
    return observation.tensor


def _send_step(
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]",
    responses: Iterator[gym_env_pb2.EnvReply],
    action: np.ndarray,
) -> dict[str, object]:
    request_queue.put(
        gym_env_pb2.EnvRequest(
            step=gym_env_pb2.Step(
                action=gym_env_pb2.TensorValue(
                    tensor=_tensor_from_array(action.astype(np.float32))
                )
            )
        )
    )
    reply = next(responses)
    if reply.WhichOneof("result") != "step":
        raise RuntimeError("expected StepReply")

    info = MessageToDict(reply.step.info, preserving_proto_field_name=True)
    if (
        len(reply.step.reward) != 1
        or len(reply.step.terminated) != 1
        or len(reply.step.truncated) != 1
    ):
        raise RuntimeError("pid move v2 demo expects single-env repeated scalars")
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


def _format_aim_error(info: dict[str, object]) -> str:
    value = info.get("aim_error")
    if value is None:
        return "n/a"
    return f"{float(value):.5f}"


def _sample_move_action(
    rng: np.random.Generator, move_scale: float, move_deadband: float
) -> np.ndarray:
    move = rng.uniform(-move_scale, move_scale, size=2).astype(np.float32)
    move[np.abs(move) < move_deadband] = 0.0
    return move


def _slew_move_action(
    current_move: np.ndarray, target_move: np.ndarray, max_delta: float
) -> np.ndarray:
    delta = np.clip(target_move - current_move, -max_delta, max_delta)
    return (current_move + delta).astype(np.float32)


def _sample_base_rot_action(
    rng: np.random.Generator, rot_scale: float, rot_deadband: float
) -> float:
    rot = float(rng.uniform(-rot_scale, rot_scale))
    if abs(rot) < rot_deadband:
        return 0.0
    return rot


def _slew_scalar(current_value: float, target_value: float, max_delta: float) -> float:
    return float(
        current_value + np.clip(target_value - current_value, -max_delta, max_delta)
    )


def _world_to_panel(
    x: float,
    y: float,
    panel_width: int,
    panel_height: int,
    padding: int = 24,
) -> tuple[int, int]:
    x0, x1 = SCHEMATIC_WORLD_X
    y0, y1 = SCHEMATIC_WORLD_Y
    px = padding + (x - x0) / (x1 - x0) * (panel_width - 2 * padding)
    py = panel_height - padding - (y - y0) / (y1 - y0) * (panel_height - 2 * padding)
    return int(px), int(py)


def _render_schematic(
    info: dict[str, object], frame_height: int, panel_width: int = 320
) -> np.ndarray:
    panel = np.full((frame_height, panel_width, 3), 248, dtype=np.uint8)
    cv2.rectangle(
        panel, (0, 0), (panel_width - 1, frame_height - 1), (210, 210, 210), 1
    )
    cv2.putText(
        panel,
        "Top View",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    tx, ty = _world_to_panel(
        SCHEMATIC_TARGET_X, SCHEMATIC_TARGET_Y, panel_width, frame_height
    )
    cv2.circle(panel, (tx, ty), 7, (30, 30, 220), -1)
    cv2.putText(
        panel,
        "target",
        (tx + 10, ty - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 220),
        1,
        cv2.LINE_AA,
    )

    qpos = info.get("qpos", [])
    if isinstance(qpos, list) and len(qpos) == 5:
        base_x = float(qpos[0])
        base_y = float(qpos[1])
        base_yaw = float(qpos[2])
        turret_yaw = float(qpos[3])
        pitch = float(qpos[4])
        facing_yaw = base_yaw + turret_yaw

        bx, by = _world_to_panel(base_x, base_y, panel_width, frame_height)
        cv2.circle(panel, (bx, by), 6, (30, 160, 30), -1)
        cv2.putText(
            panel,
            "turret",
            (bx + 10, by - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 160, 30),
            1,
            cv2.LINE_AA,
        )

        dir_len = 0.28
        dx = dir_len * np.cos(facing_yaw)
        dy = dir_len * np.sin(facing_yaw)
        ex, ey = _world_to_panel(base_x + dx, base_y + dy, panel_width, frame_height)
        cv2.line(panel, (bx, by), (ex, ey), (20, 120, 20), 2)
        cv2.circle(panel, (ex, ey), 4, (20, 120, 20), -1)

        cv2.line(panel, (bx, by), (tx, ty), (160, 160, 160), 1)
        cv2.putText(
            panel,
            f"base=({base_x:.2f}, {base_y:.2f})",
            (16, frame_height - 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"base_yaw={base_yaw:.2f} gun_yaw={turret_yaw:.2f}",
            (16, frame_height - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"facing={facing_yaw:.2f} pitch={pitch:.2f}",
            (16, frame_height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

    return panel


def _draw_overlay(
    frame: np.ndarray,
    info: dict[str, object],
    metrics: dict[str, float] | None,
    move_xy: np.ndarray,
    base_rot: float,
    mode: str,
    scan_yaw: float,
) -> None:
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

    qpos = info.get("qpos", [])
    qpos_text = "qpos=n/a"
    if isinstance(qpos, list) and len(qpos) == 5:
        qpos_text = (
            f"qpos=({float(qpos[0]):.2f}, {float(qpos[1]):.2f}, {float(qpos[2]):.2f}, "
            f"{float(qpos[3]):.2f}, {float(qpos[4]):.2f})"
        )

    lines = [
        f"mode={mode} scan_yaw={scan_yaw:.2f}",
        f"move=({float(move_xy[0]):.2f}, {float(move_xy[1]):.2f})",
        f"base_rot={base_rot:.2f}",
        qpos_text,
        f"aim_error={_format_aim_error(info)}",
    ]
    if metrics is not None:
        lines = [
            f"mode={mode} scan_yaw={scan_yaw:.2f}",
            f"az={metrics['azimuth_deg']:.2f}deg el={metrics['elevation_deg']:.2f}deg",
            f"plane=({metrics['plane_x']:.3f}, {metrics['plane_y']:.3f})",
            f"ff=({metrics['yaw_ff']:.2f}, {metrics['pitch_ff']:.2f})",
            f"fb=({metrics['yaw_fb']:.2f}, {metrics['pitch_fb']:.2f})",
            f"move=({float(move_xy[0]):.2f}, {float(move_xy[1]):.2f})",
            f"base_rot={base_rot:.2f}",
            qpos_text,
            f"aim_error={_format_aim_error(info)}",
        ]

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-addr", default=DEFAULT_SERVER_ADDR)
    parser.add_argument("--align-threshold-deg", type=float, default=0.25)
    parser.add_argument("--plane-threshold", type=float, default=0.015)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--move-scale", type=float, default=0.35)
    parser.add_argument("--move-deadband", type=float, default=0.08)
    parser.add_argument("--move-hold-steps", type=int, default=40)
    parser.add_argument("--move-slew-rate", type=float, default=0.04)
    parser.add_argument("--base-rot-scale", type=float, default=12)
    parser.add_argument("--base-rot-deadband", type=float, default=0.06)
    parser.add_argument("--base-rot-hold-steps", type=int, default=50)
    parser.add_argument("--base-rot-slew-rate", type=float, default=1.2)
    parser.add_argument("--scan-yaw-speed", type=float, default=0.9)
    parser.add_argument("--scan-sweep-steps", type=int, default=80)
    parser.add_argument("--ff-gain", type=float, default=3)
    parser.add_argument("--yaw-kp", type=float, default=1.8)
    parser.add_argument("--yaw-ki", type=float, default=0.12)
    parser.add_argument("--yaw-kd", type=float, default=0.24)
    parser.add_argument("--pitch-kp", type=float, default=1.8)
    parser.add_argument("--pitch-ki", type=float, default=0.12)
    parser.add_argument("--pitch-kd", type=float, default=0.24)
    parser.add_argument("--pid-deadband", type=float, default=0.002)
    parser.add_argument("--integral-limit", type=float, default=0.4)
    parser.add_argument("--feedback-limit", type=float, default=0.7)
    parser.add_argument("--fire-when-aligned", action="store_true", default=True)
    parser.add_argument("--no-fire", action="store_false", dest="fire_when_aligned")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    request_queue: "queue.Queue[gym_env_pb2.EnvRequest | object]" = queue.Queue()
    decoder = None
    last_info: dict[str, object] = {}
    last_metrics: dict[str, float] | None = None
    aligned_last_step = False
    move_xy = np.zeros(2, dtype=np.float32)
    target_move_xy = np.zeros(2, dtype=np.float32)
    base_rot = 0.0
    target_base_rot = 0.0
    scan_direction = 1.0
    lost_steps = 0
    controller = PidAimController(
        PidAimConfig(
            ff_gain=args.ff_gain,
            yaw_kp=args.yaw_kp,
            yaw_ki=args.yaw_ki,
            yaw_kd=args.yaw_kd,
            pitch_kp=args.pitch_kp,
            pitch_ki=args.pitch_ki,
            pitch_kd=args.pitch_kd,
            pid_deadband=args.pid_deadband,
            integral_limit=args.integral_limit,
            feedback_limit=args.feedback_limit,
        )
    )

    with grpc.insecure_channel(args.server_addr) as channel:
        stub = gym_env_pb2_grpc.GymEnvStub(channel)
        responses = stub.StreamEnv(_request_iterator(request_queue))

        request_queue.put(gym_env_pb2.EnvRequest(reset=gym_env_pb2.Reset()))
        reset_reply = next(responses)
        if reset_reply.WhichOneof("result") != "reset":
            raise RuntimeError("expected ResetReply")

        reset_info = MessageToDict(
            reset_reply.reset.info, preserving_proto_field_name=True
        )
        frame_rgb, decoder = _decode_frame(
            _extract_observation_tensor(reset_reply.reset.observation),
            reset_info,
            decoder,
        )
        cv2.namedWindow("Oracle PID Move Aim V2", cv2.WINDOW_NORMAL)
        action = IDLE_ACTION.copy()

        try:
            for step_idx in range(args.max_steps):
                if step_idx % max(args.move_hold_steps, 1) == 0:
                    target_move_xy = _sample_move_action(
                        rng, args.move_scale, args.move_deadband
                    )
                if step_idx % max(args.base_rot_hold_steps, 1) == 0:
                    target_base_rot = _sample_base_rot_action(
                        rng, args.base_rot_scale, args.base_rot_deadband
                    )
                move_xy = _slew_move_action(
                    move_xy, target_move_xy, args.move_slew_rate
                )
                base_rot = _slew_scalar(
                    base_rot, target_base_rot, args.base_rot_slew_rate
                )

                step_result = _send_step(request_queue, responses, action)
                last_info = step_result["info"]
                frame_rgb, decoder = _decode_frame(
                    step_result["observation"], last_info, decoder
                )

                computed = controller.update(last_info, frame_rgb.shape, dt=CONTROL_DT)
                action = IDLE_ACTION.copy()
                action[:2] = move_xy
                action[2] = base_rot
                last_metrics = None
                mode = "SCAN"
                scan_yaw = 0.0
                if computed is not None:
                    aim_action, metrics = computed
                    action[3:5] = aim_action[3:5]
                    last_metrics = metrics.as_dict()
                    lost_steps = 0
                    mode = "TRACK"
                else:
                    lost_steps += 1
                    if lost_steps % max(args.scan_sweep_steps, 1) == 0:
                        scan_direction *= -1.0
                    scan_yaw = scan_direction * args.scan_yaw_speed
                    action[3] = scan_yaw
                    action[4] = 0.0

                should_fire = False
                if last_metrics is not None:
                    aligned = (
                        abs(last_metrics["azimuth_deg"]) <= args.align_threshold_deg
                        and abs(last_metrics["elevation_deg"])
                        <= args.align_threshold_deg
                        and abs(last_metrics["plane_x"]) <= args.plane_threshold
                        and abs(last_metrics["plane_y"]) <= args.plane_threshold
                    )
                    should_fire = (
                        args.fire_when_aligned and aligned and not aligned_last_step
                    )
                    aligned_last_step = aligned
                else:
                    aligned_last_step = False

                if should_fire:
                    fire_action = action.copy()
                    fire_action[5] = 1.0
                    step_result = _send_step(request_queue, responses, fire_action)
                    last_info = step_result["info"]
                    frame_rgb, decoder = _decode_frame(
                        step_result["observation"], last_info, decoder
                    )
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
                    _draw_overlay(
                        frame,
                        last_info,
                        last_metrics,
                        move_xy,
                        base_rot,
                        mode,
                        scan_yaw,
                    )
                    display = np.concatenate(
                        [frame, _render_schematic(last_info, frame.shape[0])], axis=1
                    )
                    cv2.imshow("Oracle PID Move Aim V2", display)
                    key = cv2.waitKey(1)
                    if (key & 0xFF) == 27:
                        break

                time.sleep(CONTROL_DT)
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
