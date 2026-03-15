from __future__ import annotations

import argparse
import time

import numpy as np

from lockon.envs.turret import TurretEnvConfig, make_vector_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample multiple turret environments in parallel.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--steps", type=int, default=128, help="Number of vectorized environment steps.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used for environment creation.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=32,
        help="Episode horizon before an environment is truncated and auto-reset on the next step.",
    )
    parser.add_argument(
        "--qpos-noise-scale",
        type=float,
        default=0.02,
        help="Uniform reset noise scale applied to the initial qpos when no explicit qpos is provided.",
    )
    parser.add_argument(
        "--target-noise-scale",
        type=float,
        default=0.02,
        help="Uniform reset noise scale applied to initial targets when no explicit targets are provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TurretEnvConfig(
        render_mode=None,
        max_episode_steps=args.max_episode_steps,
        qpos_reset_noise_scale=args.qpos_noise_scale,
        target_reset_noise_scale=args.target_noise_scale,
    )
    vector_env = make_vector_env(args.num_envs, base_seed=args.seed, config=config)
    rng = np.random.default_rng(args.seed)

    try:
        observations, infos = vector_env.reset()
        print(
            f"reset observations shape={observations.shape} "
            f"qpos shape={np.asarray(infos['qpos']).shape}"
        )

        total_transitions = 0
        total_resets = 0
        start_time = time.perf_counter()

        for step_idx in range(args.steps):
            actions = rng.uniform(-1.0, 1.0, size=(args.num_envs, 5)).astype(np.float32)
            observations, rewards, terminated, truncated, infos = vector_env.step(actions)
            done = np.logical_or(terminated, truncated)
            reset_count = int(done.sum())
            total_transitions += args.num_envs
            total_resets += reset_count

            if step_idx == 0 or reset_count > 0:
                print(
                    f"step={step_idx + 1} obs={observations.shape} rewards={rewards.shape} "
                    f"done={reset_count} mean_reward={float(np.mean(rewards)):.4f}"
                )

        elapsed = time.perf_counter() - start_time
        print(
            f"throughput={total_transitions / max(elapsed, 1e-9):.1f} transitions/s "
            f"auto_resets={total_resets}"
        )
    finally:
        vector_env.close()


if __name__ == "__main__":
    main()
