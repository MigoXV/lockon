from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from gymnasium.vector import AsyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode

from lockon.envs.turret.env import TurretEnv, TurretEnvConfig


def _resolve_config(config: TurretEnvConfig | None, **env_kwargs: object) -> TurretEnvConfig:
    base_config = config or TurretEnvConfig()
    if not env_kwargs:
        return base_config
    return replace(base_config, **env_kwargs)


def build_env(
    index: int = 0,
    *,
    base_seed: int | None = None,
    config: TurretEnvConfig | None = None,
    **env_kwargs: object,
) -> TurretEnv:
    env_config = _resolve_config(config, **env_kwargs)
    env = TurretEnv(**env_config.to_kwargs())
    if base_seed is not None:
        env.reset(seed=base_seed + index)
    return env


def make_env(
    index: int = 0,
    *,
    base_seed: int | None = None,
    config: TurretEnvConfig | None = None,
    **env_kwargs: object,
) -> Callable[[], TurretEnv]:
    env_config = _resolve_config(config, **env_kwargs)

    def _factory() -> TurretEnv:
        return build_env(index=index, base_seed=base_seed, config=env_config)

    return _factory


def make_vector_env(
    num_envs: int,
    *,
    base_seed: int | None = 0,
    config: TurretEnvConfig | None = None,
    shared_memory: bool = True,
    copy: bool = True,
    context: str | None = None,
    daemon: bool = True,
    observation_mode: str = "same",
    autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
    **env_kwargs: object,
) -> AsyncVectorEnv:
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    env_config = _resolve_config(config, **env_kwargs)
    env_fns = [make_env(index=i, base_seed=base_seed, config=env_config) for i in range(num_envs)]
    return AsyncVectorEnv(
        env_fns,
        shared_memory=shared_memory,
        copy=copy,
        context=context,
        daemon=daemon,
        observation_mode=observation_mode,
        autoreset_mode=autoreset_mode,
    )
