# lockon

`lockon` provides a MuJoCo-based turret environment for local RL training and gRPC-based demos.

## Local vectorized training

Use local `gymnasium.vector.AsyncVectorEnv` workers as the primary training path. The helper APIs live under `lockon.envs.turret`.

```python
from lockon.envs.turret import TurretEnvConfig, make_vector_env

config = TurretEnvConfig(render_mode=None, max_episode_steps=200)
env = make_vector_env(num_envs=4, base_seed=0, config=config)
observations, infos = env.reset()
```

The default training path returns low-dimensional state observations and skips rendering. A sample script is available at `examples/training/vector_sample.py`.

## gRPC server

The gRPC server remains compatible with the existing demo clients, but it is intended for manual control, PID demos, and remote debugging rather than high-throughput RL sampling.

Each `StreamEnv` connection now owns an isolated environment session, so multiple clients can connect without sharing MuJoCo state.
