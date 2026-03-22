# lockon

`lockon` 提供了一个基于 MuJoCo 的炮塔环境，用于本地强化学习训练和基于 gRPC 的演示。

## 本地向量化训练

推荐使用本地 `gymnasium.vector.AsyncVectorEnv` worker 作为主要训练路径。相关辅助 API 位于 `lockon.envs.turret`。

```python
from lockon.envs.turret import TurretEnvConfig, make_vector_env

config = TurretEnvConfig(render_mode=None, max_episode_steps=200)
env = make_vector_env(num_envs=4, base_seed=0, config=config)
observations, infos = env.reset()
```

默认训练路径返回低维状态观测，并跳过渲染。示例脚本位于 `examples/training/vector_sample.py`。

## gRPC server

gRPC 服务器仍与现有演示客户端兼容，但其设计目标是手动控制、PID 演示和远程调试，而不是高吞吐量的强化学习采样。

现在每个 `StreamEnv` 连接都拥有独立隔离的环境会话，因此多个客户端可以同时连接，而无需共享 MuJoCo 状态。

## Gym V2 协议

`lockon.protos.gym_v2` 是一个轻量级、Gym 风格的 RPC 协议模式，适用于常见的 `Box + container` 场景。它并不打算覆盖完整的 Gymnasium space 协议。

编码约定：

- `Tuple(...)` 类型的值编码为 `TensorList`
- `Dict(...)` 类型的值编码为 `TensorDict`
- 该 proto 不支持嵌套容器；调用方必须在传输前将其展平
- `reward`、`terminated` 和 `truncated` 始终是 repeated 字段；单环境响应的长度为 `1`
- `observation_space` 和 `action_space` 不会在线协议中传输，必须由通信双方在协议外事先约定
