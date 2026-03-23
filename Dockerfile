FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/python:3.10.18-poetry-0-4-1

# 设置工作目录
WORKDIR /app

# 安装 MuJoCo 离屏渲染依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libgl1-mesa-dri \
    libgles2 \
    libosmesa6 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 拷贝必要的文件以安装依赖
COPY pyproject.toml poetry.lock README.md ./

# 安装依赖
RUN mkdir -p src/lockon && \
    touch src/lockon/__init__.py && \
    poetry install --no-root

# 拷贝 pyproject.toml 和 poetry.lock 文件
COPY . .

# 安装依赖
RUN poetry install

# 暴露 gRPC 服务端口
EXPOSE 50051

# 默认入口
CMD ["poetry", "run", "python", "-m", "lockon.commands.app"]
