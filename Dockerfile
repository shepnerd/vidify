# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 系统依赖：ffmpeg 用于抽帧/音频/裁剪；git/ca-certificates 便于拉包
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖定义以便利用 Docker layer cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝代码
COPY agent /app/agent
COPY server /app/server
COPY templates /app/templates
COPY static /app/static

# 拷贝配置文件
COPY models.yaml /app/models.yaml
COPY hooks.yaml /app/hooks.yaml
COPY workflows.yaml /app/workflows.yaml

# 默认缓存目录（可通过挂载覆盖）
RUN mkdir -p /app/cache
ENV CACHE_ROOT=/app/cache

EXPOSE 9000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "9000"]
