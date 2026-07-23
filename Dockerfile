# droneImaging — 无人机影像服务 Dockerfile
#
# 构建: docker build -t drone-imaging .
# 运行: docker run -d -p 8002:8002 -v $(pwd)/config.yaml:/app/config.yaml drone-imaging
#
# 注意: config.yaml 含密钥且被 git 忽略，运行时应通过 -v 挂载真实配置文件，
#       或通过环境变量传入（见 config.py 的 DRONE_PUBLIC_BASE_URL）。

FROM python:3.12-slim

# 避免 Python 写 .pyc 文件 + 输出不缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 系统依赖：rasterio/GDAL、Pillow 所需的运行时库
# (rasterio wheel 已内置 GDAL，此处补充 zlib/libjpeg/expat 等 Pillow/rasterio 依赖)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpng-dev \
        libjpeg-dev \
        libtiff-dev \
        libgeos-dev \
        libexpat1 \
        curl \
        gettext-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY prompts/ ./prompts/

# 复制配置模板作为启动时的兜底（真实配置建议通过 -v 挂载 config.yaml 覆盖）
COPY config.yaml.example ./config.yaml.example

# 健康检查（服务内置 /health 端点）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

EXPOSE 8002

# 生产启动：单 worker 即可（容器编排层面扩容），如需多 worker可改为 gunicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
