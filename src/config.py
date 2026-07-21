"""无人机影像服务 — 配置加载模块

从 config.yaml 读取所有外部服务连接信息和处理参数，
通过 Pydantic 模型校验，全局单例访问。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel


# ── 各配置节模型 ──────────────────────────────────────────────

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8002
    workers: int = 2
    debug: bool = False


class MinioBucketsConfig(BaseModel):
    raw: str = "drone-raw"
    cog: str = "drone-cog"
    thumb: str = "drone-thumb"


class MinioConfig(BaseModel):
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    buckets: MinioBucketsConfig = MinioBucketsConfig()


class PostgresqlConfig(BaseModel):
    url: str = "postgresql+asyncpg://postgres:password@localhost:5432/drone_imaging"


class MilvusConfig(BaseModel):
    host: str = "localhost"
    port: int = 19530
    collection: str = "image_vectors"
    vector_dim: int = 1024


class EmbeddingConfig(BaseModel):
    api_url: str = "http://localhost:8080/embed"
    api_key: str = ""
    model: str = "BAAI/bge-large-zh-v1.5"
    batch_size: int = 8


class VlmConfig(BaseModel):
    provider: str = "qwen-vl"
    api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "sk-xxx"
    model: str = "qwen-vl-max"
    prompt_file: str = "prompts/vlm_describe.txt"
    max_image_size: int = 1024
    timeout: int = 60
    max_retries: int = 3


class TiandituConfig(BaseModel):
    key: str = "your-tianditu-key"


class ProcessingConfig(BaseModel):
    cog_blocksize: int = 512
    cog_overview_levels: List[int] = [2, 4, 8, 16]
    thumbnail_size: int = 512
    max_concurrent_tasks: int = 3


# ── 顶层配置 ─────────────────────────────────────────────────

class AppConfig(BaseModel):
    """应用全局配置，对应 config.yaml 顶层结构"""
    server: ServerConfig = ServerConfig()
    minio: MinioConfig = MinioConfig()
    postgresql: PostgresqlConfig = PostgresqlConfig()
    milvus: MilvusConfig = MilvusConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vlm: VlmConfig = VlmConfig()
    tianditu: TiandituConfig = TiandituConfig()
    processing: ProcessingConfig = ProcessingConfig()


# ── 全局单例 ─────────────────────────────────────────────────

_config: AppConfig | None = None

# 项目根目录（config.yaml 所在目录）
BASE_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """加载 config.yaml 并返回 AppConfig 单例。

    Args:
        config_path: 配置文件路径，默认为项目根目录下的 config.yaml
    """
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        config_path = BASE_DIR / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _config = AppConfig(**raw)
    return _config


def get_config() -> AppConfig:
    """获取已加载的配置，未加载时自动加载默认路径"""
    if _config is None:
        return load_config()
    return _config


def get_public_base_url() -> str:
    """获取对外可访问的基础 URL。

    优先级：
    1. 环境变量 DRONE_PUBLIC_BASE_URL（Docker / 反向代理部署时显式指定，
       例如 http://drone-imaging:8002 或 https://drone.example.com）
    2. config.yaml 中的 server.host/port；绑定地址 0.0.0.0 对客户端不可达，
       需替换为 localhost，否则生成的 tile_url / thumbnail_url 浏览器和
       LLM 都无法访问。
    """
    env_url = os.environ.get("DRONE_PUBLIC_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    cfg = get_config()
    host = cfg.server.host
    if host in ("0.0.0.0", "::", ""):
        host = "localhost"
    return f"http://{host}:{cfg.server.port}"
