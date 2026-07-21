"""MinIO 对象存储操作封装

提供文件上传、下载、预签名 URL 生成等基础操作，
应用启动时自动检查/创建所需的桶。
"""

from __future__ import annotations

import io
from pathlib import Path

from loguru import logger
from minio import Minio
from minio.error import S3Error

from src.config import get_config

# 全局 MinIO 客户端
_client: Minio | None = None


def init_storage() -> None:
    """初始化 MinIO 客户端并确保所需桶存在。

    在 FastAPI lifespan 中调用。
    """
    global _client
    cfg = get_config().minio

    _client = Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )

    # 检查/创建桶
    for bucket_name in [cfg.buckets.raw, cfg.buckets.cog, cfg.buckets.thumb]:
        try:
            if not _client.bucket_exists(bucket_name):
                _client.make_bucket(bucket_name)
                logger.info(f"MinIO 桶已创建: {bucket_name}")
            else:
                logger.debug(f"MinIO 桶已存在: {bucket_name}")
        except S3Error as e:
            logger.error(f"MinIO 桶初始化失败 [{bucket_name}]: {e}")
            raise


def get_client() -> Minio:
    """获取 MinIO 客户端实例"""
    if _client is None:
        raise RuntimeError("MinIO 未初始化，请先调用 init_storage()")
    return _client


def upload_file(bucket: str, object_name: str, file_path: str | Path) -> str:
    """上传本地文件到 MinIO

    Args:
        bucket: 桶名
        object_name: 对象路径（如 task_id/image_id/file.tif）
        file_path: 本地文件路径

    Returns:
        对象路径
    """
    client = get_client()
    file_path = Path(file_path)
    client.fput_object(bucket, object_name, str(file_path))
    logger.debug(f"MinIO 上传: {bucket}/{object_name} ({file_path.stat().st_size} bytes)")
    return object_name


def upload_bytes(bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """上传字节数据到 MinIO

    Args:
        bucket: 桶名
        object_name: 对象路径
        data: 文件内容
        content_type: MIME 类型

    Returns:
        对象路径
    """
    client = get_client()
    client.put_object(bucket, object_name, io.BytesIO(data), len(data), content_type=content_type)
    logger.debug(f"MinIO 上传(bytes): {bucket}/{object_name} ({len(data)} bytes)")
    return object_name


def download_file(bucket: str, object_name: str, local_path: str | Path) -> Path:
    """从 MinIO 下载文件到本地

    Args:
        bucket: 桶名
        object_name: 对象路径
        local_path: 本地保存路径

    Returns:
        本地文件路径
    """
    client = get_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.fget_object(bucket, object_name, str(local_path))
    logger.debug(f"MinIO 下载: {bucket}/{object_name} → {local_path}")
    return local_path


def get_presigned_url(bucket: str, object_name: str, expires_hours: int = 24) -> str:
    """生成预签名访问 URL

    Args:
        bucket: 桶名
        object_name: 对象路径
        expires_hours: 有效期（小时）

    Returns:
        预签名 URL
    """
    from datetime import timedelta
    client = get_client()
    return client.presigned_get_object(bucket, object_name, expires=timedelta(hours=expires_hours))


def file_exists(bucket: str, object_name: str) -> bool:
    """检查对象是否存在"""
    client = get_client()
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error:
        return False
