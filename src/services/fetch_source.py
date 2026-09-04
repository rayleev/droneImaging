"""通用数据源拉取层

按 source_type 分发到不同适配器：
- scp:   SSH/SCP 拉取（临时方案，152 源文件服务器）
- minio: S3 协议拉取（长期方案，需配置 MinIO 源）
- nas:   本地/挂载路径直接读取

配置存储在 pa_di_fetch_source 表，供 sync 流程使用。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.fetch_source import FetchSource


async def get_fetch_source_by_name(
    session: AsyncSession, name: str
) -> Optional[FetchSource]:
    """按名称取数据源配置"""
    stmt = select(FetchSource).where(FetchSource.name == name)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_fetch_source_by_id(
    session: AsyncSession, source_id: str
) -> Optional[FetchSource]:
    """按 ID 取数据源配置"""
    stmt = select(FetchSource).where(FetchSource.id == source_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


def fetch_file(
    source: FetchSource,
    file_path: str,
    local_dir: Optional[str | Path] = None,
) -> Path:
    """按 source.source_type 分发到对应适配器拉取文件。

    Args:
        source: FetchSource 配置对象
        file_path: 文件路径（相对或绝对，按适配器解析）
        local_dir: 本地存放目录；None 则创建临时目录

    Returns:
        本地文件路径

    Raises:
        ValueError: 不支持的 source_type
        RuntimeError: 拉取失败
    """
    dispatch = {
        "scp": _fetch_via_scp,
        "minio": _fetch_via_minio,
        "nas": _fetch_via_nas,
        "http": _fetch_via_http,
    }
    adapter = dispatch.get(source.source_type)
    if adapter is None:
        raise ValueError(f"不支持的 source_type: {source.source_type}")
    return adapter(source, file_path, local_dir)


def _fetch_via_scp(
    source: FetchSource,
    file_path: str,
    local_dir: Optional[str | Path] = None,
) -> Path:
    """SSH/SCP 适配器"""
    import paramiko

    host_port = _parse_endpoint(source.endpoint)
    credentials = source.credentials or {}
    config = source.config or {}
    remote_root = config.get("remote_root", "")
    timeout = float(config.get("timeout", 30))

    # 解析远程路径
    if os.path.isabs(file_path):
        remote_path = file_path
    else:
        remote_path = os.path.join(remote_root, file_path.lstrip("/"))

    if local_dir is None:
        local_dir = Path(tempfile.mkdtemp(prefix="scp_fetch_"))
    else:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / os.path.basename(remote_path)

    logger.info(f"[SCP] {source.name}: {source.endpoint}:{remote_path} → {local_path}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host_port["host"],
            port=host_port["port"],
            username=credentials.get("user"),
            password=credentials.get("password"),
            timeout=timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        sftp = client.open_sftp()
        try:
            sftp.stat(remote_path)
            sftp.get(remote_path, str(local_path))

            # 同时拉取对应的世界文件 (.pgw / .tfw) 用于地理参考
            world_ext = ".pgw" if remote_path.lower().endswith(".png") else ".tfw"
            remote_world = remote_path.rsplit(".", 1)[0] + world_ext
            local_world = local_dir / os.path.basename(remote_world)
            try:
                sftp.stat(remote_world)
                sftp.get(remote_world, str(local_world))
                logger.info(f"[SCP] 同时拉取世界文件: {local_world.name}")
            except FileNotFoundError:
                pass  # 世界文件不存在则跳过
        finally:
            sftp.close()
    finally:
        client.close()

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError(f"SCP 拉取失败: {remote_path}")

    logger.info(f"[SCP] 完成: {local_path} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return local_path


def _fetch_via_minio(
    source: FetchSource,
    file_path: str,
    local_dir: Optional[str | Path] = None,
) -> Path:
    """MinIO/S3 适配器"""
    try:
        from minio import Minio
    except ImportError:
        raise RuntimeError("minio 库未安装，请 pip install minio")

    credentials = source.credentials or {}
    config = source.config or {}
    bucket = config.get("bucket", "source-images")
    endpoint = source.endpoint.replace("http://", "").replace("https://", "")

    client = Minio(
        endpoint=endpoint,
        access_key=credentials.get("access_key"),
        secret_key=credentials.get("secret_key"),
        secure=source.endpoint.startswith("https://"),
    )

    if local_dir is None:
        local_dir = Path(tempfile.mkdtemp(prefix="minio_fetch_"))
    else:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / os.path.basename(file_path)
    logger.info(f"[MinIO] {source.name}: {bucket}/{file_path} → {local_path}")

    client.fget_object(bucket, file_path, str(local_path))
    logger.info(f"[MinIO] 完成: {local_path} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return local_path


def _fetch_via_nas(
    source: FetchSource,
    file_path: str,
    local_dir: Optional[str | Path] = None,
) -> Path:
    """NAS/本地挂载适配器"""
    config = source.config or {}
    local_root = config.get("local_root", "")
    remote_root = config.get("remote_root", "")

    if os.path.isabs(file_path):
        src = Path(file_path)
    else:
        src = Path(local_root) / remote_root.lstrip("/") / file_path.lstrip("/")

    if not src.exists():
        raise FileNotFoundError(f"NAS 源文件不存在: {src}")

    if local_dir is None:
        # NAS 可直接读取，无需拷贝
        return src

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / src.name

    import shutil
    shutil.copy2(src, local_path)
    logger.info(f"[NAS] {source.name}: {src} → {local_path}")
    return local_path


def _fetch_via_http(
    source: FetchSource,
    file_path: str,
    local_dir: Optional[str | Path] = None,
) -> Path:
    """HTTP 适配器"""
    import httpx

    base_url = source.endpoint.rstrip("/")
    url = f"{base_url}/{file_path.lstrip('/')}"

    if local_dir is None:
        local_dir = Path(tempfile.mkdtemp(prefix="http_fetch_"))
    else:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / os.path.basename(file_path)
    logger.info(f"[HTTP] {source.name}: {url} → {local_path}")

    with httpx.stream("GET", url, timeout=60) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)

    logger.info(f"[HTTP] 完成: {local_path} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return local_path


def _parse_endpoint(endpoint: str) -> dict:
    """解析 host:port 格式的 endpoint"""
    parts = endpoint.split(":")
    port = int(parts[2]) if len(parts) > 2 else 22
    return {"host": parts[0] + ":" + parts[1] if len(parts) > 2 else parts[0], "port": port}