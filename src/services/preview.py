"""原图预览 / 下载服务

提供两个能力：
1. 从 MinIO 流式代理原始 GeoTIFF（供 field-shape-web 等外部系统拉取）
2. 把 GeoTIFF 渲染成 PNG 供前端弹窗查看（带磁盘缓存，最大边长受限）
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Iterator

from fastapi.responses import StreamingResponse
from loguru import logger

from src.utils.thumbnail import PREVIEW_MAX_EDGE, read_geotiff_rgb

_PREVIEW_CACHE_DIR = Path(tempfile.gettempdir()) / "drone_preview_cache"


def get_preview_cache_path(cache_key: str) -> Path:
    """返回预览 PNG 的缓存路径（不保证存在）。"""
    _PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _PREVIEW_CACHE_DIR / f"{cache_key}.png"


def render_preview_png(local_path: str | Path, cache_key: str) -> bytes:
    """把本地 GeoTIFF 渲染成 PNG 字节（命中缓存则直接返回）。

    同步函数，调用方应放到线程池执行（rasterio 读取为 CPU/IO 密集型）。
    """
    cache_path = get_preview_cache_path(cache_key)
    if cache_path.exists():
        logger.debug(f"预览缓存命中: {cache_path.name}")
        return cache_path.read_bytes()

    pil_img, width, height = read_geotiff_rgb(local_path, max_size=PREVIEW_MAX_EDGE)
    logger.info(f"渲染原图预览: {Path(local_path).name} {width}x{height} → PNG")

    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    data = buf.getvalue()

    try:
        cache_path.write_bytes(data)
    except OSError as e:
        logger.warning(f"预览缓存写入失败: {e}")

    return data


def _iter_object(resp) -> Iterator[bytes]:
    """迭代 MinIO 响应流并在结束时释放连接。"""
    try:
        yield from resp.stream(amt=1024 * 1024)
    finally:
        try:
            resp.close()
            resp.release_conn()
        except Exception:
            pass


def stream_object(bucket: str, object_name: str, filename: str) -> StreamingResponse:
    """从 MinIO 流式代理一个对象（用于下载原始 GeoTIFF）。

    相比返回预签名 URL，服务端代理可避免 MinIO 未配置 CORS 导致浏览器
    跨域 fetch 失败的问题。
    """
    from src.services.storage import get_client

    client = get_client()
    resp = client.get_object(bucket, object_name)

    safe_name = filename.replace('"', "_")
    return StreamingResponse(
        _iter_object(resp),
        media_type="image/tiff",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
        },
    )
