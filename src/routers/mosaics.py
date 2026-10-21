"""拼接影像路由

提供拼接影像（pa_di_image_mosaic）的列表 / 详情 / 原图下载 / 原图预览。
其中 ``mosaic_type='mission'`` 为整块试验田影像，可交给 field-shape-web
重新划分小区。
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config, get_public_base_url
from src.database import get_session
from src.models.image import ImageMosaic
from src.schemas.image import MosaicDetail, MosaicListItem, MosaicListResponse

router = APIRouter()


@router.get("", response_model=MosaicListResponse)
async def list_mosaics(
    mosaic_type: Optional[str] = None,
    mission_id: Optional[str] = None,
    plot_id: Optional[str] = None,
    product_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """拼接影像列表（可按 mosaic_type / mission_id / plot_id / product_type 过滤）"""
    conditions = []
    if mosaic_type:
        conditions.append(ImageMosaic.mosaic_type == mosaic_type)
    if mission_id:
        conditions.append(ImageMosaic.mission_id == mission_id)
    if plot_id:
        conditions.append(ImageMosaic.plot_id == plot_id)
    if product_type:
        conditions.append(ImageMosaic.product_type == product_type)

    query = select(ImageMosaic)
    count_query = select(func.count(ImageMosaic.id))
    for cond in conditions:
        query = query.where(cond)
        count_query = count_query.where(cond)

    total = (await session.execute(count_query)).scalar() or 0

    query = query.order_by(ImageMosaic.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    mosaics = (await session.execute(query)).scalars().all()

    base = get_public_base_url()
    items = []
    for m in mosaics:
        item = MosaicListItem.model_validate(m)
        item.thumbnail_url = f"{base}/api/mosaics/{m.id}/thumbnail"
        items.append(item)

    return MosaicListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{mosaic_id}", response_model=MosaicDetail)
async def get_mosaic(
    mosaic_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """拼接影像详情"""
    mosaic = await session.get(ImageMosaic, mosaic_id)
    if mosaic is None:
        raise HTTPException(status_code=404, detail="拼接影像不存在")

    detail = MosaicDetail.model_validate(mosaic)
    base = get_public_base_url()
    detail.thumbnail_url = f"{base}/api/mosaics/{mosaic.id}/thumbnail"
    return detail


@router.get("/{mosaic_id}/thumbnail")
async def get_mosaic_thumbnail(
    mosaic_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """拼接影像缩略图（JPEG）"""
    mosaic = await session.get(ImageMosaic, mosaic_id)
    if mosaic is None:
        raise HTTPException(status_code=404, detail="拼接影像不存在")
    if not mosaic.thumbnail_path:
        raise HTTPException(status_code=404, detail="缩略图尚未生成")

    from src.services.storage import get_client

    cfg = get_config()
    client = get_client()
    try:
        response = client.get_object(cfg.minio.buckets.thumb, mosaic.thumbnail_path)
        data = response.read()
        response.close()
        response.release_conn()
        return Response(content=data, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"读取拼接影像缩略图失败: {mosaic_id} | {e}")
        raise HTTPException(status_code=500, detail="读取缩略图失败")


@router.get("/{mosaic_id}/original")
async def get_mosaic_original(
    mosaic_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """下载拼接影像原始 GeoTIFF（服务端流式代理，供 field-shape-web 使用）"""
    mosaic = await session.get(ImageMosaic, mosaic_id)
    if mosaic is None:
        raise HTTPException(status_code=404, detail="拼接影像不存在")
    if not mosaic.file_path:
        raise HTTPException(status_code=404, detail="该拼接影像没有文件（file_path 为空）")

    from src.services.preview import stream_object

    filename = f"mosaic_{mosaic.mosaic_type}_{mosaic.product_type}_{mosaic.id}.tif"
    return stream_object(get_config().minio.buckets.cog, mosaic.file_path, filename)


@router.get("/{mosaic_id}/original.png")
async def get_mosaic_original_png(
    mosaic_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """拼接影像原图预览 PNG（最大边长 8192，带磁盘缓存）"""
    mosaic = await session.get(ImageMosaic, mosaic_id)
    if mosaic is None:
        raise HTTPException(status_code=404, detail="拼接影像不存在")
    if not mosaic.file_path:
        raise HTTPException(status_code=404, detail="该拼接影像没有文件（file_path 为空）")

    from src.services.preview import render_preview_png
    from src.services.storage import download_file

    cfg = get_config()
    loop = asyncio.get_event_loop()
    tmp_dir = Path(tempfile.mkdtemp(prefix="preview_mosaic_"))
    try:
        local_path = await loop.run_in_executor(
            None,
            download_file,
            cfg.minio.buckets.cog,
            mosaic.file_path,
            str(tmp_dir / "mosaic.tif"),
        )
        data = await loop.run_in_executor(
            None, render_preview_png, str(local_path), f"mosaic_{mosaic.id}"
        )
        return Response(content=data, media_type="image/png")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
