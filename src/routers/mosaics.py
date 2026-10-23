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


@router.get("/create")
async def create_mosaics(
    session: AsyncSession = Depends(get_session),
):
    """手动触发拼接影像创建（GET 方式，绕过 nginx POST 限制）。

    返回中附带 errors：每个拼接失败的原因，便于定位静默失败。
    """
    from src.services.mosaic import create_mosaics_for_session
    errors: list = []
    count = await create_mosaics_for_session(session, errors=errors)
    return {
        "created": count,
        "error_count": len(errors),
        "errors": errors[:50],
        "message": f"创建了 {count} 个拼接影像（失败 {len(errors)}）",
    }


@router.get("/diagnostic")
async def diagnostic(
    session: AsyncSession = Depends(get_session),
):
    """诊断拼接影像创建条件：统计就绪影像中具备/缺失关键字段的数量。"""
    from src.models.image import Image as ImageModel

    base_stmt = select(ImageModel).where(
        ImageModel.status == 'ready',
        ImageModel.deleted_at.is_(None),
    )
    all_ready = (await session.execute(base_stmt)).scalars().all()

    total = len(all_ready)
    with_cog = sum(1 for img in all_ready if img.cog_path)
    without_cog = total - with_cog
    with_plot = sum(1 for img in all_ready if img.plot_id)
    with_mission = sum(1 for img in all_ready if img.mission_id)
    with_product_type = sum(1 for img in all_ready if img.product_type)

    # 按 mission_id + product_type 分组，统计可拼接的飞行任务数
    from collections import defaultdict
    mission_groups = defaultdict(set)
    for img in all_ready:
        if img.mission_id and img.plot_id and img.cog_path:
            key = (str(img.mission_id), img.product_type or '')
            mission_groups[key].add(str(img.plot_id))

    stitchable_missions = {
        k: len(v) for k, v in mission_groups.items() if len(v) >= 2
    }

    return {
        "total_ready": total,
        "with_cog_path": with_cog,
        "without_cog_path": without_cog,
        "with_plot_id": with_plot,
        "with_mission_id": with_mission,
        "with_product_type": with_product_type,
        "stitchable_missions": len(stitchable_missions),
        "mission_detail": [
            {"mission_id": k[0], "product_type": k[1], "unique_plots": v}
            for k, v in stitchable_missions.items()
        ],
    }


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
        item.tile_url = f"{base}/api/mosaics/{m.id}/tiles/{{z}}/{{x}}/{{y}}.png"
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
    detail.tile_url = f"{base}/api/mosaics/{mosaic.id}/tiles/{{z}}/{{x}}/{{y}}.png"
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


@router.get("/{mosaic_id}/tiles/{z}/{x}/{y}.png")
async def get_mosaic_tile(
    mosaic_id: uuid.UUID,
    z: int,
    x: int,
    y: int,
    session: AsyncSession = Depends(get_session),
):
    """获取拼接影像的 XYZ 瓦片（256×256 PNG）。

    复用影像瓦片服务的渲染逻辑，从 MinIO 中的拼接 COG 按需生成瓦片，
    使整块田影像可以叠加在卫星地图底图上显示。
    """
    mosaic = await session.get(ImageMosaic, mosaic_id)
    if mosaic is None:
        raise HTTPException(status_code=404, detail="拼接影像不存在")
    if not mosaic.file_path:
        raise HTTPException(status_code=404, detail="该拼接影像没有文件（file_path 为空）")

    from src.routers.tiles import _render_tile, _tile_bounds, _transparent_tile
    from src.services.storage import get_presigned_url

    cfg = get_config()
    try:
        cog_url = get_presigned_url(cfg.minio.buckets.cog, mosaic.file_path)
    except Exception as e:
        logger.error(f"生成拼接影像预签名 URL 失败: {e}")
        raise HTTPException(status_code=500, detail="存储访问失败")

    tile_bbox = _tile_bounds(z, x, y)
    try:
        png_bytes = _render_tile(cog_url, tile_bbox, None)
    except Exception as e:
        logger.error(f"拼接影像瓦片渲染失败 [{mosaic_id}] z={z} x={x} y={y}: {e}")
        png_bytes = _transparent_tile()

    return Response(content=png_bytes, media_type="image/png")


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


@router.get("/{mosaic_id}/original.tif")
async def get_mosaic_original_tif(
    mosaic_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """同 /original，仅让 URL 带 .tif 后缀。

    field-shape-web 是根据 URL 最后一段推断文件名的，若没有扩展名会被判为
    400 "Unsupported file format"，因此这里提供一个带后缀的等价地址。
    """
    return await get_mosaic_original(mosaic_id, session)


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
    tmp_dir = Path(tempfile.mkdtemp(prefix="preview_mosaic_"))
    try:
        local_path = await asyncio.get_running_loop().run_in_executor(
            None,
            download_file,
            cfg.minio.buckets.cog,
            mosaic.file_path,
            str(tmp_dir / "mosaic.tif"),
        )
        data = await asyncio.get_running_loop().run_in_executor(
            None, render_preview_png, str(local_path), f"mosaic_{mosaic.id}"
        )
        return Response(content=data, media_type="image/png")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
