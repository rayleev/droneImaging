"""影像管理路由

提供影像上传、详情查询、列表查询、状态查询、缩略图等接口。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config, get_public_base_url
from src.database import get_session
from src.models.image import Image
from src.schemas.image import (
    ImageDetail,
    ImageListItem,
    ImageListResponse,
    ImageStatusResponse,
    ImageUploadResponse,
)

router = APIRouter()


@router.post("/upload", response_model=ImageUploadResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(..., description="GeoTIFF 文件"),
    task_id: str = Form(..., description="任务编号"),
    field_group: Optional[str] = Form(None),
    field_name: Optional[str] = Form(None),
    survey_stage: Optional[str] = Form(None),
    device_model: Optional[str] = Form(None),
    data_type: Optional[str] = Form(None),
    surveyor: Optional[str] = Form(None),
    survey_time: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """上传无人机 GeoTIFF 影像

    接收文件和业务字段，创建数据库记录后触发后台处理 Pipeline。
    """
    # 校验文件类型
    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="仅支持 GeoTIFF 文件（.tif/.tiff）")

    # 校验文件大小（防 OOM）：先读入内存但设上限
    max_upload_bytes = get_config().server.max_upload_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content) / 1024 / 1024:.1f} MB），上限 {get_config().server.max_upload_mb} MB",
        )

    # 解析调查时间
    parsed_survey_time = None
    if survey_time:
        try:
            parsed_survey_time = datetime.fromisoformat(survey_time)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"调查时间格式无效: {survey_time}，请使用 ISO 8601")

    # 创建数据库记录
    image = Image(
        task_id=task_id,
        field_group=field_group,
        field_name=field_name,
        survey_stage=survey_stage,
        device_model=device_model,
        data_type=data_type,
        surveyor=surveyor,
        survey_time=parsed_survey_time,
        original_filename=file.filename,
        status="uploaded",
        source="upload",
    )
    session.add(image)
    await session.commit()
    await session.refresh(image)

    logger.info(f"影像已上传: {image.id} | file={file.filename} task={task_id}")

    # 保存临时文件并触发后台 Pipeline
    import tempfile
    from pathlib import Path
    from src.services.pipeline import process_image

    # 安全：仅取文件名 basename，防止路径遍历（如 ../../etc/passwd）
    safe_name = Path(file.filename).name if file.filename else "upload.tif"
    if not safe_name:
        safe_name = "upload.tif"

    tmp_dir = Path(tempfile.mkdtemp(prefix="drone_"))
    tmp_path = tmp_dir / safe_name
    tmp_path.write_bytes(content)

    # 异步后台处理（不阻塞响应）
    import asyncio
    asyncio.create_task(_run_pipeline(image.id, str(tmp_path)))

    return ImageUploadResponse(
        id=image.id,
        status="uploaded",
        message="影像已上传，后台处理中",
    )


async def _run_pipeline(image_id: uuid.UUID, file_path: str):
    """后台执行入库 Pipeline（独立 session）"""
    from src.database import async_session_factory
    from src.services.pipeline import process_image

    async with async_session_factory() as session:
        await process_image(session, image_id, file_path)


@router.get("/{image_id}", response_model=ImageDetail)
async def get_image_detail(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """获取单张影像的完整详情"""
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")

    detail = ImageDetail.model_validate(image)
    # 补充 URL
    base = get_public_base_url()
    detail.thumbnail_url = f"{base}/api/images/{image_id}/thumbnail"
    detail.tile_url = f"{base}/api/tiles/{image_id}/{{z}}/{{x}}/{{y}}.png"
    return detail


@router.get("", response_model=ImageListResponse)
async def list_images(
    task_id: Optional[str] = None,
    field_name: Optional[str] = None,
    survey_stage: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """影像列表（支持过滤 + 分页）"""
    query = select(Image)
    count_query = select(func.count(Image.id))

    # 过滤条件
    if task_id:
        query = query.where(Image.task_id == task_id)
        count_query = count_query.where(Image.task_id == task_id)
    if field_name:
        query = query.where(Image.field_name == field_name)
        count_query = count_query.where(Image.field_name == field_name)
    if survey_stage:
        query = query.where(Image.survey_stage == survey_stage)
        count_query = count_query.where(Image.survey_stage == survey_stage)
    if status:
        query = query.where(Image.status == status)
        count_query = count_query.where(Image.status == status)

    # 总数
    total = (await session.execute(count_query)).scalar() or 0

    # 分页
    query = query.order_by(Image.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(query)
    images = result.scalars().all()

    # 补充 URL
    base = get_public_base_url()
    items = []
    for img in images:
        item = ImageListItem.model_validate(img)
        item.thumbnail_url = f"{base}/api/images/{img.id}/thumbnail"
        item.tile_url = f"{base}/api/tiles/{img.id}/{{z}}/{{x}}/{{y}}.png"
        items.append(item)

    return ImageListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{image_id}/status", response_model=ImageStatusResponse)
async def get_image_status(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """查询影像处理状态"""
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")

    # 状态描述映射
    progress_map = {
        "uploaded": "已上传，等待处理",
        "parsing": "正在解析 GeoTIFF 元数据",
        "converting": "正在转换为 COG 格式",
        "describing": "正在生成 VLM 描述",
        "embedding": "正在生成向量并写入 Milvus",
        "ready": "处理完成",
        "error": "处理失败",
    }

    return ImageStatusResponse(
        id=image.id,
        status=image.status,
        progress=progress_map.get(image.status, image.status),
        error_message=image.error_message,
    )


@router.get("/{image_id}/thumbnail")
async def get_thumbnail(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """获取影像缩略图（JPEG）"""
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    if not image.thumbnail_path:
        raise HTTPException(status_code=404, detail="缩略图尚未生成")

    from src.services.storage import get_client
    cfg = get_config()
    client = get_client()

    try:
        response = client.get_object(cfg.minio.buckets.thumb, image.thumbnail_path)
        data = response.read()
        response.close()
        response.release_conn()
        return Response(content=data, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"读取缩略图失败: {image_id} | {e}")
        raise HTTPException(status_code=500, detail="读取缩略图失败")
