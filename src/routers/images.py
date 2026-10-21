"""影像管理路由

提供影像上传、详情查询、列表查询、状态查询、缩略图、搜索、新增、编辑、软删除等接口。
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from loguru import logger
from sqlalchemy import func, select, or_, and_
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


@router.post("/upload", response_model=ImageUploadResponse, status_code=201, deprecated=True)
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
    """[已废弃] 上传无人机 GeoTIFF 影像

    数据源已切换为从 agri_research.flight_missions 同步小区裁剪片，
    请改用 POST /api/images/sync。此接口保留仅供临时补传/调试。
    """
    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="仅支持 GeoTIFF 文件（.tif/.tiff）")

    max_upload_bytes = get_config().server.max_upload_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content) / 1024 / 1024:.1f} MB），上限 {get_config().server.max_upload_mb} MB",
        )

    parsed_survey_time = None
    if survey_time:
        try:
            parsed_survey_time = datetime.fromisoformat(survey_time)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"调查时间格式无效: {survey_time}，请使用 ISO 8601")

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

    import tempfile
    from pathlib import Path
    from src.services.pipeline import process_image

    safe_name = Path(file.filename).name if file.filename else "upload.tif"
    if not safe_name:
        safe_name = "upload.tif"

    tmp_dir = Path(tempfile.mkdtemp(prefix="drone_"))
    tmp_path = tmp_dir / safe_name
    tmp_path.write_bytes(content)

    import asyncio
    asyncio.create_task(_run_pipeline(image.id, str(tmp_path)))

    return ImageUploadResponse(
        id=image.id,
        status="uploaded",
        message="影像已上传，后台处理中",
    )


async def _run_pipeline(image_id: uuid.UUID, file_path: str):
    """后台执行入库 Pipeline（独立 session）"""
    import src.database as _db
    from src.services.pipeline import process_image

    async with _db.async_session_factory() as session:
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
    base = get_public_base_url()
    detail.thumbnail_url = f"{base}/api/images/{image_id}/thumbnail"
    detail.tile_url = f"{base}/api/tiles/{image_id}/{{z}}/{{x}}/{{y}}.png"
    return detail


@router.get("", response_model=ImageListResponse)
async def list_images(
    search: Optional[str] = None,
    task_id: Optional[str] = None,
    field_name: Optional[str] = None,
    field_group: Optional[str] = None,
    survey_stage: Optional[str] = None,
    data_type: Optional[str] = None,
    surveyor: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """影像列表（支持搜索 + 过滤 + 分页）"""
    query = select(Image).where(Image.deleted_at.is_(None))
    count_query = select(func.count(Image.id)).where(Image.deleted_at.is_(None))

    if search:
        search_pattern = f"%{search}%"
        search_filter = or_(
            Image.task_id.ilike(search_pattern),
            Image.field_group.ilike(search_pattern),
            Image.field_name.ilike(search_pattern),
            Image.survey_stage.ilike(search_pattern),
            Image.device_model.ilike(search_pattern),
            Image.data_type.ilike(search_pattern),
            Image.surveyor.ilike(search_pattern),
            Image.original_filename.ilike(search_pattern),
            Image.vlm_description.ilike(search_pattern),
            Image.vlm_model.ilike(search_pattern),
            Image.embedding_id.ilike(search_pattern),
            Image.error_message.ilike(search_pattern),
            Image.status.ilike(search_pattern),
            Image.source.ilike(search_pattern),
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    if task_id:
        query = query.where(Image.task_id == task_id)
        count_query = count_query.where(Image.task_id == task_id)
    if field_name:
        query = query.where(Image.field_name == field_name)
        count_query = count_query.where(Image.field_name == field_name)
    if field_group:
        query = query.where(Image.field_group == field_group)
        count_query = count_query.where(Image.field_group == field_group)
    if survey_stage:
        query = query.where(Image.survey_stage == survey_stage)
        count_query = count_query.where(Image.survey_stage == survey_stage)
    if data_type:
        query = query.where(Image.data_type == data_type)
        count_query = count_query.where(Image.data_type == data_type)
    if surveyor:
        query = query.where(Image.surveyor == surveyor)
        count_query = count_query.where(Image.surveyor == surveyor)
    if status:
        query = query.where(Image.status == status)
        count_query = count_query.where(Image.status == status)

    total = (await session.execute(count_query)).scalar() or 0

    query = query.order_by(Image.upload_time.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(query)
    images = result.scalars().all()

    base = get_public_base_url()
    items = []
    for img in images:
        item = ImageListItem.model_validate(img)
        item.thumbnail_url = f"{base}/api/images/{img.id}/thumbnail"
        item.tile_url = f"{base}/api/tiles/{img.id}/{{z}}/{{x}}/{{y}}.png"
        items.append(item)

    return ImageListResponse(items=items, total=total, page=page, page_size=page_size)


@router.post("", response_model=ImageDetail, status_code=201)
async def create_image(
    task_id: str = Form(..., description="任务编号"),
    field_group: Optional[str] = Form(None),
    field_name: Optional[str] = Form(None),
    survey_stage: Optional[str] = Form(None),
    device_model: Optional[str] = Form(None),
    data_type: Optional[str] = Form(None),
    surveyor: Optional[str] = Form(None),
    survey_time: Optional[str] = Form(None),
    status: str = Form("uploaded"),
    source: str = Form("upload"),
    error_message: Optional[str] = Form(None),
    vlm_description: Optional[str] = Form(None),
    extra_metadata: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """新增影像记录"""
    parsed_survey_time = None
    if survey_time:
        try:
            parsed_survey_time = datetime.fromisoformat(survey_time)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"调查时间格式无效: {survey_time}")

    parsed_extra = None
    if extra_metadata:
        import json
        try:
            parsed_extra = json.loads(extra_metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="extra_metadata 格式无效，应为 JSON 字符串")

    image = Image(
        task_id=task_id,
        field_group=field_group,
        field_name=field_name,
        survey_stage=survey_stage,
        device_model=device_model,
        data_type=data_type,
        surveyor=surveyor,
        survey_time=parsed_survey_time,
        original_filename=task_id + ".tif",
        status=status,
        source=source,
        error_message=error_message,
        vlm_description=vlm_description,
        extra_metadata=parsed_extra,
    )
    session.add(image)
    await session.commit()
    await session.refresh(image)

    logger.info(f"影像已创建: {image.id} | task={task_id}")

    detail = ImageDetail.model_validate(image)
    base = get_public_base_url()
    detail.thumbnail_url = f"{base}/api/images/{image.id}/thumbnail"
    detail.tile_url = f"{base}/api/tiles/{image.id}/{{z}}/{{x}}/{{y}}.png"
    return detail


@router.put("/{image_id}", response_model=ImageDetail)
async def update_image(
    image_id: uuid.UUID,
    task_id: Optional[str] = Form(None),
    field_group: Optional[str] = Form(None),
    field_name: Optional[str] = Form(None),
    survey_stage: Optional[str] = Form(None),
    device_model: Optional[str] = Form(None),
    data_type: Optional[str] = Form(None),
    surveyor: Optional[str] = Form(None),
    survey_time: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    error_message: Optional[str] = Form(None),
    vlm_description: Optional[str] = Form(None),
    extra_metadata: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """编辑影像记录"""
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")

    if task_id is not None:
        image.task_id = task_id
    if field_group is not None:
        image.field_group = field_group
    if field_name is not None:
        image.field_name = field_name
    if survey_stage is not None:
        image.survey_stage = survey_stage
    if device_model is not None:
        image.device_model = device_model
    if data_type is not None:
        image.data_type = data_type
    if surveyor is not None:
        image.surveyor = surveyor
    if survey_time is not None:
        try:
            image.survey_time = datetime.fromisoformat(survey_time) if survey_time else None
        except ValueError:
            raise HTTPException(status_code=400, detail=f"调查时间格式无效: {survey_time}")
    if status is not None:
        image.status = status
    if source is not None:
        image.source = source
    if error_message is not None:
        image.error_message = error_message
    if vlm_description is not None:
        image.vlm_description = vlm_description
    if extra_metadata is not None:
        import json
        try:
            image.extra_metadata = json.loads(extra_metadata) if extra_metadata else None
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="extra_metadata 格式无效，应为 JSON 字符串")

    await session.commit()
    await session.refresh(image)

    logger.info(f"影像已更新: {image.id}")

    detail = ImageDetail.model_validate(image)
    base = get_public_base_url()
    detail.thumbnail_url = f"{base}/api/images/{image.id}/thumbnail"
    detail.tile_url = f"{base}/api/tiles/{image.id}/{{z}}/{{x}}/{{y}}.png"
    return detail


@router.delete("/{image_id}")
async def delete_image(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """软删除影像记录"""
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")

    image.deleted_at = datetime.now()
    await session.commit()

    logger.info(f"影像已软删除: {image.id}")

    return {"message": "删除成功", "id": str(image.id)}


@router.get("/{image_id}/status", response_model=ImageStatusResponse)
async def get_image_status(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """查询影像处理状态"""
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")

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


@router.get("/{image_id}/original")
async def get_original_image(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """下载原始 GeoTIFF（服务端流式代理）。

    供外部系统（如 field-shape-web）拉取原始影像使用。
    """
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    if not image.original_path:
        raise HTTPException(status_code=404, detail="该影像没有原始文件（original_path 为空）")

    from src.services.preview import stream_object

    return stream_object(
        get_config().minio.buckets.raw,
        image.original_path,
        image.original_filename,
    )


@router.get("/{image_id}/original.png")
async def get_original_png(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """原图预览 PNG（最大边长 8192，带磁盘缓存）。

    浏览器无法直接显示 GeoTIFF，前端弹窗通过该端点获取可缩放查看的 PNG。
    """
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    if not image.original_path:
        raise HTTPException(status_code=404, detail="该影像没有原始文件（original_path 为空）")

    from src.services.preview import render_preview_png
    from src.services.storage import download_file

    cfg = get_config()
    loop = asyncio.get_event_loop()
    tmp_dir = Path(tempfile.mkdtemp(prefix="preview_"))
    try:
        local_path = await loop.run_in_executor(
            None,
            download_file,
            cfg.minio.buckets.raw,
            image.original_path,
            str(tmp_dir / image.original_filename),
        )
        data = await loop.run_in_executor(
            None, render_preview_png, str(local_path), f"img_{image.id}"
        )
        return Response(content=data, media_type="image/png")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
