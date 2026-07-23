"""语义检索路由

提供基于自然语言的影像语义检索接口，
通过 Milvus 向量搜索 + PostgreSQL 元数据补全实现。
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config, get_public_base_url
from src.database import get_session
from src.models.image import Image
from src.schemas.image import (
    ImageSearchRequest,
    ImageSearchResponse,
    ImageSearchResultItem,
)

router = APIRouter()


@router.post("/search", response_model=ImageSearchResponse)
async def search_images(
    req: ImageSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """语义检索无人机影像

    将自然语言 query 转为向量，在 Milvus 中搜索相似影像，
    再从 PostgreSQL 补全元数据，返回带瓦片 URL 的结果列表。
    """
    from src.services.embedding import embed_text
    from src.services.milvus_client import search_vectors

    # 1. query → embedding
    try:
        query_vector = await embed_text(req.query)
    except Exception as e:
        logger.error(f"Embedding 失败: {e}")
        raise HTTPException(status_code=500, detail=f"文本向量化失败: {e}")

    # 2. Milvus 向量搜索
    filters = {}
    if req.filters:
        if req.filters.task_id:
            filters["task_id"] = req.filters.task_id
        if req.filters.field_name:
            filters["field_name"] = req.filters.field_name
        if req.filters.survey_stage:
            filters["survey_stage"] = req.filters.survey_stage
        if req.filters.crop_type:
            filters["crop_type"] = req.filters.crop_type
        if req.filters.growth_stage:
            filters["growth_stage"] = req.filters.growth_stage

    try:
        search_results = await search_vectors(query_vector, top_k=req.limit, filters=filters)
    except Exception as e:
        logger.error(f"Milvus 搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"向量检索失败: {e}")

    if not search_results:
        return ImageSearchResponse(results=[], total=0)

    # 3. 从 PostgreSQL 补全元数据
    image_ids = [uuid.UUID(r["id"]) for r in search_results]
    score_map = {r["id"]: r["score"] for r in search_results}

    stmt = select(Image).where(Image.id.in_(image_ids))
    result = await session.execute(stmt)
    images = {str(img.id): img for img in result.scalars().all()}

    # 4. 组装响应
    base = get_public_base_url()
    items = []
    for r in search_results:
        img = images.get(r["id"])
        if img is None:
            continue  # Milvus 有但 PG 没有（数据不一致），跳过

        # 从 extra_metadata 提取 VLM 结构化字段（crop_type、growth_stage）
        extra = img.extra_metadata or {}
        crop_type = extra.get("crop_type")
        growth_stage = extra.get("growth_stage")

        items.append(ImageSearchResultItem(
            id=img.id,
            task_id=img.task_id,
            field_name=img.field_name,
            survey_stage=img.survey_stage,
            crop_type=crop_type,
            growth_stage=growth_stage,
            survey_time=img.survey_time,
            bbox=img.bbox,
            center=[img.center_lon, img.center_lat] if img.center_lon else None,
            thumbnail_url=f"{base}/api/images/{img.id}/thumbnail",
            tile_url=f"{base}/api/tiles/{img.id}/{{z}}/{{x}}/{{y}}.png",
            vlm_description=img.vlm_description,
            score=score_map[r["id"]],
        ))

    return ImageSearchResponse(results=items, total=len(items))
