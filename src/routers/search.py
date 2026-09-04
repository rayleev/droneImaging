"""语义检索路由

提供基于自然语言的影像语义检索接口，通过 Milvus 向量搜索 + 标量过滤 +
两阶段性状过滤实现。

检索流程：
1. 若请求带 nl_query 且未自带 query/filters → LLM 分解为 text_query + filters + trait_filters
   若调用方已传 query/filters → 跳过 LLM 分解
2. 若有 trait_filters → 第一阶段查 agri_research.observations 得 plot_id 集合
3. text_query → embedding → Milvus 向量搜索（带 filters + plot_id in [...] 范围过滤）
4. 从 PostgreSQL 补全完整元数据 + plot_geom，组装响应
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_public_base_url
from src.database import get_session
from src.models.image import Image
from src.schemas.image import (
    ImageSearchRequest,
    ImageSearchResponse,
    ImageSearchResultItem,
    TraitFilter,
)

router = APIRouter()


@router.post("/search", response_model=ImageSearchResponse)
async def search_images(
    req: ImageSearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """语义检索无人机影像

    支持两种调用方式：
    - 传 nl_query：服务端用 LLM 分解为结构化条件（decomposed=true/false 标记是否成功）
    - 直接传 query + filters + trait_filters：跳过 LLM 分解（decomposed=true）
    """
    from src.services.embedding import embed_text
    from src.services.milvus_client import search_vectors

    decomposed = True
    text_query = req.query or ""
    filters = req.filters
    trait_filters = req.trait_filters

    # 1. 需要 LLM 分解的情况
    if req.nl_query and not (req.query or req.filters):
        from src.services.query_decompose import decompose_query
        result = await decompose_query(req.nl_query)
        decomposed = result["decomposed"]
        text_query = result["text_query"] or req.nl_query
        if filters is None:
            filters = result["filters"]
        if trait_filters is None:
            trait_filters = result["trait_filters"]

    if not text_query and not filters and not trait_filters:
        return ImageSearchResponse(results=[], total=0, decomposed=decomposed)

    # 2. 两阶段性状过滤：第一阶段查 agri_research 得 plot_id 集合
    in_filters: Optional[dict] = None
    if trait_filters:
        try:
            from src.services.research_sync import resolve_plot_ids_by_traits
            tf_dicts = [tf.model_dump() for tf in trait_filters]
            plot_ids = await resolve_plot_ids_by_traits(tf_dicts)
        except Exception as e:
            logger.warning(f"性状过滤第一阶段失败，忽略性状条件: {e}")
            plot_ids = []
        if plot_ids:
            in_filters = {"plot_id": plot_ids}
        elif trait_filters:
            # 性状条件存在但命中的 plot_id 为空 → 直接返回空结果
            return ImageSearchResponse(results=[], total=0, decomposed=decomposed)

    # 3. query → embedding
    query_vector = None
    if text_query:
        try:
            query_vector = await embed_text(text_query)
        except Exception as e:
            logger.error(f"Embedding 失败: {e}")
            raise HTTPException(status_code=500, detail=f"文本向量化失败: {e}")

    # 4. 构建标量过滤
    eq_filters: dict = {}
    range_filters: dict = {}
    if filters:
        for key in [
            "task_id", "field_name", "survey_stage", "crop_type", "growth_stage",
            "mission_id", "plot_id", "experiment_id", "site_id", "plot_code",
            "product_type", "variety", "treatment", "data_source_type",
        ]:
            v = getattr(filters, key, None)
            if v:
                eq_filters[key] = str(v)
        if filters.flight_date:
            eq_filters["flight_date"] = filters.flight_date
        range_ops = {}
        if filters.flight_date_from:
            range_ops["gte"] = filters.flight_date_from
        if filters.flight_date_to:
            range_ops["lte"] = filters.flight_date_to
        if range_ops:
            range_filters["flight_date"] = range_ops

    # 5. Milvus 搜索
    try:
        search_results = await search_vectors(
            query_vector=query_vector or [],
            top_k=req.limit,
            filters=eq_filters or None,
            in_filters=in_filters,
            range_filters=range_filters or None,
        )
    except Exception as e:
        logger.error(f"Milvus 搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"向量检索失败: {e}")

    if not search_results:
        return ImageSearchResponse(results=[], total=0, decomposed=decomposed)

    # 6. 从 PG 补全元数据
    image_ids = [uuid.UUID(r["id"]) for r in search_results]
    score_map = {r["id"]: r["score"] for r in search_results}

    stmt = select(Image).where(Image.id.in_(image_ids))
    result = await session.execute(stmt)
    images = {str(img.id): img for img in result.scalars().all()}

    # 7. 组装响应
    base = get_public_base_url()
    items = []
    for r in search_results:
        img = images.get(r["id"])
        if img is None:
            continue
        extra = img.extra_metadata or {}
        crop_type = extra.get("crop_type") or r.get("crop_type") or None
        growth_stage = extra.get("growth_stage") or r.get("growth_stage") or None
        plot_geom = extra.get("plot_geom")

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
            mission_id=str(img.mission_id) if img.mission_id else None,
            plot_id=str(img.plot_id) if img.plot_id else None,
            experiment_id=str(img.experiment_id) if img.experiment_id else None,
            site_id=str(img.site_id) if img.site_id else None,
            plot_code=img.plot_code,
            flight_date=img.flight_date,
            product_type=img.product_type,
            variety=img.variety,
            treatment=img.treatment,
            plot_geom=plot_geom,
        ))

    # 8. group_by 聚合（可选）
    if req.group_by == "mission":
        seen_missions: set = set()
        grouped = []
        for it in items:
            key = it.mission_id or str(it.id)
            if key in seen_missions:
                continue
            seen_missions.add(key)
            grouped.append(it)
        items = grouped

    return ImageSearchResponse(results=items, total=len(items), decomposed=decomposed)
