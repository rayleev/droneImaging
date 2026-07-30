"""智能补全路由

基于用户绘制的示例区域和自然语言描述，自动补全剩余试验小区。
支持通过配置切换策略：sam_llm、vlm 或 sam_vlm。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.models.image import Image
from src.schemas.image import CompleteRequest, CompleteResponse
from src.services.plot_completion import (
    SamLLMStrategy,
    VLMStrategy,
    SamVLMStrategy,
    SamTemplateStrategy,
    VLMDirectStrategy,
    get_completion_config,
    CompletionRequest as InternalCompletionRequest,
)

router = APIRouter()


def _get_strategy():
    """根据配置选择补全策略"""
    config = get_completion_config()
    if config.strategy == "vlm_direct":
        return VLMDirectStrategy(config)
    elif config.strategy == "sam_template":
        return SamTemplateStrategy(config)
    elif config.strategy == "sam_vlm":
        return SamVLMStrategy(config)
    elif config.strategy == "vlm":
        return VLMStrategy(config)
    else:
        return SamLLMStrategy(config)


@router.post("/complete", response_model=CompleteResponse)
async def complete_plots(
    req: CompleteRequest,
    session: AsyncSession = Depends(get_session),
):
    """智能补全试验小区"""
    try:
        image_id = uuid.UUID(str(req.image_id))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"无效的 image_id: {req.image_id}")

    stmt = select(Image).where(Image.id == image_id)
    result = await session.execute(stmt)
    image = result.scalar_one_or_none()

    if image is None:
        raise HTTPException(status_code=404, detail=f"影像不存在: {req.image_id}")

    if image.status != "ready":
        raise HTTPException(status_code=400, detail=f"影像未就绪，当前状态: {image.status}")

    strategy = _get_strategy()
    logger.info(f"使用策略: {strategy.name} - {strategy.description}")

    internal_req = InternalCompletionRequest(
        image_id=image_id,
        example_region=req.example_region,
        description=req.description,
        image_bbox=image.bbox,
        cog_path=image.cog_path,
        nodata=image.nodata,
    )

    try:
        comp_result = await strategy.complete(internal_req)
    except Exception as e:
        logger.exception("智能补全失败: {}", e)
        raise HTTPException(status_code=500, detail="智能补全失败: " + str(e))

    plots_data = []
    for p in comp_result.plots:
        plots_data.append({
            "id": p.id,
            "label": p.label,
            "row": p.row,
            "col": p.col,
            "bbox": p.bbox,
            "polygon": p.polygon,
            "area_m2": p.area_m2,
        })

    logger.info(f"智能补全完成: {comp_result.total} 个小区 ({comp_result.n_rows}行 x {comp_result.n_cols}列)")

    response_data = {
        "image_id": str(image.id),
        "total": comp_result.total,
        "n_rows": comp_result.n_rows,
        "n_cols": comp_result.n_cols,
        "region": comp_result.region,
        "example_size_m": comp_result.example_size_m,
        "plots": plots_data,
    }

    if comp_result.debug_info:
        response_data["debug_info"] = comp_result.debug_info

    from fastapi.responses import JSONResponse
    return JSONResponse(content=response_data)
