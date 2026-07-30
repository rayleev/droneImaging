"""试验小区划分路由

提供将无人机影像区域划分为试验小区的接口。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.models.image import Image
from src.schemas.image import PlotDivideRequest, PlotDivideResponse
from src.services.plot_divider import divide_plots

router = APIRouter()


@router.post("/divide", response_model=PlotDivideResponse)
async def divide_plots_endpoint(
    req: PlotDivideRequest,
    session: AsyncSession = Depends(get_session),
):
    """划分试验小区

    将指定影像的区域按行列数或小区面积划分为网格状小区，
    返回每个小区的坐标、面积和编号。
    """
    # 1. 加载影像
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

    # 2. 校验参数
    if req.n_rows is None and req.n_cols is None and req.plot_width_m is None and req.plot_height_m is None:
        raise HTTPException(status_code=400, detail="必须指定 n_rows/n_cols 或 plot_width_m/plot_height_m 之一")

    # 3. 执行划分
    try:
        response = divide_plots(
            image=image,
            region=req.region,
            n_rows=req.n_rows,
            n_cols=req.n_cols,
            plot_width_m=req.plot_width_m,
            plot_height_m=req.plot_height_m,
            rotation_deg=req.rotation_deg,
            label_scheme=req.label_scheme,
        )
    except ValueError as e:
        logger.error(f"划分失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"划分异常: {e}")
        raise HTTPException(status_code=500, detail=f"划分失败: {e}")

    return PlotDivideResponse(**response)
