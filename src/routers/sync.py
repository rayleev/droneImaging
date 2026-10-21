"""从 agri_research 同步影像的路由

数据源已切换为从 flight_missions 同步小区裁剪片（image_products）。
旧的 POST /api/images/upload 接口保留供临时补传，标记为 deprecated。

提供：
- POST /api/images/sync        触发同步（可选 mission_id / experiment_id / force）
- GET  /api/images/sync/status 查询最近一次同步进度
- PATCH /api/images/{id}/refresh-metadata  轻量回填元数据 + 重建向量（不重跑 VLM）
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config
from src.database import get_session
import src.database as _db
from src.models.image import Image
from src.schemas.image import (
    ImageSyncRequest,
    ImageSyncResponse,
    ImageSyncStatusResponse,
    RefreshMetadataResponse,
)

router = APIRouter()


# ── 同步进度内存态（单实例足够；多 worker 部署需换 Redis） ────

_sync_status: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "total": 0,
    "processed": 0,
    "succeeded": 0,
    "failed": 0,
    "skipped": 0,
    "last_error": None,
    "last_image_product_id": None,
}


def _reset_sync_status(total: int) -> None:
    _sync_status.update(
        running=True,
        started_at=datetime.now().isoformat(),
        finished_at=None,
        total=total,
        processed=0,
        succeeded=0,
        failed=0,
        skipped=0,
        last_error=None,
        last_image_product_id=None,
    )


def _finish_sync_status(last_error: Optional[str] = None) -> None:
    _sync_status.update(
        running=False,
        finished_at=datetime.now().isoformat(),
        last_error=last_error,
    )


@router.post("/sync/trigger", response_model=ImageSyncResponse, status_code=202)
async def sync_images(
    req: ImageSyncRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """触发从 agri_research 同步影像。

    - 不传 mission_id/experiment_id = 增量同步所有 processed=true 且本地未存在的 image_products
    - 传 mission_id = 单次飞行任务
    - 传 experiment_id = 该试验下所有 mission
    - force=true = 已存在的也强制重跑（重跑 VLM + embedding）

    同步在后台执行，立即返回 202；通过 GET /api/images/sync/status 查进度。
    """
    cfg = get_config()
    if not cfg.research_db.enabled:
        raise HTTPException(status_code=503, detail="research_db 未启用，无法同步")

    if _sync_status["running"]:
        raise HTTPException(status_code=409, detail="已有同步任务在运行中")

    # 拉候选
    from src.services.research_sync import list_sync_candidates

    # 本地已存在的 image_product_id（force=false 时跳过）
    existing_ids: list[str] = []
    if not req.force:
        stmt = select(Image.image_product_id).where(Image.image_product_id.is_not(None))
        result = await session.execute(stmt)
        existing_ids = [str(r[0]) for r in result.fetchall() if r[0] is not None]

    try:
        candidates = await list_sync_candidates(
            mission_id=str(req.mission_id) if req.mission_id else None,
            experiment_id=str(req.experiment_id) if req.experiment_id else None,
            product_type=req.product_type,
            existing_image_product_ids=existing_ids if not req.force else None,
        )
    except Exception as e:
        logger.exception(f"拉取同步候选失败: {e}")
        raise HTTPException(status_code=502, detail=f"拉取 agri_research 候选失败: {e}")

    if not candidates:
        return ImageSyncResponse(
            accepted=False,
            total=0,
            message="无可同步的新影像（本地已全部存在）",
        )

    _reset_sync_status(total=len(candidates))
    background_tasks.add_task(_run_sync, candidates, req.force)
    return ImageSyncResponse(
        accepted=True,
        total=len(candidates),
        message=f"已接受同步任务，共 {len(candidates)} 张影像待处理",
    )


async def _run_sync(candidates: list[dict], force: bool) -> None:
    """后台执行同步：逐条富化 + 入库 pipeline。"""
    from src.services.research_sync import (
        fetch_plot_enrichment,
        fetch_observations_snapshot,
    )
    from src.services.pipeline import process_sync_image

    sem = asyncio.Semaphore(get_config().processing.max_concurrent_tasks)

    async def _one(candidate: dict) -> None:
        async with sem:
            image_product_id = str(candidate["image_product_id"])
            try:
                # 富化
                plot_id = str(candidate["plot_id"]) if candidate.get("plot_id") else None
                enrichment: dict = {}
                obs_snapshot: dict = {}
                if plot_id:
                    capture_date = (
                        candidate.get("capture_date")
                        or candidate.get("flight_date")
                        or datetime.now().strftime("%Y-%m-%d")
                    )
                    capture_date_str = (
                        capture_date.strftime("%Y-%m-%d")
                        if isinstance(capture_date, datetime)
                        else str(capture_date)
                    )
                    try:
                        enrichment = await fetch_plot_enrichment(plot_id)
                        obs_snapshot = await fetch_observations_snapshot(
                            plot_id, capture_date_str
                        )
                    except Exception as e:
                        logger.warning(
                            f"富化失败 image_product={image_product_id}: {e}（降级继续）"
                        )

                # 入库 pipeline（独立 session，跳过 VLM/Embedding 以提升速度）
                async with _db.async_session_factory() as sess:
                    await process_sync_image(
                        session=sess,
                        candidate=candidate,
                        enrichment=enrichment,
                        observations_snapshot=obs_snapshot,
                        force=force,
                        skip_vlm_embedding=True,  # 跳过 VLM/Embedding，后续并行处理
                    )
                _sync_status["succeeded"] += 1
            except Exception as e:
                logger.exception(f"同步失败 image_product={image_product_id}: {e}")
                _sync_status["failed"] += 1
                _sync_status["last_error"] = str(e)[:500]
            finally:
                _sync_status["processed"] += 1
                _sync_status["last_image_product_id"] = image_product_id

    try:
        # 阶段 1：入库（跳过 VLM/Embedding，提升速度）
        await asyncio.gather(*[_one(c) for c in candidates])
    except Exception as e:
        logger.exception(f"同步任务异常: {e}")
        _finish_sync_status(last_error=str(e)[:500])
        return

    # 阶段 2：Mosaic 拼接（VLM/Embedding 由 processing.skip_vlm_embedding 控制，
    # 禁用时只拼接、不生成描述与向量）
    logger.info("入库完成，开始 Mosaic 拼接...")

    async def _run_mosaic() -> None:
        """创建拼接影像（VLM/Embedding 按配置决定是否执行）。"""
        try:
            from src.services.mosaic import create_mosaics_for_session
            async with _db.async_session_factory() as mosaic_session:
                mosaic_count = await create_mosaics_for_session(mosaic_session)
                logger.info(f"创建了 {mosaic_count} 个拼接影像")
        except Exception as e:
            logger.exception(f"创建拼接影像失败: {e}")

    try:
        await _run_mosaic()
    except Exception as e:
        logger.exception(f"Mosaic 拼接异常: {e}")

    _finish_sync_status()
    logger.info(
        f"同步完成: total={_sync_status['total']} "
        f"succeeded={_sync_status['succeeded']} failed={_sync_status['failed']}"
    )


@router.get("/sync-status", response_model=ImageSyncStatusResponse)
async def get_sync_status():
    """查询最近一次同步进度"""
    return ImageSyncStatusResponse(**_sync_status)


@router.patch("/refresh-metadata/{image_id}", response_model=RefreshMetadataResponse)
async def refresh_metadata(
    image_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """轻量回填元数据 + 重建向量（不重跑 VLM）。

    用于 flight_missions / plots / observations 元数据变更后刷新本地标量列和向量，
    复用现有 vlm_description，不重新调用 VLM。
    """
    cfg = get_config()
    if not cfg.research_db.enabled:
        raise HTTPException(status_code=503, detail="research_db 未启用")

    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    if not image.plot_id:
        raise HTTPException(
            status_code=400, detail="该影像非 sync 来源，无 plot_id 可回填"
        )

    from src.services.research_sync import (
        fetch_plot_enrichment,
        fetch_observations_snapshot,
    )
    from src.services.embedding import build_embedding_text, embed_text
    from src.services.milvus_client import delete_vector, insert_vector

    capture_date = image.capture_date or image.flight_date or datetime.now().strftime("%Y-%m-%d")
    try:
        enrichment = await fetch_plot_enrichment(str(image.plot_id))
        obs_snapshot = await fetch_observations_snapshot(str(image.plot_id), capture_date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"拉取 agri_research 元数据失败: {e}")

    # 回填标量列
    if enrichment.get("plot_code"):
        image.plot_code = enrichment["plot_code"]
    if enrichment.get("variety") is not None:
        image.variety = enrichment["variety"]
    if enrichment.get("treatment") is not None:
        image.treatment = enrichment["treatment"]

    # 合并 extra_metadata（保留 VLM 输出，更新 observations / plot_geom / site）
    extra = image.extra_metadata or {}
    extra["plot_geom"] = enrichment.get("plot_geom")
    extra["experiment_name"] = enrichment.get("experiment_name")
    extra["site_name"] = enrichment.get("site_name")
    extra["observations_snapshot"] = obs_snapshot.get("observations", [])
    extra["growth_stage_at_capture"] = obs_snapshot.get("growth_stage_at_capture")
    image.extra_metadata = extra

    # 重建向量
    growth_stage = obs_snapshot.get("growth_stage_at_capture") or ""
    experiment_name = enrichment.get("experiment_name") or ""
    crop_type = enrichment.get("experiment_crop_type") or ""
    site_name = enrichment.get("site_name") or ""
    embed_input = build_embedding_text(
        vlm_description=image.vlm_description or "",
        task_id=image.task_id or "",
        field_name=image.field_name or "",
        survey_stage=image.survey_stage or "",
        device_model=image.device_model or "",
        data_type=image.data_type or "",
        experiment_name=experiment_name,
        crop_type=crop_type,
        site_name=site_name,
        plot_code=image.plot_code or "",
        variety=image.variety or "",
        treatment=image.treatment or "",
        flight_date=image.flight_date or "",
        growth_stage=growth_stage,
    )
    try:
        vector = await embed_text(embed_input)
        try:
            await delete_vector(str(image.id))
        except Exception:
            pass
        await insert_vector(
            image_id=str(image.id),
            text_vector=vector,
            task_id=image.task_id or "",
            field_name=image.field_name or "",
            survey_stage=image.survey_stage or "",
            crop_type=crop_type,
            growth_stage=growth_stage,
            mission_id=str(image.mission_id) if image.mission_id else "",
            plot_id=str(image.plot_id) if image.plot_id else "",
            experiment_id=str(image.experiment_id) if image.experiment_id else "",
            site_id=str(image.site_id) if image.site_id else "",
            plot_code=image.plot_code or "",
            flight_date=image.flight_date or "",
            product_type=image.product_type or "",
            variety=image.variety or "",
            treatment=image.treatment or "",
            data_source_type=(obs_snapshot.get("observations") or [{}])[0].get("data_source_type", "") if obs_snapshot.get("observations") else "",
        )
    except Exception as e:
        logger.exception(f"重建向量失败: {e}")
        raise HTTPException(status_code=500, detail=f"重建向量失败: {e}")

    await session.commit()
    return RefreshMetadataResponse(
        id=image.id,
        refreshed_at=datetime.now(),
        message="元数据已回填，向量已重建",
    )
