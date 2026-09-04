"""影像 VLM 描述和 Embedding 后处理模块

用于为已入库（跳过 VLM/Embedding）的影像补充生成 VLM 描述和向量。
可与 Mosaic 拼接并行执行，提升整体吞吐量。
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config
from src.models.image import Image
import src.database as _db


async def finalize_pending_vlm_embeddings(session: AsyncSession) -> int:
    """为所有 status='ready' 但缺少 vlm_description 的影像补充 VLM/Embedding。

    Args:
        session: 数据库 Session

    Returns:
        处理的影像数量
    """
    stmt = select(Image).where(
        Image.status == 'ready',
        Image.vlm_description.is_(None),
        Image.thumbnail_path.is_not(None),
    )
    result = await session.execute(stmt)
    images = result.scalars().all()

    if not images:
        logger.info("[VLM Finalize] 无需处理的影像")
        return 0

    logger.info(f"[VLM Finalize] 找到 {len(images)} 张待处理影像")
    sem = asyncio.Semaphore(get_config().processing.max_concurrent_tasks)

    async def _one(img: Image) -> None:
        async with sem:
            await _finalize_single_image(img)

    await asyncio.gather(*[_one(img) for img in images])
    logger.info(f"[VLM Finalize] 完成 {len(images)} 张影像")
    return len(images)


async def _finalize_single_image(image: Image) -> None:
    """为单张影像生成 VLM 描述和 Embedding。"""
    from src.services.storage import download_file
    from src.services.vlm import describe_image
    from src.services.embedding import build_embedding_text, embed_text
    from src.services.milvus_client import insert_vector

    cfg = get_config()
    tmp_dir = Path(tempfile.mkdtemp(prefix="vlm_finalize_"))

    try:
        # 使用独立 session
        async with _db.async_session_factory() as session:
            img = await session.get(Image, image.id)
            if img is None or img.vlm_description is not None:
                return  # 已被其他任务处理

            # 下载缩略图
            if not img.thumbnail_path:
                return
            local_thumb = tmp_dir / f"thumb_{img.id}.jpg"
            download_file(cfg.minio.buckets.thumb, img.thumbnail_path, str(local_thumb))

            # VLM 描述
            logger.info(f"[VLM Finalize] 开始 {img.id}")
            result = await describe_image(str(local_thumb))
            img.vlm_description = result.summary
            img.vlm_model = cfg.vlm.model
            img.vlm_time = datetime.now()

            existing_extra = img.extra_metadata or {}
            existing_extra.update(result.raw)
            img.extra_metadata = existing_extra
            await session.commit()

            # Embedding + Milvus
            embed_input = build_embedding_text(
                vlm_description=result.summary,
                task_id=img.task_id or "",
                field_name=img.field_name or "",
                survey_stage=img.survey_stage or "",
                device_model=img.device_model or "",
                data_type=img.data_type or "",
                experiment_name=img.extra_metadata.get("experiment_name", "") if img.extra_metadata else "",
                crop_type=result.crop_type or (img.extra_metadata.get("experiment_crop_type", "") if img.extra_metadata else ""),
                site_name=img.extra_metadata.get("site_name", "") if img.extra_metadata else "",
                plot_code=img.plot_code or "",
                variety=img.variety or "",
                treatment=img.treatment or "",
                flight_date=img.flight_date or "",
                growth_stage=result.growth_stage or (img.extra_metadata.get("growth_stage_at_capture", "") if img.extra_metadata else ""),
            )
            vector = await embed_text(embed_input)
            img.embedding_id = await insert_vector(
                image_id=str(img.id),
                text_vector=vector,
                task_id=img.task_id or "",
                field_name=img.field_name or "",
                survey_stage=img.survey_stage or "",
                crop_type=result.crop_type or "",
                growth_stage=result.growth_stage or "",
                mission_id=img.mission_id or "",
                plot_id=img.plot_id or "",
                experiment_id=img.experiment_id or "",
                site_id=img.site_id or "",
                plot_code=img.plot_code or "",
                flight_date=img.flight_date or "",
                product_type=img.product_type or "",
                variety=img.variety or "",
                treatment=img.treatment or "",
                data_source_type="",
                granularity=img.granularity or "plot",
            )
            await session.commit()
            logger.info(f"[VLM Finalize] 完成 {img.id}")
        except RuntimeError as e:
            # Embedding API 限流或失败，跳过该影像（后续可补跑）
            logger.warning(f"[VLM Finalize] Embedding 失败，跳过 {img.id}: {e}")
            # 不标记为 error，保持 ready 状态以便后续补跑

    except Exception as e:
        logger.exception(f"[VLM Finalize] 失败 {image.id}: {e}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
