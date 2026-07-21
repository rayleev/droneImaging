"""影像入库异步 Pipeline 编排模块

编排完整的入库流程：
  上传原始文件 → 解析元数据 → COG 转换 → 缩略图 → VLM 描述 → Embedding → Milvus

任一步骤失败不影响数据库记录，status 标记为 error 并记录错误信息。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config
from src.models.image import Image

# 全局并发信号量，限制同时处理的影像数
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """获取并发控制信号量（延迟初始化）"""
    global _semaphore
    if _semaphore is None:
        cfg = get_config().processing
        _semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
    return _semaphore


async def process_image(
    session: AsyncSession,
    image_id: uuid.UUID,
    file_path: str,
) -> None:
    """执行完整的影像入库 Pipeline。

    Args:
        session: 数据库 Session
        image_id: 影像记录 ID
        file_path: 本地临时文件路径
    """
    sem = _get_semaphore()
    async with sem:
        await _do_process(session, image_id, file_path)


async def _do_process(
    session: AsyncSession,
    image_id: uuid.UUID,
    file_path: str,
) -> None:
    """Pipeline 实际执行逻辑"""
    cfg = get_config()
    file_path = Path(file_path)

    # 加载影像记录
    image = await session.get(Image, image_id)
    if image is None:
        logger.error(f"Pipeline: 影像记录不存在 {image_id}")
        return

    try:
        # ── Step 1: 上传原始文件到 MinIO ──
        await _update_status(session, image, "parsing", "上传原始文件到 MinIO")
        from src.services.storage import upload_file

        object_name = f"{image.task_id}/{image.id}/{image.original_filename}"
        upload_file(cfg.minio.buckets.raw, object_name, file_path)
        image.original_path = object_name
        image.file_size_bytes = file_path.stat().st_size
        await session.commit()
        logger.info(f"[{image_id}] Step 1 完成: 原始文件已上传 MinIO")

        # ── Step 2: 解析 GeoTIFF 元数据 ──
        from src.services.geotiff import parse_geotiff

        metadata = parse_geotiff(file_path)
        image.image_width = metadata.image_width
        image.image_height = metadata.image_height
        image.bands = metadata.bands
        image.crs = metadata.crs
        image.bbox = metadata.bbox
        image.center_lon = metadata.center_lon
        image.center_lat = metadata.center_lat
        image.pixel_scale_x = metadata.pixel_scale_x
        image.pixel_scale_y = metadata.pixel_scale_y
        image.nodata = metadata.nodata
        image.geotransform = {"values": metadata.geotransform} if metadata.geotransform else None
        await session.commit()
        logger.info(f"[{image_id}] Step 2 完成: GeoTIFF 元数据已解析")

        # ── Step 3: COG 转换 ──
        await _update_status(session, image, "converting", "COG 转换中")
        from src.services.cog import convert_to_cog

        # COG 转换是 CPU 密集型操作，放到线程池
        loop = asyncio.get_event_loop()
        cog_path = await loop.run_in_executor(None, convert_to_cog, file_path)

        # 上传 COG 到 MinIO
        cog_object_name = f"{image.task_id}/{image.id}/{file_path.stem}.tif"
        upload_file(cfg.minio.buckets.cog, cog_object_name, cog_path)
        image.cog_path = cog_object_name
        await session.commit()
        logger.info(f"[{image_id}] Step 3 完成: COG 转换并上传")

        # ── Step 4: 生成缩略图 ──
        from src.utils.thumbnail import generate_thumbnail

        thumb_path = await loop.run_in_executor(None, generate_thumbnail, file_path)
        thumb_object_name = f"{image.task_id}/{image.id}/{file_path.stem}_thumb.jpg"
        upload_file(cfg.minio.buckets.thumb, thumb_object_name, thumb_path)
        image.thumbnail_path = thumb_object_name
        await session.commit()
        logger.info(f"[{image_id}] Step 4 完成: 缩略图已生成")

        # ── Step 5: VLM 描述 ──
        await _update_status(session, image, "describing", "VLM 描述生成中")
        from src.services.vlm import describe_image

        description = await describe_image(thumb_path)
        image.vlm_description = description
        image.vlm_model = cfg.vlm.model
        image.vlm_time = datetime.now()
        await session.commit()
        logger.info(f"[{image_id}] Step 5 完成: VLM 描述已生成")

        # ── Step 6: Embedding + Milvus ──
        await _update_status(session, image, "embedding", "向量生成中")
        from src.services.embedding import build_embedding_text, embed_text
        from src.services.milvus_client import insert_vector

        embed_text_input = build_embedding_text(
            vlm_description=description,
            task_id=image.task_id or "",
            field_name=image.field_name or "",
            survey_stage=image.survey_stage or "",
            device_model=image.device_model or "",
            data_type=image.data_type or "",
        )
        vector = await embed_text(embed_text_input)
        await insert_vector(
            image_id=str(image.id),
            text_vector=vector,
            task_id=image.task_id or "",
            field_name=image.field_name or "",
            survey_stage=image.survey_stage or "",
        )
        image.embedding_id = str(image.id)
        await session.commit()
        logger.info(f"[{image_id}] Step 6 完成: 向量已写入 Milvus")

        # ── 完成 ──
        image.status = "ready"
        image.error_message = None
        await session.commit()
        logger.info(f"[{image_id}] Pipeline 全部完成 ✓")

    except Exception as e:
        logger.exception(f"[{image_id}] Pipeline 失败: {e}")
        image.status = "error"
        image.error_message = str(e)[:2000]  # 截断避免过长
        await session.commit()

    finally:
        # 清理临时文件
        try:
            if file_path.exists():
                file_path.unlink()
                # 尝试清理临时目录
                parent = file_path.parent
                if parent.name.startswith("drone_") or parent.name.startswith("cog_") or parent.name.startswith("thumb_"):
                    import shutil
                    shutil.rmtree(parent, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning(f"[{image_id}] 临时文件清理失败: {cleanup_err}")


async def _update_status(
    session: AsyncSession,
    image: Image,
    status: str,
    progress: str = "",
) -> None:
    """更新影像处理状态"""
    image.status = status
    await session.commit()
    logger.debug(f"[{image.id}] 状态更新: {status} | {progress}")
