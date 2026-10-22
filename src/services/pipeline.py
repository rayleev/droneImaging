"""影像入库异步 Pipeline 编排模块

编排完整的入库流程：
  上传原始文件 → 解析元数据 → COG 转换 → 缩略图 → VLM 描述 → Embedding → Milvus

任一步骤失败不影响数据库记录，status 标记为 error 并记录错误信息。
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    skip_vlm_embedding: bool | None = None,
) -> None:
    """执行完整的影像入库 Pipeline。

    Args:
        session: 数据库 Session
        image_id: 影像记录 ID
        file_path: 本地临时文件路径
        skip_vlm_embedding: 是否跳过 VLM 描述 / Embedding / Milvus 步骤；
            None 时取配置 ``processing.skip_vlm_embedding``
    """
    sem = _get_semaphore()
    async with sem:
        await _do_process(session, image_id, file_path, skip_vlm_embedding)


async def _do_process(
    session: AsyncSession,
    image_id: uuid.UUID,
    file_path: str,
    skip_vlm_embedding: bool | None = None,
) -> None:
    """Pipeline 实际执行逻辑"""
    cfg = get_config()
    if skip_vlm_embedding is None:
        skip_vlm_embedding = cfg.processing.skip_vlm_embedding
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
        if not skip_vlm_embedding:
            await _update_status(session, image, "describing", "VLM 描述生成中")
            from src.services.vlm import describe_image

            result = await describe_image(thumb_path)
            # summary 用于 embedding 和展示；完整结构化 JSON 存入 extra_metadata
            image.vlm_description = result.summary
            # 合并而非覆盖 extra_metadata，避免抹掉之前写入的数据
            existing_extra = image.extra_metadata or {}
            existing_extra.update(result.raw)
            image.extra_metadata = existing_extra
            image.vlm_model = cfg.vlm.model
            image.vlm_time = datetime.now()
            await session.commit()
            logger.info(f"[{image_id}] Step 5 完成: VLM 描述已生成 (parsed_ok={result.parsed_ok})")

            # ── Step 6: Embedding + Milvus ──
            await _update_status(session, image, "embedding", "向量生成中")
            from src.services.embedding import build_embedding_text, embed_text
            from src.services.milvus_client import insert_vector

            # 用 summary（而非 JSON 原文）做 embedding，避免字段名污染向量语义
            embed_text_input = build_embedding_text(
                vlm_description=result.summary,
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
                crop_type=result.crop_type or "",
                growth_stage=result.growth_stage or "",
            )
            image.embedding_id = str(image.id)
            await session.commit()
            logger.info(f"[{image_id}] Step 6 完成: 向量已写入 Milvus")
        else:
            logger.info(f"[{image_id}] 已跳过 VLM / Embedding / Milvus（skip_vlm_embedding=True）")

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


# ── sync 路径：从 agri_research 同步来的小区裁剪片 ────────────

async def process_sync_image(
    session: AsyncSession,
    candidate: dict,
    enrichment: dict,
    observations_snapshot: dict,
    force: bool = False,
    skip_vlm_embedding: bool | None = None,
) -> None:
    """处理从 agri_research 同步来的单张小区裁剪片。

    与 process_image（upload 路径）的差异：
    - 元数据（mission/plot/experiment/site/obs）已由 sync 路由富化好，直接写入字段
    - 源文件是 PNG+pgw 或 TIFF+tfw，先用 geo_convert 归一化为 GeoTIFF
    - source='sync'，task_id 用 mission_id 前缀
    - force=false 且 image_product_id 已存在则跳过（去重由上游完成，此处兜底）

    Args:
        session: 本服务 DB Session
        candidate: list_sync_candidates 返回的行（含 file_path / mission_id / plot_id 等）
        enrichment: fetch_plot_enrichment 返回的小区富化信息
        observations_snapshot: fetch_observations_snapshot 返回的性状快照
        force: 强制重跑
        skip_vlm_embedding: 是否跳过 VLM/Embedding 步骤；None 时取配置
            ``processing.skip_vlm_embedding``
    """
    sem = _get_semaphore()
    async with sem:
        await _do_process_sync(session, candidate, enrichment, observations_snapshot, force, skip_vlm_embedding)


async def _do_process_sync(
    session: AsyncSession,
    candidate: dict,
    enrichment: dict,
    observations_snapshot: dict,
    force: bool,
    skip_vlm_embedding: bool | None = None,
) -> None:
    cfg = get_config()
    if skip_vlm_embedding is None:
        skip_vlm_embedding = cfg.processing.skip_vlm_embedding
    image_product_id = str(candidate["image_product_id"])
    mission_id = str(candidate["mission_id"]) if candidate.get("mission_id") else None
    plot_id = str(candidate["plot_id"]) if candidate.get("plot_id") else None
    experiment_id = str(candidate["experiment_id"]) if candidate.get("experiment_id") else None
    site_id = None  # 由 enrichment 不直接返回 site_id；plot→site 关系在 enrichment 里未含 id
    product_type = candidate.get("product_type") or ""
    file_path_str = candidate.get("file_path") or ""
    capture_date_raw = candidate.get("capture_date")
    flight_date_raw = candidate.get("flight_date")
    image_id: uuid.UUID | None = None

    def _iso(d) -> Optional[str]:
        if d is None:
            return None
        if isinstance(d, datetime):
            return d.strftime("%Y-%m-%d")
        return str(d)

    capture_date = _iso(capture_date_raw)
    flight_date = _iso(flight_date_raw)

    def _to_dt(d: Optional[str]) -> Optional[datetime]:
        if not d:
            return None
        try:
            return datetime.fromisoformat(d)
        except ValueError:
            return None

    # 调查时间：优先 capture_date，其次 flight_date（前端时间条按此排序展示）
    survey_time = _to_dt(capture_date) or _to_dt(flight_date)

    # 已存在则按 force 决定跳过/重跑
    from sqlalchemy import select as _sel
    existing = (
        await session.execute(
            _sel(Image).where(Image.image_product_id == image_product_id)
        )
    ).scalar_one_or_none()
    if existing and not force:
        logger.info(f"[sync] 跳过已存在 image_product={image_product_id}")
        return
    if existing and force:
        # 删除旧的（含向量）后重建
        from src.services.milvus_client import delete_vector
        try:
            await delete_vector(str(existing.id))
        except Exception:
            pass
        await session.delete(existing)
        await session.commit()

    # 源文件：通过通用数据源层拉取到本地临时目录
    # 配置来源：pa_di_fetch_source 表（名称由 cfg.fetch_source_name 指定）
    from src.services.fetch_source import fetch_file, get_fetch_source_by_name

    src_path: Path
    _tmp_dir: Optional[Path] = None
    loop = asyncio.get_event_loop()
    try:
        if not cfg.fetch_source_enabled:
            raise RuntimeError("数据源拉取未启用（fetch_source_enabled=false）")

        fetch_source = await get_fetch_source_by_name(
            session, cfg.fetch_source_name
        )
        if fetch_source is None:
            raise RuntimeError(
                f"pa_di_fetch_source 表中未找到数据源: {cfg.fetch_source_name}"
            )
        logger.info(f"[sync] 使用数据源: {fetch_source.name} (type={fetch_source.source_type})")
        _tmp_dir = Path(tempfile.mkdtemp(prefix="fetch_"))
        src_path = await loop.run_in_executor(
            None, fetch_file, fetch_source, file_path_str, _tmp_dir
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"远程源文件不存在: {e}")
    except Exception as e:
        raise RuntimeError(f"拉取源文件失败: {e}")

    try:
        # ── 创建 Image 记录（元数据直接写入） ──
        plot_code = enrichment.get("plot_code")
        variety = enrichment.get("variety")
        treatment = enrichment.get("treatment")
        experiment_name = enrichment.get("experiment_name") or ""
        crop_type = enrichment.get("experiment_crop_type") or ""
        site_name = enrichment.get("site_name") or ""
        growth_stage_at_capture = observations_snapshot.get("growth_stage_at_capture") or ""

        # task_id：mission 级编号（用 mission_id 前 8 位 + product_type）
        task_id = f"mission_{mission_id[:8]}_{product_type}" if mission_id else f"sync_{image_product_id[:8]}"
        original_filename = src_path.name

        image = Image(
            task_id=task_id,
            field_name=site_name or None,
            original_filename=original_filename,
            status="parsing",
            source="sync",
            granularity="plot",
            image_product_id=image_product_id,
            mission_id=mission_id,
            plot_id=plot_id,
            experiment_id=experiment_id,
            site_id=site_id,
            plot_code=plot_code,
            flight_date=flight_date,
            product_type=product_type,
            variety=variety,
            treatment=treatment,
            capture_date=capture_date,
            survey_time=survey_time,
            data_type="可见光" if product_type == "rgb" else ("多光谱" if product_type == "ms" else product_type),
            extra_metadata={
                "experiment_name": experiment_name,
                "experiment_crop_type": crop_type,
                "site_name": site_name,
                "plot_geom": enrichment.get("plot_geom"),
                "observations_snapshot": observations_snapshot.get("observations", []),
                "growth_stage_at_capture": growth_stage_at_capture or None,
                "raw_data_path": candidate.get("raw_data_path"),
                "image_products_file_path": file_path_str,
            },
        )
        session.add(image)
        await session.commit()
        await session.refresh(image)
        image_id = image.id
        logger.info(f"[sync] 创建 Image {image_id} image_product={image_product_id}")

        # ── 归一化为 GeoTIFF（线程池） ──
        loop = asyncio.get_event_loop()
        from src.services.geo_convert import to_geotiff, NoGeoReferenceError
        try:
            geotiff_path = await loop.run_in_executor(None, to_geotiff, str(src_path))
        except NoGeoReferenceError as e:
            logger.warning(f"[sync] image_product={image_product_id} 无地理参考，跳过: {e}")
            if image_id is not None:
                try:
                    skip_img = await session.get(Image, image_id)
                    if skip_img is not None:
                        skip_img.status = "no_geo_ref"
                        skip_img.error_message = str(e)[:2000]
                        await session.commit()
                except Exception:
                    pass
            return  # 跳过后续步骤

        # ── 解析 GeoTIFF 元数据 ──
        from src.services.geotiff import parse_geotiff
        metadata = parse_geotiff(geotiff_path)
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
        image.file_size_bytes = src_path.stat().st_size
        await session.commit()

        # ── 上传原始（归一化后）到 MinIO ──
        from src.services.storage import upload_file
        raw_object_name = f"missions/{mission_id}/{product_type}/{src_path.stem}.tif"
        upload_file(cfg.minio.buckets.raw, raw_object_name, geotiff_path)
        image.original_path = raw_object_name
        await session.commit()

        # ── COG 转换 ──
        await _update_status(session, image, "converting", "COG 转换中")
        from src.services.cog import convert_to_cog
        cog_path = await loop.run_in_executor(None, convert_to_cog, str(geotiff_path))
        cog_object_name = f"missions/{mission_id}/{product_type}/{src_path.stem}_cog.tif"
        upload_file(cfg.minio.buckets.cog, cog_object_name, cog_path)
        image.cog_path = cog_object_name
        await session.commit()

        # ── 缩略图 ──
        from src.utils.thumbnail import generate_thumbnail
        thumb_path = await loop.run_in_executor(None, generate_thumbnail, str(geotiff_path))
        thumb_object_name = f"missions/{mission_id}/{product_type}/{src_path.stem}_thumb.jpg"
        upload_file(cfg.minio.buckets.thumb, thumb_object_name, thumb_path)
        image.thumbnail_path = thumb_object_name
        await session.commit()

        # ── VLM 描述（rgb 必跑；ms 也跑，VLM 会按可见光处理或降级） ──
        if not skip_vlm_embedding:
            await _update_status(session, image, "describing", "VLM 描述生成中")
            from src.services.vlm import describe_image
            result = await describe_image(thumb_path)
            image.vlm_description = result.summary
            existing_extra = image.extra_metadata or {}
            existing_extra.update(result.raw)
            image.extra_metadata = existing_extra
            image.vlm_model = cfg.vlm.model
            image.vlm_time = datetime.now()
            await session.commit()

            # ── Embedding + Milvus ──
            await _update_status(session, image, "embedding", "向量生成中")
            from src.services.embedding import build_embedding_text, embed_text
            from src.services.milvus_client import insert_vector

            embed_input = build_embedding_text(
                vlm_description=result.summary,
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
                growth_stage=growth_stage_at_capture,
            )
            vector = await embed_text(embed_input)
            data_source_type = ""
            obs_list = observations_snapshot.get("observations") or []
            if obs_list:
                data_source_type = obs_list[0].get("data_source_type") or ""
            await insert_vector(
                image_id=str(image.id),
                text_vector=vector,
                task_id=image.task_id or "",
                field_name=image.field_name or "",
                survey_stage=image.survey_stage or "",
                crop_type=result.crop_type or crop_type,
                growth_stage=result.growth_stage or growth_stage_at_capture,
                mission_id=mission_id or "",
                plot_id=plot_id or "",
                experiment_id=experiment_id or "",
                site_id=site_id or "",
                plot_code=image.plot_code or "",
                flight_date=image.flight_date or "",
                product_type=product_type or "",
                variety=image.variety or "",
                treatment=image.treatment or "",
                data_source_type=data_source_type,
                granularity="plot",
            )
            image.embedding_id = str(image.id)
            await session.commit()
        else:
            # 跳过 VLM/Embedding，不生成描述与向量
            logger.info(
                f"[sync] 已跳过 VLM / Embedding / Milvus "
                f"image_product={image_product_id}（skip_vlm_embedding=True）"
            )

        image.status = "ready"
        image.error_message = None
        await session.commit()
        logger.info(f"[sync] {image_id} Pipeline 全部完成 ✓")

    except Exception as e:
        logger.exception(f"[sync] image_product={image_product_id} Pipeline 失败: {e}")
        if image_id is not None:
            try:
                err_img = await session.get(Image, image_id)
                if err_img is not None:
                    err_img.status = "error"
                    err_img.error_message = str(e)[:2000]
                    await session.commit()
            except Exception:
                pass

    finally:
        # 清理 SCP 拉取的临时源文件（COG/缩略图临时文件由各自函数创建，系统 tmp 清理）
        if _tmp_dir is not None:
            import shutil

            try:
                shutil.rmtree(_tmp_dir, ignore_errors=True)
                logger.debug(f"清理临时目录: {_tmp_dir}")
            except Exception:
                pass
