"""影像拼接服务

支持两种拼接模式：
1. 小区级拼接：同一小区（plot）的多张影像拼接成该小区的完整影像
2. 飞行任务级拼接：同一飞行任务（mission）下多个小区的影像拼接成完整试验田影像

拼接完成后，自动生成 VLM 描述并写入 Milvus 向量库。
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config
from src.models.image import Image, ImageMosaic


async def create_mosaics_for_session(session: AsyncSession) -> int:
    """为所有有足够图片的小区和飞行任务创建拼接影像。

    Args:
        session: 数据库 Session

    Returns:
        创建的拼接影像数量
    """
    # 获取所有就绪状态的图片
    stmt = select(
        Image.id, Image.plot_id, Image.plot_code, Image.mission_id,
        Image.product_type, Image.cog_path, Image.bbox, Image.crs,
        Image.image_width, Image.image_height, Image.field_name,
        Image.variety, Image.treatment, Image.experiment_id,
    ).where(
        Image.status == 'ready',
        Image.plot_id.is_not(None),
        Image.cog_path.is_not(None),
    )
    result = await session.execute(stmt)
    images = result.fetchall()

    if not images:
        logger.info("[Mosaic] 没有可用的就绪图片")
        return 0

    created_count = 0

    # ── 1. 小区级拼接 (plot_id + product_type) ──
    plot_groups: dict[tuple, list] = defaultdict(list)
    for img in images:
        key = (img.plot_id, img.product_type)
        plot_groups[key].append(img)

    for (plot_id, product_type), imgs in plot_groups.items():
        if len(imgs) < 2:
            continue  # 单张图片不需要拼接

        # 检查是否已有拼接记录
        existing = await session.execute(
            select(ImageMosaic).where(
                ImageMosaic.plot_id == plot_id,
                ImageMosaic.product_type == product_type,
                ImageMosaic.mosaic_type == 'plot',
            )
        )
        if existing.scalar_one_or_none():
            continue

        try:
            mosaic = await create_mosaic_for_plot(
                session, plot_id, product_type, imgs
            )
            if mosaic:
                created_count += 1
                # 为拼接影像生成 VLM 描述和 Embedding
                await process_mosaic_vlm_and_embedding(session, mosaic)
        except Exception as e:
            logger.exception(f"[Mosaic] 小区级拼接失败 plot={plot_id}: {e}")

    # ── 2. 飞行任务级拼接 (mission_id + product_type) ──
    mission_groups: dict[tuple, list] = defaultdict(list)
    for img in images:
        if img.mission_id:
            key = (str(img.mission_id), img.product_type)
            mission_groups[key].append(img)

    for (mission_id, product_type), imgs in mission_groups.items():
        # 至少需要 2 个不同的小区才能拼接
        unique_plots = set(str(img.plot_id) for img in imgs if img.plot_id)
        if len(unique_plots) < 2:
            continue

        # 检查是否已有拼接记录
        existing = await session.execute(
            select(ImageMosaic).where(
                ImageMosaic.mission_id == mission_id,
                ImageMosaic.product_type == product_type,
                ImageMosaic.mosaic_type == 'mission',
            )
        )
        if existing.scalar_one_or_none():
            continue

        try:
            mosaic = await create_mosaic_for_mission(
                session, mission_id, product_type, imgs
            )
            if mosaic:
                created_count += 1
                # 为拼接影像生成 VLM 描述和 Embedding
                await process_mosaic_vlm_and_embedding(session, mosaic)
        except Exception as e:
            logger.exception(f"[Mosaic] 飞行任务级拼接失败 mission={mission_id}: {e}")

    logger.info(f"[Mosaic] 创建了 {created_count} 个拼接影像")
    return created_count


async def create_mosaic_for_plot(
    session: AsyncSession,
    plot_id: str,
    product_type: str,
    images: list,
) -> Optional[ImageMosaic]:
    """为指定小区创建拼接影像。

    Args:
        session: 数据库 Session
        plot_id: 小区ID
        product_type: 产品类型
        images: 图片记录列表

    Returns:
        创建的 ImageMosaic 记录，失败返回 None
    """
    from src.services.storage import download_file, upload_file
    import rasterio
    from rasterio.merge import merge
    import tempfile

    cfg = get_config()

    # 下载所有 COG 到临时目录
    tmp_dir = Path(tempfile.mkdtemp(prefix="mosaic_plot_"))
    local_files = []
    image_ids = []

    try:
        for img in images:
            if not img.cog_path:
                continue
            local_path = tmp_dir / f"{img.id}.tif"
            try:
                download_file(cfg.minio.buckets.cog, img.cog_path, str(local_path))
                local_files.append(local_path)
                image_ids.append(str(img.id))
            except Exception as e:
                logger.warning(f"[Mosaic] 下载失败 {img.id}: {e}")

        if len(local_files) < 2:
            logger.info(f"[Mosaic] 可用图片不足2张，跳过 plot={plot_id}")
            return None

        # 拼接影像
        logger.info(f"[Mosaic] 开始拼接 {len(local_files)} 张图片 plot={plot_id}")

        try:
            mosaic_path, bbox, crs, width, height = _merge_rasterio(local_files, tmp_dir, f"mosaic_{plot_id}_{product_type}.tif")
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"[Mosaic] 拼接失败 plot={plot_id}: {e}")
            return None

        # 上传到 MinIO
        mosaic_minio_path = f"missions/{plot_id}/{product_type}/mosaic_{plot_id}_{product_type}.tif"
        upload_file(cfg.minio.buckets.cog, mosaic_minio_path, str(mosaic_path))

        # 生成缩略图
        from src.utils.thumbnail import generate_thumbnail
        thumb_path = tmp_dir / f"mosaic_{plot_id}_{product_type}_thumb.jpg"
        generate_thumbnail(str(mosaic_path), str(thumb_path))
        thumb_minio_path = f"missions/{plot_id}/{product_type}/mosaic_{plot_id}_{product_type}_thumb.jpg"
        upload_file(cfg.minio.buckets.thumb, thumb_minio_path, str(thumb_path))

        # 创建数据库记录
        mosaic = ImageMosaic(
            plot_id=plot_id,
            plot_code=images[0].plot_code,
            mission_id=images[0].mission_id,
            product_type=product_type,
            mosaic_type='plot',
            image_ids=image_ids,
            image_count=len(image_ids),
            file_path=mosaic_minio_path,
            thumbnail_path=thumb_minio_path,
            bbox=bbox,
            crs=str(crs),
            width_pixels=width,
            height_pixels=height,
        )
        session.add(mosaic)
        await session.commit()

        logger.info(
            f"[Mosaic] 小区级拼接完成 plot={plot_id} "
            f"images={len(image_ids)} size={width}x{height}"
        )
        return mosaic

    except Exception as e:
        logger.exception(f"[Mosaic] 拼接异常 plot={plot_id}: {e}")
        return None

    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def create_mosaic_for_mission(
    session: AsyncSession,
    mission_id: str,
    product_type: str,
    images: list,
) -> Optional[ImageMosaic]:
    """为指定飞行任务创建拼接影像（多个小区拼成完整试验田）。

    Args:
        session: 数据库 Session
        mission_id: 飞行任务ID
        product_type: 产品类型
        images: 图片记录列表

    Returns:
        创建的 ImageMosaic 记录，失败返回 None
    """
    from src.services.storage import download_file, upload_file
    import tempfile

    cfg = get_config()

    # 下载所有 COG 到临时目录
    tmp_dir = Path(tempfile.mkdtemp(prefix="mosaic_mission_"))
    local_files = []
    image_ids = []

    try:
        for img in images:
            if not img.cog_path:
                continue
            local_path = tmp_dir / f"{img.id}.tif"
            try:
                download_file(cfg.minio.buckets.cog, img.cog_path, str(local_path))
                local_files.append(local_path)
                image_ids.append(str(img.id))
            except Exception as e:
                logger.warning(f"[Mosaic] 下载失败 {img.id}: {e}")

        if len(local_files) < 2:
            logger.info(f"[Mosaic] 可用图片不足2张，跳过 mission={mission_id}")
            return None

        # 拼接影像
        logger.info(f"[Mosaic] 开始飞行任务级拼接 {len(local_files)} 张图片 mission={mission_id}")

        try:
            mosaic_path, bbox, crs, width, height = _merge_rasterio(
                local_files, tmp_dir, f"mosaic_mission_{mission_id}_{product_type}.tif"
            )
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"[Mosaic] 拼接失败 mission={mission_id}: {e}")
            return None

        # 上传到 MinIO
        mosaic_minio_path = f"missions/{mission_id}/{product_type}/mosaic_mission_{mission_id}_{product_type}.tif"
        upload_file(cfg.minio.buckets.cog, mosaic_minio_path, str(mosaic_path))

        # 生成缩略图
        from src.utils.thumbnail import generate_thumbnail
        thumb_path = tmp_dir / f"mosaic_mission_{mission_id}_{product_type}_thumb.jpg"
        generate_thumbnail(str(mosaic_path), str(thumb_path))
        thumb_minio_path = f"missions/{mission_id}/{product_type}/mosaic_mission_{mission_id}_{product_type}_thumb.jpg"
        upload_file(cfg.minio.buckets.thumb, thumb_minio_path, str(thumb_path))

        # 收集所有小区信息
        plot_ids = list(set(str(img.plot_id) for img in images if img.plot_id))
        plot_codes = list(set(img.plot_code for img in images if img.plot_code))

        # 创建数据库记录
        mosaic = ImageMosaic(
            plot_id=plot_ids[0] if plot_ids else None,  # 主小区
            plot_code=','.join(plot_codes) if plot_codes else None,
            mission_id=mission_id,
            product_type=product_type,
            mosaic_type='mission',
            image_ids=image_ids,
            image_count=len(image_ids),
            file_path=mosaic_minio_path,
            thumbnail_path=thumb_minio_path,
            bbox=bbox,
            crs=str(crs),
            width_pixels=width,
            height_pixels=height,
        )
        session.add(mosaic)
        await session.commit()

        logger.info(
            f"[Mosaic] 飞行任务级拼接完成 mission={mission_id} "
            f"plots={len(plot_ids)} images={len(image_ids)} size={width}x{height}"
        )
        return mosaic

    except Exception as e:
        logger.exception(f"[Mosaic] 飞行任务级拼接异常 mission={mission_id}: {e}")
        return None

    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _merge_rasterio(
    local_files: list[Path],
    tmp_dir: Path,
    output_filename: str,
) -> tuple[Path, list, str, int, int]:
    """使用 rasterio 拼接多个影像文件。

    Args:
        local_files: 本地文件路径列表
        tmp_dir: 临时目录
        output_filename: 输出文件名

    Returns:
        (拼接后的文件路径, bbox, crs, width, height)
    """
    import rasterio
    from rasterio.merge import merge

    if len(local_files) < 2:
        raise ValueError(f"可用影像不足2张，无法拼接")

    datasets = [rasterio.open(str(f)) for f in local_files]
    try:
        # 拼接
        mosaic_array, transform = merge(datasets)
        crs = datasets[0].crs
        height, width = mosaic_array.shape[1], mosaic_array.shape[0] if len(mosaic_array.shape) == 3 else mosaic_array.shape[:2]

        # 计算边界
        bounds = datasets[0].bounds
        for ds in datasets[1:]:
            bounds = (
                min(bounds.left, ds.bounds.left),
                min(bounds.bottom, ds.bounds.bottom),
                max(bounds.right, ds.right),
                max(bounds.top, ds.top),
            )
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]

    finally:
        for ds in datasets:
            ds.close()

    # 保存拼接结果
    mosaic_path = tmp_dir / output_filename

    profile = datasets[0].profile.copy()
    profile.update(
        driver='GTiff',
        width=width,
        height=height,
        count=mosaic_array.shape[0] if len(mosaic_array.shape) == 3 else 1,
        transform=transform,
        crs=crs,
    )

    with rasterio.open(str(mosaic_path), 'w', **profile) as dst:
        if len(mosaic_array.shape) == 3:
            for i in range(mosaic_array.shape[0]):
                dst.write(mosaic_array[i], i + 1)
        else:
            dst.write(mosaic_array, 1)

    return mosaic_path, bbox, crs, width, height


async def process_mosaic_vlm_and_embedding(
    session: AsyncSession,
    mosaic: ImageMosaic,
) -> None:
    """为拼接影像生成 VLM 描述并写入 Milvus 向量库。

    Args:
        session: 数据库 Session
        mosaic: ImageMosaic 记录
    """
    from src.services.storage import download_file
    from src.services.vlm import describe_image
    from src.services.embedding import build_embedding_text, embed_text
    from src.services.milvus_client import insert_vector
    import tempfile

    cfg = get_config()
    tmp_dir = Path(tempfile.mkdtemp(prefix="mosaic_vlm_"))

    try:
        # 下载缩略图
        if not mosaic.thumbnail_path:
            logger.warning(f"[Mosaic VLM] 无缩略图，跳过 mosaic={mosaic.id}")
            return

        local_thumb = tmp_dir / f"mosaic_thumb_{mosaic.id}.jpg"
        download_file(cfg.minio.buckets.thumb, mosaic.thumbnail_path, str(local_thumb))

        # VLM 描述生成
        logger.info(f"[Mosaic VLM] 开始生成描述 mosaic={mosaic.id}")
        result = await describe_image(str(local_thumb))
        mosaic.vlm_description = result.summary
        mosaic.vlm_model = cfg.vlm.model
        mosaic.vlm_time = datetime.now()
        await session.commit()

        # Embedding + Milvus
        logger.info(f"[Mosaic VLM] 开始生成向量 mosaic={mosaic.id}")
        embed_input = build_embedding_text(
            vlm_description=result.summary,
            task_id=f"mosaic_{mosaic.mosaic_type}",
            field_name=mosaic.plot_code or "",
            survey_stage="",
            device_model="",
            data_type="可见光" if mosaic.product_type == "rgb" else mosaic.product_type or "",
            experiment_name="",
            crop_type="",
            site_name="",
            plot_code=mosaic.plot_code or "",
            variety="",
            treatment="",
            flight_date="",
            growth_stage="",
        )
        vector = await embed_text(embed_input)
        mosaic.embedding_id = await insert_vector(
            vector=vector,
            entity={
                "type": "mosaic",
                "mosaic_id": str(mosaic.id),
                "mosaic_type": mosaic.mosaic_type or "",
                "plot_id": mosaic.plot_id or "",
                "mission_id": mosaic.mission_id or "",
                "product_type": mosaic.product_type or "",
            }
        )
        await session.commit()

        logger.info(f"[Mosaic VLM] 完成 mosaic={mosaic.id}")

    except Exception as e:
        logger.exception(f"[Mosaic VLM] 失败 mosaic={mosaic.id}: {e}")

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
