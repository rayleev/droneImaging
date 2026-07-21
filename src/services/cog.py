"""COG（Cloud Optimized GeoTIFF）转换模块

将原始 GeoTIFF 转换为 COG 格式，内置多级 overview 金字塔，
支持 HTTP Range 按需读取，供瓦片服务使用。

使用 rasterio 内置的 COG 驱动（无需外部 gdal_translate CLI）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import rasterio
from rasterio.shutil import copy as rio_copy
from loguru import logger

from src.config import get_config


def convert_to_cog(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """将 GeoTIFF 转换为 COG 格式。

    使用 rasterio 内置 COG 驱动，自动生成 overview 金字塔。

    Args:
        input_path: 原始 GeoTIFF 路径
        output_path: 输出 COG 路径，默认为临时目录

    Returns:
        COG 文件路径

    Raises:
        RuntimeError: COG 转换失败
    """
    input_path = Path(input_path)
    cfg = get_config().processing

    if output_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="cog_"))
        output_path = tmp_dir / f"{input_path.stem}_cog.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"COG 转换开始: {input_path.name} → {output_path.name}")

    try:
        with rasterio.open(input_path) as src:
            rio_copy(
                src,
                str(output_path),
                driver="COG",
                blocksize=cfg.cog_blocksize,
                overviews="AUTO",
                overview_count=len(cfg.cog_overview_levels),
                compress="DEFLATE",
                predictor="YES",
            )
    except Exception as e:
        logger.error(f"COG 转换失败: {e}")
        raise RuntimeError(f"COG 转换失败: {e}") from e

    output_size = output_path.stat().st_size
    logger.info(f"COG 转换完成: {output_path.name} ({output_size / 1024 / 1024:.1f} MB)")
    return output_path
