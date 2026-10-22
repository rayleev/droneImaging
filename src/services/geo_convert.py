"""栅格文件地理参考归一化模块

将 agri_research 同步来的小区裁剪片归一化为带内嵌地理参考的 GeoTIFF，
便于复用现有 pipeline（parse_geotiff / convert_to_cog / generate_thumbnail）。

支持两种输入：
- PNG + 同名 .pgw 世界文件（rgb 产品）
- TIFF + 同名 .tfw 世界文件（ms 产品；多数 TIFF 已内嵌参考，tfw 优先级低）

世界文件仅含 6 参数仿射变换（无 CRS），统一按 WGS84 (EPSG:4326) 处理，
与 plots.geom / sites 经纬度的 SRID=4326 一致。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from affine import Affine
from loguru import logger
from rasterio.shutil import copy as rio_copy


class NoGeoReferenceError(Exception):
    """栅格文件无地理参考（无内嵌 CRS/transform，也无世界文件）"""
    pass


def _read_world_file(world_path: Path) -> Optional[Affine]:
    """解析世界文件（.pgw / .tfw），返回仿射变换。

    世界文件 6 行依次为：
        pixel_x_size
        rot_x (通常 0)
        rot_y (通常 0)
        pixel_y_size (负值，向下)
        upper_left_x
        upper_left_y
    """
    try:
        vals = [float(line.strip()) for line in world_path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as e:
        logger.warning(f"世界文件解析失败 {world_path}: {e}")
        return None
    if len(vals) < 6:
        logger.warning(f"世界文件行数不足 {world_path}: {vals}")
        return None
    px, rot_x, rot_y, py, ulx, uly = vals[:6]
    # 世界文件行序为 A, D, B, E, C, F；Affine 参数序为 (a, b, c, d, e, f)
    # 即 (A, B, C, D, E, F) = (px, rot_y, ulx, rot_x, py, uly)
    return Affine(px, rot_y, ulx, rot_x, py, uly)


def _find_world_file(raster_path: Path) -> Optional[Path]:
    """查找配套的世界文件。

    优先级：
    1. 同名 + .pgw / .tfw（按扩展名）
    2. 去掉原扩展名后 + .pgw / .tfw
    """
    candidates = []
    suffix = raster_path.suffix.lower()
    world_ext = ".tfw" if suffix in (".tif", ".tiff") else ".pgw"
    candidates.append(raster_path.with_suffix(world_ext))
    candidates.append(raster_path.with_suffix(raster_path.suffix + world_ext))
    for c in candidates:
        if c.exists():
            return c
    return None


def to_geotiff(
    src_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """将带世界文件的栅格片归一化为 GeoTIFF。

    若输入 TIFF 已内嵌 CRS/transform，直接拷贝为 GeoTIFF；
    否则读取同名 .pgw / .tfw，按 EPSG:4326 写入仿射变换。

    Args:
        src_path: 输入 PNG / TIFF 路径
        output_path: 输出 GeoTIFF 路径，默认临时目录

    Returns:
        GeoTIFF 文件路径

    Raises:
        NoGeoReferenceError: 无地理参考（无内嵌 CRS/transform，也无世界文件）
        RuntimeError: 世界文件解析失败等其他错误
    """
    src_path = Path(src_path)

    if output_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="geoconv_"))
        output_path = tmp_dir / f"{src_path.stem}_geo.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"地理参考归一化: {src_path.name} → {output_path.name}")

    with rasterio.open(src_path) as src:
        profile = src.profile
        has_crs = src.crs is not None
        has_transform = src.transform is not None and not src.transform == Affine.identity()

        if has_crs and has_transform:
            # 已是合法 GeoTIFF，直接拷贝（保证输出为 GTiff 驱动）
            profile.update(driver="GTiff")
            rio_copy(src, str(output_path), **profile)
            logger.debug(f"输入已内嵌地理参考，直接拷贝: {output_path.name}")
            return output_path

        # 缺地理参考，尝试世界文件
        world_file = _find_world_file(src_path)
        if world_file is None:
            raise NoGeoReferenceError(
                f"无法确定地理参考：{src_path.name} 既无内嵌 CRS/transform，也无同名 .pgw/.tfw"
            )

        transform = _read_world_file(world_file)
        if transform is None:
            raise RuntimeError(f"世界文件解析失败: {world_file}")

        profile.update(
            driver="GTiff",
            crs="EPSG:4326",
            transform=transform,
        )
        data = src.read()
        if src.nodata is not None:
            profile.update(nodata=src.nodata)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    logger.info(f"GeoTIFF 归一化完成: {output_path.name} (crs=EPSG:4326)")
    return output_path
