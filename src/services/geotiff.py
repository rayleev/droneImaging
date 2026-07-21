"""GeoTIFF 元数据解析模块

使用 rasterio 读取 GeoTIFF 的地理配准信息、
波段信息、坐标系等，返回结构化的元数据对象。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import rasterio
from loguru import logger
from pydantic import BaseModel


class GeoTiffMetadata(BaseModel):
    """GeoTIFF 元数据解析结果"""
    image_width: int
    image_height: int
    bands: int
    crs: Optional[str] = None
    bbox: Optional[List[float]] = None       # [min_lon, min_lat, max_lon, max_lat]
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    pixel_scale_x: Optional[float] = None    # 度/像素
    pixel_scale_y: Optional[float] = None
    nodata: Optional[float] = None
    geotransform: Optional[List[float]] = None  # 仿射变换 6 元素
    dtype: Optional[str] = None              # 像素数据类型


def parse_geotiff(file_path: str | Path) -> GeoTiffMetadata:
    """解析 GeoTIFF 文件，提取地理配准和波段元数据。

    Args:
        file_path: GeoTIFF 文件路径

    Returns:
        GeoTiffMetadata 结构化元数据

    Raises:
        rasterio.errors.RasterioIOError: 文件无法打开
        ValueError: 文件不是有效的 GeoTIFF
    """
    file_path = Path(file_path)
    logger.info(f"解析 GeoTIFF: {file_path.name}")

    with rasterio.open(file_path) as ds:
        # 基本维度
        width = ds.width
        height = ds.height
        bands = ds.count
        dtype = str(ds.dtypes[0]) if ds.dtypes else None

        # 坐标系
        crs = str(ds.crs) if ds.crs else None

        # NoData
        nodata = ds.nodata

        # 仿射变换（geotransform）
        transform = ds.transform
        geotransform = list(transform)[:6] if transform else None

        # 像素分辨率（从仿射变换提取）
        pixel_scale_x = abs(transform.a) if transform else None
        pixel_scale_y = abs(transform.e) if transform else None

        # 地理范围（bbox）
        bounds = ds.bounds  # (left, bottom, right, top)
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top] if bounds else None

        # 中心点
        center_lon = (bounds.left + bounds.right) / 2 if bounds else None
        center_lat = (bounds.bottom + bounds.top) / 2 if bounds else None

    metadata = GeoTiffMetadata(
        image_width=width,
        image_height=height,
        bands=bands,
        crs=crs,
        bbox=bbox,
        center_lon=center_lon,
        center_lat=center_lat,
        pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y,
        nodata=nodata,
        geotransform=geotransform,
        dtype=dtype,
    )

    logger.info(
        f"GeoTIFF 解析完成: {width}x{height} bands={bands} "
        f"crs={crs} bbox={bbox}"
    )
    return metadata
