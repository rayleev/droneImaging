"""瓦片服务路由

从 MinIO 中的 COG 文件按需生成 XYZ 瓦片（256×256 PNG），
利用 COG 的内部 tiling 和 overview 金字塔实现高效读取。

瓦片 URL 格式: /api/tiles/{image_id}/{z}/{x}/{y}.png
"""

from __future__ import annotations

import io
import math
import uuid

import numpy as np
import rasterio
from fastapi import APIRouter, Depends, HTTPException, Response
from loguru import logger
from rasterio.enums import Resampling
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_config
from src.database import get_session
from src.models.image import Image
from src.services.storage import get_presigned_url

router = APIRouter()

TILE_SIZE = 256


def _tile_bounds(z: int, x: int, y: int) -> tuple:
    """计算 XYZ 瓦片的地理范围（Web Mercator → WGS84）。

    返回 (min_lon, min_lat, max_lon, max_lat)。
    """
    n = 2 ** z
    # 瓦片边界（Web Mercator 归一化坐标）
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (lon_min, lat_min, lon_max, lat_max)


@router.get("/{image_id}/{z}/{x}/{y}.png")
async def get_tile(
    image_id: uuid.UUID,
    z: int,
    x: int,
    y: int,
    session: AsyncSession = Depends(get_session),
):
    """获取指定影像的 XYZ 瓦片（256×256 PNG）。

    从 MinIO 读取 COG 文件，按瓦片坐标裁剪并缩放，
    利用 COG overview 金字塔避免全分辨率读取。
    """
    # 1. 查询影像记录，获取 COG 路径
    image = await session.get(Image, image_id)
    if image is None:
        logger.warning(f"Tile request: image not found {image_id}")
        raise HTTPException(status_code=404, detail="影像不存在")
    if not image.cog_path:
        logger.warning(f"Tile request: COG path empty {image_id}")
        raise HTTPException(status_code=404, detail="COG 尚未生成")
    if image.status != "ready":
        logger.warning(f"Tile request: image not ready {image_id} status={image.status}")
        raise HTTPException(status_code=409, detail=f"影像处理中: {image.status}")

    # 2. 获取 MinIO 预签名 URL
    cfg = get_config()
    try:
        cog_url = get_presigned_url(cfg.minio.buckets.cog, image.cog_path)
        logger.debug(f"Tile presigned URL generated for {image_id}")
    except Exception as e:
        logger.error(f"生成预签名 URL 失败: {e}")
        raise HTTPException(status_code=500, detail="存储访问失败")

    # 3. 计算瓦片地理范围
    tile_bbox = _tile_bounds(z, x, y)

    # 4. 用 rasterio 读取对应区域
    try:
        png_bytes = _render_tile(cog_url, tile_bbox, image.nodata)
    except Exception as e:
        logger.error(f"瓦片渲染失败 [{image_id}] z={z} x={x} y={y}: {e}")
        # 返回透明瓦片而非报错（前端体验更好）
        png_bytes = _transparent_tile()

    return Response(content=png_bytes, media_type="image/png")


def _render_tile(cog_url: str, tile_bbox: tuple, nodata: float | None) -> bytes:
    """从 COG 渲染单个瓦片。

    关键点：瓦片的地理范围在低缩放级别下往往比影像本身还大。如果直接把整个
    瓦片范围读出来拉伸到 256×256，rasterio 会把越界窗口裁剪到影像边界、再把
    整幅影像拉伸填满瓦片，导致"缩小地图时图像大小不变"的错误现象。

    正确做法：
    1. 计算瓦片范围与影像范围的交集；
    2. 只读取交集部分（窗口落在影像内，读取高效，能利用 COG overview）；
    3. 把交集渲染到 256×256 瓦片画布中正确的像素位置（其余区域透明）。
    这样影像才会随地图缩放正确变大/变小。
    """
    with rasterio.open(cog_url) as ds:
        from rasterio.windows import from_bounds

        # 影像范围（与瓦片同为 WGS84）
        rleft, rbottom, rright, rtop = ds.bounds
        tmin_lon, tmin_lat, tmax_lon, tmax_lat = tile_bbox

        # 1. 瓦片与影像的地理交集
        imin_lon = max(tmin_lon, rleft)
        imax_lon = min(tmax_lon, rright)
        imin_lat = max(tmin_lat, rbottom)
        imax_lat = min(tmax_lat, rtop)
        if imin_lon >= imax_lon or imin_lat >= imax_lat:
            return _transparent_tile()  # 无交集 → 透明瓦片

        # 2. 交集对应的影像像素窗口（完全落在影像内）
        window = from_bounds(imin_lon, imin_lat, imax_lon, imax_lat, transform=ds.transform)

        # 3. 交集在 256×256 瓦片画布中占据的像素区域（y 轴向下，顶部为 max_lat）
        lon_span = tmax_lon - tmin_lon
        lat_span = tmax_lat - tmin_lat
        x0 = (imin_lon - tmin_lon) / lon_span * TILE_SIZE
        x1 = (imax_lon - tmin_lon) / lon_span * TILE_SIZE
        y0 = (tmax_lat - imax_lat) / lat_span * TILE_SIZE
        y1 = (tmax_lat - imin_lat) / lat_span * TILE_SIZE

        x0i, y0i = max(0, int(round(x0))), max(0, int(round(y0)))
        x1i, y1i = min(TILE_SIZE, int(round(x1))), min(TILE_SIZE, int(round(y1)))
        sub_w, sub_h = x1i - x0i, y1i - y0i
        if sub_w <= 0 or sub_h <= 0:
            return _transparent_tile()  # 该级别下影像过小，无法渲染

        # 读取前 3 波段（RGB），缩放到交集子区域大小
        bands_to_read = min(3, ds.count)
        data = ds.read(
            indexes=list(range(1, bands_to_read + 1)),
            window=window,
            out_shape=(bands_to_read, sub_h, sub_w),
            resampling=Resampling.bilinear,
        )

    # 转为 HWC
    img = np.transpose(data, (1, 2, 0))

    # 处理 NoData → alpha 通道
    if nodata is not None:
        # 所有波段都是 nodata 的像素设为透明
        nodata_mask = np.all(img == nodata, axis=2)
    else:
        nodata_mask = np.all(img == 0, axis=2)

    # 值域拉伸（百分位）
    img_float = img.astype(np.float32)
    for b in range(img_float.shape[2]):
        band = img_float[:, :, b]
        valid = band[~nodata_mask]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                img_float[:, :, b] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)

    img_uint8 = img_float.astype(np.uint8)

    # 构建 RGBA（添加 alpha 通道）
    alpha = np.where(nodata_mask, 0, 255).astype(np.uint8)
    rgba_sub = np.dstack([img_uint8, alpha])

    # 4. 贴到 256×256 透明画布的正确位置
    from PIL import Image as PILImage
    canvas = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    canvas[y0i:y0i + sub_h, x0i:x0i + sub_w] = rgba_sub
    pil_img = PILImage.fromarray(canvas, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _transparent_tile() -> bytes:
    """生成全透明 256×256 PNG 瓦片"""
    from PIL import Image as PILImage
    img = PILImage.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
