"""缩略图生成模块

从 GeoTIFF 生成 JPEG 缩略图，用于 VLM 输入和前端列表展示。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from PIL import Image as PILImage
from rasterio.enums import Resampling

from src.config import get_config


def generate_thumbnail(
    input_path: str | Path,
    output_path: str | Path | None = None,
    max_size: int | None = None,
) -> Path:
    """从 GeoTIFF 生成 JPEG 缩略图。

    读取前 3 个波段（RGB），降采样到指定尺寸，保存为 JPEG。

    Args:
        input_path: GeoTIFF 文件路径
        output_path: 输出 JPEG 路径，默认为临时目录
        max_size: 长边最大像素数，默认取配置值

    Returns:
        缩略图文件路径
    """
    input_path = Path(input_path)
    cfg = get_config().processing

    if max_size is None:
        max_size = cfg.thumbnail_size

    if output_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="thumb_"))
        output_path = tmp_dir / f"{input_path.stem}_thumb.jpg"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"生成缩略图: {input_path.name} → max_size={max_size}")

    with rasterio.open(input_path) as ds:
        # 计算降采样比例
        width = ds.width
        height = ds.height
        scale = max_size / max(width, height)
        out_width = max(1, int(width * scale))
        out_height = max(1, int(height * scale))

        # 读取前 3 个波段（RGB），降采样
        bands_to_read = min(3, ds.count)
        data = ds.read(
            indexes=list(range(1, bands_to_read + 1)),
            out_shape=(bands_to_read, out_height, out_width),
            resampling=Resampling.average,
        )

    # 转为 HWC 格式 (height, width, channels)
    img_array = np.transpose(data, (1, 2, 0))

    # 处理 NoData 和值域
    # 将 0 值（NoData）设为黑色，其余线性拉伸到 0-255
    if img_array.dtype != np.uint8:
        # 百分位拉伸（2%-98%），避免极端值影响
        for b in range(img_array.shape[2]):
            band = img_array[:, :, b].astype(np.float32)
            valid = band[band > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    band = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    band = np.zeros_like(band)
            img_array[:, :, b] = band
        img_array = img_array.astype(np.uint8)

    # 如果只有 1 个波段，复制为 3 通道
    if img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=2)

    # 保存 JPEG
    pil_img = PILImage.fromarray(img_array, mode="RGB")
    pil_img.save(str(output_path), "JPEG", quality=85)

    file_size = output_path.stat().st_size
    logger.info(f"缩略图生成完成: {output_path.name} ({file_size / 1024:.1f} KB)")
    return output_path
