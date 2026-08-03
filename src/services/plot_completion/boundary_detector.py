"""田块边界检测

VLM_SAM 策略的几何分割模块：
- 优先远程 SAM 服务（自动实例分割）
- 次选本地 SAM 模型
- 兜底 OpenCV Canny 边缘
- 全部失败回退影像 bbox

所有 _detect_* 函数支持 focus_bbox 参数：按地理范围外扩 50% 裁剪影像后送分割，
降低 SAM 计算量并提升局部精度。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds
from loguru import logger


# ── COG 加载 ──────────────────────────────────────────────


def _load_cog_with_focus(
    cog_path: str,
    focus_bbox: Optional[list] = None,
    max_size: int = 2048,
) -> Tuple[np.ndarray, object]:
    """加载 COG，可选按 focus_bbox 裁剪。

    改动原因：SAM 在大图上计算开销大且边缘精度差，按 focus_bbox 外扩 50% 裁剪后
    只送局部图给 SAM，降低计算量 + 提升局部精度。裁剪后 bounds 为窗口地理范围，
    后续 pixel_to_geo 直接用该 bounds 转换即可，无需手动加偏移。

    Args:
        cog_path: COG 在 MinIO 的路径
        focus_bbox: [min_lon, min_lat, max_lon, max_lat]，None 则读整图
        max_size: 最长边上限，超过则降采样

    Returns:
        (data, bounds)：data 形状 (3, H, W) uint8；bounds 为 rasterio Bounds
    """
    from src.services.storage import get_presigned_url
    from src.config import get_config

    cfg = get_config()
    cog_url = get_presigned_url(cfg.minio.buckets.cog, cog_path)

    with rasterio.open(cog_url) as ds:
        if focus_bbox is not None:
            try:
                # 改动原因：focus_bbox 外扩 50%，避免分割时边界效应导致小区被截断
                min_lon, min_lat, max_lon, max_lat = focus_bbox
                w_geo = max_lon - min_lon
                h_geo = max_lat - min_lat
                expanded = [
                    min_lon - w_geo * 0.5,
                    min_lat - h_geo * 0.5,
                    max_lon + w_geo * 0.5,
                    max_lat + h_geo * 0.5,
                ]
                # 裁剪到影像范围内
                expanded[0] = max(expanded[0], ds.bounds.left)
                expanded[1] = max(expanded[1], ds.bounds.bottom)
                expanded[2] = min(expanded[2], ds.bounds.right)
                expanded[3] = min(expanded[3], ds.bounds.top)

                window = from_bounds(*expanded, transform=ds.transform)
                window = window.round_offsets(op="floor").round_lengths(op="ceil")
                win_w = int(window.width)
                win_h = int(window.height)

                if win_w <= 0 or win_h <= 0:
                    raise ValueError("focus_bbox 窗口无效")

                # 降采样到 max_size 内
                scale = max(1, max(win_w, win_h) // max_size)
                if scale > 1:
                    out_shape = (3, win_h // scale, win_w // scale)
                    data = ds.read(
                        indexes=[1, 2, 3],
                        window=window,
                        out_shape=out_shape,
                        resampling=Resampling.cubic,
                    )
                else:
                    data = ds.read(indexes=[1, 2, 3], window=window)

                bounds = ds.window_bounds(window)
                # window_bounds 返回 tuple，统一包装为 BoundingBox 以支持 .left/.right/.top/.bottom 属性访问
                bounds = BoundingBox(*bounds)
                return data, bounds
            except Exception as e:
                logger.warning(f"focus_bbox 裁剪失败，回退整图: {e}")
                # 落到下方整图分支

        # 整图 + overview 降采样
        ov = min(3, len(ds.overviews(1)) - 1) if ds.overviews(1) else 0
        if ov > 0:
            data = ds.read(
                indexes=[1, 2, 3],
                out_shape=(3, ds.height // (2 ** ov), ds.width // (2 ** ov)),
                resampling=Resampling.cubic,
            )
        else:
            data = ds.read(
                indexes=[1, 2, 3],
                out_shape=(3, min(ds.height, max_size), min(ds.width, max_size)),
            )
        return data, ds.bounds


# ── mask 后处理 ──────────────────────────────────────────


def _filter_plot_masks(masks: list, target_area_px: float, tolerance: float = 0.3) -> list:
    """按示例小区面积筛选 SAM mask。

    改动原因：SAM 自动分割会返回道路、树冠、裸地等碎片，按"目标小区面积 ±30%"
    先验过滤，保留尺寸接近真实小区的 mask。

    Args:
        masks: SAM 返回的 mask 列表，每项含 area_pixels 字段
        target_area_px: 示例小区的像素面积
        tolerance: 容差比例（0.3 = ±30%）

    Returns:
        筛选后的 mask 列表
    """
    if target_area_px <= 0:
        return masks
    lower = target_area_px * (1 - tolerance)
    upper = target_area_px * (1 + tolerance)
    return [m for m in masks if lower <= m.get("area_pixels", 0) <= upper]


def _refine_mask_to_rectangle(polygon: list) -> list:
    """对 mask 多边形取最小外接矩形。

    改动原因：农田小区是矩形的先验，SAM 边缘常有锯齿，最小外接矩形可修正形状
    并对齐到矩形网格。

    Args:
        polygon: [[x, y], ...] 像素坐标点列表

    Returns:
        矩形化后的 4 角点 + 闭合点 [[x, y], ...] 共 5 点
    """
    if not polygon or len(polygon) < 3:
        return polygon
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = [[float(p[0]), float(p[1])] for p in box]
    box.append(box[0])  # 闭合
    return box


# ── SAM 远程 ─────────────────────────────────────────────


async def _detect_with_sam_remote(
    cog_path: str,
    sam_service_url: str,
    timeout: float,
    nodata,
    focus_bbox: Optional[list] = None,
    target_area_pixels: float = 0,
    area_tolerance: float = 0.7,
    sam_params: Optional[dict] = None,
) -> dict:
    """调用远程 SAM 服务进行自动实例分割，按目标面积筛选。

    改动原因：农田影像重复纹理多，SAM 默认参数（points_per_side=32,
    pred_iou_thresh=0.86）漏检严重。改为更激进的默认值，并支持通过
    sam_params 覆盖。同时增加 mask 数量日志，便于调参。

    Args:
        area_tolerance: 面积过滤宽容度，0.7 表示保留 [0.3, 1.7] 倍目标面积的 mask
                        （比旧的 0.5 更宽，因为真实小区面积会有变化）
        sam_params: SAM auto_segment 参数覆盖。None 用默认激进参数
    """
    import requests
    import base64
    import io
    from PIL import Image

    data, bounds = _load_cog_with_focus(cog_path, focus_bbox)

    img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
    h, w = img.shape[:2]

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 农田影像适配的激进默认参数（曾用 debug_sam_params.py 验证：
    # 在重复纹理农田上能找出更多小面积 mask，包含单个小区）
    default_params = {
        "points_per_side": 96,        # 默认 32 → 96，更多采样点
        "pred_iou_thresh": 0.60,      # 默认 0.86 → 0.60，放过低质量 mask
        "stability_score_thresh": 0.80,  # 默认 0.92 → 0.80
        "min_mask_region_area": 30,   # 默认 100 → 30，保留小小区
    }
    if sam_params:
        default_params.update(sam_params)

    logger.info(
        f"SAM auto_segment 调用: img={w}x{h}, params={default_params}, "
        f"target_area={target_area_pixels:.0f}px², tolerance=±{area_tolerance*100:.0f}%"
    )

    resp = requests.post(
        f"{sam_service_url}/sam/auto_segment",
        json={
            "image_base64": img_base64,
            **default_params,
        },
        timeout=timeout,
    )

    effective_bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]

    if resp.status_code != 200:
        logger.error(f"SAM auto segment error: {resp.status_code} - {resp.text}")
        return {
            "method": "bbox",
            "effective_bbox": effective_bbox,
            "boundaries": [],
            "detected_plots": [],
        }

    result = resp.json()
    masks = result.get("masks", [])
    logger.info(f"SAM 返回 {len(masks)} 个 mask（过滤前）")

    if not masks:
        return {
            "method": "bbox",
            "effective_bbox": effective_bbox,
            "boundaries": [],
            "detected_plots": [],
        }

    def pixel_to_geo(px, py):
        lon = bounds.left + (px / w) * (bounds.right - bounds.left)
        lat = bounds.top - (py / h) * (bounds.top - bounds.bottom)
        return [float(lon), float(lat)]

    # 按目标面积筛选 + 矩形化修正
    detected_plots = []
    if target_area_pixels > 0:
        filtered = _filter_plot_masks(masks, target_area_pixels, area_tolerance)
        logger.info(
            f"SAM mask 过滤: {len(masks)} → {len(filtered)} "
            f"(target={target_area_pixels:.0f}px², tolerance=±{area_tolerance*100:.0f}%)"
        )
        # 改动原因：筛选为空时不再无脑保留所有 mask（会引入大量噪声），
        # 改为按面积相近度排序保留 top-20，至少给策略层一些候选
        if not filtered:
            logger.warning(
                f"SAM 过滤后为空，按面积相近度保留 top-20 兜底（非最佳，建议检查 target_area_pixels）"
            )
            filtered = sorted(
                masks,
                key=lambda m: abs(m.get("area_pixels", 0) - target_area_pixels),
            )[:20]
        for m in filtered:
            polygon = m.get("polygon", [[]])[0]
            if polygon:
                refined = _refine_mask_to_rectangle(polygon)
                geo_pts = [pixel_to_geo(px, py) for px, py in refined]
                detected_plots.append({
                    "polygon": geo_pts,
                    "area_pixels": m.get("area_pixels", 0),
                    "confidence": m.get("confidence", 0),
                })

    # 取最大 mask 作为大边界
    best_mask = max(masks, key=lambda m: m.get("area_pixels", 0))
    boundary_polygon = best_mask.get("polygon", [[]])[0]
    boundaries = [pixel_to_geo(px, py) for px, py in boundary_polygon] if boundary_polygon else []

    return {
        "method": "sam_remote",
        "effective_bbox": effective_bbox,
        "boundaries": boundaries,
        "detected_plots": detected_plots,
    }


# ── 主入口 ───────────────────────────────────────────────


async def detect_field_boundaries(
    cog_path: Optional[str],
    image_bbox: Optional[list],
    sam_model=None,
    nodata: Optional[float] = None,
    sam_service_url: Optional[str] = None,
    sam_service_timeout: float = 60.0,
    target_area_pixels: float = 0,
    focus_bbox: Optional[list] = None,
    area_tolerance: float = 0.7,
    sam_params: Optional[dict] = None,
) -> dict:
    """检测田块边界，支持 focus_bbox 局部裁剪。

    Args:
        cog_path: COG 在 MinIO 的路径
        image_bbox: 影像整体 bbox [min_lon, min_lat, max_lon, max_lat]（fallback 用）
        sam_model: 本地 SAM 模型实例（可选）
        nodata: 影像 nodata 值
        sam_service_url: 远程 SAM 服务地址
        sam_service_timeout: SAM 服务超时
        target_area_pixels: 目标小区像素面积（用于 mask 筛选）
        focus_bbox: [min_lon, min_lat, max_lon, max_lat]，按其外扩 50% 裁剪影像送 SAM
        area_tolerance: 面积过滤宽容度，0.7 表示保留 [0.3, 1.7] 倍目标面积的 mask
        sam_params: SAM auto_segment 参数覆盖（如 points_per_side 等）
    """
    if not cog_path:
        return {"method": "bbox", "effective_bbox": image_bbox, "boundaries": []}
    try:
        if sam_service_url:
            return await _detect_with_sam_remote(
                cog_path, sam_service_url, sam_service_timeout, nodata,
                focus_bbox=focus_bbox,
                target_area_pixels=target_area_pixels,
                area_tolerance=area_tolerance,
                sam_params=sam_params,
            )
        elif sam_model is not None:
            return await _detect_with_sam(cog_path, sam_model, nodata, focus_bbox=focus_bbox)
        else:
            return await _detect_with_opencv(cog_path, image_bbox, nodata, focus_bbox=focus_bbox)
    except Exception as e:
        logger.error(f"boundary detection failed: {e}")
        return {"method": "bbox", "effective_bbox": image_bbox, "boundaries": []}


# ── 本地 SAM ─────────────────────────────────────────────


async def _detect_with_sam(cog_path, sam_model, nodata, focus_bbox: Optional[list] = None) -> dict:
    data, bounds = _load_cog_with_focus(cog_path, focus_bbox)
    img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
    sam_model.set_image(img)
    masks, scores, logits = sam_model.predict(point_coords=None, point_labels=None, multimask_output=True)
    combined = np.any(masks, axis=0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        h, w = img.shape[:2]
        pts = []
        for pt in approx:
            px, py = pt[0]
            lon = bounds.left + (px / w) * (bounds.right - bounds.left)
            lat = bounds.top - (py / h) * (bounds.top - bounds.bottom)
            pts.append([float(lon), float(lat)])
        return {
            "method": "sam",
            "effective_bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "boundaries": pts,
        }
    return {
        "method": "bbox",
        "effective_bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "boundaries": [],
    }


# ── OpenCV 兜底 ──────────────────────────────────────────


async def _detect_with_opencv(cog_path, image_bbox, nodata, focus_bbox: Optional[list] = None) -> dict:
    data, bounds = _load_cog_with_focus(cog_path, focus_bbox, max_size=1024)
    img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"method": "bbox", "effective_bbox": image_bbox, "boundaries": []}
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    h, w = img.shape[:2]
    pts = []
    for pt in approx:
        px, py = pt[0]
        lon = bounds.left + (px / w) * (bounds.right - bounds.left)
        lat = bounds.top - (py / h) * (bounds.top - bounds.bottom)
        pts.append([float(lon), float(lat)])
    return {
        "method": "opencv",
        "effective_bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "boundaries": pts,
    }
