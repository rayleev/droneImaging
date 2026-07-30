"""田块边界检测"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from loguru import logger



async def _detect_with_sam_remote(cog_path: str, sam_service_url: str, timeout: float, nodata, target_area_pixels: float = 0, area_tolerance: float = 0.5):
    """调用远程 SAM 服务进行自动实例分割，按目标面积筛选"""
    import requests
    import base64
    import io
    from PIL import Image
    
    from src.services.storage import get_presigned_url
    from src.config import get_config
    cfg = get_config()
    cog_url = get_presigned_url(cfg.minio.buckets.cog, cog_path)
    
    # 读取 COG 图像
    with rasterio.open(cog_url) as ds:
        ov = min(3, len(ds.overviews(1)) - 1) if ds.overviews(1) else 0
        if ov > 0:
            data = ds.read(indexes=[1,2,3], out_shape=(3, ds.height//(2**ov), ds.width//(2**ov)), resampling=Resampling.cubic)
        else:
            data = ds.read(indexes=[1,2,3], out_shape=(3, min(ds.height,2048), min(ds.width,2048)))
        transform = ds.bounds
    
    img = np.transpose(data, (1,2,0)).astype(np.uint8)
    h, w = img.shape[:2]
    
    # 编码为 base64
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # 调用 SAM 自动分割接口
    resp = requests.post(
        f"{sam_service_url}/sam/auto_segment",
        json={
            "image_base64": img_base64,
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "min_mask_region_area": 100,
        },
        timeout=timeout,
    )
    
    if resp.status_code != 200:
        logger.error(f"SAM auto segment error: {resp.status_code} - {resp.text}")
        return {"method": "bbox", "effective_bbox": [transform.left, transform.bottom, transform.right, transform.top], "boundaries": [], "detected_plots": []}
    
    result = resp.json()
    masks = result.get("masks", [])
    
    if not masks:
        return {"method": "bbox", "effective_bbox": [transform.left, transform.bottom, transform.right, transform.top], "boundaries": [], "detected_plots": []}
    
    # 像素坐标转地理坐标函数
    def pixel_to_geo(px, py):
        lon = transform.left + (px / w) * (transform.right - transform.left)
        lat = transform.top - (py / h) * (transform.top - transform.bottom)
        return [float(lon), float(lat)]
    
    # 按目标面积筛选（如果提供了目标面积）
    detected_plots = []
    if target_area_pixels > 0:
        lower = target_area_pixels * (1 - area_tolerance)
        upper = target_area_pixels * (1 + area_tolerance)
        for m in masks:
            area = m.get("area_pixels", 0)
            if lower <= area <= upper:
                polygon = m.get("polygon", [[]])[0]
                if polygon:
                    geo_pts = [pixel_to_geo(px, py) for px, py in polygon]
                    detected_plots.append({
                        "polygon": geo_pts,
                        "area_pixels": area,
                        "confidence": m.get("confidence", 0),
                    })
    
    # 如果没有筛选到，返回所有 mask 中面积最接近的
    if not detected_plots and masks:
        # 按面积排序，取中位数面积的 mask
        sorted_masks = sorted(masks, key=lambda m: m.get("area_pixels", 0))
        mid_idx = len(sorted_masks) // 2
        for m in sorted_masks[max(0, mid_idx-2):mid_idx+3]:
            polygon = m.get("polygon", [[]])[0]
            if polygon:
                geo_pts = [pixel_to_geo(px, py) for px, py in polygon]
                detected_plots.append({
                    "polygon": geo_pts,
                    "area_pixels": m.get("area_pixels", 0),
                    "confidence": m.get("confidence", 0),
                })
    
    # 取最大的 mask 作为边界
    best_mask = max(masks, key=lambda m: m.get("area_pixels", 0))
    boundary_polygon = best_mask.get("polygon", [[]])[0]
    boundaries = [pixel_to_geo(px, py) for px, py in boundary_polygon] if boundary_polygon else []
    
    return {
        "method": "sam_remote",
        "effective_bbox": [transform.left, transform.bottom, transform.right, transform.top],
        "boundaries": boundaries,
        "detected_plots": detected_plots,
    }

async def detect_field_boundaries(
    cog_path: Optional[str],
    image_bbox: Optional[list],
    sam_model=None,
    nodata: Optional[float] = None,
    sam_service_url: Optional[str] = None,
    sam_service_timeout: float = 60.0,
    target_area_pixels: float = 0,
) -> dict:
    if not cog_path:
        return {"method": "bbox", "effective_bbox": image_bbox, "boundaries": []}
    try:
        if sam_service_url:
            return await _detect_with_sam_remote(cog_path, sam_service_url, sam_service_timeout, nodata, target_area_pixels)
        elif sam_model is not None:
            return await _detect_with_sam(cog_path, sam_model, nodata)
        else:
            return await _detect_with_opencv(cog_path, image_bbox, nodata)
    except Exception as e:
        logger.error(f"boundary detection failed: {e}")
        return {"method": "bbox", "effective_bbox": image_bbox, "boundaries": []}


async def _detect_with_sam(cog_path, sam_model, nodata):
    from src.services.storage import get_presigned_url
    from src.config import get_config
    cfg = get_config()
    cog_url = get_presigned_url(cfg.minio.buckets.cog, cog_path)
    with rasterio.open(cog_url) as ds:
        ov = min(3, len(ds.overviews(1)) - 1) if ds.overviews(1) else 0
        if ov > 0:
            data = ds.read(indexes=[1,2,3], out_shape=(3, ds.height//(2**ov), ds.width//(2**ov)), resampling=Resampling.cubic)
        else:
            data = ds.read(indexes=[1,2,3], out_shape=(3, min(ds.height,2048), min(ds.width,2048)))
        transform = ds.bounds
    img = np.transpose(data, (1,2,0)).astype(np.uint8)
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
            lon = transform.left + (px/w)*(transform.right-transform.left)
            lat = transform.top - (py/h)*(transform.top-transform.bottom)
            pts.append([float(lon), float(lat)])
        return {"method": "sam", "effective_bbox": [transform.left, transform.bottom, transform.right, transform.top], "boundaries": pts}
    return {"method": "bbox", "effective_bbox": [transform.left, transform.bottom, transform.right, transform.top], "boundaries": []}


async def _detect_with_opencv(cog_path, image_bbox, nodata):
    from src.services.storage import get_presigned_url
    from src.config import get_config
    cfg = get_config()
    cog_url = get_presigned_url(cfg.minio.buckets.cog, cog_path)
    with rasterio.open(cog_url) as ds:
        ov = min(2, len(ds.overviews(1)) - 1) if ds.overviews(1) else 0
        if ov > 0:
            data = ds.read(indexes=[1,2,3], out_shape=(3, ds.height//(2**ov), ds.width//(2**ov)), resampling=Resampling.cubic)
        else:
            data = ds.read(indexes=[1,2,3], out_shape=(3, min(ds.height,1024), min(ds.width,1024)))
        bounds = ds.bounds
    img = np.transpose(data, (1,2,0)).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
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
        lon = bounds.left + (px/w)*(bounds.right-bounds.left)
        lat = bounds.top - (py/h)*(bounds.top-bounds.bottom)
        pts.append([float(lon), float(lat)])
    return {"method": "opencv", "effective_bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top], "boundaries": pts}
