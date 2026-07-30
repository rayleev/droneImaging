"""SAM 实例分割 + 模板匹配策略

流程：
1. SAM 实例分割：识别图像中所有候选区域
2. 用户画框：作为模板（面积、形状）
3. 代码模板匹配：计算相似度，筛选匹配区域
4. 返回匹配区域的地理坐标
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from loguru import logger

from .base import (
    CompletionRequest,
    CompletionResult,
    PlotCell,
    PlotCompletionStrategy,
)


def polygon_iou(poly1: list, poly2: list) -> float:
    """计算两个多边形的 IoU（近似）"""
    # 使用包围盒近似计算
    def bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)
    
    b1 = bbox(poly1)
    b2 = bbox(poly2)
    
    # 计算交集
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])
    
    if x_right < x_left or y_top < y_bottom:
        return 0.0
    
    intersection = (x_right - x_left) * (y_top - y_bottom)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def polygon_area(poly: list) -> float:
    """计算多边形面积（Shoelace 公式）"""
    n = len(poly)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2


def centroid(poly: list) -> tuple:
    """计算多边形 centroid"""
    x = sum(p[0] for p in poly) / len(poly)
    y = sum(p[1] for p in poly) / len(poly)
    return x, y


class SamTemplateStrategy(PlotCompletionStrategy):
    """SAM 实例分割 + 模板匹配策略"""

    def __init__(self, config=None):
        from .config import get_completion_config
        self.config = config or get_completion_config()

    @property
    def name(self) -> str:
        return "sam_template"

    @property
    def description(self) -> str:
        return "SAM(实例分割) + 模板匹配"

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """执行 SAM 实例分割 + 模板匹配"""
        from src.services.storage import get_presigned_url
        from src.config import get_config
        
        cfg = get_config()
        cog_url = get_presigned_url(cfg.minio.buckets.cog, request.cog_path)
        
        # 步骤 1: 读取图像
        logger.info("步骤 1: 读取图像...")
        with rasterio.open(cog_url) as ds:
            ov = min(3, len(ds.overviews(1)) - 1) if ds.overviews(1) else 0
            if ov > 0:
                from rasterio.enums import Resampling
                data = ds.read(indexes=[1, 2, 3], out_shape=(3, ds.height // (2 ** ov), ds.width // (2 ** ov)), resampling=Resampling.cubic)
            else:
                data = ds.read(indexes=[1, 2, 3], out_shape=(3, min(ds.height, 1024), min(ds.width, 1024)))
            transform = ds.bounds
            img_h, img_w = data.shape[1], data.shape[2]
        
        # 步骤 2: SAM 实例分割
        logger.info("步骤 2: SAM 实例分割...")
        img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        
        import base64, io
        from PIL import Image
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        resp = requests.post(
            f"{self.config.sam_service_url}/sam/auto_segment",
            json={
                "image_base64": img_base64,
                "points_per_side": 32,
                "pred_iou_thresh": 0.86,
                "stability_score_thresh": 0.92,
                "min_mask_region_area": 500,  # 过滤太小的区域
            },
            timeout=self.config.sam_service_timeout,
        )
        
        if resp.status_code != 200:
            logger.error(f"SAM 分割失败: {resp.status_code}")
            raise RuntimeError(f"SAM 分割失败: {resp.status_code}")
        
        result = resp.json()
        masks = result.get("masks", [])
        logger.info(f"SAM 返回 {len(masks)} 个候选实例")
        
        if not masks:
            raise RuntimeError("SAM 未识别到任何区域")
        
        # 步骤 3: 将用户画框转换为像素坐标
        logger.info("步骤 3: 处理用户画框模板...")
        user_box_pixels = None
        if request.example_region:
            coords = request.example_region.get("coordinates", [[]])[0]
            if coords and len(coords) > 0:
                ex_lons = [c[0] for c in coords]
                ex_lats = [c[1] for c in coords]
                # WGS84 转像素
                left = (min(ex_lons) - transform.left) / (transform.right - transform.left) * img_w
                right = (max(ex_lons) - transform.left) / (transform.right - transform.left) * img_w
                top = (transform.top - max(ex_lats)) / (transform.top - transform.bottom) * img_h
                bottom = (transform.top - min(ex_lats)) / (transform.top - transform.bottom) * img_h
                user_box_pixels = {
                    'left': left, 'top': top, 'right': right, 'bottom': bottom,
                    'width': right - left, 'height': bottom - top,
                    'area': (right - left) * (bottom - top),
                    'polygon': [[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]
                }
                logger.info(f"用户框: {user_box_pixels['width']:.0f}x{user_box_pixels['height']:.0f} 像素")
        
        if not user_box_pixels:
            raise RuntimeError("无法识别用户绘制的示例区域")
        
        # 步骤 4: 模板匹配 - 计算每个候选区域与用户框的相似度
        logger.info("步骤 4: 模板匹配...")
        
        candidates = []
        for i, mask in enumerate(masks):
            polygon = mask.get("polygon", [[]])[0]
            area_pixels = mask.get("area_pixels", 0)
            
            if not polygon or area_pixels < 100:
                continue
            
            # 像素坐标转 WGS84
            geo_polygon = []
            for px, py in polygon:
                lon = transform.left + (px / img_w) * (transform.right - transform.left)
                lat = transform.top - (py / img_h) * (transform.top - transform.bottom)
                geo_polygon.append([lon, lat])
            
            # 计算相似度
            area_ratio = area_pixels / user_box_pixels['area'] if user_box_pixels['area'] > 0 else 0
            area_similarity = 1 - abs(1 - area_ratio)  # 面积越接近越好
            
            # 形状相似度（IoU）
            iou = polygon_iou(geo_polygon, user_box_pixels['polygon'])
            
            # 综合得分
            score = 0.5 * area_similarity + 0.5 * iou
            
            candidates.append({
                'index': i,
                'polygon': geo_polygon,
                'area_pixels': area_pixels,
                'area_ratio': area_ratio,
                'iou': iou,
                'score': score,
            })
        
        # 按得分排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"找到 {len(candidates)} 个候选区域")
        for c in candidates[:5]:
            logger.info(f"  候选 {c['index']}: 面积比={c['area_ratio']:.2f}, IoU={c['iou']:.2f}, 得分={c['score']:.2f}")
        
        # 步骤 5: 筛选匹配区域（得分 > 阈值）
        threshold = 0.3  # 相似度阈值
        matched = [c for c in candidates if c['score'] > threshold]
        
        # 限制数量
        max_plots = 200
        if len(matched) > max_plots:
            matched = matched[:max_plots]
        
        logger.info(f"匹配到 {len(matched)} 个区域（阈值={threshold}）")
        
        # 步骤 6: 构建结果
        plots = []
        for i, m in enumerate(matched):
            poly = m['polygon']
            # 计算包围盒
            lons = [p[0] for p in poly]
            lats = [p[1] for p in poly]
            bbox = [min(lons), min(lats), max(lons), max(lats)]
            
            # 构建 GeoJSON polygon（闭合）
            if poly[0] != poly[-1]:
                poly = poly + [poly[0]]
            
            plots.append(PlotCell(
                id=f"plot-{i}",
                label=f"P{i+1}",
                row=0,
                col=i,
                bbox=bbox,
                polygon=[poly],
                area_m2=0.0,
            ))
        
        return CompletionResult(
            image_id=request.image_id,
            total=len(plots),
            n_rows=1,
            n_cols=len(plots),
            region={"type": "bbox", "coordinates": request.image_bbox or [0, 0, 0, 0]},
            example_size_m={"width": 0, "height": 0},
            plots=plots,
            debug_info={
                "strategy": "sam_template",
                "sam_instances": len(masks),
                "candidates": len(candidates),
                "matched": len(matched),
                "threshold": threshold,
            },
        )
