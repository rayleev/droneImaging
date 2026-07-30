"""SAM + VLM 混合策略：SAM 识别边界 + VLM 理解用户意图 + 代码计算网格"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import rasterio
import requests
from loguru import logger
from PIL import Image

from .base import (
    CompletionRequest,
    CompletionResult,
    PlotCell,
    PlotCompletionStrategy,
)


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """射线法判断点是否在多边形内"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class SamVLMStrategy(PlotCompletionStrategy):
    """SAM + VLM 混合策略"""

    def __init__(self, config=None):
        from .config import get_completion_config
        self.config = config or get_completion_config()

    @property
    def name(self) -> str:
        return "sam_vlm"

    @property
    def description(self) -> str:
        return "SAM(边界) + VLM(理解) + 代码(计算)"

    def _load_prompt(self) -> str:
        """从文件加载提示词"""
        prompt_file = Path("prompts/vlm_understand_drawing.txt")
        if not prompt_file.is_absolute():
            from src.config import BASE_DIR
            prompt_file = BASE_DIR / prompt_file
        return prompt_file.read_text(encoding="utf-8").strip()

    def _build_prompt(self, request: CompletionRequest, transform, user_box_pixels: dict) -> str:
        """构建提示词"""
        template = self._load_prompt()
        image_info = f"影像像素尺寸: 1024x1024\n影像地理范围: 左下角({transform.left:.6f}, {transform.bottom:.6f}), 右上角({transform.right:.6f}, {transform.top:.6f})"
        user_desc = request.description
        example_info = f"""用户绘制的示例区域像素坐标:
- 左: {user_box_pixels['left']:.0f}
- 上: {user_box_pixels['top']:.0f}
- 右: {user_box_pixels['right']:.0f}
- 下: {user_box_pixels['bottom']:.0f}
- 宽度: {user_box_pixels['width']:.0f}像素
- 高度: {user_box_pixels['height']:.0f}像素"""
        prompt = template.replace("{{image_info}}", image_info)
        prompt = prompt.replace("{{user_desc}}", user_desc)
        prompt = prompt.replace("{{example_info}}", example_info)
        return prompt

    def _wgs84_to_pixel(self, lon: float, lat: float, transform) -> tuple:
        """WGS84 转像素坐标"""
        px = (lon - transform.left) / (transform.right - transform.left) * 1024
        py = (transform.top - lat) / (transform.top - transform.bottom) * 1024
        return px, py

    def _pixel_to_wgs84(self, px: float, py: float, transform) -> tuple:
        """像素坐标转 WGS84"""
        lon = transform.left + (px / 1024) * (transform.right - transform.left)
        lat = transform.top - (py / 1024) * (transform.top - transform.bottom)
        return lon, lat

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """执行 SAM+VLM 混合补全"""
        from src.services.storage import get_presigned_url
        from src.config import get_config
        from .boundary_detector import detect_field_boundaries
        
        cfg = get_config()
        cog_url = get_presigned_url(cfg.minio.buckets.cog, request.cog_path)
        
        # 步骤 1: SAM 识别农田边界
        logger.info("步骤 1: SAM 识别农田边界...")
        boundary_info = await detect_field_boundaries(
            request.cog_path, request.image_bbox,
            sam_model=None, nodata=request.nodata,
            sam_service_url=self.config.sam_service_url,
            sam_service_timeout=self.config.sam_service_timeout,
        )
        
        with rasterio.open(cog_url) as ds:
            ov = min(3, len(ds.overviews(1)) - 1) if ds.overviews(1) else 0
            if ov > 0:
                from rasterio.enums import Resampling
                data = ds.read(indexes=[1, 2, 3], out_shape=(3, ds.height // (2 ** ov), ds.width // (2 ** ov)), resampling=Resampling.cubic)
            else:
                data = ds.read(indexes=[1, 2, 3], out_shape=(3, min(ds.height, 1024), min(ds.width, 1024)))
            transform = ds.bounds
        
        # SAM 返回的边界是地理坐标，需要转换为像素坐标
        sam_boundary_geo = boundary_info.get("boundaries", [])
        if sam_boundary_geo:
            sam_boundary = []
            for lon, lat in sam_boundary_geo:
                px = (lon - transform.left) / (transform.right - transform.left) * 1024
                py = (transform.top - lat) / (transform.top - transform.bottom) * 1024
                sam_boundary.append([px, py])
            logger.info(f"SAM 识别边界: {len(sam_boundary)} 个点（已转换为像素坐标）")
        else:
            logger.warning("SAM 未识别到边界，使用影像全图范围")
            sam_boundary = [[0, 0], [1024, 0], [1024, 1024], [0, 1024], [0, 0]]
        
        # 步骤 2: 将用户画框转为像素坐标
        logger.info("步骤 2: 转换用户画框为像素坐标...")
        user_box_pixels = None
        if request.example_region:
            coords = request.example_region.get("coordinates", [[]])[0]
            if coords and len(coords) > 0:
                ex_lons = [c[0] for c in coords]
                ex_lats = [c[1] for c in coords]
                left_px, top_px = self._wgs84_to_pixel(min(ex_lons), max(ex_lats), transform)
                right_px, bottom_px = self._wgs84_to_pixel(max(ex_lons), min(ex_lats), transform)
                user_box_pixels = {
                    'left': left_px,
                    'top': top_px,
                    'right': right_px,
                    'bottom': bottom_px,
                    'width': right_px - left_px,
                    'height': bottom_px - top_px,
                }
                logger.info(f"用户画框像素坐标: {user_box_pixels}")
        
        if not user_box_pixels or user_box_pixels['width'] <= 0 or user_box_pixels['height'] <= 0:
            raise RuntimeError("无法识别用户绘制的示例区域")
        
        # 步骤 3: 可选 VLM 理解用户意图
        logger.info("步骤 3: VLM 理解用户意图...")
        vlm_result = None
        try:
            img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{img_base64}"
            
            prompt = self._build_prompt(request, transform, user_box_pixels)
            
            vlm_cfg = cfg.vlm
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {vlm_cfg.api_key}",
            }
            
            payload = {
                "model": vlm_cfg.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            }
            
            resp = requests.post(
                f"{vlm_cfg.api_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.vlm_timeout,
            )
            
            if resp.status_code == 200:
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"VLM 响应: {content[:200]}")
                vlm_result = content
            else:
                logger.warning(f"VLM 调用失败: {resp.status_code}")
        except Exception as e:
            logger.warning(f"VLM 调用异常: {e}")
        
        # 步骤 4: 解析 VLM 意图，计算网格
        logger.info("步骤 4: 解析 VLM 意图并计算网格...")
        
        # 解析 VLM 结果
        intent = "repeat"  # 默认策略
        if vlm_result:
            try:
                vlm_json = json.loads(vlm_result)
                intent = vlm_json.get("intent", "repeat")
                logger.info(f"VLM 意图: {intent}, 说明: {vlm_json.get('notes', '')}")
            except json.JSONDecodeError:
                logger.warning(f"VLM 结果解析失败，使用默认策略: repeat")
        
        # 计算 SAM 边界的包围盒
        boundary_xs = [p[0] for p in sam_boundary]
        boundary_ys = [p[1] for p in sam_boundary]
        boundary_bounds = [min(boundary_xs), min(boundary_ys), max(boundary_xs), max(boundary_ys)]
        
        # 计算 SAM 边界面积（像素）
        boundary_area_pixels = 0
        n = len(sam_boundary)
        for i in range(n):
            j = (i + 1) % n
            boundary_area_pixels += sam_boundary[i][0] * sam_boundary[j][1]
            boundary_area_pixels -= sam_boundary[j][0] * sam_boundary[i][1]
        boundary_area_pixels = abs(boundary_area_pixels) / 2
        
        user_box_area = user_box_pixels['width'] * user_box_pixels['height']
        
        if intent == "fill":
            # fill 策略：根据 SAM 边界面积自动计算小区大小
            logger.info("使用 fill 策略：自动调整小区大小铺满区域")
            # 估算合适的小区数量（基于边界面积和用户画框面积的比例）
            target_cells = max(1, int(boundary_area_pixels / user_box_area))
            # 限制小区数量
            target_cells = min(target_cells, 200)
            # 计算行列数（近似正方形分割）
            aspect_ratio = (boundary_bounds[2] - boundary_bounds[0]) / (boundary_bounds[3] - boundary_bounds[1])
            max_cols = max(1, int((target_cells * aspect_ratio) ** 0.5))
            max_rows = max(1, int(target_cells / max_cols))
            # 根据行列数计算单元格大小
            cell_width = (boundary_bounds[2] - boundary_bounds[0]) / max_cols
            cell_height = (boundary_bounds[3] - boundary_bounds[1]) / max_rows
            # 原点为 SAM 边界包围盒的左上角
            origin_x = boundary_bounds[0]
            origin_y = boundary_bounds[1]
        else:
            # repeat 策略：使用用户画框大小
            logger.info("使用 repeat 策略：按照用户画框大小重复排列")
            cell_width = user_box_pixels['width']
            cell_height = user_box_pixels['height']
            origin_x = user_box_pixels['left']
            origin_y = user_box_pixels['top']
            
            max_cols = max(1, int((boundary_bounds[2] - origin_x) / cell_width) + 1)
            max_rows = max(1, int((boundary_bounds[3] - origin_y) / cell_height) + 1)
            
            max_cells = 200
            if max_rows * max_cols > max_cells:
                scale = (max_cells / (max_rows * max_cols)) ** 0.5
                max_rows = max(1, int(max_rows * scale))
                max_cols = max(1, int(max_cols * scale))
        
        logger.info(f"网格参数: {max_rows}行 x {max_cols}列, 单元格 {cell_width:.0f}x{cell_height:.0f}像素, 意图: {intent}")
        
        # 步骤 5: 生成小区并裁剪到 SAM 边界内
        logger.info("步骤 5: 生成小区并裁剪...")
        
        plots = []
        for row in range(max_rows):
            for col in range(max_cols):
                px_left = origin_x + col * cell_width
                px_top = origin_y + row * cell_height
                px_right = px_left + cell_width
                px_bottom = px_top + cell_height
                
                # 计算中心点
                center_x = (px_left + px_right) / 2
                center_y = (px_top + px_bottom) / 2
                
                # 检查中心点是否在 SAM 边界内
                if not point_in_polygon(center_x, center_y, sam_boundary):
                    continue
                
                # 裁剪到 SAM 边界包围盒
                px_left_clipped = max(px_left, boundary_bounds[0])
                px_top_clipped = max(px_top, boundary_bounds[1])
                px_right_clipped = min(px_right, boundary_bounds[2])
                px_bottom_clipped = min(px_bottom, boundary_bounds[3])
                
                # 检查裁剪后尺寸
                if (px_right_clipped - px_left_clipped) < cell_width * 0.3 or (px_bottom_clipped - px_top_clipped) < cell_height * 0.3:
                    continue
                
                min_lon, max_lat = self._pixel_to_wgs84(px_left_clipped, px_top_clipped, transform)
                max_lon, min_lat = self._pixel_to_wgs84(px_right_clipped, px_bottom_clipped, transform)
                
                bbox = [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)]
                polygon = [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]
                
                label = f"{chr(65 + row)}{col + 1}" if row < 26 else f"P{row}-{col}"
                plots.append(PlotCell(
                    id=f"plot-{row}-{col}",
                    label=label,
                    row=row,
                    col=col,
                    bbox=bbox,
                    polygon=polygon,
                    area_m2=0.0,
                ))
        
        logger.info(f"生成 {len(plots)} 个小区")
        
        return CompletionResult(
            image_id=request.image_id,
            total=len(plots),
            n_rows=max_rows,
            n_cols=max_cols,
            region={"type": "bbox", "coordinates": request.image_bbox or [0, 0, 0, 0]},
            example_size_m={"width": 0, "height": 0},
            plots=plots,
            debug_info={
                "strategy": "sam_vlm",
                "sam_boundary_points": len(sam_boundary),
                "user_box_pixels": user_box_pixels,
                "vlm_result": vlm_result,
                "cell_size": [cell_width, cell_height],
            },
        )
