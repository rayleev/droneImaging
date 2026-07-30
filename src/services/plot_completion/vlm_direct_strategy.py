"""VLM 关键点识别策略"""

from __future__ import annotations

import json
import re
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


class VLMDirectStrategy(PlotCompletionStrategy):
    """VLM 关键点识别策略"""

    def __init__(self, config=None):
        from .config import get_completion_config
        self.config = config or get_completion_config()

    @property
    def name(self) -> str:
        return "vlm_direct"

    @property
    def description(self) -> str:
        return "VLM(关键点识别) + 代码(计算)"

    def _load_prompt(self) -> str:
        prompt_file = Path("prompts/vlm_keypoints.txt")
        if not prompt_file.is_absolute():
            from src.config import BASE_DIR
            prompt_file = BASE_DIR / prompt_file
        return prompt_file.read_text(encoding="utf-8").strip()

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        from src.services.storage import get_presigned_url
        from src.config import get_config
        
        cfg = get_config()
        cog_url = get_presigned_url(cfg.minio.buckets.cog, request.cog_path)
        
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
        
        logger.info("步骤 2: 处理用户画框模板...")
        user_box = None
        if request.example_region:
            coords = request.example_region.get("coordinates", [[]])[0]
            if coords and len(coords) > 0:
                ex_lons = [c[0] for c in coords]
                ex_lats = [c[1] for c in coords]
                left = (min(ex_lons) - transform.left) / (transform.right - transform.left) * img_w
                right = (max(ex_lons) - transform.left) / (transform.right - transform.left) * img_w
                top = (transform.top - max(ex_lats)) / (transform.top - transform.bottom) * img_h
                bottom = (transform.top - min(ex_lats)) / (transform.top - transform.bottom) * img_h
                user_box = {
                    'left': left, 'top': top, 'right': right, 'bottom': bottom,
                    'width': right - left, 'height': bottom - top,
                }
                logger.info(f"用户框: {user_box['width']:.0f}x{user_box['height']:.0f} 像素")
        
        if not user_box:
            raise RuntimeError("无法识别用户绘制的示例区域")
        
        logger.info("步骤 3: 批次调用 VLM 识别关键点...")
        all_keypoints = await self._batch_detect_keypoints(data, img_w, img_h, user_box)
        logger.info(f"共识别到 {len(all_keypoints)} 个关键点")
        
        logger.info("步骤 4: 基于关键点计算小区...")
        plots = self._calculate_plots_from_keypoints(all_keypoints, user_box, img_w, img_h, transform)
        logger.info(f"计算得到 {len(plots)} 个小区")
        
        return CompletionResult(
            image_id=request.image_id,
            total=len(plots),
            n_rows=1,
            n_cols=len(plots),
            region={"type": "bbox", "coordinates": request.image_bbox or [0, 0, 0, 0]},
            example_size_m={"width": 0, "height": 0},
            plots=plots,
            debug_info={
                "strategy": "vlm_direct",
                "keypoints_count": len(all_keypoints),
                "user_box": user_box,
            },
        )
    
    async def _batch_detect_keypoints(self, data, img_w, img_h, user_box):
        all_keypoints = []
        rows, cols = 2, 2
        region_w = img_w // cols
        region_h = img_h // rows
        
        for row in range(rows):
            for col in range(cols):
                x1 = col * region_w
                y1 = row * region_h
                x2 = (col + 1) * region_w
                y2 = (row + 1) * region_h
                
                logger.info(f"  识别区域 [{row},{col}]: ({x1},{y1})-({x2},{y2})")
                region_data = data[:, y1:y2, x1:x2]
                
                keypoints = await self._detect_keypoints_in_region(
                    region_data, x1, y1, img_w, img_h, user_box
                )
                all_keypoints.extend(keypoints)
        
        unique_keypoints = []
        seen = set()
        for kp in all_keypoints:
            key = (round(kp['x'], 1), round(kp['y'], 1))
            if key not in seen:
                seen.add(key)
                unique_keypoints.append(kp)
        
        return unique_keypoints
    
    async def _detect_keypoints_in_region(self, region_data, offset_x, offset_y, img_w, img_h, user_box):
        import base64, io
        from PIL import Image as PILImage
        
        img = np.transpose(region_data, (1, 2, 0)).astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        prompt = self._load_prompt()
        prompt = prompt.replace("{{image_info}}", f"区域像素尺寸: {region_data.shape[2]}x{region_data.shape[1]}")
        prompt = prompt.replace("{{user_width}}", f"{user_box['width']:.0f}")
        prompt = prompt.replace("{{user_height}}", f"{user_box['height']:.0f}")
        prompt = prompt.replace("{{offset_x}}", str(offset_x))
        prompt = prompt.replace("{{offset_y}}", str(offset_y))
        
        from src.config import get_config
        cfg = get_config()
        vlm_cfg = cfg.vlm
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {vlm_cfg.api_key}",
        }
        
        payload = {
            "model": vlm_cfg.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        
        try:
            resp = requests.post(f"{vlm_cfg.api_url}/chat/completions", json=payload, headers=headers, timeout=60)
            if resp.status_code != 200:
                return []
            
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
            
            json_str = json_str.strip()
            if not json_str.startswith('{'):
                start = json_str.find('{')
                if start >= 0:
                    json_str = json_str[start:]
            if not json_str.endswith('}'):
                end = json_str.rfind('}')
                if end >= 0:
                    json_str = json_str[:end+1]
            
            data = json.loads(json_str)
            keypoints = []
            for c in data.get("field_corners", []):
                keypoints.append({'x': c.get('x', 0) + offset_x, 'y': c.get('y', 0) + offset_y, 'type': 'field_corner'})
            for s in data.get("separators", []):
                for p in s.get("points", []):
                    keypoints.append({'x': p.get('x', 0) + offset_x, 'y': p.get('y', 0) + offset_y, 'type': 'separator'})
            return keypoints
        except Exception as e:
            logger.error(f"VLM 解析失败: {e}")
            return []
    
    def _calculate_plots_from_keypoints(self, keypoints, user_box, img_w, img_h, transform):
        plots = []
        if not keypoints:
            return self._default_grid_plots(user_box, img_w, img_h, transform)
        
        corners = [kp for kp in keypoints if kp['type'] == 'field_corner']
        separators = [kp for kp in keypoints if kp['type'] == 'separator']
        
        if len(corners) >= 4:
            xs = [c['x'] for c in corners]
            ys = [c['y'] for c in corners]
            field_left, field_top = min(xs), min(ys)
            field_right, field_bottom = max(xs), max(ys)
            field_width = field_right - field_left
            field_height = field_bottom - field_top
            
            cols = max(1, round(field_width / user_box['width']))
            rows = max(1, round(field_height / user_box['height']))
            
            max_plots = 200
            if rows * cols > max_plots:
                scale = (max_plots / (rows * cols)) ** 0.5
                rows = max(1, int(rows * scale))
                cols = max(1, int(cols * scale))
            
            cell_w = field_width / cols
            cell_h = field_height / rows
            
            for row in range(rows):
                for col in range(cols):
                    px_left = field_left + col * cell_w
                    px_top = field_top + row * cell_h
                    px_right = px_left + cell_w
                    px_bottom = px_top + cell_h
                    
                    lon1 = transform.left + (px_left / img_w) * (transform.right - transform.left)
                    lat1 = transform.top - (px_top / img_h) * (transform.top - transform.bottom)
                    lon2 = transform.left + (px_right / img_w) * (transform.right - transform.left)
                    lat2 = transform.top - (px_bottom / img_h) * (transform.top - transform.bottom)
                    
                    bbox = [round(lon1, 6), round(lat2, 6), round(lon2, 6), round(lat1, 6)]
                    polygon = [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]
                    plots.append(PlotCell(id=f"plot-{row}-{col}", label=f"P{row*cols+col+1}", row=row, col=col, bbox=bbox, polygon=polygon, area_m2=0.0))
        
        elif separators:
            xs = [s['x'] for s in separators]
            ys = [s['y'] for s in separators]
            unique_xs = sorted(list(set([round(x, -1) for x in xs])))
            unique_ys = sorted(list(set([round(y, -1) for y in ys])))
            
            if len(unique_xs) >= 2 and len(unique_ys) >= 2:
                for i, x in enumerate(unique_xs[:-1]):
                    for j, y in enumerate(unique_ys[:-1]):
                        px_left, px_top = x, y
                        px_right, px_bottom = unique_xs[i+1], unique_ys[j+1]
                        lon1 = transform.left + (px_left / img_w) * (transform.right - transform.left)
                        lat1 = transform.top - (px_top / img_h) * (transform.top - transform.bottom)
                        lon2 = transform.left + (px_right / img_w) * (transform.right - transform.left)
                        lat2 = transform.top - (px_bottom / img_h) * (transform.top - transform.bottom)
                        bbox = [round(lon1, 6), round(lat2, 6), round(lon2, 6), round(lat1, 6)]
                        polygon = [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]
                        plots.append(PlotCell(id=f"plot-{j}-{i}", label=f"P{j*(len(unique_xs)-1)+i+1}", row=j, col=i, bbox=bbox, polygon=polygon, area_m2=0.0))
            else:
                return self._default_grid_plots(user_box, img_w, img_h, transform)
        else:
            return self._default_grid_plots(user_box, img_w, img_h, transform)
        
        return plots
    
    def _default_grid_plots(self, user_box, img_w, img_h, transform):
        plots = []
        cols = max(1, int(img_w / user_box['width']))
        rows = max(1, int(img_h / user_box['height']))
        
        max_plots = 200
        if rows * cols > max_plots:
            scale = (max_plots / (rows * cols)) ** 0.5
            rows = max(1, int(rows * scale))
            cols = max(1, int(cols * scale))
        
        for row in range(rows):
            for col in range(cols):
                px_left = col * user_box['width']
                px_top = row * user_box['height']
                px_right = px_left + user_box['width']
                px_bottom = px_top + user_box['height']
                
                lon1 = transform.left + (px_left / img_w) * (transform.right - transform.left)
                lat1 = transform.top - (px_top / img_h) * (transform.top - transform.bottom)
                lon2 = transform.left + (px_right / img_w) * (transform.right - transform.left)
                lat2 = transform.top - (px_bottom / img_h) * (transform.top - transform.bottom)
                
                bbox = [round(lon1, 6), round(lat2, 6), round(lon2, 6), round(lat1, 6)]
                polygon = [[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]
                plots.append(PlotCell(id=f"plot-{row}-{col}", label=f"P{row*cols+col+1}", row=row, col=col, bbox=bbox, polygon=polygon, area_m2=0.0))
        
        return plots
