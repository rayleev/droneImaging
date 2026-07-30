"""VLM 端到端策略 - 边界识别版本"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from PIL import Image

from .base import (
    CompletionRequest,
    CompletionResult,
    PlotCell,
    PlotCompletionStrategy,
)


class VLMStrategy(PlotCompletionStrategy):
    """VLM 端到端小区划分策略"""

    def __init__(self, config=None):
        from .config import get_completion_config
        self.config = config or get_completion_config()

    @property
    def name(self) -> str:
        return "vlm"

    @property
    def description(self) -> str:
        return "VLM端到端智能划分"

    def _load_prompt(self) -> str:
        """从文件加载提示词"""
        prompt_file = Path("prompts/vlm_plot_completion.txt")
        if not prompt_file.is_absolute():
            from src.config import BASE_DIR
            prompt_file = BASE_DIR / prompt_file
        return prompt_file.read_text(encoding="utf-8").strip()

    def _build_prompt(self, request: CompletionRequest, transform) -> str:
        """构建提示词"""
        template = self._load_prompt()
        
        # 影像信息
        image_info = f"""## 影像信息
- 影像像素尺寸: 1024x1024
- 影像地理范围（WGS84）: 左下角({transform.left:.6f}, {transform.bottom:.6f}), 右上角({transform.right:.6f}, {transform.top:.6f})"""
        
        # 用户描述
        user_desc = f"""## 用户描述
{request.description}"""
        
        # 示例区域信息（转换为像素坐标）
        example_info = ""
        if request.example_region:
            coords = request.example_region.get("coordinates", [[]])[0]
            if coords and len(coords) > 0:
                ex_lons = [c[0] for c in coords]
                ex_lats = [c[1] for c in coords]
                min_lon, max_lon = min(ex_lons), max(ex_lons)
                min_lat, max_lat = min(ex_lats), max(ex_lats)
                # 转换为像素坐标
                px_left = (min_lon - transform.left) / (transform.right - transform.left) * 1024
                px_right = (max_lon - transform.left) / (transform.right - transform.left) * 1024
                px_top = (transform.top - max_lat) / (transform.top - transform.bottom) * 1024
                px_bottom = (transform.top - min_lat) / (transform.top - transform.bottom) * 1024
                example_info = f"""
## 用户绘制的示例区域
- 示例区域像素坐标: 左={px_left:.0f}, 上={px_top:.0f}, 右={px_right:.0f}, 下={px_bottom:.0f}
- 示例区域像素大小: {px_right - px_left:.0f}x{px_bottom - px_top:.0f}像素"""
        
        prompt = template.replace("{{image_info}}", image_info)
        prompt = prompt.replace("{{user_desc}}", user_desc)
        prompt = prompt.replace("{{example_info}}", example_info)
        
        return prompt

    def _pixel_to_geo(self, px, py, transform):
        """像素坐标转地理坐标"""
        lon = transform.left + (px / 1024) * (transform.right - transform.left)
        lat = transform.top - (py / 1024) * (transform.top - transform.bottom)
        return float(lon), float(lat)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """执行 VLM 端到端补全"""
        import requests

        from src.services.storage import get_presigned_url
        from src.config import get_config
        cfg = get_config()
        cog_url = get_presigned_url(cfg.minio.buckets.cog, request.cog_path)

        with rasterio.open(cog_url) as ds:
            ov = min(3, len(ds.overviews(1)) - 1) if ds.overviews(1) else 0
            if ov > 0:
                from rasterio.enums import Resampling
                data = ds.read(indexes=[1, 2, 3], out_shape=(3, ds.height // (2 ** ov), ds.width // (2 ** ov)), resampling=Resampling.cubic)
            else:
                data = ds.read(indexes=[1, 2, 3], out_shape=(3, min(ds.height, 1024), min(ds.width, 1024)))
            transform = ds.bounds

        img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{img_base64}"

        prompt = self._build_prompt(request, transform)

        from src.config import get_config
        cfg = get_config()
        vlm_cfg = cfg.vlm

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {vlm_cfg.api_key}",
        }

        payload = {
            "model": self.config.vlm_model or vlm_cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        vlm_api_url = self.config.vlm_api_url or vlm_cfg.api_url
        resp = requests.post(
            f"{vlm_api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.vlm_timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"VLM API error: {resp.status_code} - {resp.text[:500]}")

        result = resp.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"VLM response: {content[:500]}")
        return self._parse_vlm_output(request.image_id, content, transform, request.example_region)

    def _parse_vlm_output(self, image_id, content: str, transform, example_region) -> CompletionResult:
        """解析 VLM 输出，代码计算精确网格"""
        data = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if match:
                data = json.loads(match.group(1))
            else:
                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    data = json.loads(match.group(0))
        if data is None:
            raise RuntimeError(f"VLM output parse failed: {content[:500]}")

        if "error" in data:
            raise RuntimeError(f"VLM returned error: {data['error']}")

        # 从 VLM 获取田块边界（相对坐标 0-1）
        field_bbox = data.get("field_bbox", [0, 0, 1, 1])
        n_rows = data.get("n_rows", 1)
        n_cols = data.get("n_cols", 1)
        
        # 将相对坐标转换为像素坐标
        field_left = field_bbox[0] * 1024
        field_top = field_bbox[1] * 1024
        field_right = field_bbox[2] * 1024
        field_bottom = field_bbox[3] * 1024
        
        # 从示例区域获取单元格大小
        cell_width = 50  # 默认值
        cell_height = 50
        if example_region:
            coords = example_region.get("coordinates", [[]])[0]
            if coords and len(coords) > 0:
                ex_lons = [c[0] for c in coords]
                ex_lats = [c[1] for c in coords]
                min_lon, max_lon = min(ex_lons), max(ex_lons)
                min_lat, max_lat = min(ex_lats), max(ex_lats)
                px_left = (min_lon - transform.left) / (transform.right - transform.left) * 1024
                px_right = (max_lon - transform.left) / (transform.right - transform.left) * 1024
                px_top = (transform.top - max_lat) / (transform.top - transform.bottom) * 1024
                px_bottom = (transform.top - min_lat) / (transform.top - transform.bottom) * 1024
                cell_width = px_right - px_left
                cell_height = px_bottom - px_top
        
        # 计算起始位置（田块左上角）
        origin_x = field_left
        origin_y = field_top
        
        # 如果 VLM 返回了行列数，使用它；否则根据田块大小和单元格大小计算
        if n_rows <= 0:
            n_rows = max(1, int((field_bottom - field_top) / cell_height))
        if n_cols <= 0:
            n_cols = max(1, int((field_right - field_left) / cell_width))
        
        # 限制行列数，防止超出边界
        max_cells = 200
        if n_rows * n_cols > max_cells:
            scale = (max_cells / (n_rows * n_cols)) ** 0.5
            n_rows = max(1, int(n_rows * scale))
            n_cols = max(1, int(n_cols * scale))
        
        plots = []
        for row in range(n_rows):
            for col in range(n_cols):
                px_left = origin_x + col * cell_width
                px_top = origin_y + row * cell_height
                px_right = px_left + cell_width
                px_bottom = px_top + cell_height
                
                # 裁剪到田块边界
                px_left = max(px_left, field_left)
                px_top = max(px_top, field_top)
                px_right = min(px_right, field_right)
                px_bottom = min(px_bottom, field_bottom)
                
                if px_right <= px_left or px_bottom <= px_top:
                    continue
                
                # 转换为地理坐标
                min_lon, max_lat = self._pixel_to_geo(px_left, px_top, transform)
                max_lon, min_lat = self._pixel_to_geo(px_right, px_bottom, transform)
                
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

        return CompletionResult(
            image_id=image_id,
            total=len(plots),
            n_rows=n_rows,
            n_cols=n_cols,
            region={"type": "bbox", "coordinates": [0, 0, 0, 0]},
            example_size_m={"width": 0, "height": 0},
            plots=plots,
            debug_info={"strategy": "vlm", "raw_vlm_output": content, "grid_params": data},
        )
