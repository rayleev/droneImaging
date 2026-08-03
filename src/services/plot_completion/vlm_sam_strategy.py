"""VLM_SAM 智能补全策略

执行顺序（关键）：先 SAM 后 VLM，最后 SAM 批量分割
1. SAM 精确分割用户画框区域 → 提取单个小区的精确地理尺寸
   （修正旧策略"用降采样图直接给 VLM 找角点"的精度问题）
2. VLM 一次性识别整块田的大边界 field_boundary + 异常区域 anomalies + 旋转角度
   （单次调用，降低 VLM 开销；VLM 只识别大尺度结构，不识别每个小区）
3. SAM 批量分割：在 VLM 大边界（或影像 bbox 兜底）内调用 SAM auto_segment，
   按 Step 1 测得的 target_area_pixels 筛选，直接用 SAM mask 作为小区边界
   （不再用代码均分网格，避免"平铺"效果）
4. 若 SAM 批量分割失败或返回空，回退到 Step 1 尺寸 + 代码网格复制
   与 anomalies 求交标记 status=skip
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import numpy as np
from loguru import logger
from PIL import Image

from src.services.plot_divider import (
    _polygon_area_m2,
    _rotate_point,
    _wgs84_to_meters_factor,
)
from .base import (
    CompletionRequest,
    CompletionResult,
    PlotCell,
    PlotCompletionStrategy,
)
from .boundary_detector import (
    _load_cog_with_focus,
    detect_field_boundaries,
)


class VLMSamStrategy(PlotCompletionStrategy):
    """VLM_SAM 策略：SAM 精确几何 + VLM 语义理解"""

    def __init__(self, config=None):
        from .config import get_completion_config
        self.config = config or get_completion_config()

    @property
    def name(self) -> str:
        return "vlm_sam"

    @property
    def description(self) -> str:
        return "SAM(精确分割示例小区) + VLM(识别大边界+异常) + 网格复制"

    def _load_prompt(self) -> str:
        prompt_file = Path("prompts/vlm_sam_field_boundary.txt")
        if not prompt_file.is_absolute():
            from src.config import BASE_DIR
            prompt_file = BASE_DIR / prompt_file
        return prompt_file.read_text(encoding="utf-8").strip()

    # ── 几何工具 ──────────────────────────────────────────

    @staticmethod
    def _extract_bbox(geojson: Optional[dict]) -> Optional[list]:
        """从 GeoJSON Polygon 或 bbox 提取 [min_lon, min_lat, max_lon, max_lat]"""
        if geojson is None:
            return None
        if isinstance(geojson, list) and len(geojson) == 4:
            return geojson
        if geojson.get("type") == "Polygon":
            coords = geojson["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            return [min(lons), min(lats), max(lons), max(lats)]
        if "bbox" in geojson and len(geojson["bbox"]) == 4:
            return geojson["bbox"]
        return None

    @staticmethod
    def _polygon_to_bbox(polygon: list) -> list:
        """多边形点列表转 bbox [min_lon, min_lat, max_lon, max_lat]"""
        if not polygon:
            return [0, 0, 0, 0]
        lons = [p[0] for p in polygon]
        lats = [p[1] for p in polygon]
        return [min(lons), min(lats), max(lons), max(lats)]

    @staticmethod
    def _point_in_polygon(point: list, polygon: list) -> bool:
        """射线法判断点是否在多边形内（用于 anomaly 命中判断）"""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _validate_field_boundary(
        field_boundary: list,
        image_bbox: Optional[list],
    ) -> Tuple[list, str]:
        """校验 VLM 返回的 field_boundary 是否为有效的地理多边形。

        改动原因：VLM（尤其 GLM-4V）经常返回无效坐标——
        - 像素坐标（值在 [0, image_w/h] 整数范围）
        - 归一化坐标（值在 [0, 1] 范围）
        - 超出影像 bbox 的坐标（VLM 编造）
        - 4 个点全相同或共线（VLM 偷懒）
        - 面积过小（不像整块田的外边界）
        - 面积过大（接近或超过影像 bbox，等于没识别）

        Args:
            field_boundary: [[lon, lat], ...] 点列表
            image_bbox: [min_lon, min_lat, max_lon, max_lat]

        Returns:
            (valid_boundary, reason)：valid_boundary 为 [] 表示无效；
            reason 是说明（用于日志）
        """
        if not field_boundary or len(field_boundary) < 4:
            return [], "点数 < 4，无法构成多边形"

        # 去掉闭合点后看唯一性
        pts = field_boundary
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            return [], "唯一角点 < 3"

        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        lon_range = max(lons) - min(lons)
        lat_range = max(lats) - min(lats)

        # 1) 检测归一化坐标 [0, 1]
        if max(lons) <= 1.0 and min(lons) >= 0.0 and max(lats) <= 1.0 and min(lats) >= 0.0:
            return [], f"坐标疑似归一化值（lon∈[{min(lons):.4f},{max(lons):.4f}], lat∈[{min(lats):.4f},{max(lats):.4f}]），应为 WGS84 经纬度"

        # 2) 检测像素坐标（整数 + 值范围像像素）
        all_int = all(float(p[0]).is_integer() and float(p[1]).is_integer() for p in pts)
        if all_int and lon_range > 50:
            # 经度跨度 > 50 不可能是经纬度，疑似像素坐标
            return [], f"坐标疑似像素整数（lon_range={lon_range:.0f}），应为 WGS84 经纬度"

        # 没有 image_bbox 时只能做基本校验
        if not image_bbox or len(image_bbox) != 4:
            return field_boundary, "无 image_bbox 对照，跳过深度校验"

        img_min_lon, img_min_lat, img_max_lon, img_max_lat = image_bbox
        img_w = img_max_lon - img_min_lon
        img_h = img_max_lat - img_min_lat
        if img_w <= 0 or img_h <= 0:
            return field_boundary, "image_bbox 无效"

        # 3) 检测越界：所有点必须落在 image_bbox 外扩 5% 范围内
        #    （VLM 可能略微外推，但不能跑太远）
        ext = 0.05
        bound_min_lon = img_min_lon - img_w * ext
        bound_max_lon = img_max_lon + img_w * ext
        bound_min_lat = img_min_lat - img_h * ext
        bound_max_lat = img_max_lat + img_h * ext

        out_of_bounds = []
        for i, (lon, lat) in enumerate(pts):
            if not (bound_min_lon <= lon <= bound_max_lon and bound_min_lat <= lat <= bound_max_lat):
                out_of_bounds.append((i, lon, lat))
        if out_of_bounds:
            sample = out_of_bounds[0]
            return [], f"点 {sample[0]} 坐标 ({sample[1]:.6f}, {sample[2]:.6f}) 越出影像 bbox 外扩 5% 范围，共 {len(out_of_bounds)} 个点越界"

        # 4) 面积校验：field_boundary 应该是"整块田的外边界"，
        #    合理范围是 image_bbox 面积的 20%~100%
        fb_area = lon_range * lat_range
        img_area = img_w * img_h
        ratio = fb_area / img_area if img_area > 0 else 0
        if ratio < 0.2:
            return [], f"多边形面积仅为影像的 {ratio*100:.1f}%（<20%），不像整块田外边界"
        if ratio > 0.99:
            return [], f"多边形面积 {ratio*100:.1f}% 接近或等于影像 bbox（VLM 偷懒），丢弃"

        # 5) 形状校验：长宽比不能太离谱（避免 VLM 返回一条线）
        aspect = max(lon_range / max(lat_range, 1e-12), lat_range / max(lon_range, 1e-12))
        if aspect > 20:
            return [], f"多边形长宽比 {aspect:.1f} 过大（像一条线），不像田块"

        return field_boundary, f"有效（面积占影像 {ratio*100:.1f}%）"

    # ── Step 1: SAM 精确分割示例小区 ─────────────────────

    def _estimate_target_area_pixels(
        self,
        cog_path: str,
        example_bbox: list,
        focus_bbox: Optional[list] = None,
    ) -> float:
        """估算示例小区在 focus 图上的像素面积（用于 SAM mask 筛选）。

        改动原因：focus_bbox 决定了 _load_cog_with_focus 裁剪出的影像范围，
        影像可能被降采样到 max_size。同一个 example_bbox 在不同 focus 图中
        对应的像素面积完全不同。Step 1 用 example_bbox 作为 focus 测面积，
        Step 3 必须用 field_bbox 作为 focus 重新算，否则过滤阈值全错。

        Args:
            cog_path: COG 路径
            example_bbox: 示例小区的地理 bbox
            focus_bbox: 实际送给 SAM 的 focus 范围。None 时默认 = example_bbox
                        （保持旧逻辑，向后兼容）

        Returns:
            示例小区在 focus 图上的像素面积
        """
        try:
            actual_focus = focus_bbox if focus_bbox is not None else example_bbox
            data, bounds = _load_cog_with_focus(cog_path, focus_bbox=actual_focus)
            h, w = data.shape[1], data.shape[2]
            focus_w_geo = bounds.right - bounds.left
            focus_h_geo = bounds.top - bounds.bottom
            if focus_w_geo <= 0 or focus_h_geo <= 0:
                return 0
            ex_min_lon, ex_min_lat, ex_max_lon, ex_max_lat = example_bbox
            ex_w_geo = ex_max_lon - ex_min_lon
            ex_h_geo = ex_max_lat - ex_min_lat
            ex_w_px = ex_w_geo / focus_w_geo * w
            ex_h_px = ex_h_geo / focus_h_geo * h
            return ex_w_px * ex_h_px
        except Exception as e:
            logger.warning(f"估算 target_area_pixels 失败: {e}")
            return 0

    def _extract_plot_size_from_sam(
        self, sam_result: dict, example_bbox: list
    ) -> Tuple[float, float, dict]:
        """从 SAM 结果提取单个小区的地理尺寸

        Returns:
            (plot_w_deg, plot_h_deg, example_size_m)
        """
        detected = sam_result.get("detected_plots", [])

        # 优先用 SAM 分割出的 mask 算尺寸
        if detected:
            mask_polygon = detected[0].get("polygon", [])
            if mask_polygon and len(mask_polygon) >= 3:
                bbox = self._polygon_to_bbox(mask_polygon)
                plot_w_deg = bbox[2] - bbox[0]
                plot_h_deg = bbox[3] - bbox[1]
                center_lat = (bbox[1] + bbox[3]) / 2
                m_per_deg_lon, m_per_deg_lat = _wgs84_to_meters_factor(center_lat)
                example_size_m = {
                    "width": round(plot_w_deg * m_per_deg_lon, 2),
                    "height": round(plot_h_deg * m_per_deg_lat, 2),
                }
                return plot_w_deg, plot_h_deg, example_size_m

        # 回退：用 example_bbox 作为小区尺寸
        ex_min_lon, ex_min_lat, ex_max_lon, ex_max_lat = example_bbox
        plot_w_deg = ex_max_lon - ex_min_lon
        plot_h_deg = ex_max_lat - ex_min_lat
        center_lat = (ex_min_lat + ex_max_lat) / 2
        m_per_deg_lon, m_per_deg_lat = _wgs84_to_meters_factor(center_lat)
        example_size_m = {
            "width": round(plot_w_deg * m_per_deg_lon, 2),
            "height": round(plot_h_deg * m_per_deg_lat, 2),
        }
        return plot_w_deg, plot_h_deg, example_size_m

    # ── Step 2: VLM 识别大尺度结构 ───────────────────────

    async def _vlm_detect_field_boundary(self, request: CompletionRequest) -> dict:
        """VLM 一次性调用识别大边界 field_boundary + 异常 anomalies + 旋转 rotation_deg"""
        try:
            # 降采样整图（focus_bbox=None 读整图 + overview 降采样）
            data, bounds = _load_cog_with_focus(request.cog_path, focus_bbox=None, max_size=1024)
            img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
            h, w = img.shape[:2]

            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            prompt = self._load_prompt()
            image_info = (
                f"影像尺寸: {w}x{h} 像素, "
                f"地理范围: [{bounds.left}, {bounds.bottom}, {bounds.right}, {bounds.top}]"
            )
            prompt = prompt.replace("{{image_info}}", image_info)
            prompt = prompt.replace("{{description}}", request.description or "按示例小区大小铺满整块田")

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
                "max_tokens": 1024,  # 智谱 GLM-4V 限制 [1,1024]
            }

            async with httpx.AsyncClient(timeout=vlm_cfg.timeout) as client:
                resp = await client.post(
                    f"{vlm_cfg.api_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code >= 400:
                    logger.error(f"VLM API {resp.status_code} 响应体: {resp.text[:500]}")
                    resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]

            json_data = self._extract_json(content)
            if json_data is None:
                logger.warning(f"VLM 返回非 JSON: {content[:200]}")
                return {
                    "field_boundary": [],
                    "anomalies": [],
                    "rotation_deg": 0,
                    "summary": content[:200],
                }

            field_boundary = json_data.get("field_boundary", [])
            logger.info(f"VLM field_boundary 原始值: {json.dumps(field_boundary, ensure_ascii=False)}")
            logger.info(f"VLM anomalies 原始值: {json.dumps(json_data.get('anomalies', []), ensure_ascii=False)}")
            logger.info(f"VLM content 前 300 字: {content[:300]}")

            # 强化校验：检测 VLM 偷懒/错误坐标（像素坐标、归一化、越界、面积异常）
            if field_boundary:
                field_boundary, reason = self._validate_field_boundary(
                    field_boundary, request.image_bbox
                )
                if not field_boundary:
                    logger.warning(f"VLM field_boundary 无效：{reason}")
                else:
                    logger.info(f"VLM field_boundary 校验通过：{reason}")

            return {
                "field_boundary": field_boundary,
                "anomalies": json_data.get("anomalies", []),
                "rotation_deg": float(json_data.get("rotation_deg", 0) or 0),
                "summary": json_data.get("summary", ""),
            }
        except Exception as e:
            logger.warning(f"VLM 识别大边界失败: {e}")
            return {
                "field_boundary": [],
                "anomalies": [],
                "rotation_deg": 0,
                "summary": f"VLM 失败: {e}",
            }

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """从 VLM 返回文本提取 JSON（支持纯 JSON / markdown 代码块 / 首尾花括号）"""
        if not text:
            return None
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

    # ── Step 3: 网格复制 + 异常标记 ───────────────────────

    def _generate_plots_in_field(
        self,
        field_bbox: list,
        plot_w_deg: float,
        plot_h_deg: float,
        rotation_deg: float,
        anomalies: list,
    ) -> Tuple[List[PlotCell], int, int]:
        """在 field_bbox 内用 plot_w/h 铺网格，与 anomalies 求交标记 skip

        Returns:
            (plots, n_rows, n_cols)
        """
        if plot_w_deg <= 0 or plot_h_deg <= 0:
            return [], 0, 0

        min_lon, min_lat, max_lon, max_lat = field_bbox
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2

        field_w_deg = max_lon - min_lon
        field_h_deg = max_lat - min_lat

        n_cols = max(1, round(field_w_deg / plot_w_deg))
        n_rows = max(1, round(field_h_deg / plot_h_deg))

        # 限制最大小区数，避免大图 + 小区尺寸导致数量爆炸
        max_plots = 200
        if n_rows * n_cols > max_plots:
            scale = (max_plots / (n_rows * n_cols)) ** 0.5
            n_rows = max(1, int(n_rows * scale))
            n_cols = max(1, int(n_cols * scale))

        cell_w_deg = field_w_deg / n_cols
        cell_h_deg = field_h_deg / n_rows

        plots = []
        for row in range(n_rows):
            for col in range(n_cols):
                # 单元格 bbox（未旋转），row=0 在最北
                cell_min_lon = min_lon + col * cell_w_deg
                cell_max_lon = cell_min_lon + cell_w_deg
                cell_max_lat = max_lat - row * cell_h_deg
                cell_min_lat = cell_max_lat - cell_h_deg

                # 4 角点（顺时针闭合）
                corners = [
                    [cell_min_lon, cell_min_lat],
                    [cell_max_lon, cell_min_lat],
                    [cell_max_lon, cell_max_lat],
                    [cell_min_lon, cell_max_lat],
                    [cell_min_lon, cell_min_lat],
                ]

                # 按 rotation_deg 旋转（复用 plot_divider._rotate_point）
                if rotation_deg != 0:
                    corners = [
                        list(_rotate_point(c[0], c[1], center_lon, center_lat, rotation_deg))
                        for c in corners
                    ]

                # 旋转后 bbox
                lons = [c[0] for c in corners]
                lats = [c[1] for c in corners]
                cell_bbox = [min(lons), min(lats), max(lons), max(lats)]

                # 面积（复用 plot_divider._polygon_area_m2）
                area = _polygon_area_m2(corners, center_lat)

                # 判断中心点是否落入任一 anomaly 多边形
                cell_center = [
                    (cell_bbox[0] + cell_bbox[2]) / 2,
                    (cell_bbox[1] + cell_bbox[3]) / 2,
                ]
                status = "ok"
                for anomaly in anomalies:
                    if self._point_in_polygon(cell_center, anomaly):
                        status = "skip"
                        break

                label = f"P{row * n_cols + col + 1}"
                plots.append(PlotCell(
                    id=f"plot-{row}-{col}",
                    label=label,
                    row=row,
                    col=col,
                    bbox=cell_bbox,
                    polygon=[corners],
                    area_m2=round(area, 2),
                    status=status,
                ))

        return plots, n_rows, n_cols

    def _plots_from_sam_masks(
        self,
        sam_plots: list,
        field_bbox: list,
        plot_w_deg: float,
        plot_h_deg: float,
        anomalies: list,
    ) -> Tuple[List[PlotCell], int, int]:
        """把 SAM 批量分割的 mask 转成 PlotCell 列表

        改动原因：SAM 在重复纹理农田上常把多个小区合并成一条长 mask
        （如 1 列 3 个小区被识别成 1 个细长 mask）。直接用面积过滤会
        把这些长 mask 全砍掉，只剩噪声。改为按 target 尺寸拆分长 mask：
        - 计算 n_w = round(mask_w / target_w), n_h = round(mask_h / target_h)
        - 把 mask 的 bbox 切成 n_w × n_h 个子矩形，每个作为一个小区
        - 太小（双维度均 < 0.5x target）的 mask 跳过（噪声）
        - 太大（单维度 > 20x target）的 mask 跳过（整块田/背景）

        Returns:
            (plots, n_rows, n_cols)
        """
        if not sam_plots or plot_w_deg <= 0 or plot_h_deg <= 0:
            return [], 0, 0

        min_lon, min_lat, max_lon, max_lat = field_bbox
        center_lat = (min_lat + max_lat) / 2

        valid_plots: List[PlotCell] = []
        plot_counter = 0
        split_stats = {"single": 0, "split": 0, "skipped_small": 0, "skipped_large": 0}

        for idx, sp in enumerate(sam_plots):
            polygon = sp.get("polygon", [])
            if not polygon or len(polygon) < 3:
                continue

            # 去掉闭合点
            if polygon[0] == polygon[-1]:
                polygon = polygon[:-1]
            if len(polygon) < 3:
                continue

            # 用 bbox 做拆分（axis-aligned），忽略 mask 的旋转角度
            lons = [p[0] for p in polygon]
            lats = [p[1] for p in polygon]
            mask_min_lon, mask_max_lon = min(lons), max(lons)
            mask_min_lat, mask_max_lat = min(lats), max(lats)
            mask_w = mask_max_lon - mask_min_lon
            mask_h = mask_max_lat - mask_min_lat

            if mask_w <= 0 or mask_h <= 0:
                continue

            # 计算横向/纵向能拆几个小区
            n_w = max(1, round(mask_w / plot_w_deg))
            n_h = max(1, round(mask_h / plot_h_deg))

            # 跳过太大（>20x 单维度，可能是整块田/背景）
            if n_w > 20 or n_h > 20:
                split_stats["skipped_large"] += 1
                logger.debug(
                    f"mask {idx} 跳过(太大): {n_w}x{n_h}, "
                    f"mask={mask_w*111320:.1f}m x {mask_h*110540:.1f}m"
                )
                continue

            # 跳过太小（双维度均 < 0.5x target，噪声碎片）
            if mask_w < plot_w_deg * 0.5 and mask_h < plot_h_deg * 0.5:
                split_stats["skipped_small"] += 1
                logger.debug(
                    f"mask {idx} 跳过(太小): mask={mask_w*111320:.1f}m x {mask_h*110540:.1f}m, "
                    f"target={plot_w_deg*111320:.1f}m x {plot_h_deg*110540:.1f}m"
                )
                continue

            if n_w == 1 and n_h == 1:
                split_stats["single"] += 1
            else:
                split_stats["split"] += 1
                logger.info(
                    f"mask {idx} 拆分: {n_w}x{n_h} = {n_w*n_h} 个子小区 "
                    f"(mask={mask_w*111320:.1f}m x {mask_h*110540:.1f}m)"
                )

            # 把 mask bbox 切成 n_w × n_h 个子矩形
            sub_w = mask_w / n_w
            sub_h = mask_h / n_h
            for row in range(n_h):
                for col in range(n_w):
                    sub_min_lon = mask_min_lon + col * sub_w
                    sub_max_lon = sub_min_lon + sub_w
                    # row=0 在最北（lat 最大）
                    sub_max_lat = mask_max_lat - row * sub_h
                    sub_min_lat = sub_max_lat - sub_h

                    corners = [
                        [sub_min_lon, sub_min_lat],
                        [sub_max_lon, sub_min_lat],
                        [sub_max_lon, sub_max_lat],
                        [sub_min_lon, sub_max_lat],
                        [sub_min_lon, sub_min_lat],
                    ]
                    cell_bbox = [sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat]
                    area_m2 = _polygon_area_m2(corners, center_lat)

                    # 判断中心点是否落入任一 anomaly 多边形
                    cell_center = [
                        (sub_min_lon + sub_max_lon) / 2,
                        (sub_min_lat + sub_max_lat) / 2,
                    ]
                    status = "ok"
                    for anomaly in anomalies:
                        if self._point_in_polygon(cell_center, anomaly):
                            status = "skip"
                            break

                    plot_counter += 1
                    valid_plots.append(PlotCell(
                        id=f"plot-sam-{idx}-{row}-{col}",
                        label=f"P{plot_counter}",
                        row=row,
                        col=col,
                        bbox=cell_bbox,
                        polygon=[corners],
                        area_m2=round(area_m2, 2),
                        status=status,
                    ))

        logger.info(
            f"SAM mask 拆分统计: 单小区={split_stats['single']}, "
            f"拆分={split_stats['split']}, "
            f"跳过(太小)={split_stats['skipped_small']}, "
            f"跳过(太大)={split_stats['skipped_large']}, "
            f"共产出 {len(valid_plots)} 个小区（去重前）"
        )

        # 去重：SAM 的横条/竖条 mask 互相重叠，拆分后会有大量重叠子小区。
        # 按"面积最接近 target 的优先"排序，依次保留；与已保留小区重叠 >50% 的丢弃。
        if len(valid_plots) > 1:
            valid_plots = self._deduplicate_overlapping_plots(
                valid_plots, plot_w_deg * plot_h_deg, center_lat
            )

        # 估算 n_rows/n_cols（仅用于响应字段）
        field_w_deg = max_lon - min_lon
        field_h_deg = max_lat - min_lat
        n_cols = max(1, round(field_w_deg / plot_w_deg)) if plot_w_deg > 0 else 1
        n_rows = max(1, round(field_h_deg / plot_h_deg)) if plot_h_deg > 0 else 1

        return valid_plots, n_rows, n_cols

    @staticmethod
    def _deduplicate_overlapping_plots(
        plots: List[PlotCell],
        target_area_deg2: float,
        center_lat: float,
    ) -> List[PlotCell]:
        """去重：移除重叠 >50% 的小区，优先保留面积最接近 target 的。

        改动原因：SAM auto_segment 在重复纹理农田上常同时返回横条 mask
        （一行小区）和竖条 mask（一列小区），它们覆盖同一区域。拆分后
        产生大量重叠子小区，必须去重否则小区数量翻倍。

        策略：
        1. 按面积与 target 的比值排序（越接近 1.0 越优先）
        2. 依次加入结果列表，加入前检查与已有小区的重叠率
        3. 重叠率 >50%（基于较小小区面积）则跳过
        """
        if not plots:
            return plots

        # 计算每个 plot 的面积（度²），用于排序
        def plot_area_deg2(p: PlotCell) -> float:
            w = p.bbox[2] - p.bbox[0]
            h = p.bbox[3] - p.bbox[1]
            return w * h

        # 按面积接近 target 程度排序（比值越接近 1.0 越优先）
        def area_closeness(p: PlotCell) -> float:
            a = plot_area_deg2(p)
            if target_area_deg2 <= 0:
                return 0
            return abs(a / target_area_deg2 - 1.0)

        sorted_plots = sorted(plots, key=area_closeness)

        kept: List[PlotCell] = []
        for p in sorted_plots:
            overlap = False
            for k in kept:
                ratio = VLMSamStrategy._bbox_overlap_ratio(p.bbox, k.bbox)
                if ratio > 0.5:
                    overlap = True
                    break
            if not overlap:
                kept.append(p)

        logger.info(
            f"去重: {len(plots)} → {len(kept)} 个小区 "
            f"(移除 {len(plots) - len(kept)} 个重叠)"
        )
        return kept

    @staticmethod
    def _bbox_overlap_ratio(bbox_a: list, bbox_b: list) -> float:
        """计算两个 bbox 的重叠率（基于较小 bbox 的面积）。

        Returns:
            0.0~1.0，重叠面积 / 较小 bbox 面积
        """
        a_min_lon, a_min_lat, a_max_lon, a_max_lat = bbox_a
        b_min_lon, b_min_lat, b_max_lon, b_max_lat = bbox_b

        # 重叠区域
        ov_min_lon = max(a_min_lon, b_min_lon)
        ov_max_lon = min(a_max_lon, b_max_lon)
        ov_min_lat = max(a_min_lat, b_min_lat)
        ov_max_lat = min(a_max_lat, b_max_lat)

        if ov_min_lon >= ov_max_lon or ov_min_lat >= ov_max_lat:
            return 0.0  # 无重叠

        overlap_area = (ov_max_lon - ov_min_lon) * (ov_max_lat - ov_min_lat)
        area_a = (a_max_lon - a_min_lon) * (a_max_lat - a_min_lat)
        area_b = (b_max_lon - b_min_lon) * (b_max_lat - b_min_lat)
        smaller = min(area_a, area_b)
        if smaller <= 0:
            return 0.0
        return overlap_area / smaller

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        debug_info = {
            "strategy": "vlm_sam",
            "steps": [],
        }

        # 提取示例小区 bbox（SAM focus 用）
        example_bbox = self._extract_bbox(request.example_region)
        if not example_bbox:
            example_bbox = request.image_bbox or [0, 0, 0, 0]
            debug_info["steps"].append({
                "step": "extract_example",
                "warning": "无示例区域，用影像 bbox",
            })

        # ── Step 1: SAM 精确分割示例小区 ──
        logger.info("Step 1: SAM 精确分割示例小区...")
        target_area_px = self._estimate_target_area_pixels(request.cog_path, example_bbox)

        sam_result = await detect_field_boundaries(
            cog_path=request.cog_path,
            image_bbox=request.image_bbox,
            sam_service_url=self.config.sam_service_url,
            sam_service_timeout=self.config.sam_service_timeout,
            target_area_pixels=target_area_px,
            focus_bbox=example_bbox,
        )

        plot_w_deg, plot_h_deg, example_size_m = self._extract_plot_size_from_sam(
            sam_result, example_bbox
        )
        debug_info["steps"].append({
            "step": "sam",
            "method": sam_result.get("method"),
            "detected_plots_count": len(sam_result.get("detected_plots", [])),
            "plot_w_deg": round(plot_w_deg, 8),
            "plot_h_deg": round(plot_h_deg, 8),
            "target_area_pixels": round(target_area_px, 2),
        })
        logger.info(
            f"SAM 完成: method={sam_result.get('method')}, "
            f"检测到 {len(sam_result.get('detected_plots', []))} 个小区, "
            f"尺寸 {plot_w_deg:.6f}°x{plot_h_deg:.6f}°"
        )

        # ── Step 2: VLM 识别大尺度结构 ──
        logger.info("Step 2: VLM 识别大边界...")
        vlm_result = await self._vlm_detect_field_boundary(request)
        field_boundary = vlm_result.get("field_boundary", [])
        anomalies = vlm_result.get("anomalies", [])
        rotation_deg = vlm_result.get("rotation_deg", 0)

        debug_info["steps"].append({
            "step": "vlm",
            "field_boundary_points": len(field_boundary),
            "anomalies_count": len(anomalies),
            "rotation_deg": rotation_deg,
            "summary": vlm_result.get("summary", ""),
        })
        logger.info(
            f"VLM 完成: 大边界 {len(field_boundary)} 点, "
            f"异常 {len(anomalies)} 个, 旋转 {rotation_deg}°"
        )

        # ── Step 3: SAM 批量分割（用真实田埂边界，不再代码均分）──
        logger.info("Step 3: SAM 批量分割...")
        if field_boundary and len(field_boundary) >= 3:
            field_bbox = self._polygon_to_bbox(field_boundary)
            # 用 field_bbox 作为 focus 重新算 target_area_pixels，
            # 否则用 Step 1 的值会因 focus 图尺寸不同导致过滤全错
            target_area_px_field = self._estimate_target_area_pixels(
                request.cog_path, example_bbox, focus_bbox=field_bbox,
            )
            logger.info(
                f"Step 3 重新算 target_area_pixels: example_focus={target_area_px:.0f}px², "
                f"field_focus={target_area_px_field:.0f}px²"
            )
        else:
            # VLM 失败或无大边界，用影像 bbox 兜底
            field_bbox = request.image_bbox or [0, 0, 0, 0]
            target_area_px_field = self._estimate_target_area_pixels(
                request.cog_path, example_bbox, focus_bbox=field_bbox,
            )
            debug_info["steps"].append({
                "step": "sam_batch",
                "warning": "VLM 无大边界，用影像 bbox 作为批量分割范围",
            })

        # 在 field_bbox 内调 SAM 批量分割
        batch_result = await detect_field_boundaries(
            cog_path=request.cog_path,
            image_bbox=request.image_bbox,
            sam_service_url=self.config.sam_service_url,
            sam_service_timeout=self.config.sam_service_timeout,
            target_area_pixels=target_area_px_field,
            focus_bbox=field_bbox,
        )

        batch_plots = batch_result.get("detected_plots", [])
        batch_method = batch_result.get("method", "")
        logger.info(
            f"SAM 批量分割完成: method={batch_method}, 检测到 {len(batch_plots)} 个小区"
        )
        # 调试：打印每个 mask 的面积和 bbox，便于排查过滤逻辑
        for i, sp in enumerate(batch_plots):
            poly = sp.get("polygon", [])
            if poly and len(poly) >= 3:
                lons = [p[0] for p in poly]
                lats = [p[1] for p in poly]
                w_deg = max(lons) - min(lons)
                h_deg = max(lats) - min(lats)
                logger.info(
                    f"  mask[{i}]: area_pixels={sp.get('area_pixels', 0):.0f}, "
                    f"w={w_deg:.8f}° h={h_deg:.8f}°, "
                    f"target=[{plot_w_deg:.8f}° x {plot_h_deg:.8f}°]"
                )

        if batch_plots:
            # 用 SAM mask 作为小区边界
            plots, n_rows, n_cols = self._plots_from_sam_masks(
                sam_plots=batch_plots,
                field_bbox=field_bbox,
                plot_w_deg=plot_w_deg,
                plot_h_deg=plot_h_deg,
                anomalies=anomalies,
            )
            debug_info["steps"].append({
                "step": "sam_batch",
                "method": batch_method,
                "detected_plots_count": len(batch_plots),
                "used_grid_fallback": False,
            })
        else:
            # SAM 批量分割失败，回退到代码网格复制
            logger.warning("SAM 批量分割无结果，回退到代码网格复制")
            plots, n_rows, n_cols = self._generate_plots_in_field(
                field_bbox=field_bbox,
                plot_w_deg=plot_w_deg,
                plot_h_deg=plot_h_deg,
                rotation_deg=rotation_deg,
                anomalies=anomalies,
            )
            debug_info["steps"].append({
                "step": "sam_batch",
                "method": batch_method,
                "detected_plots_count": 0,
                "used_grid_fallback": True,
                "warning": "SAM 批量分割无结果，用代码网格复制兜底",
            })

        skip_count = sum(1 for p in plots if p.status == "skip")
        debug_info["total_plots"] = len(plots)
        debug_info["skip_count"] = skip_count
        debug_info["ok_count"] = len(plots) - skip_count
        debug_info["n_rows"] = n_rows
        debug_info["n_cols"] = n_cols

        logger.info(f"小区生成完成: 共 {len(plots)} 个小区 (ok {len(plots) - skip_count}, skip {skip_count})")

        return CompletionResult(
            image_id=request.image_id,
            total=len(plots),
            n_rows=n_rows,
            n_cols=n_cols,
            region={"type": "bbox", "coordinates": field_bbox},
            example_size_m=example_size_m,
            plots=plots,
            debug_info=debug_info,
        )
