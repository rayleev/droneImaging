"""SAM + LLM hybrid strategy"""
from __future__ import annotations
import math
from loguru import logger
from .base import CompletionRequest, CompletionResult, PlotCell, PlotCompletionStrategy

class SamLLMStrategy(PlotCompletionStrategy):
    def __init__(self, config=None):
        from .config import get_completion_config
        self.config = config or get_completion_config()
        self._sam_model = None
        self._sam_available = False
        # 不再加载本地 SAM，使用远程服务

    @property
    def name(self): return "sam_llm"

    @property
    def description(self):
        return "SAM (remote) + LLM" if self.config.sam_service_url else "OpenCV + LLM (no SAM)"

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        from .llm_parser import parse_description_with_llm
        from .boundary_detector import detect_field_boundaries
        layout_hint = await parse_description_with_llm(request.description, self.config)

        # 计算用户手绘区域的面积（像素）用于筛选
        target_area_pixels = 0
        if request.example_region:
            try:
                coords = request.example_region.get("coordinates", [])
                if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                    coords = coords[0]
                if coords:
                    # 用地理坐标估算像素面积（需要图像尺寸，这里用近似值）
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    geo_w = max(lons) - min(lons)
                    geo_h = max(lats) - min(lats)
                    # 假设图像约 2000x2000 像素覆盖整图
                    bbox = request.image_bbox
                    if bbox and len(bbox) == 4:
                        total_geo_w = bbox[2] - bbox[0]
                        total_geo_h = bbox[3] - bbox[1]
                        # 估算像素面积（假设 SAM 处理的是 1024x1024 的降采样图）
                        scale = 1024.0 / max(total_geo_w, total_geo_h)
                        target_area_pixels = (geo_w * scale) * (geo_h * scale)
            except Exception:
                pass

        boundary_info = await detect_field_boundaries(
            request.cog_path, request.image_bbox,
            self._sam_model if self._sam_available else None, request.nodata,
            sam_service_url=self.config.sam_service_url,
            sam_service_timeout=self.config.sam_service_timeout,
            target_area_pixels=target_area_pixels)

        # 如果有 SAM 检测到的实际区域，直接使用；否则 fallback 到网格生成
        detected_plots = boundary_info.get("detected_plots", [])
        if detected_plots:
            plots = self._convert_detected_plots(detected_plots)
            n_rows = max(p.row for p in plots) + 1 if plots else 1
            n_cols = max(p.col for p in plots) + 1 if plots else 1
        else:
            plots = self._generate_plots(boundary_info, layout_hint, request.example_region, request.image_bbox)
            n_rows = layout_hint.get("n_rows", 1) or 1
            n_cols = layout_hint.get("n_cols", 1) or 1

        example_size_m = self._compute_example_size(request.example_region) if request.example_region else {"width":0,"height":0}
        return CompletionResult(
            image_id=request.image_id, total=len(plots),
            n_rows=n_rows, n_cols=n_cols,
            region={"type":"bbox","coordinates":request.image_bbox or [0,0,0,0]},
            example_size_m=example_size_m, plots=plots,
            debug_info={"strategy":self.name,"layout_hint":layout_hint,"boundary_info":boundary_info} if self.config.return_debug_info else {})

    @staticmethod
    def _point_in_polygon(x, y, polygon):
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

    def _generate_plots(self, boundary_info, layout_hint, example_region, image_bbox):
        plots = []
        eff = boundary_info.get("effective_bbox") if boundary_info else None
        if eff:
            eff_bbox = eff
        elif image_bbox and len(image_bbox)==4:
            eff_bbox = image_bbox
        else:
            return plots
        min_lon,min_lat,max_lon,max_lat = eff_bbox
        center_lat = (min_lat+max_lat)/2
        m_per_deg_lon = 111_320.0*math.cos(math.radians(center_lat))
        m_per_deg_lat = 111_320.0
        n_rows_hint = layout_hint.get("n_rows")
        n_cols_hint = layout_hint.get("n_cols")

        # SAM 识别的边界多边形（地理坐标）
        sam_boundary = None
        if boundary_info and boundary_info.get("method") == "sam_remote":
            boundaries = boundary_info.get("boundaries")
            if boundaries and len(boundaries) > 2:
                sam_boundary = boundaries

        # 计算示例区域的大小（米）和锚点地理坐标
        anchor_lon, anchor_lat = min_lon, max_lat
        cell_w_m = (max_lon-min_lon)*m_per_deg_lon
        cell_h_m = (max_lat-min_lat)*m_per_deg_lat
        if example_region:
            try:
                coords = example_region.get("coordinates", [])
                if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                    coords = coords[0]
                ex_lons = [c[0] for c in coords]
                ex_lats = [c[1] for c in coords]
                anchor_lon = min(ex_lons)
                anchor_lat = max(ex_lats)
                ex_center_lat = (min(ex_lats)+max(ex_lats))/2
                ex_m_per_deg_lon = 111_320.0*math.cos(math.radians(ex_center_lat))
                ex_m_per_deg_lat = 111_320.0
                cell_w_m = (max(ex_lons)-min(ex_lons))*ex_m_per_deg_lon
                cell_h_m = (max(ex_lats)-min(ex_lats))*ex_m_per_deg_lat
            except Exception:
                pass

        cell_w_deg = cell_w_m/m_per_deg_lon if m_per_deg_lon > 0 else 0
        cell_h_deg = cell_h_m/m_per_deg_lat if m_per_deg_lat > 0 else 0
        if cell_w_deg <= 0 or cell_h_deg <= 0:
            return plots

        # 计算行列数：从锚点到边界右边/下边的距离
        dist_right = (max_lon-anchor_lon)*m_per_deg_lon
        dist_down = (anchor_lat-min_lat)*m_per_deg_lat
        if n_cols_hint is None:
            n_cols_hint = max(1, round(dist_right/cell_w_m)) if cell_w_m > 0 else 1
        if n_rows_hint is None:
            n_rows_hint = max(1, round(dist_down/cell_h_m)) if cell_h_m > 0 else 1

        n_rows = n_rows_hint or 1
        n_cols = n_cols_hint or 1

        for row in range(n_rows):
            for col in range(n_cols):
                c_min_lon = anchor_lon + col*cell_w_deg
                c_max_lon = c_min_lon + cell_w_deg
                c_max_lat = anchor_lat - row*cell_h_deg
                c_min_lat = c_max_lat - cell_h_deg

                # 裁剪到有效范围
                c_min_lon = max(c_min_lon, min_lon)
                c_min_lat = max(c_min_lat, min_lat)
                c_max_lon = min(c_max_lon, max_lon)
                c_max_lat = min(c_max_lat, max_lat)
                if c_max_lon<=c_min_lon or c_max_lat<=c_min_lat:
                    continue

                # 检查是否在 SAM 边界内（如果有的话）
                if sam_boundary:
                    center_lon = (c_min_lon + c_max_lon) / 2
                    center_lat = (c_min_lat + c_max_lat) / 2
                    if not self._point_in_polygon(center_lon, center_lat, sam_boundary):
                        continue

                corners=[[c_min_lon,c_min_lat],[c_max_lon,c_min_lat],[c_max_lon,c_max_lat],[c_min_lon,c_max_lat],[c_min_lon,c_min_lat]]
                area=(c_max_lon-c_min_lon)*m_per_deg_lon*(c_max_lat-c_min_lat)*m_per_deg_lat
                plots.append(PlotCell(id=f"plot-{row}-{col}",label=f"{chr(65+row)}{col+1}",row=row,col=col,bbox=[c_min_lon,c_min_lat,c_max_lon,c_max_lat],polygon=[corners],area_m2=round(area,2)))
        return plots

    def _convert_detected_plots(self, detected_plots):
        """将 SAM 检测到的区域转换为 PlotCell"""
        from .base import PlotCell
        plots = []
        # 按中心点位置排序，分配行列号
        sorted_plots = sorted(detected_plots, key=lambda p: (-p["polygon"][0][1], p["polygon"][0][0]))
        for idx, dp in enumerate(sorted_plots):
            polygon = dp["polygon"]
            if not polygon:
                continue
            lons = [p[0] for p in polygon]
            lats = [p[1] for p in polygon]
            bbox = [min(lons), min(lats), max(lons), max(lats)]
            # 确保多边形闭合
            if polygon[0] != polygon[-1]:
                polygon = polygon + [polygon[0]]
            corners = [[p[0], p[1]] for p in polygon]
            plots.append(PlotCell(
                id=f"plot-{idx}",
                label=f"P{idx+1}",
                row=idx,
                col=0,
                bbox=bbox,
                polygon=[corners],
                area_m2=dp.get("area_pixels", 0),
            ))
        return plots

    def _compute_example_size(self, example_region):
        coords = example_region.get("coordinates",[])
        if not coords: return {"width":0,"height":0}
        if isinstance(coords[0],list) and isinstance(coords[0][0],list): coords=coords[0]
        if not coords or len(coords)<3: return {"width":0,"height":0}
        lons=[c[0] for c in coords]; lats=[c[1] for c in coords]
        min_lon,max_lon=min(lons),max(lons); min_lat,max_lat=min(lats),max(lats)
        center_lat=(min_lat+max_lat)/2
        m_per_deg_lon=111_320.0*math.cos(math.radians(center_lat))
        m_per_deg_lat=111_320.0
        return {"width":round((max_lon-min_lon)*m_per_deg_lon,2),"height":round((max_lat-min_lat)*m_per_deg_lat,2)}
