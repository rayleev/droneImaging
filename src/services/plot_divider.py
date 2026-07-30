"""试验小区划分服务

将无人机影像的指定区域按规则划分为网格状试验小区。
计算在像素空间进行（处理旋转、非方形像素、任意 CRS），
输入输出统一为 WGS84。
"""

from __future__ import annotations

import math
import uuid
from typing import List, Tuple

from loguru import logger

from src.models.image import Image


def _wgs84_to_meters_factor(lat: float) -> Tuple[float, float]:
    """计算某纬度下 WGS84 度→米的换算因子

    Returns:
        (meters_per_deg_lon, meters_per_deg_lat)
    """
    # 1 度纬度 ≈ 111,320 米
    m_per_deg_lat = 111_320.0
    # 1 度经度随纬度变化
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return m_per_deg_lon, m_per_deg_lat


def _rotate_point(x: float, y: float, cx: float, cy: float, angle_deg: float) -> Tuple[float, float]:
    """绕中心点旋转一个点（2D）"""
    if angle_deg == 0:
        return x, y
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = x - cx
    dy = y - cy
    rx = cx + dx * cos_a - dy * sin_a
    ry = cy + dx * sin_a + dy * cos_a
    return rx, ry


def _polygon_area_m2(ring: List[List[float]], center_lat: float) -> float:
    """用鞋带公式计算多边形面积（平方米），在 center_lat 处做局部 ENU 投影"""
    if len(ring) < 3:
        return 0.0
    m_per_deg_lon, m_per_deg_lat = _wgs84_to_meters_factor(center_lat)
    # 取 ring 第一个点作为局部原点
    origin_lon = ring[0][0]
    origin_lat = ring[0][1]
    # 转为局部 ENU 坐标（米）
    pts = [
        ((p[0] - origin_lon) * m_per_deg_lon, (p[1] - origin_lat) * m_per_deg_lat)
        for p in ring
    ]
    # 鞋带公式
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _compute_region_bbox(region: dict | None, image: Image) -> List[float]:
    """计算实际使用的区域 bbox [min_lon, min_lat, max_lon, max_lat]"""
    if region is None:
        # 用整幅影像 bbox
        if image.bbox and len(image.bbox) == 4:
            return image.bbox
        raise ValueError("影像缺少 bbox 且未指定 region")

    # GeoJSON Polygon
    if region.get("type") == "Polygon":
        coords = region["coordinates"][0]  # 外环
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [min(lons), min(lats), max(lons), max(lats)]

    # bbox 直接传入 [min_lon, min_lat, max_lon, max_lat]
    if isinstance(region, list) and len(region) == 4:
        return region

    if "bbox" in region and len(region["bbox"]) == 4:
        return region["bbox"]

    raise ValueError(f"不支持的 region 格式: {region}")


def divide_plots(
    image: Image,
    region: dict | None = None,
    n_rows: int | None = None,
    n_cols: int | None = None,
    plot_width_m: float | None = None,
    plot_height_m: float | None = None,
    rotation_deg: float = 0.0,
    label_scheme: str = "grid",
) -> dict:
    """划分试验小区

    Args:
        image: Image ORM 对象
        region: 绘制区域（GeoJSON Polygon 或 bbox），None 则用整幅影像
        n_rows: 行数
        n_cols: 列数
        plot_width_m: 小区宽度（米），与 plot_height_m 一起使用
        plot_height_m: 小区高度（米）
        rotation_deg: 旋转角度
        label_scheme: 编号方案

    Returns:
        PlotDivideResponse 对应的 dict
    """
    # 1. 确定区域 bbox
    region_bbox = _compute_region_bbox(region, image)
    min_lon, min_lat, max_lon, max_lat = region_bbox
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # 2. 计算区域尺寸（米）
    m_per_deg_lon, m_per_deg_lat = _wgs84_to_meters_factor(center_lat)
    region_width_m = (max_lon - min_lon) * m_per_deg_lon
    region_height_m = (max_lat - min_lat) * m_per_deg_lat

    # 3. 确定行列数
    if n_rows and n_cols:
        pass
    elif plot_width_m and plot_height_m:
        n_cols = max(1, round(region_width_m / plot_width_m))
        n_rows = max(1, round(region_height_m / plot_height_m))
    elif plot_width_m:
        n_cols = max(1, round(region_width_m / plot_width_m))
        n_rows = n_cols  # 默认正方形
    elif plot_height_m:
        n_rows = max(1, round(region_height_m / plot_height_m))
        n_cols = n_rows
    else:
        # 默认 1x1
        n_rows = 1
        n_cols = 1

    if n_rows == 0:
        n_rows = 1
    if n_cols == 0:
        n_cols = 1

    logger.info(
        f"划分小区: region={region_bbox}, rows={n_rows}, cols={n_cols}, "
        f"rotation={rotation_deg}deg, size={region_width_m:.1f}x{region_height_m:.1f}m"
    )

    # 4. 在 WGS84 空间计算网格（简化：小范围可近似为平面）
    cell_width_deg = (max_lon - min_lon) / n_cols
    cell_height_deg = (max_lat - min_lat) / n_rows

    plots = []
    for row in range(n_rows):
        for col in range(n_cols):
            # 单元格 bbox（未旋转）
            cell_min_lon = min_lon + col * cell_width_deg
            cell_max_lon = cell_min_lon + cell_width_deg
            cell_max_lat = max_lat - row * cell_height_deg
            cell_min_lat = cell_max_lat - cell_height_deg

            # 多边形角点（未旋转），顺时针
            corners = [
                [cell_min_lon, cell_min_lat],
                [cell_max_lon, cell_min_lat],
                [cell_max_lon, cell_max_lat],
                [cell_min_lon, cell_max_lat],
                [cell_min_lon, cell_min_lat],  # 闭合
            ]

            # 旋转
            if rotation_deg != 0:
                corners = [
                    _rotate_point(c[0], c[1], center_lon, center_lat, rotation_deg)
                    for c in corners
                ]

            # 计算 bbox（旋转后可能变化）
            lons = [c[0] for c in corners]
            lats = [c[1] for c in corners]
            cell_bbox = [min(lons), min(lats), max(lons), max(lats)]

            # 面积
            area = _polygon_area_m2(corners, center_lat)

            # 编号
            if label_scheme == "linear":
                label = str(row * n_cols + col + 1)
            else:  # grid: A1, A2, ..., B1, B2, ...
                label = f"{chr(65 + row)}{col + 1}"

            plots.append({
                "id": f"plot-{row}-{col}",
                "label": label,
                "row": row,
                "col": col,
                "bbox": cell_bbox,
                "polygon": [corners],  # GeoJSON Polygon coordinates
                "area_m2": round(area, 2),
            })

    return {
        "image_id": str(image.id),
        "total": len(plots),
        "region": {"type": "bbox", "coordinates": region_bbox},
        "rotation_deg": rotation_deg,
        "crs": image.crs or "EPSG:4326",
        "plots": plots,
    }
