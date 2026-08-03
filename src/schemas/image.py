"""影像相关 Pydantic 请求/响应模型"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── 请求模型 ─────────────────────────────────────────────────

class ImageUploadMeta(BaseModel):
    """上传影像时附带的业务字段（multipart/form-data 中的非文件部分）"""
    task_id: str = Field(..., description="任务编号")
    field_group: Optional[str] = Field(None, description="试验田分组")
    field_name: Optional[str] = Field(None, description="试验田名称")
    survey_stage: Optional[str] = Field(None, description="调查阶段")
    device_model: Optional[str] = Field(None, description="设备型号")
    data_type: Optional[str] = Field(None, description="数据类型")
    surveyor: Optional[str] = Field(None, description="调查员")
    survey_time: Optional[str] = Field(None, description="调查时间（ISO 8601）")


class ImageSearchFilters(BaseModel):
    """检索时的结构化过滤条件"""
    task_id: Optional[str] = None
    field_name: Optional[str] = None
    survey_stage: Optional[str] = None
    crop_type: Optional[str] = Field(None, description="作物类型 ISO 代码，如 ZM=玉米")
    growth_stage: Optional[str] = Field(None, description="生长阶段 BBCH 代码或中文")


class ImageSearchRequest(BaseModel):
    """语义检索请求"""
    query: str = Field(..., description="自然语言查询")
    limit: int = Field(10, ge=1, le=50, description="返回数量上限")
    filters: Optional[ImageSearchFilters] = Field(None, description="结构化过滤条件")


# ── 响应模型 ─────────────────────────────────────────────────

class ImageUploadResponse(BaseModel):
    """上传响应"""
    id: uuid.UUID
    status: str
    message: str


class ImageStatusResponse(BaseModel):
    """处理状态响应"""
    id: uuid.UUID
    status: str
    progress: Optional[str] = None
    error_message: Optional[str] = None


class ImageListItem(BaseModel):
    deleted_at: Optional[datetime] = None
    """影像列表项（精简字段）"""
    id: uuid.UUID
    task_id: str
    field_group: Optional[str] = None
    field_name: Optional[str] = None
    survey_stage: Optional[str] = None
    device_model: Optional[str] = None
    data_type: Optional[str] = None
    surveyor: Optional[str] = None
    survey_time: Optional[datetime] = None
    upload_time: datetime
    original_filename: str
    status: str
    bbox: Optional[list] = None
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    thumbnail_url: Optional[str] = None
    tile_url: Optional[str] = None

    model_config = {"from_attributes": True}


class ImageDetail(ImageListItem):
    """影像完整详情"""
    file_size_bytes: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    bands: Optional[int] = None
    crs: Optional[str] = None
    pixel_scale_x: Optional[float] = None
    pixel_scale_y: Optional[float] = None
    nodata: Optional[float] = None
    geotransform: Optional[dict] = None
    vlm_description: Optional[str] = None
    vlm_model: Optional[str] = None
    vlm_time: Optional[datetime] = None
    source: str = "upload"
    extra_metadata: Optional[dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ImageSearchResultItem(BaseModel):
    """语义检索结果项"""
    id: uuid.UUID
    task_id: str
    field_name: Optional[str] = None
    survey_stage: Optional[str] = None
    crop_type: Optional[str] = None
    growth_stage: Optional[str] = None
    survey_time: Optional[datetime] = None
    bbox: Optional[list] = None
    center: Optional[list] = None
    thumbnail_url: Optional[str] = None
    tile_url: Optional[str] = None
    vlm_description: Optional[str] = None
    score: float


class ImageSearchResponse(BaseModel):
    """语义检索响应"""
    results: List[ImageSearchResultItem]
    total: int


class ImageListResponse(BaseModel):
    """影像列表响应（分页）"""
    items: List[ImageListItem]
    total: int
    page: int
    page_size: int


# ── 试验小区划分 ─────────────────────────────────────────────

class PlotDivideRequest(BaseModel):
    """试验小区划分请求"""
    image_id: uuid.UUID = Field(..., description="影像 ID")
    region: Optional[dict] = Field(None, description="绘制区域 GeoJSON Polygon 或 bbox [minLon,minLat,maxLon,maxLat]，为空则用整幅影像")
    n_rows: Optional[int] = Field(None, description="行数（与 n_cols 一起使用）")
    n_cols: Optional[int] = Field(None, description="列数（与 n_rows 一起使用）")
    plot_width_m: Optional[float] = Field(None, description="每个小区宽度（米），替代 n_cols")
    plot_height_m: Optional[float] = Field(None, description="每个小区高度（米），替代 n_rows")
    rotation_deg: float = Field(0.0, description="旋转角度（度），绕区域中心")
    label_scheme: str = Field("grid", description="编号方案: grid(A1,A2...) 或 linear(1,2,3...)")


class PlotCell(BaseModel):
    """单个小区"""
    id: str = Field(..., description="唯一标识")
    label: str = Field(..., description="编号标签（如 A1、B2）")
    row: int = Field(..., description="行索引")
    col: int = Field(..., description="列索引")
    bbox: list = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    polygon: list = Field(..., description="WGS84 多边形坐标环 [[[lon,lat],...]]")
    area_m2: float = Field(..., description="面积（平方米）")
    status: str = Field("ok", description="ok / skip（异常小区如被树遮挡、非试验田）")


class PlotDivideResponse(BaseModel):
    """试验小区划分响应"""
    image_id: uuid.UUID
    total: int = Field(..., description="小区总数")
    region: dict = Field(..., description="实际使用的区域")
    rotation_deg: float = Field(..., description="旋转角度")
    crs: str = Field(..., description="坐标系")
    plots: List[PlotCell]

# ── 智能补全 ─────────────────────────────────────────────────

class CompleteRequest(BaseModel):
    """智能补全请求"""
    image_id: uuid.UUID = Field(..., description="影像 ID")
    example_region: dict = Field(..., description="用户绘制的示例区域 GeoJSON Polygon")
    description: str = Field(..., description="自然语言描述，如'按这个大小，5行3列布满整个区域'")


class CompleteResponse(BaseModel):
    """智能补全响应"""
    image_id: uuid.UUID
    total: int = Field(..., description="小区总数")
    n_rows: int = Field(..., description="行数")
    n_cols: int = Field(..., description="列数")
    region: dict = Field(..., description="实际使用的区域")
    example_size_m: dict = Field(..., description="示例区域尺寸（米）")
    plots: List[PlotCell]
