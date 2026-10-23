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
    # agri_research 关联键
    mission_id: Optional[str] = None
    plot_id: Optional[str] = None
    experiment_id: Optional[str] = None
    site_id: Optional[str] = None
    plot_code: Optional[str] = None
    # 分面过滤
    flight_date: Optional[str] = Field(None, description="飞行日期 YYYY-MM-DD（等值）")
    flight_date_from: Optional[str] = Field(None, description="飞行日期下界 YYYY-MM-DD（含）")
    flight_date_to: Optional[str] = Field(None, description="飞行日期上界 YYYY-MM-DD（含）")
    product_type: Optional[str] = Field(None, description="rgb / ms")
    variety: Optional[str] = None
    treatment: Optional[str] = None
    data_source_type: Optional[str] = Field(None, description="remote_sensing / model_prediction / manual")


class TraitFilter(BaseModel):
    """性状观测值过滤条件（两阶段性状检索用）"""
    trait_code: str = Field(..., description="性状代码，如 ROLLING_LEAF_SCORE")
    op: str = Field("gte", description="比较运算: gt / gte / lt / lte / eq")
    value: float = Field(..., description="比较值")


class ImageSearchRequest(BaseModel):
    """语义检索请求"""
    query: str = Field("", description="自然语言查询（已结构化分解的文本维度）")
    nl_query: Optional[str] = Field(
        None,
        description="原始自然语言查询；提供时服务端会用 LLM 分解为 query+filters+trait_filters，"
                    "若同时提供 query/filters 则跳过 LLM 分解",
    )
    limit: int = Field(10, ge=1, le=50, description="返回数量上限")
    filters: Optional[ImageSearchFilters] = Field(None, description="结构化过滤条件")
    trait_filters: Optional[List[TraitFilter]] = Field(
        None, description="性状观测值过滤（两阶段：先查 agri_research.observations 得 plot_id 再过滤）"
    )
    group_by: Optional[str] = Field(
        None, description="结果聚合: mission(按飞行任务) / plot(按小区，默认) / none(平铺)"
    )


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
    # agri_research 富化字段
    mission_id: Optional[str] = None
    plot_id: Optional[str] = None
    experiment_id: Optional[str] = None
    site_id: Optional[str] = None
    plot_code: Optional[str] = None
    flight_date: Optional[str] = None
    product_type: Optional[str] = None
    variety: Optional[str] = None
    treatment: Optional[str] = None
    plot_geom: Optional[dict] = Field(None, description="小区边界 GeoJSON Polygon（来自 plots.geom）")


class ImageSearchResponse(BaseModel):
    """语义检索响应"""
    results: List[ImageSearchResultItem]
    total: int
    decomposed: bool = Field(True, description="是否成功进行了 LLM 查询分解；false 表示降级为纯向量召回")


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
    plot_id: Optional[uuid.UUID] = None
    plot_code: Optional[str] = None
    mission_id: Optional[uuid.UUID] = None
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
    # 文件路径（MinIO）
    original_path: Optional[str] = None
    cog_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    # agri_research 富化字段
    image_product_id: Optional[uuid.UUID] = None
    mission_id: Optional[uuid.UUID] = None
    plot_id: Optional[uuid.UUID] = None
    experiment_id: Optional[uuid.UUID] = None
    site_id: Optional[uuid.UUID] = None
    plot_code: Optional[str] = None
    flight_date: Optional[str] = None
    product_type: Optional[str] = None
    variety: Optional[str] = None
    treatment: Optional[str] = None
    capture_date: Optional[str] = None

    model_config = {"from_attributes": True}


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


# ── agri_research 同步 ──────────────────────────────────────

class ImageSyncRequest(BaseModel):
    """从 agri_research 同步影像请求"""
    mission_id: Optional[uuid.UUID] = Field(None, description="限定单次飞行任务")
    experiment_id: Optional[uuid.UUID] = Field(None, description="限定试验（下所有 mission）")
    product_type: Optional[str] = Field(None, description="按产品类型过滤: rgb / ms")
    force: bool = Field(False, description="已存在的也强制重跑 VLM + embedding")


class ImageSyncResponse(BaseModel):
    """同步触发响应"""
    accepted: bool
    total: int = Field(..., description="本次待处理影像数")
    message: str


class ImageSyncStatusResponse(BaseModel):
    """同步进度响应"""
    running: bool
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    last_error: Optional[str] = None
    last_image_product_id: Optional[str] = None


class RefreshMetadataResponse(BaseModel):
    """元数据回填响应"""
    id: uuid.UUID
    refreshed_at: datetime
    message: str


# ── 拼接影像（pa_di_image_mosaic） ────────────────────────────

class MosaicListItem(BaseModel):
    """拼接影像列表项"""
    id: uuid.UUID
    mosaic_type: Optional[str] = None
    plot_id: Optional[uuid.UUID] = None
    plot_code: Optional[str] = None
    mission_id: Optional[uuid.UUID] = None
    product_type: Optional[str] = None
    image_count: int = 0
    survey_time: Optional[datetime] = None
    bbox: Optional[list] = None
    crs: Optional[str] = None
    width_pixels: Optional[int] = None
    height_pixels: Optional[int] = None
    thumbnail_url: Optional[str] = None
    tile_url: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class MosaicDetail(MosaicListItem):
    """拼接影像详情"""
    image_ids: Optional[list] = None
    vlm_description: Optional[str] = None
    vlm_model: Optional[str] = None
    vlm_time: Optional[datetime] = None


class MosaicListResponse(BaseModel):
    """拼接影像列表响应（分页）"""
    items: List[MosaicListItem]
    total: int
    page: int
    page_size: int
