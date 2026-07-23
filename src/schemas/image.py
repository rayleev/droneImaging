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
