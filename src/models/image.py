"""影像注册表 ORM 模型

存储无人机影像的业务信息和 GeoTIFF 元数据。
每条记录对应一张入库的无人机影像。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Double,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class Image(Base):
    """无人机影像注册表

    记录影像的业务属性（任务、试验田、调查阶段等）和
    GeoTIFF 技术元数据（bbox、分辨率、坐标系等），
    以及 VLM 描述和向量检索关联信息。
    """

    __tablename__ = "pa_di_image"
    __table_args__ = (
        {"comment": "无人机影像注册表"},
    )

    # ── 主键 ──
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        comment="主键",
    )

    # ── 业务字段 ──
    task_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True,
        comment="任务编号",
    )
    field_group: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
        comment="试验田分组",
    )
    field_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True, index=True,
        comment="试验田名称",
    )
    survey_stage: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True,
        comment="调查阶段（如分蘖期、抽穗期）",
    )
    device_model: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
        comment="设备型号",
    )
    data_type: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="数据类型（如可见光、多光谱）",
    )
    surveyor: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="调查员",
    )
    survey_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="调查时间",
    )
    upload_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(),
        comment="上传/入库时间",
    )

    # ── 文件路径（MinIO） ──
    original_filename: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="原始文件名",
    )
    original_path: Mapped[str | None] = mapped_column(
        String(1000), nullable=True,
        comment="MinIO 中原始文件路径",
    )
    cog_path: Mapped[str | None] = mapped_column(
        String(1000), nullable=True,
        comment="MinIO 中 COG 文件路径",
    )
    thumbnail_path: Mapped[str | None] = mapped_column(
        String(1000), nullable=True,
        comment="MinIO 中缩略图路径",
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True,
        comment="文件大小（字节）",
    )

    # ── GeoTIFF 元数据 ──
    image_width: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="像素宽度",
    )
    image_height: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="像素高度",
    )
    bands: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="波段数",
    )
    crs: Mapped[str | None] = mapped_column(
        String(50), nullable=True,
        comment="坐标系（如 EPSG:4326）",
    )
    bbox: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="地理范围 [min_lon, min_lat, max_lon, max_lat]",
    )
    center_lon: Mapped[float | None] = mapped_column(
        Double, nullable=True,
        comment="中心经度",
    )
    center_lat: Mapped[float | None] = mapped_column(
        Double, nullable=True,
        comment="中心纬度",
    )
    pixel_scale_x: Mapped[float | None] = mapped_column(
        Double, nullable=True,
        comment="X方向像素分辨率（度/像素）",
    )
    pixel_scale_y: Mapped[float | None] = mapped_column(
        Double, nullable=True,
        comment="Y方向像素分辨率（度/像素）",
    )
    nodata: Mapped[float | None] = mapped_column(
        Double, nullable=True,
        comment="NoData 值",
    )
    geotransform: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="完整仿射变换参数（6元素）",
    )

    # ── VLM 描述 ──
    vlm_description: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="VLM 生成的自然语言摘要（用于 embedding 和展示）",
    )
    vlm_model: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="使用的 VLM 模型名",
    )
    vlm_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="VLM 描述生成时间",
    )

    # ── 向量检索关联 ──
    embedding_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="Milvus 中对应的向量 ID",
    )

    # ── 状态 ──
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="uploaded", index=True,
        comment="处理状态: uploaded/parsing/converting/describing/embedding/ready/error",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="错误信息（status=error 时）",
    )
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="upload",
        comment="来源: upload/fetch",
    )

    # ── 扩展 ──
    extra_metadata: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="扩展元数据（含 VLM 结构化输出：crop_type、growth_stage、canopy_coverage 等）",
    )

    # ── 时间戳 ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        comment="记录创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now(),
        comment="记录更新时间",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="软删除时间",
    )

    def __repr__(self) -> str:
        return f"<Image {self.id} task={self.task_id} status={self.status}>"
