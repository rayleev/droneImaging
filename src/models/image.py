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

    # ── agri_research 关联字段（sync 来源影像专用） ──
    image_product_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, unique=True, index=True,
        comment="agri_research.image_products.id（去重键，sync 来源必填）",
    )
    mission_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="agri_research.flight_missions.id",
    )
    plot_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="agri_research.plots.id",
    )
    experiment_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="agri_research.experiments.id",
    )
    site_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="agri_research.sites.id",
    )
    plot_code: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True,
        comment="小区编号（如 P01）",
    )
    flight_date: Mapped[str | None] = mapped_column(
        String(10), nullable=True, index=True,
        comment="飞行日期 ISO 格式 YYYY-MM-DD",
    )
    product_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True,
        comment="产品类型：rgb / ms",
    )
    variety: Mapped[str | None] = mapped_column(
        String(200), nullable=True, index=True,
        comment="品种",
    )
    treatment: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="处理（plots.treatment jsonb 转字符串）",
    )
    capture_date: Mapped[str | None] = mapped_column(
        String(10), nullable=True,
        comment="影像拍摄日期 ISO 格式 YYYY-MM-DD",
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
        comment="来源: upload/sync",
    )
    granularity: Mapped[str] = mapped_column(
        String(20), nullable=False, default="plot", index=True,
        comment="粒度: plot(小区级影像) / mission(整架次正射)",
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


class ImageMosaic(Base):
    """拼接后的整体影像表

    存储同一小区（plot）的多张影像拼接后的整体影像信息。
    """

    __tablename__ = "pa_di_image_mosaic"
    __table_args__ = (
        {"comment": "拼接后的整体影像表"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        comment="主键",
    )

    # ── 关联信息 ──
    plot_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="关联的小区ID",
    )
    plot_code: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="小区编号",
    )
    mission_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), nullable=True,
        comment="关联的飞行任务ID",
    )
    product_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True,
        comment="产品类型：rgb / ms",
    )
    mosaic_type: Mapped[str | None] = mapped_column(
        String(20), nullable=True, index=True,
        comment="拼接类型: plot(小区级) / mission(飞行任务级)",
    )

    # ── 拼接信息 ──
    image_ids: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True,
        comment="参与拼接的图片ID列表",
    )
    image_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="参与拼接的图片数量",
    )

    # ── 文件路径 ──
    file_path: Mapped[str | None] = mapped_column(
        String(1000), nullable=True,
        comment="MinIO中的拼接影像路径",
    )
    thumbnail_path: Mapped[str | None] = mapped_column(
        String(1000), nullable=True,
        comment="MinIO中的缩略图路径",
    )

    # ── 地理参考 ──
    bbox: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="拼接影像的地理范围 [min_lon, min_lat, max_lon, max_lat]",
    )
    crs: Mapped[str | None] = mapped_column(
        String(50), nullable=True,
        comment="坐标系（如 EPSG:4326）",
    )
    width_pixels: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="拼接影像宽度（像素）",
    )
    height_pixels: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="拼接影像高度（像素）",
    )

    # ── VLM 描述 ──
    vlm_description: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="VLM 生成的自然语言摘要",
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

    # ── 时间戳 ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        comment="记录创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now(),
        comment="记录更新时间",
    )

    def __repr__(self) -> str:
        return f"<ImageMosaic {self.id} plot={self.plot_code} images={self.image_count}>"
