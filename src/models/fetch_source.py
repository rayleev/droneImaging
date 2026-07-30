"""外部存储源配置 ORM 模型（P1 预留）

记录外部 MinIO/NAS/HTTP 存储源的连接信息，
供 fetch_images 接口拉取影像时使用。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class FetchSource(Base):
    """外部存储源配置表

    存储外部 MinIO、NAS 共享目录或 HTTP 文件服务的连接信息，
    用于按元数据从外部系统拉取原始影像。
    """

    __tablename__ = "pa_di_fetch_source"
    __table_args__ = (
        {"comment": "外部存储源配置表（P1 预留）"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        comment="主键",
    )
    name: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True,
        comment="存储源名称（唯一标识）",
    )
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="类型: minio / nas / http",
    )
    endpoint: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="连接地址",
    )
    credentials: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="认证信息（access_key/secret_key 等，加密存储）",
    )
    config: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="额外配置（桶名、根路径等）",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        comment="创建时间",
    )

    def __repr__(self) -> str:
        return f"<FetchSource {self.name} type={self.source_type}>"
