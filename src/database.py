"""无人机影像服务 — 数据库连接模块

基于 SQLAlchemy 2.0 async 引擎，提供异步 Session 工厂。
应用启动时调用 init_db() 自动建表。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.config import get_config


class Base(DeclarativeBase):
    """ORM 模型基类"""
    pass


# 延迟初始化，在 init_db() 中赋值
engine = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """初始化数据库引擎并自动建表。

    在 FastAPI lifespan 中调用。
    """
    global engine, async_session_factory

    cfg = get_config().postgresql
    engine = create_async_engine(
        cfg.url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # 连接健康检查
    )
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # 自动建表（生产环境建议用 Alembic 迁移）
    async with engine.begin() as conn:
        # 导入所有模型，确保 metadata 注册
        from src.models import image, fetch_source  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """关闭数据库连接池"""
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None


async def get_session() -> AsyncSession:
    """FastAPI 依赖注入：获取数据库 Session"""
    if async_session_factory is None:
        raise RuntimeError("数据库未初始化，请先调用 init_db()")
    async with async_session_factory() as session:
        yield session
