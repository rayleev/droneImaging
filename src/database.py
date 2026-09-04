"""无人机影像服务 — 数据库连接模块

基于 SQLAlchemy 2.0 async 引擎，提供异步 Session 工厂。
应用启动时调用 init_db() 自动建表。

除本服务自身数据库外，还维护一个对外部 agri_research 库的只读连接池，
用于 sync 流程拉取飞行任务/小区/性状观测元数据。
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

# 外部 agri_research 只读引擎（sync 用）
research_engine = None
research_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """初始化数据库引擎并自动建表。

    在 FastAPI lifespan 中调用。
    """
    global engine, async_session_factory, research_engine, research_session_factory

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

    # 外部 agri_research 只读连接池
    rcfg = get_config().research_db
    if rcfg.enabled:
        try:
            research_engine = create_async_engine(
                rcfg.url,
                echo=False,
                pool_size=rcfg.pool_size,
                max_overflow=0,
                pool_pre_ping=True,
                connect_args={"timeout": rcfg.connect_timeout},
            )
            research_session_factory = async_sessionmaker(
                research_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        except Exception:
            # 解析失败时不阻断主库启动，sync 功能在调用时报错
            research_engine = None
            research_session_factory = None


async def close_db() -> None:
    """关闭数据库连接池"""
    global engine, research_engine
    if engine is not None:
        await engine.dispose()
        engine = None
    if research_engine is not None:
        await research_engine.dispose()
        research_engine = None


async def get_session() -> AsyncSession:
    """FastAPI 依赖注入：获取数据库 Session"""
    if async_session_factory is None:
        raise RuntimeError("数据库未初始化，请先调用 init_db()")
    async with async_session_factory() as session:
        yield session


def get_research_session_factory() -> async_sessionmaker[AsyncSession]:
    """获取 agri_research 只读 Session 工厂。

    sync 服务在拉取元数据时调用。若未启用或初始化失败则抛 RuntimeError。
    """
    if research_session_factory is None:
        raise RuntimeError(
            "agri_research 只读连接池未初始化（research_db.enabled=false 或连接失败）"
        )
    return research_session_factory
