"""无人机影像服务 — FastAPI 应用入口

启动方式: uvicorn src.main:app --port 8002 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import get_config, load_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时释放"""
    # ── 启动 ──
    cfg = load_config()
    logger.info(f"droneImaging 服务启动 | port={cfg.server.port} debug={cfg.server.debug}")

    # 初始化数据库（自动建表）
    from src.database import init_db
    await init_db()
    logger.info("PostgreSQL 数据库初始化完成")

    # 初始化 MinIO（检查/创建桶）
    from src.services.storage import init_storage
    init_storage()
    logger.info("MinIO 存储初始化完成")

    # 初始化 Milvus（检查/创建 collection）
    try:
        from src.services.milvus_client import ensure_collection
        ensure_collection()
        logger.info("Milvus collection 初始化完成")
    except Exception as e:
        logger.warning(f"Milvus 初始化失败（服务仍可启动，检索功能不可用）: {e}")

    yield

    # ── 关闭 ──
    from src.database import close_db
    await close_db()
    logger.info("droneImaging 服务已关闭")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    cfg = get_config()

    app = FastAPI(
        title="droneImaging — 无人机影像服务",
        description="为 phenomicsAgentCC 提供无人机影像入库、语义检索和瓦片服务",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（内网部署，按需收紧）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 注册路由 ──
    from src.routers.images import router as images_router
    from src.routers.search import router as search_router
    from src.routers.tiles import router as tiles_router

    app.include_router(images_router, prefix="/api/images", tags=["影像管理"])
    app.include_router(search_router, prefix="/api/images", tags=["语义检索"])
    app.include_router(tiles_router, prefix="/api/tiles", tags=["瓦片服务"])

    # 健康检查（/api/health 供前端经 /api/drone 代理访问）
    @app.get("/health", tags=["系统"])
    @app.get("/api/health", tags=["系统"], include_in_schema=False)
    async def health_check():
        return {"status": "ok", "service": "droneImaging"}

    return app


app = create_app()
