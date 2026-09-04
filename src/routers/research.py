"""agri_research 元数据查询路由

提供试验地点(sites)和试验(experiments)的查询接口，
供 phenomicsAgent 在页面上展示试验地点供用户查看和对话。

提供：
- GET /api/research/sites           列出所有试验地点
- GET /api/research/sites/{site_id}  单个地点详情（含试验列表）
- GET /api/research/experiments     列出所有试验（可按 site_id 过滤）
- GET /api/research/experiments/{exp_id}  单个试验详情
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from sqlalchemy import text

from src.database import get_research_session_factory


router = APIRouter()


# ── sites ─────────────────────────────────────────────────────


@router.get("/sites")
async def list_sites():
    """列出所有试验地点"""
    factory = get_research_session_factory()
    async with factory() as session:
        result = await session.execute(
            text("""
                SELECT id, name, latitude, longitude
                FROM sites
                ORDER BY name
            """)
        )
        rows = result.fetchall()

    sites = [
        {
            "id": str(row[0]),
            "name": row[1],
            "latitude": float(row[2]) if row[2] is not None else None,
            "longitude": float(row[3]) if row[3] is not None else None,
        }
        for row in rows
    ]
    return {"total": len(sites), "items": sites}


@router.get("/sites/{site_id}")
async def get_site(site_id: UUID):
    """获取单个地点详情（含该地点下的试验列表）"""
    factory = get_research_session_factory()
    async with factory() as session:
        # 地点基本信息
        result = await session.execute(
            text("SELECT id, name, latitude, longitude FROM sites WHERE id = CAST(:sid AS uuid)"),
            {"sid": str(site_id)},
        )
        row = result.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="地点不存在")

        site = {
            "id": str(row[0]),
            "name": row[1],
            "latitude": float(row[2]) if row[2] is not None else None,
            "longitude": float(row[3]) if row[3] is not None else None,
        }

        # 该地点下的试验列表
        result = await session.execute(
            text("""
                SELECT e.id, e.name, e.crop_type
                FROM experiments e
                JOIN plots p ON p.experiment_id = e.id
                WHERE p.site_id = CAST(:sid AS uuid)
                GROUP BY e.id, e.name, e.crop_type
                ORDER BY e.name
            """),
            {"sid": str(site_id)},
        )
        exp_rows = result.fetchall()

    experiments = [
        {
            "id": str(r[0]),
            "name": r[1],
            "crop_type": r[2],
        }
        for r in exp_rows
    ]

    return {
        **site,
        "experiments": experiments,
        "experiment_count": len(experiments),
    }


# ── experiments ──────────────────────────────────────────────


@router.get("/experiments")
async def list_experiments(
    site_id: Optional[UUID] = Query(None, description="按地点过滤"),
):
    """列出所有试验（可按 site_id 过滤）"""
    factory = get_research_session_factory()
    async with factory() as session:
        if site_id:
            result = await session.execute(
                text("""
                    SELECT DISTINCT e.id, e.name, e.crop_type
                    FROM experiments e
                    JOIN plots p ON p.experiment_id = e.id
                    WHERE p.site_id = CAST(:sid AS uuid)
                    ORDER BY e.name
                """),
                {"sid": str(site_id)},
            )
        else:
            result = await session.execute(
                text("SELECT id, name, crop_type FROM experiments ORDER BY name")
            )
        rows = result.fetchall()

    experiments = [
        {
            "id": str(row[0]),
            "name": row[1],
            "crop_type": row[2],
        }
        for row in rows
    ]
    return {"total": len(experiments), "items": experiments}


@router.get("/experiments/{exp_id}")
async def get_experiment(exp_id: UUID):
    """获取单个试验详情（含关联的地点和小区数量）"""
    factory = get_research_session_factory()
    async with factory() as session:
        # 试验基本信息
        result = await session.execute(
            text("SELECT id, name, crop_type FROM experiments WHERE id = CAST(:eid AS uuid)"),
            {"eid": str(exp_id)},
        )
        row = result.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="试验不存在")

        experiment = {
            "id": str(row[0]),
            "name": row[1],
            "crop_type": row[2],
        }

        # 关联的地点
        result = await session.execute(
            text("""
                SELECT DISTINCT s.id, s.name, s.latitude, s.longitude
                FROM sites s
                JOIN plots p ON p.site_id = s.id
                WHERE p.experiment_id = CAST(:eid AS uuid)
            """),
            {"eid": str(exp_id)},
        )
        site_rows = result.fetchall()

        # 小区数量
        result = await session.execute(
            text("SELECT COUNT(*) FROM plots WHERE experiment_id = CAST(:eid AS uuid)"),
            {"eid": str(exp_id)},
        )
        plot_count = result.scalar()

        # 飞行任务数量
        result = await session.execute(
            text("""
                SELECT COUNT(*) FROM flight_missions
                WHERE experiment_id = CAST(:eid AS uuid) AND processed = true
            """),
            {"eid": str(exp_id)},
        )
        mission_count = result.scalar()

    sites = [
        {
            "id": str(r[0]),
            "name": r[1],
            "latitude": float(r[2]) if r[2] is not None else None,
            "longitude": float(r[3]) if r[3] is not None else None,
        }
        for r in site_rows
    ]

    return {
        **experiment,
        "sites": sites,
        "plot_count": plot_count,
        "mission_count": mission_count,
    }
