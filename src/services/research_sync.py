"""agri_research 元数据拉取与富化模块

只读访问外部 agri_research 数据库，拉取飞行任务/小区/性状观测等元数据，
供 sync 服务在影像入库前富化 Image 记录。

所有 SQL 均为 SELECT；写入只发生在本服务自身的 drone_imaging 库。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_research_session_factory


# ── 待同步影像候选查询 ───────────────────────────────────────

CANDIDATES_SQL = text(
    """
    SELECT
        ip.id            AS image_product_id,
        ip.mission_id,
        ip.plot_id,
        ip.sensor_id,
        ip.product_type,
        ip.file_path,
        ip.file_format,
        ip.capture_date,
        fm.flight_date,
        fm.experiment_id,
        fm.site_id,
        fm.raw_data_path
    FROM image_products ip
    JOIN flight_missions fm ON fm.id = ip.mission_id
    WHERE fm.processed = true
      AND (CAST(:mission_id AS uuid) IS NULL OR ip.mission_id = CAST(:mission_id AS uuid))
      AND (CAST(:experiment_id AS uuid) IS NULL OR fm.experiment_id = CAST(:experiment_id AS uuid))
      AND (CAST(:product_type AS varchar) IS NULL OR ip.product_type = :product_type)
    ORDER BY fm.flight_date, ip.product_type, ip.id
    """
)


# ── 单条影像富化查询（plot/experiment/site + 最近观测快照） ────

ENRICH_SQL = text(
    """
    SELECT
        p.plot_code,
        p.treatment   AS treatment_json,
        p.variety,
        ST_AsGeoJSON(p.geom) AS plot_geom_geojson,
        e.name         AS experiment_name,
        e.crop_type    AS experiment_crop_type,
        s.name         AS site_name,
        s.latitude     AS site_lat,
        s.longitude    AS site_lon
    FROM plots p
    LEFT JOIN experiments e ON e.id = p.experiment_id
    LEFT JOIN sites s ON s.id = p.site_id
    WHERE p.id = CAST(:plot_id AS uuid)
    """
)


OBSERVATIONS_SNAPSHOT_SQL = text(
    """
    SELECT
        o.trait_id,
        t.trait_code,
        t.name        AS trait_name,
        o.value_num,
        o.value_cat,
        o.value_text,
        o.unit,
        o.obs_date,
        o.growth_stage,
        ds.type       AS data_source_type,
        ds.name       AS data_source_name
    FROM observations o
    JOIN traits t ON t.id = o.trait_id
    LEFT JOIN data_sources ds ON ds.id = o.data_source_id
    WHERE o.plot_id = CAST(:plot_id AS uuid)
      AND o.obs_date <= CAST(:capture_date AS date)
    ORDER BY o.obs_date DESC, o.updated_at DESC
    """
)


async def list_sync_candidates(
    mission_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    product_type: Optional[str] = None,
    existing_image_product_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """列出待同步的 image_products 候选。

    Args:
        mission_id: 限定单次飞行任务
        experiment_id: 限定试验
        product_type: 按产品类型过滤 (rgb / ms)
        existing_image_product_ids: 本地已存在（且不强制重跑）的 image_products.id 列表，
            用于增量同步去重

    Returns:
        候选行列表，每行含 image_product_id / mission_id / plot_id / file_path 等
    """
    factory = get_research_session_factory()
    async with factory() as session:
        result = await session.execute(
            CANDIDATES_SQL,
            {
                "mission_id": mission_id,
                "experiment_id": experiment_id,
                "product_type": product_type,
            },
        )
        rows = [dict(r._mapping) for r in result.fetchall()]

    if existing_image_product_ids:
        existing_set = {str(x) for x in existing_image_product_ids}
        rows = [r for r in rows if str(r["image_product_id"]) not in existing_set]

    logger.info(
        f"agri_research 同步候选: {len(rows)} 行"
        f"{' (mission=' + mission_id + ')' if mission_id else ''}"
        f"{' (experiment=' + experiment_id + ')' if experiment_id else ''}"
    )
    return rows


async def fetch_plot_enrichment(plot_id: str) -> Dict[str, Any]:
    """拉取小区富化信息：plot_code / treatment / variety / geom + 试验名 / 作物 / 地点。"""
    factory = get_research_session_factory()
    async with factory() as session:
        result = await session.execute(ENRICH_SQL, {"plot_id": plot_id})
        row = result.fetchone()
        if row is None:
            return {}
        mapping = dict(row._mapping)

    treatment_json = mapping.get("treatment_json")
    treatment_str: Optional[str]
    if treatment_json is None:
        treatment_str = None
    elif isinstance(treatment_json, (dict, list)):
        treatment_str = json.dumps(treatment_json, ensure_ascii=False)
    else:
        # psycopg 异步驱动通常已返回 dict/list；兜底按字符串处理
        treatment_str = str(treatment_json)

    plot_geom_geojson = mapping.get("plot_geom_geojson")
    plot_geom: Optional[dict]
    if isinstance(plot_geom_geojson, str):
        try:
            plot_geom = json.loads(plot_geom_geojson)
        except (json.JSONDecodeError, ValueError):
            plot_geom = None
    elif isinstance(plot_geom_geojson, dict):
        plot_geom = plot_geom_geojson
    else:
        plot_geom = None

    return {
        "plot_code": mapping.get("plot_code"),
        "treatment": treatment_str,
        "variety": mapping.get("variety"),
        "plot_geom": plot_geom,
        "experiment_name": mapping.get("experiment_name"),
        "experiment_crop_type": mapping.get("experiment_crop_type"),
        "site_name": mapping.get("site_name"),
        "site_lat": float(mapping["site_lat"]) if mapping.get("site_lat") is not None else None,
        "site_lon": float(mapping["site_lon"]) if mapping.get("site_lon") is not None else None,
    }


async def fetch_observations_snapshot(
    plot_id: str,
    capture_date: str,
) -> List[Dict[str, Any]]:
    """拉取该 plot 上 obs_date <= capture_date 的最近观测快照。

    每个 trait 取最新一条（obs_date DESC, updated_at DESC）。
    实测/模型预测都保留，标注 data_source_type。
    """
    factory = get_research_session_factory()
    async with factory() as session:
        result = await session.execute(
            OBSERVATIONS_SNAPSHOT_SQL,
            {"plot_id": plot_id, "capture_date": capture_date},
        )
        rows = [dict(r._mapping) for r in result.fetchall()]

    # 每个 trait_code 只保留最新一条
    seen: set = set()
    snapshot: List[Dict[str, Any]] = []
    for r in rows:
        tc = r.get("trait_code")
        if not tc or tc in seen:
            continue
        seen.add(tc)
        snapshot.append({
            "trait_code": tc,
            "trait_name": r.get("trait_name"),
            "value_num": float(r["value_num"]) if r.get("value_num") is not None else None,
            "value_cat": r.get("value_cat"),
            "value_text": r.get("value_text"),
            "unit": r.get("unit"),
            "obs_date": str(r["obs_date"]) if r.get("obs_date") else None,
            "growth_stage": r.get("growth_stage"),
            "data_source_type": r.get("data_source_type"),
            "data_source_name": r.get("data_source_name"),
        })

    # 推断拍摄时生育期：取快照里最近一条带 growth_stage 的观测
    growth_stage_at_capture: Optional[str] = None
    for r in rows:
        if r.get("growth_stage"):
            growth_stage_at_capture = r["growth_stage"]
            break

    return {
        "observations": snapshot,
        "growth_stage_at_capture": growth_stage_at_capture,
    }


async def resolve_plot_ids_by_traits(
    trait_filters: List[Dict[str, Any]],
) -> List[str]:
    """两阶段性状检索第一阶段：按性状数值条件反查 plot_id 集合。

    Args:
        trait_filters: [{"trait_code": "ROLLING_LEAF_SCORE", "op": "gte", "value": 3}, ...]

    Returns:
        命中的 plot_id（字符串）列表，用于 Milvus plot_id in [...] 过滤。
        多个 trait_filter 之间为 AND 关系（取交集）。
    """
    if not trait_filters:
        return []

    factory = get_research_session_factory()
    op_sql_map = {
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "eq": "=",
    }

    # 每个 trait_filter 求一个 plot_id 集合，最后取交集
    sets: List[set] = []
    for tf in trait_filters:
        trait_code = tf.get("trait_code")
        op = op_sql_map.get(tf.get("op", "gte"), ">=")
        value = tf.get("value")
        if not trait_code or value is None:
            continue
        sql = text(
            f"""
            SELECT DISTINCT o.plot_id::text
            FROM observations o
            JOIN traits t ON t.id = o.trait_id
            WHERE t.trait_code = :trait_code
              AND o.value_num {op} CAST(:value AS numeric)
              AND o.plot_id IS NOT NULL
            """
        )
        async with factory() as session:
            result = await session.execute(
                sql, {"trait_code": trait_code, "value": value}
            )
            ids = {r[0] for r in result.fetchall()}
        sets.append(ids)

    if not sets:
        return []

    intersection = set.intersection(*sets)
    return sorted(intersection)
