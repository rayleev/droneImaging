"""Milvus 向量数据库操作封装

管理 image_vectors collection 的创建、插入、搜索和删除。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from src.config import get_config

# 全局 Milvus 连接标识
_ALIAS = "drone_imaging"


def _ensure_connection() -> None:
    """确保 Milvus 连接已建立"""
    cfg = get_config().milvus
    if not connections.has_connection(_ALIAS):
        connections.connect(alias=_ALIAS, host=cfg.host, port=cfg.port)
        logger.info(f"Milvus 已连接: {cfg.host}:{cfg.port}")


def ensure_collection() -> None:
    """确保 image_vectors collection 存在，不存在则创建。

    Schema:
        - id: VARCHAR(64), 主键
        - text_vector: FLOAT_VECTOR(dim), 文本 embedding
        - image_vector: FLOAT_VECTOR(dim), 图片 embedding（预留）
        - task_id: VARCHAR(100), 标量过滤
        - field_name: VARCHAR(200), 标量过滤
        - survey_stage: VARCHAR(100), 标量过滤
    """
    _ensure_connection()
    cfg = get_config().milvus
    dim = cfg.vector_dim
    collection_name = cfg.collection

    if utility.has_collection(collection_name, using=_ALIAS):
        logger.debug(f"Milvus collection 已存在: {collection_name}")
        return

    # 定义 schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="field_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="survey_stage", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="crop_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="growth_stage", dtype=DataType.VARCHAR, max_length=100),
    ]
    schema = CollectionSchema(fields=fields, description="无人机影像语义向量")

    collection = Collection(name=collection_name, schema=schema, using=_ALIAS)

    # 创建向量索引（text_vector 用 HNSW，image_vector 预留）
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256},
    }
    collection.create_index(field_name="text_vector", index_params=index_params)
    collection.create_index(field_name="image_vector", index_params=index_params)

    # 创建标量索引（加速过滤）
    collection.create_index(field_name="task_id", index_name="idx_task_id")
    collection.create_index(field_name="field_name", index_name="idx_field_name")
    collection.create_index(field_name="survey_stage", index_name="idx_survey_stage")
    collection.create_index(field_name="crop_type", index_name="idx_crop_type")
    collection.create_index(field_name="growth_stage", index_name="idx_growth_stage")

    logger.info(f"Milvus collection 已创建: {collection_name} (dim={dim})")


def reset_collection() -> None:
    """删除旧 collection 并用新 schema 重建。

    用于迁移：schema 变更（如新增标量字段）后一次性调用。
    注意：会丢失所有已有向量数据，需确保数据可重新入库。
    """
    _ensure_connection()
    cfg = get_config().milvus
    collection_name = cfg.collection

    if utility.has_collection(collection_name, using=_ALIAS):
        utility.drop_collection(collection_name, using=_ALIAS)
        logger.info(f"Milvus collection 已删除: {collection_name}")

    # 复用 ensure_collection 创建新 schema
    ensure_collection()
    logger.info(f"Milvus collection 已重建: {collection_name}")


async def insert_vector(
    image_id: str,
    text_vector: List[float],
    task_id: str = "",
    field_name: str = "",
    survey_stage: str = "",
    crop_type: str = "",
    growth_stage: str = "",
    image_vector: Optional[List[float]] = None,
) -> None:
    """插入一条向量记录。

    Args:
        image_id: 影像 UUID（字符串）
        text_vector: 文本 embedding 向量
        task_id: 任务编号（标量过滤用）
        field_name: 试验田名称
        survey_stage: 调查阶段
        crop_type: 作物类型（标量过滤用，来自 VLM 结构化输出）
        growth_stage: 生长阶段（标量过滤用，来自 VLM 结构化输出）
        image_vector: 图片 embedding（预留，默认零向量）
    """
    _ensure_connection()
    cfg = get_config().milvus
    dim = cfg.vector_dim

    collection = Collection(cfg.collection, using=_ALIAS)

    # image_vector 预留，暂用零向量
    if image_vector is None:
        image_vector = [0.0] * dim

    data = [
        [image_id],           # id
        [text_vector],        # text_vector
        [image_vector],       # image_vector
        [task_id],            # task_id
        [field_name],         # field_name
        [survey_stage],       # survey_stage
        [crop_type],          # crop_type
        [growth_stage],       # growth_stage
    ]

    collection.insert(data)
    collection.flush()
    logger.debug(f"Milvus 向量已插入: id={image_id}")


# 允许作为 Milvus 标量过滤的字段白名单（防注入）
_FILTERABLE_FIELDS = frozenset({"task_id", "field_name", "survey_stage", "crop_type", "growth_stage"})


def _build_filter_expr(filters: Dict[str, str]) -> Optional[str]:
    """构建 Milvus 过滤表达式（安全版本）。

    - 仅接受白名单内的字段名（防 key 注入）
    - 对值中的双引号做转义（防 value 注入逃逸字符串字面量）
    """
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if not value or key not in _FILTERABLE_FIELDS:
            continue
        # 转义双引号，防止值逃逸出字符串字面量
        safe_value = value.replace('"', "")
        conditions.append(f'{key} == "{safe_value}"')
    return " and ".join(conditions) if conditions else None


async def search_vectors(
    query_vector: List[float],
    top_k: int = 10,
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """向量相似度搜索。

    Args:
        query_vector: 查询向量
        top_k: 返回数量
        filters: 标量过滤条件（如 {"task_id": "xxx"}）

    Returns:
        [{"id": "...", "score": 0.87}, ...] 按相似度降序
    """
    _ensure_connection()
    cfg = get_config().milvus

    collection = Collection(cfg.collection, using=_ALIAS)
    collection.load()

    # 构建过滤表达式（安全版本：字段白名单 + 值转义）
    expr = _build_filter_expr(filters)

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    results = collection.search(
        data=[query_vector],
        anns_field="text_vector",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["task_id", "field_name", "survey_stage"],
    )

    # 解析结果
    items = []
    for hits in results:
        for hit in hits:
            items.append({
                "id": hit.id,
                "score": hit.score,
                "task_id": hit.entity.get("task_id", ""),
                "field_name": hit.entity.get("field_name", ""),
                "survey_stage": hit.entity.get("survey_stage", ""),
            })

    logger.debug(f"Milvus 搜索完成: query_dim={len(query_vector)} results={len(items)}")
    return items


async def delete_vector(image_id: str) -> None:
    """删除指定影像的向量记录"""
    _ensure_connection()
    cfg = get_config().milvus

    collection = Collection(cfg.collection, using=_ALIAS)
    collection.delete(expr=f'id == "{image_id}"')
    collection.flush()
    logger.debug(f"Milvus 向量已删除: id={image_id}")
