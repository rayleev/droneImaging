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

    logger.info(f"Milvus collection 已创建: {collection_name} (dim={dim})")


async def insert_vector(
    image_id: str,
    text_vector: List[float],
    task_id: str = "",
    field_name: str = "",
    survey_stage: str = "",
    image_vector: Optional[List[float]] = None,
) -> None:
    """插入一条向量记录。

    Args:
        image_id: 影像 UUID（字符串）
        text_vector: 文本 embedding 向量
        task_id: 任务编号（标量过滤用）
        field_name: 试验田名称
        survey_stage: 调查阶段
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
    ]

    collection.insert(data)
    collection.flush()
    logger.debug(f"Milvus 向量已插入: id={image_id}")


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

    # 构建过滤表达式
    expr = None
    if filters:
        conditions = []
        for key, value in filters.items():
            if value:
                conditions.append(f'{key} == "{value}"')
        if conditions:
            expr = " and ".join(conditions)

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
