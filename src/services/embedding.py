"""文本 Embedding 模块

将 VLM 描述 + 业务元数据拼接后调用 embedding API，
生成固定维度的文本向量，供 Milvus 检索使用。
"""

from __future__ import annotations

import asyncio
from typing import List

import httpx
from loguru import logger

from src.config import get_config


async def _embed_with_retry(payload: dict, max_retries: int = 3) -> dict:
    """调用 embedding API，遇到 429/5xx 时指数退避重试。"""
    cfg = get_config().embedding
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["X-API-Key"] = cfg.api_key

    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(cfg.api_url, json=payload, headers=headers)
                if resp.status_code == 429:
                    # 配额/限流，退避后重试
                    logger.warning(f"Embedding 429 限流，第 {attempt} 次重试（退避 {backoff:.0f}s）")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                if resp.status_code >= 500:
                    logger.warning(f"Embedding {resp.status_code} 服务端错误，第 {attempt} 次重试")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
        except httpx.RequestError as e:
            if attempt == max_retries:
                raise
            logger.warning(f"Embedding 网络错误 {e}，第 {attempt} 次重试")
            await asyncio.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Embedding API 在 {max_retries} 次重试后仍失败（可能是日配额用尽）")


async def embed_text(text: str) -> List[float]:
    """将文本转为向量。

    Args:
        text: 输入文本

    Returns:
        浮点数向量（维度由模型决定，默认 1024）

    Raises:
        RuntimeError: embedding API 调用失败
    """
    payload = {"texts": [text]}
    try:
        data = await _embed_with_retry(payload)

        # 自定义格式: {"embeddings": [[...]], "dimension": N, ...}
        if "embeddings" in data and isinstance(data["embeddings"], list):
            vector = data["embeddings"][0]
        # 兼容 OpenAI 格式
        elif "data" in data and isinstance(data["data"], list):
            vector = data["data"][0]["embedding"]
        else:
            raise ValueError(f"未知的 embedding 响应格式: {list(data.keys())}")

        logger.debug(f"Embedding 生成成功: dim={len(vector)}")
        return vector

    except Exception as e:
        logger.error(f"Embedding API 调用失败: {e}")
        raise RuntimeError(f"文本向量化失败: {e}")


async def embed_batch(texts: List[str]) -> List[List[float]]:
    """批量文本向量化。

    Args:
        texts: 文本列表

    Returns:
        向量列表
    """
    payload = {"texts": texts}
    try:
        data = await _embed_with_retry(payload)

        if "embeddings" in data and isinstance(data["embeddings"], list):
            vectors = data["embeddings"]
        elif "data" in data and isinstance(data["data"], list):
            vectors = [item["embedding"] for item in data["data"]]
        else:
            raise ValueError(f"未知的批量 embedding 响应格式")

        logger.debug(f"批量 Embedding 生成成功: count={len(vectors)}")
        return vectors

    except Exception as e:
        logger.error(f"批量 Embedding API 调用失败: {e}")
        raise RuntimeError(f"批量文本向量化失败: {e}")
