"""自然语言查询分解模块

将自然语言查询分解为三部分：
- text_query：走向量的文本语义维度
- filters：结构化等值/范围过滤（mission_id / plot_id / flight_date / product_type 等）
- trait_filters：性状观测值过滤（两阶段：先查 agri_research 得 plot_id 再过滤）

LLM 不可用时降级：整句当纯向量文本，过滤为空，返回 decomposed=false。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from src.config import get_config
from src.schemas.image import ImageSearchFilters, TraitFilter


_DECOMPOSE_PROMPT = """你是一个无人机农业影像检索系统的查询分解器。
把用户的自然语言查询分解为结构化检索条件，输出严格的 JSON。

可用的结构化过滤字段（仅这些字段可填，其他一律留空）：
- crop_type: 作物类型（如 水稻/玉米/小麦，或 ISO 代码 rice/ZM/wheat）
- growth_stage: 生育期（如 分蘖期/抽穗期/灌浆期，或 BBCH 代码）
- mission_id / experiment_id / site_id / plot_id: UUID（用户通常不会直接给，留空）
- plot_code: 小区编号（如 P01、P02）
- flight_date: 飞行日期 YYYY-MM-DD（等值）
- flight_date_from / flight_date_to: 日期范围 YYYY-MM-DD
- product_type: rgb（可见光）/ ms（多光谱）
- variety: 品种名
- treatment: 处理（如 干旱/对照/氮处理）
- data_source_type: remote_sensing / model_prediction / manual

性状过滤（trait_filters 数组，每个元素 {trait_code, op, value}）：
- trait_code 用大写英文代码，常见：ROLLING_LEAF_SCORE(卷叶评分) / TOTAL_BIOMASS(总生物量) /
  ABOVEGROUND_BIOMASS(地上部生物量) / TOTAL_WATER_CONTENT(总含水量) /
  WATER_CONTENT(植株相对含水量) / WATER_CONTENT_PER_AREA(单位面积含水量) / BIOMASS_DENSITY(生物量密度)
- op: gt / gte / lt / lte / eq
- value: 数字

text_query：去掉结构化条件后剩下的语义描述（用于向量召回，如"叶片卷曲严重"、"长势稀疏"）。
若用户查询完全是结构化条件没有语义部分，text_query 填空字符串。

只输出 JSON，不要任何解释。格式：
{"text_query": "...", "filters": {...}, "trait_filters": [{"trait_code":"...","op":"...","value":0}]}
"""


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 返回中提取 JSON 对象。"""
    if not text:
        return None
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        try:
            data = json.loads(code_block.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


async def decompose_query(nl_query: str) -> Dict[str, Any]:
    """用 LLM 分解自然语言查询。

    Returns:
        {
            "text_query": str,
            "filters": ImageSearchFilters | None,
            "trait_filters": List[TraitFilter] | None,
            "decomposed": bool,
        }
    """
    cfg = get_config().llm
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": _DECOMPOSE_PROMPT},
            {"role": "user", "content": nl_query},
        ],
        "max_tokens": 800,
        "temperature": 0.1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.post(
                f"{cfg.api_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"查询分解 LLM 调用失败，降级为纯向量召回: {e}")
        return {
            "text_query": nl_query,
            "filters": None,
            "trait_filters": None,
            "decomposed": False,
        }

    parsed = _extract_json(content)
    if parsed is None:
        logger.warning(f"查询分解 LLM 返回非 JSON，降级: {content[:200]}")
        return {
            "text_query": nl_query,
            "filters": None,
            "trait_filters": None,
            "decomposed": False,
        }

    # 解析 filters
    filters: Optional[ImageSearchFilters] = None
    raw_filters = parsed.get("filters") or {}
    if isinstance(raw_filters, dict) and any(v for v in raw_filters.values()):
        try:
            filters = ImageSearchFilters(**{
                k: v for k, v in raw_filters.items()
                if k in ImageSearchFilters.model_fields and v
            })
        except Exception as e:
            logger.warning(f"filters 解析失败: {e}")

    # 解析 trait_filters
    trait_filters: Optional[List[TraitFilter]] = None
    raw_tf = parsed.get("trait_filters") or []
    if isinstance(raw_tf, list) and raw_tf:
        parsed_tf: List[TraitFilter] = []
        for item in raw_tf:
            if not isinstance(item, dict):
                continue
            try:
                parsed_tf.append(TraitFilter(
                    trait_code=item.get("trait_code", ""),
                    op=item.get("op", "gte"),
                    value=float(item.get("value")),
                ))
            except Exception:
                continue
        if parsed_tf:
            trait_filters = parsed_tf

    text_query = parsed.get("text_query") or nl_query
    return {
        "text_query": text_query,
        "filters": filters,
        "trait_filters": trait_filters,
        "decomposed": True,
    }
