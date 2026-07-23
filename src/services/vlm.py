"""VLM 视觉语言模型描述生成模块

调用可配置的 VLM（默认 Qwen-VL）对无人机影像缩略图生成结构化输出：
- 自然语言摘要（summary）：用于 embedding 和人工阅读
- 结构化字段（crop_type、growth_stage、canopy_coverage 等）：用于过滤和展示
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.config import get_config


# ── 结构化输出模型 ────────────────────────────────────────────


class VLMOutput(BaseModel):
    """VLM 结构化输出的 Pydantic 校验模型。"""

    summary: str = Field(..., min_length=10, description="自然语言摘要")
    crop_type: Optional[str] = Field(None, description="作物类型 ISO 代码")
    growth_stage: Optional[str] = Field(None, description="生长阶段 BBCH 代码或中文")
    canopy_coverage: Optional[int] = Field(None, ge=0, le=100, description="冠层覆盖度 0-100")
    color_features: List[str] = Field(default_factory=list)
    anomalies: List[dict] = Field(default_factory=list)
    management_traces: List[str] = Field(default_factory=list)
    image_quality: Optional[str] = Field(None, description="high/medium/low/unusable")
    shooting_angle: Optional[str] = Field(None, description="nadir/oblique/unknown")

    @field_validator("image_quality")
    @classmethod
    def _validate_quality(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"high", "medium", "low", "unusable"}
        if v not in allowed:
            raise ValueError(f"image_quality 必须是 {allowed} 之一，得到: {v}")
        return v

    @field_validator("shooting_angle")
    @classmethod
    def _validate_angle(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"nadir", "oblique", "unknown"}
        if v not in allowed:
            raise ValueError(f"shooting_angle 必须是 {allowed} 之一，得到: {v}")
        return v


@dataclass
class VLMResult:
    """VLM 解析结果。"""

    summary: str
    crop_type: Optional[str] = None
    growth_stage: Optional[str] = None
    canopy_coverage: Optional[int] = None
    color_features: List[str] = field(default_factory=list)
    anomalies: List[dict] = field(default_factory=list)
    management_traces: List[str] = field(default_factory=list)
    image_quality: Optional[str] = None
    shooting_angle: Optional[str] = None
    raw: dict = field(default_factory=dict)
    parsed_ok: bool = True  # False 表示降级为纯文本 summary


# ── JSON 提取工具 ──────────────────────────────────────────────


def _extract_json(text: str) -> Optional[dict]:
    """从 VLM 返回文本中提取 JSON 对象。

    处理以下情况：
    - 纯 JSON
    - markdown 代码块包裹（```json ... ```）
    - 前后有多余文本
    - 非法输入（返回 None）
    """
    if not text:
        return None

    # 1. 尝试直接解析
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 2. 尝试从 markdown 代码块提取
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        try:
            data = json.loads(code_block.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # 3. 尝试找到第一个 { 到最后一个 }
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


def _fallback_result(raw_text: str, json_data: Optional[dict] = None) -> VLMResult:
    """解析失败时的降级结果。

    优先从已解析 JSON 中提取 summary 字段（即使其他字段校验失败），
    否则把原始文本作为 summary。
    """
    # JSON 解析成功但校验失败时，尽量提取 summary 避免存 JSON 原文
    if json_data is not None:
        summary = json_data.get("summary") or raw_text.strip()
    else:
        summary = raw_text.strip()
    return VLMResult(
        summary=summary,
        raw=json_data or {},
        parsed_ok=False,
    )


# ── 主函数 ─────────────────────────────────────────────────────


async def describe_image(thumbnail_path: str | Path) -> VLMResult:
    """对影像缩略图生成 VLM 结构化描述。

    将图片编码为 base64，连同提示词发送给 VLM API，
    解析返回的 JSON 为结构化 VLMResult。

    解析失败时会重试一次（在 max_retries 内），
    仍失败则降级：把原始文本作为 summary，结构化字段置空。

    Args:
        thumbnail_path: 缩略图 JPEG 文件路径

    Returns:
        VLMResult 结构化结果

    Raises:
        RuntimeError: API 调用失败（重试耗尽后）
    """
    cfg = get_config().vlm
    thumbnail_path = Path(thumbnail_path)

    # 读取提示词
    prompt_file = Path(cfg.prompt_file)
    if not prompt_file.is_absolute():
        from src.config import BASE_DIR

        prompt_file = BASE_DIR / prompt_file
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()

    # 图片 → base64
    image_b64 = base64.b64encode(thumbnail_path.read_bytes()).decode("utf-8")

    # 构建 OpenAI 兼容格式请求
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]

    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }

    # 带重试的 API 调用
    last_error = None
    raw_text = ""
    for attempt in range(1, cfg.max_retries + 1):
        try:
            logger.info(f"VLM 描述生成 (attempt {attempt}/{cfg.max_retries}): {cfg.model}")
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                resp = await client.post(
                    f"{cfg.api_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"].strip()
                logger.info(f"VLM 描述生成成功: {len(raw_text)} 字")

                # 尝试解析 JSON
                json_data = _extract_json(raw_text)
                if json_data is not None:
                    # 处理非农田场景
                    if "error" in json_data and json_data["error"] == "non-agricultural scene":
                        logger.warning("VLM 判定为非农田场景")
                        return VLMResult(
                            summary=json_data.get("summary", raw_text),
                            raw=json_data,
                            parsed_ok=True,
                        )
                    try:
                        output = VLMOutput(**json_data)
                        return VLMResult(
                            summary=output.summary,
                            crop_type=output.crop_type,
                            growth_stage=output.growth_stage,
                            canopy_coverage=output.canopy_coverage,
                            color_features=output.color_features,
                            anomalies=output.anomalies,
                            management_traces=output.management_traces,
                            image_quality=output.image_quality,
                            shooting_angle=output.shooting_angle,
                            raw=json_data,
                            parsed_ok=True,
                        )
                    except Exception as validate_err:
                        logger.warning(f"VLM 输出校验失败 (attempt {attempt}): {validate_err}")
                        # 校验失败，继续重试
                        last_error = validate_err
                        if attempt < cfg.max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        # 重试耗尽，降级（尽量从 JSON 提取 summary）
                        return _fallback_result(raw_text, json_data=json_data)
                else:
                    logger.warning(f"VLM 输出非 JSON 格式 (attempt {attempt})")
                    # 非 JSON，继续重试
                    if attempt < cfg.max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # 重试耗尽，降级
                    return _fallback_result(raw_text)

        except Exception as e:
            last_error = e
            logger.warning(f"VLM API 调用失败 (attempt {attempt}): {e}")
            if attempt < cfg.max_retries:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"VLM 描述生成失败（{cfg.max_retries} 次重试后）: {last_error}")
