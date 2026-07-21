"""VLM 视觉语言模型描述生成模块

调用 Qwen-VL（或可配置的本地模型）对无人机影像缩略图
生成自由文本描述，后续可替换为结构化 JSON 输出。
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
from loguru import logger

from src.config import get_config


async def describe_image(thumbnail_path: str | Path) -> str:
    """对影像缩略图生成 VLM 描述。

    将图片编码为 base64，连同提示词发送给 VLM API，
    返回自由文本描述（200-400 字中文）。

    Args:
        thumbnail_path: 缩略图 JPEG 文件路径

    Returns:
        VLM 生成的描述文本

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
                description = data["choices"][0]["message"]["content"].strip()
                logger.info(f"VLM 描述生成成功: {len(description)} 字")
                return description

        except Exception as e:
            last_error = e
            logger.warning(f"VLM API 调用失败 (attempt {attempt}): {e}")
            if attempt < cfg.max_retries:
                import asyncio
                await asyncio.sleep(2 ** attempt)  # 指数退避

    raise RuntimeError(f"VLM 描述生成失败（{cfg.max_retries} 次重试后）: {last_error}")
