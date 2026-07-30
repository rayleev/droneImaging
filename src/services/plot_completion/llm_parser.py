"""LLM 自然语言解析"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional
from loguru import logger


def _load_prompt_template() -> str:
    """加载 LLM 提示词模板"""
    # 从项目根目录查找
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    prompt_path = project_root / "prompts" / "llm_plot_completion.txt"
    
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # fallback 到内联提示词
        return (
            '解析以下农业试验小区描述，提取布局信息。\n'
            '描述："{description}"\n\n'
            '请输出 JSON 格式：\n'
            '- n_rows: 行数（整数或 null）\n'
            '- n_cols: 列数（整数或 null）\n'
            '- fill_mode: "repeat" 或 "fill"\n'
            '- layout: "grid"（网格）或 "along_edge"（沿边缘）或 "custom"\n'
            '- extra: 其他约束说明\n\n'
            '只输出 JSON，不要其他文字。'
        )


def parse_description_regex(description: str) -> dict:
    """正则解析描述（快速、无需 LLM）"""
    result = {
        "n_rows": None,
        "n_cols": None,
        "fill_mode": "repeat",
        "layout": "grid",
    }

    patterns = [
        r"(\d+)\s*[行\u00d7xX*\u00d7]\s*(\d+)\s*列?",
        r"(\d+)\s*行\s*(\d+)\s*列",
        r"横[\u5411\u5411]?\s*(\d+)\s*\u4e2a?.*?\u7eb5[\u5411\u5411]?\s*(\d+)\s*\u4e2a?",
    ]

    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            n1, n2 = int(match.group(1)), int(match.group(2))
            if "\u6a2a" in description[:match.start() + 5]:
                result["n_cols"] = n1
                result["n_rows"] = n2
            else:
                result["n_rows"] = n1
                result["n_cols"] = n2
            break

    if "\u5e03\u6ee1" in description or "\u586b\u6ee1" in description or "\u94fa\u6ee1" in description:
        result["fill_mode"] = "fill"
    elif "\u91cd\u590d" in description or "\u590d\u5236" in description:
        result["fill_mode"] = "repeat"

    return result


async def parse_description_with_llm(description: str, config=None) -> dict:
    """使用 LLM 解析描述（更智能）"""
    from .config import CompletionConfig
    cfg = config or CompletionConfig()

    regex_result = parse_description_regex(description)
    if regex_result["n_rows"] and regex_result["n_cols"]:
        return regex_result

    try:
        import requests

        prompt_template = _load_prompt_template()
        prompt = prompt_template.replace("{description}", description)

        resp = requests.post(
            cfg.llm_api_url,
            json={
                "model": cfg.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=cfg.llm_timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                llm_result = json.loads(content)
                return {**regex_result, **{k: v for k, v in llm_result.items() if v is not None}}
            except json.JSONDecodeError:
                # 尝试从 JSON 代码块中提取
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    try:
                        llm_result = json.loads(json_match.group(1))
                        return {**regex_result, **{k: v for k, v in llm_result.items() if v is not None}}
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        logger.warning(f"LLM parse failed: {e}")

    return regex_result
