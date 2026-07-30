"""补全策略配置管理"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.config import get_config


@dataclass
class CompletionConfig:
    """补全服务配置"""
    
    # 策略选择: "sam_llm" 或 "vlm"
    strategy: str = "sam_llm"
    
    # SAM 远程服务配置
    sam_service_url: str = "http://127.0.0.1:8003"
    sam_service_timeout: float = 60.0
    
    # SAM 模型配置（本地模式，与 sam_service 互斥）
    sam_model_type: str = "vit_h"
    sam_checkpoint: Optional[str] = None
    
    # LLM 配置（用于 sam_llm 策略）
    llm_api_url: str = ""
    llm_model: str = ""
    llm_timeout: float = 30.0
    
    # VLM 配置（用于 vlm 策略）
    vlm_api_url: Optional[str] = None
    vlm_model: str = ""
    vlm_timeout: float = 60.0
    
    # 输出配置
    return_debug_info: bool = True
    
    # 缓存配置
    cache_sam_results: bool = True


def get_completion_config() -> CompletionConfig:
    """从主配置加载，环境变量覆盖
    
    Fallback 逻辑：
    - completion.llm 为 null → 使用顶层 llm
    - completion.vlm 为 null → 使用顶层 vlm
    """
    cfg = get_config()
    comp_cfg = cfg.completion
    
    # 解析 LLM 配置（fallback 到顶层）
    resolved_llm = comp_cfg.resolve_llm(cfg.llm)
    llm_api_url = os.getenv("LLM_API_URL", resolved_llm.api_url)
    llm_model = os.getenv("LLM_MODEL", resolved_llm.model)
    llm_timeout = float(os.getenv("LLM_TIMEOUT", str(resolved_llm.timeout)))
    
    # 解析 VLM 配置（fallback 到顶层）
    resolved_vlm = comp_cfg.resolve_vlm(cfg.vlm)
    vlm_api_url = os.getenv("VLM_API_URL", resolved_vlm.api_url)
    vlm_model = os.getenv("VLM_MODEL", resolved_vlm.model)
    vlm_timeout = float(os.getenv("VLM_TIMEOUT", str(resolved_vlm.timeout)))
    
    return CompletionConfig(
        strategy=os.getenv("COMPLETION_STRATEGY", comp_cfg.strategy),
        sam_service_url=os.getenv("SAM_SERVICE_URL", comp_cfg.sam_service.url),
        sam_service_timeout=float(os.getenv("SAM_SERVICE_TIMEOUT", str(comp_cfg.sam_service.timeout))),
        sam_model_type=os.getenv("SAM_MODEL_TYPE", comp_cfg.sam.model_type),
        sam_checkpoint=os.getenv("SAM_CHECKPOINT", comp_cfg.sam.checkpoint),
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
        vlm_api_url=vlm_api_url,
        vlm_model=vlm_model,
        vlm_timeout=vlm_timeout,
        return_debug_info=os.getenv("RETURN_DEBUG_INFO", "true").lower() == "true",
        cache_sam_results=os.getenv("CACHE_SAM_RESULTS", "true").lower() == "true",
    )
