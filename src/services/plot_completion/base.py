"""试验小区智能补全 - 策略基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import uuid


@dataclass
class PlotCell:
    """单个小区"""
    id: str
    label: str
    row: int
    col: int
    bbox: list  # [min_lon, min_lat, max_lon, max_lat]
    polygon: list  # [[[lon, lat], ...]]
    area_m2: float
    status: str = "ok"  # ok / skip（被树遮挡、非试验田等异常）


@dataclass
class CompletionRequest:
    """补全请求"""
    image_id: uuid.UUID
    example_region: Optional[dict] = None  # 用户绘制的示例区域 GeoJSON
    description: str = ""  # 自然语言描述
    image_bbox: Optional[list] = None  # 整幅影像的 bbox [min_lon, min_lat, max_lon, max_lat]
    cog_path: Optional[str] = None  # COG 文件路径（用于 SAM）
    nodata: Optional[float] = None


@dataclass
class CompletionResult:
    """补全结果"""
    image_id: uuid.UUID
    total: int
    n_rows: int
    n_cols: int
    region: dict
    example_size_m: dict
    plots: List[PlotCell]
    debug_info: dict = field(default_factory=dict)  # 调试信息（如 SAM 边界）


class PlotCompletionStrategy(ABC):
    """试验小区补全策略基类"""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """执行智能补全"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """策略描述"""
        ...
