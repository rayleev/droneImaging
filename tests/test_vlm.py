"""VLM 服务单元测试"""

import pytest

from src.services.vlm import (
    VLMOutput,
    VLMResult,
    _extract_json,
    _fallback_result,
)


# ── _extract_json 测试 ─────────────────────────────────────────


class TestExtractJson:
    """测试 _extract_json 工具函数。"""

    def test_pure_json(self):
        """纯 JSON 字符串。"""
        text = '{"crop_type": "ZM", "summary": "测试摘要"}'
        result = _extract_json(text)
        assert result == {"crop_type": "ZM", "summary": "测试摘要"}

    def test_markdown_code_block(self):
        """markdown 代码块包裹的 JSON。"""
        text = '```json\n{"crop_type": "WH", "summary": "小麦田"}\n```'
        result = _extract_json(text)
        assert result == {"crop_type": "WH", "summary": "小麦田"}

    def test_markdown_code_block_without_json_hint(self):
        """markdown 代码块（无 json 提示）包裹的 JSON。"""
        text = '```\n{"crop_type": "RI"}\n```'
        result = _extract_json(text)
        assert result == {"crop_type": "RI"}

    def test_json_with_surrounding_text(self):
        """前后有多余文本的 JSON。"""
        text = '这是分析结果\n{"crop_type": "SB"}\n谢谢'
        result = _extract_json(text)
        assert result == {"crop_type": "SB"}

    def test_empty_string(self):
        """空字符串返回 None。"""
        assert _extract_json("") is None

    def test_no_json(self):
        """无 JSON 内容返回 None。"""
        assert _extract_json("这是纯文本，没有 JSON") is None

    def test_invalid_json(self):
        """非法 JSON 返回 None。"""
        assert _extract_json("{invalid json}") is None

    def test_nested_json(self):
        """嵌套 JSON 对象。"""
        text = '{"anomalies": [{"type": "病虫害", "location": "西北"}], "summary": "测试"}'
        result = _extract_json(text)
        assert result["anomalies"] == [{"type": "病虫害", "location": "西北"}]


# ── VLMOutput 校验测试 ─────────────────────────────────────────


class TestVLMOutput:
    """测试 VLMOutput Pydantic 校验模型。"""

    def test_valid_full(self):
        """完整合法数据。"""
        data = {
            "summary": "这是一块玉米田的正射影像，处于 5-6 叶期。",
            "crop_type": "ZM",
            "growth_stage": "15",
            "canopy_coverage": 75,
            "color_features": ["绿色主导"],
            "anomalies": [],
            "management_traces": ["灌溉行"],
            "image_quality": "high",
            "shooting_angle": "nadir",
        }
        output = VLMOutput(**data)
        assert output.crop_type == "ZM"
        assert output.canopy_coverage == 75

    def test_valid_minimal(self):
        """最小必填数据（只有 summary）。"""
        output = VLMOutput(summary="这是一个测试摘要内容")
        assert output.summary == "这是一个测试摘要内容"
        assert output.crop_type is None
        assert output.canopy_coverage is None
        assert output.color_features == []

    def test_canopy_coverage_boundary(self):
        """canopy_coverage 边界值。"""
        summary = "这是一段用于测试的摘要文本"
        assert VLMOutput(summary=summary, canopy_coverage=0).canopy_coverage == 0
        assert VLMOutput(summary=summary, canopy_coverage=100).canopy_coverage == 100

    def test_canopy_coverage_out_of_range(self):
        """canopy_coverage 越界应失败。"""
        with pytest.raises(Exception):
            VLMOutput(summary="t", canopy_coverage=101)
        with pytest.raises(Exception):
            VLMOutput(summary="t", canopy_coverage=-1)

    def test_invalid_image_quality(self):
        """非法 image_quality 枚举值应失败。"""
        with pytest.raises(Exception):
            VLMOutput(summary="t", image_quality="excellent")

    def test_invalid_shooting_angle(self):
        """非法 shooting_angle 枚举值应失败。"""
        with pytest.raises(Exception):
            VLMOutput(summary="t", shooting_angle="side")

    def test_summary_too_short(self):
        """summary 太短应失败（min_length=10）。"""
        with pytest.raises(Exception):
            VLMOutput(summary="短")


# ── _fallback_result 测试 ──────────────────────────────────────


class TestFallbackResult:
    """测试 _fallback_result 降级函数。"""

    def test_fallback_preserves_text(self):
        """降级结果应保留原始文本作为 summary。"""
        result = _fallback_result("原始 VLM 输出文本")
        assert result.summary == "原始 VLM 输出文本"
        assert result.parsed_ok is False
        assert result.crop_type is None

    def test_fallback_strips_whitespace(self):
        """降级结果应去除首尾空白。"""
        result = _fallback_result("  文本  \n")
        assert result.summary == "文本"


# ── VLMResult dataclass 测试 ───────────────────────────────────


class TestVLMResult:
    """测试 VLMResult dataclass 默认值。"""

    def test_default_values(self):
        """默认值正确。"""
        result = VLMResult(summary="测试")
        assert result.crop_type is None
        assert result.canopy_coverage is None
        assert result.color_features == []
        assert result.anomalies == []
        assert result.management_traces == []
        assert result.parsed_ok is True
        assert result.raw == {}
