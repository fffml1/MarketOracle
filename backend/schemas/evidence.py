from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceCategory(str, Enum):
    """证据所属维度，便于调研员与辩手按图索骥"""

    TECHNICAL = "technical"        # 量价、均线、技术指标、资金流
    FUNDAMENTAL = "fundamental"    # 财务指标、业绩预告、主营业务
    NEWS_CATALYST = "news_catalyst"# 新闻、行业政策、产业链催化
    RISK_EVENT = "risk_event"      # 减持、质押、解禁、立案/问询/负面舆情


class EvidenceItem(BaseModel):
    """单条可追溯证据"""

    evidence_id: str = Field(..., description="全局唯一证据编号，如 EV_TECH_001")
    category: EvidenceCategory = Field(..., description="证据类别")
    symbol: str = Field(..., description="股票代码，如 300750.SZ 或 600519.SH")
    title: str = Field(..., description="证据短标题，便于辩论时快速定位")
    summary: str = Field(..., description="核心事实摘要（含关键数值与日期）")
    payload: dict[str, Any] = Field(default_factory=dict, description="原始结构化数值或明细")

    source: str = Field(..., description="数据源名称，如 akshare_kline / eastmoney_notice")
    published_at: str | None = Field(default=None, description="事实发生/发布时间（ISO 或 YYYY-MM-DD）")
    fetched_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"),
        description="抓取时间戳",
    )
    is_degraded: bool = Field(default=False, description="该条证据是否为降级/兜底结果")
    degrade_reason: str | None = Field(default=None, description="降级原因说明")


class EvidenceBundle(BaseModel):
    """单次工具调用返回的证据集合"""

    status: Literal["ok", "partial", "error"] = Field(default="ok")
    tool_name: str
    symbol: str
    as_of_date: str = Field(..., description="研究基准日期 YYYY-MM-DD")
    evidences: list[EvidenceItem] = Field(default_factory=list)
    is_degraded: bool = Field(default=False)
    degrade_reason: str | None = Field(default=None)
    error_message: str | None = Field(default=None)

    def to_prompt_text(self) -> str:
        """将证据集合格式化为适合喂给 LLM 的文本块，保留 evidence_id"""
        if self.status == "error" and not self.evidences:
            return (
                f"[{self.tool_name}] 基准日 {self.as_of_date} 取数失败: "
                f"{self.error_message or '未知错误'} (降级={self.is_degraded})"
            )

        lines: list[str] = [
            f"=== 工具: {self.tool_name} | 标的: {self.symbol} | 基准日: {self.as_of_date} "
            f"| 状态: {self.status} | 降级: {self.is_degraded} ==="
        ]
        if self.degrade_reason:
            lines.append(f"[降级提示] {self.degrade_reason}")

        for item in self.evidences:
            deg_tag = f" [降级: {item.degrade_reason or '部分字段缺失'}]" if item.is_degraded else ""
            pub_tag = f" | 时间: {item.published_at}" if item.published_at else ""
            lines.append(
                f"- [{item.evidence_id}] ({item.category.value}{pub_tag} | 来源: {item.source}){deg_tag}\n"
                f"  标题: {item.title}\n"
                f"  摘要: {item.summary}"
            )
        return "\n".join(lines)