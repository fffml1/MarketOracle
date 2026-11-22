from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ActionPlan(BaseModel):
    """可执行的交易与风控预案"""

    stance: Literal["积极看多", "逢低分批布局", "观望等待确认", "减仓/规避风险"] = Field(
        ..., description="操作立场"
    )
    entry_zone: str = Field(..., description="建议关注/介入价格区间，若观望则写明触发条件")
    target_levels: list[str] = Field(default_factory=list, description="上方阻力/目标价位列表")
    stop_loss_rule: str = Field(..., description="硬性止损价位或触发离场条件")
    position_advice: str = Field(..., description="仓位管理建议（如轻仓试探/空仓观察/持仓降仓）")


class DebateSynthesis(BaseModel):
    """多空辩论要点归纳"""

    consensus_points: list[str] = Field(default_factory=list, description="多空双方已达成共识的客观事实")
    unresolved_conflicts: list[str] = Field(default_factory=list, description="尚未解决的核心分歧点")
    bull_strongest_argument: str = Field(..., description="多头最具说服力的核心论据（附证据ID）")
    bear_strongest_argument: str = Field(..., description="空头最具杀伤力的核心质疑（附证据ID）")


class FinalResearchReport(BaseModel):
    """单股深度研究与多Agent辩论最终决策报告"""

    symbol: str
    as_of_date: str
    headline_verdict: str = Field(..., description="一句话核心决策结论")
    rating: Literal["强烈推荐", "偏多关注", "中性观望", "谨慎减持", "回避"]
    confidence_score: int = Field(..., ge=0, le=100, description="决策置信度 0-100（若存在数据降级需相应扣分）")
    confidence_rationale: str = Field(..., description="置信度评分依据（含数据完整度说明）")

    executive_summary: str = Field(..., description="200字以内的核心逻辑综述")
    debate_synthesis: DebateSynthesis
    action_plan: ActionPlan
    key_risk_warnings: list[str] = Field(default_factory=list, description="核心风险红线列表（附证据ID）")
    cited_evidence_ids: list[str] = Field(default_factory=list, description="报告引用的全量证据ID")

    # 审计与元数据字段（由程序自动填充）
    is_data_degraded: bool = False
    reflection_passed: bool = True
    reflection_notes: str = ""

    def to_markdown(self) -> str:
        """渲染为适合终端、飞书/企微推送或前端展示的 Markdown 决策卡片"""
        deg_banner = (
            "\n> ⚠️ **数据降级提示**：本次调研部分数据源触发降级或覆盖受限，置信度已作折价处理。\n"
            if self.is_data_degraded
            else ""
        )
        consensus_md = "\n".join(f"- {c}" for c in self.debate_synthesis.consensus_points) or "- 无"
        conflict_md = "\n".join(f"- {c}" for c in self.debate_synthesis.unresolved_conflicts) or "- 无"
        targets_str = " / ".join(self.action_plan.target_levels) if self.action_plan.target_levels else "暂无明确目标"
        risks_md = "\n".join(f"- {r}" for r in self.key_risk_warnings) or "- 暂无"
        cites_str = ", ".join(self.cited_evidence_ids) if self.cited_evidence_ids else "无"

        return (
            f"# 投研决策报告：{self.symbol}（基准日 {self.as_of_date}）\n"
            f"{deg_banner}\n"
            f"**核心结论**：{self.headline_verdict}\n"
            f"**综合评级**：`{self.rating}` | **置信度**：`{self.confidence_score}/100`（{self.confidence_rationale}）\n\n"
            f"## 一、 决策综述\n{self.executive_summary}\n\n"
            f"## 二、 多空辩论核心复盘\n"
            f"- **多头最强论据**：{self.debate_synthesis.bull_strongest_argument}\n"
            f"- **空头最强质疑**：{self.debate_synthesis.bear_strongest_argument}\n\n"
            f"**双方共识**：\n{consensus_md}\n\n"
            f"**未决核心分歧**：\n{conflict_md}\n\n"
            f"## 三、 交易与风控预案\n"
            f"- **操作立场**：{self.action_plan.stance}\n"
            f"- **仓位建议**：{self.action_plan.position_advice}\n"
            f"- **关注/介入区间**：{self.action_plan.entry_zone}\n"
            f"- **目标/阻力位**：{targets_str}\n"
            f"- **硬止损/离场条件**：{self.action_plan.stop_loss_rule}\n\n"
            f"## 四、 核心风险红线\n{risks_md}\n\n"
            f"---\n"
            f"*核查状态：{'通过' if self.reflection_passed else '已修订'}（{self.reflection_notes}） | "
            f"溯源证据：[{cites_str}]*"
        )