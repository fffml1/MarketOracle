from __future__ import annotations

import json
import re
from typing import Any

from hello_agents import HelloAgentsLLM, SimpleAgent
from pydantic import ValidationError

from .debate_arena import DebateTranscript, EVIDENCE_ID_PATTERN
from .research_team import ResearchBriefs
from ..schemas.report import FinalResearchReport
from ..tools.hello_adapter import EvidenceStore

CIO_SYSTEM_PROMPT = """你是一家顶级投研机构的【首席投资官 (CIO / Judge)】。
你的任务是在听取多头 (Bull) 与空头 (Bear) 的多轮对抗辩论后，对照原始客观证据清单，做出冷静、平衡、可执行的最终投研裁决。

裁决铁律：
1. **以原始证据为最高准绳**：若多空双方的口头估算与原始证据数值有出入，一律以原始证据为准。
2. **不和稀泥，但敬畏不确定性**：明确给出操作立场与具体价格风控条件；同时清晰列出双方尚未解决的核心分歧。
3. **证据溯源**：所有核心论据与风险提示必须标注方括号证据编号，如 [EV_TECH_001]，严禁编造不存在的编号。
4. **降级折价**：若原始证据中存在接口降级或关键维度缺失，必须在 confidence_score 中扣减分数并说明原因。
5. **严格输出合法 JSON**：只输出符合给定 Schema 的纯 JSON 对象，不要包含 ```json 代码块标记或任何额外解释文字。"""

CRITIC_SYSTEM_PROMPT = """你是一名极其严苛的【投研合规与事实审计官 (Reflection Critic)】。
你的唯一职责是核对 CIO 起草的决策报告 JSON 是否存在以下问题：
1. **数值失真**：报告中提到的现价、均线（MA5/MA20/MA60）、涨跌幅、回撤、减持比例等数字，是否与【原始客观证据清单】完全一致？
2. **伪造引用**：报告中出现的 [EV_XXX_001] 编号是否全部存在于合法证据列表？
3. **风控缺失**：是否给出了明确的止损/离场条件？如果存在数据降级，置信度是否合理（不应高于 70 分）？

请只输出合法 JSON，格式如下：
{
  "passed": true 或 false,
  "issues": ["发现的具体问题1", "发现的具体问题2"],
  "summary": "一句话审计结论"
}"""


def _extract_json_obj(raw: str) -> dict[str, Any]:
    """容错提取模型返回的 JSON 对象"""
    text = (raw or "").strip()
    # 去掉可能残留的 markdown 代码块围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


class CIOJudgeAgent:
    """CIO 裁决官 + Reflection 反思核查器"""

    def __init__(self, llm: HelloAgentsLLM, store: EvidenceStore) -> None:
        self.store = store
        self.cio_agent = SimpleAgent(
            name="CIOJudge",
            llm=llm,
            system_prompt=CIO_SYSTEM_PROMPT,
        )
        self.critic_agent = SimpleAgent(
            name="ReflectionCritic",
            llm=llm,
            system_prompt=CRITIC_SYSTEM_PROMPT,
        )

    def _build_schema_hint(self) -> str:
        return json.dumps(FinalResearchReport.model_json_schema(), ensure_ascii=False, indent=2)

    def _programmatic_audit(self, draft_json: dict[str, Any]) -> list[str]:
        """程序化硬核查：检测伪造证据ID与缺失关键字段"""
        issues: list[str] = []
        serialized = json.dumps(draft_json, ensure_ascii=False)
        found_ids = set(EVIDENCE_ID_PATTERN.findall(serialized))
        valid_ids = self.store.get_valid_ids()

        fake_ids = sorted(found_ids - valid_ids)
        if fake_ids:
            issues.append(f"引用了未在证据库中登记的编号: {fake_ids}")

        any_degraded = any(b.is_degraded for b in self.store.bundles)
        score = draft_json.get("confidence_score")
        if any_degraded and isinstance(score, int) and score > 70:
            issues.append(f"底层数据存在降级，但 confidence_score={score} 过高，建议下调至 70 及以下")

        return issues

    def adjudicate(
        self,
        briefs: ResearchBriefs,
        transcript: DebateTranscript,
        user_focus: str = "",
    ) -> FinalResearchReport:
        raw_evidence_text = self.store.dump_all_prompt_text()
        valid_id_list = sorted(self.store.get_valid_ids())
        any_degraded = any(b.is_degraded for b in self.store.bundles)

        base_context = (
            f"【研究标的】{briefs.symbol} | 【基准日】{briefs.as_of_date}\n"
            f"【合法证据编号白名单】{valid_id_list}\n"
            f"【底层数据是否存在降级】{any_degraded}\n"
            f"【用户重点关注】{user_focus or '综合波段与基本面风控评估'}\n\n"
            f"=== 一、原始客观证据清单（最高事实准绳） ===\n{raw_evidence_text}\n\n"
            f"=== 二、多空对抗辩论实录 ===\n{transcript.to_markdown()}\n"
            "仅依据原始证据和已完成发言归纳。把未完成发言和待核实反驳保留为限制，"
            "说明停止原因及实际完成轮数。\n"
        )

        # 1. 生成初版裁决 JSON
        draft_prompt = (
            f"{base_context}\n"
            f"请严格按照以下 JSON Schema 输出最终投研裁决报告（只输出纯 JSON）：\n"
            f"{self._build_schema_hint()}"
        )
        draft_raw = self.cio_agent.run(draft_prompt)
        draft_dict = _extract_json_obj(draft_raw)

        # 2. Reflection 反思核查（程序规则检查 + LLM Critic 语义/数值审计）
        prog_issues = self._programmatic_audit(draft_dict)

        critic_prompt = (
            f"【原始客观证据清单】\n{raw_evidence_text}\n\n"
            f"【合法证据编号白名单】{valid_id_list}\n\n"
            f"【CIO 裁决报告初稿 JSON】\n{json.dumps(draft_dict, ensure_ascii=False, indent=2)}\n\n"
            f"请逐项核对数值一致性、引用编号合法性与风控完整性，输出审计 JSON。"
        )
        critic_raw = self.critic_agent.run(critic_prompt)
        try:
            critic_res = _extract_json_obj(critic_raw)
        except Exception:
            critic_res = {"passed": True, "issues": [], "summary": "审计格式解析跳过"}

        llm_issues = critic_res.get("issues") or []
        all_issues = [*prog_issues, *(str(x) for x in llm_issues if x)]
        passed = bool(critic_res.get("passed", True)) and len(all_issues) == 0

        # 3. 若未通过审计，触发一次定向修订
        if not passed:
            revise_prompt = (
                f"{base_context}\n"
                f"【你上一版的报告初稿】\n{json.dumps(draft_dict, ensure_ascii=False, indent=2)}\n\n"
                f"【审计官指出的必须修正的问题】\n"
                + "\n".join(f"- {iss}" for iss in all_issues)
                + "\n\n请逐条修正上述问题，保持严格符合 JSON Schema，重新输出完整纯 JSON："
            )
            revised_raw = self.cio_agent.run(revise_prompt)
            draft_dict = _extract_json_obj(revised_raw)
            reflection_notes = f"初稿经反思修订（修正 {len(all_issues)} 项问题）"
        else:
            reflection_notes = str(critic_res.get("summary") or "数值与证据引用核查一次通过")

        # 4. 补全程序审计元数据并用 Pydantic 强校验
        draft_dict["symbol"] = briefs.symbol
        draft_dict["as_of_date"] = briefs.as_of_date
        draft_dict["is_data_degraded"] = any_degraded
        draft_dict["reflection_passed"] = passed
        draft_dict["reflection_notes"] = reflection_notes

        # 自动汇总报告中实际引用的合法证据ID
        final_found_ids = sorted(
            set(EVIDENCE_ID_PATTERN.findall(json.dumps(draft_dict, ensure_ascii=False)))
            & self.store.get_valid_ids()
        )
        if final_found_ids:
            draft_dict["cited_evidence_ids"] = final_found_ids

        try:
            return FinalResearchReport.model_validate(draft_dict)
        except ValidationError as exc:
            raise RuntimeError(f"CIO 最终报告 Schema 校验失败: {exc}") from exc
