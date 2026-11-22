from __future__ import annotations

import re
from contextlib import closing
from pydantic import BaseModel, Field
from hello_agents.core.budget import BudgetExceeded

from hello_agents import HelloAgentsLLM, ReActAgent
from hello_agents.tools import ToolRegistry

from .research_team import ResearchBriefs
from .debate_stream import DebateEvent, DebateListener, ResearchRun

from ..tools.hello_adapter import EvidenceStore

EVIDENCE_ID_PATTERN = re.compile(r"\[(EV_[A-Z]+_\d{3,})\]")


class DebateTurn(BaseModel):
    """单轮辩论记录"""

    round_num: int
    speaker: str  # "Bull" | "Bear"
    content: str
    cited_ids: list[str] = Field(default_factory=list, description="发言中引用的合法证据ID")
    invalid_ids: list[str] = Field(default_factory=list, description="发言中出现的未登记/伪造证据ID")
    completed: bool = True


class DebateTranscript(BaseModel):
    """完整多空辩论实录"""

    symbol: str
    as_of_date: str
    rounds: int
    turns: list[DebateTurn] = Field(default_factory=list)
    stop_reason: str | None = None
    unresolved: list[str] = Field(default_factory=list)

    def completed_rounds(self) -> int:
        bull = {t.round_num for t in self.turns if t.completed and t.speaker == "Bull"}
        bear = {t.round_num for t in self.turns if t.completed and t.speaker == "Bear"}
        return len(bull & bear)

    def to_markdown(self) -> str:
        lines: list[str] = [f"## 多空对抗辩论实录（{self.symbol} | 基准日 {self.as_of_date}）\n"]
        lines.append(f"已完成 {self.completed_rounds()} 轮。停止原因：{self.stop_reason or '未记录'}")
        lines.extend(f"- 未决：{issue}" for issue in self.unresolved)
        for t in self.turns:
            role_label = "多头主理人 (Bull)" if t.speaker == "Bull" else "空头质询官 (Bear)"
            cite_str = ", ".join(t.cited_ids) if t.cited_ids else "无"
            warn_str = f" | ⚠️ 疑似无效引用: {', '.join(t.invalid_ids)}" if t.invalid_ids else ""
            status = "" if t.completed else "（发言未完成）"
            lines.append(f"### 第 {t.round_num} 轮 - {role_label}{status}")
            lines.append(f"> 引用证据: [{cite_str}]{warn_str}\n")
            lines.append(t.content.strip())
            lines.append("")
        return "\n".join(lines)


BULL_SYSTEM_PROMPT = """你是一名专业机构【多头主理人 (Bull)】。
你的任务是在多空投研辩论中，基于事实证据发掘标的的上涨逻辑、趋势优势、催化剂与赔率空间，并有力回击空头的质疑。

辩论铁律：
1. **证据驱动**：每一个数据、事件或论点必须引用方括号证据编号，如 [EV_TECH_001]、[EV_NEWS_002]。严禁编造编号或数字！
2. **直面交锋**：如果是第 2 轮及以后，必须逐条回应空头上一轮提出的核心质疑，不能避重就轻。
3. **按需补证**：若现有底稿缺少回应空头所需的细粒度信息，可调用可用工具补充检索（新证据会自动分配编号）。
4. **保持客观克制**：承认确实存在的短期风险点，但论证为何当前上涨催化或趋势结构足以覆盖这些风险。
5. 每次发言控制在 350 字以内，结构清晰（【核心论点】、【对空头反驳】、【关键观察位】）。"""

BEAR_SYSTEM_PROMPT = """你是一名机构【空头质询官 / 首席风控辩手 (Bear)】。
你的任务是专门挑刺，寻找多头逻辑中的漏洞、利好兑现风险、技术面背离、估值透支或公告隐患。

辩论铁律：
1. **证据驱动**：每一个质疑和反驳必须引用方括号证据编号，如 [EV_RISK_001]、[EV_TECH_001]。严禁编造编号或数字！
2. **精准打击**：紧盯多头上一轮的立论进行拆解（例如：质疑催化是否已反映在股价中、均线是否走弱、是否有减持/回撤压制）。
3. **按需补证**：若发现风险底稿信息不足，可调用可用工具进一步排查。
4. **保持专业理性**：不为了反对而反对，认可多头有硬数据支撑的亮点，但重点论证当前介入的潜在回撤风险与交易拥挤度。
5. 每次发言控制在 350 字以内，结构清晰（【核心质疑】、【风险证据拆解】、【触发看空条件】）。"""


class DebateArena:
    """多空对抗辩论状态机"""

    def __init__(
        self,
        llm: HelloAgentsLLM,
        debater_registry: ToolRegistry,
        store: EvidenceStore,
        max_rounds: int = 2,
    ) -> None:
        self.store = store
        self.llm = llm
        self.max_rounds = max_rounds

        self.bull_agent = ReActAgent(
            name="BullDebater",
            llm=llm,
            tool_registry=debater_registry,
            system_prompt=BULL_SYSTEM_PROMPT,
            max_steps=3,
        )
        self.bear_agent = ReActAgent(
            name="BearDebater",
            llm=llm,
            tool_registry=debater_registry,
            system_prompt=BEAR_SYSTEM_PROMPT,
            max_steps=3,
        )

    def _validate_citations(self, text: str) -> tuple[list[str], list[str]]:
        """提取发言中的 [EV_XXX_001] 并与 EvidenceStore 核对真伪"""
        found = list(dict.fromkeys(EVIDENCE_ID_PATTERN.findall(text)))
        valid_pool = self.store.get_valid_ids()
        valid_ids = [eid for eid in found if eid in valid_pool]
        invalid_ids = [eid for eid in found if eid not in valid_pool]
        return valid_ids, invalid_ids

    @staticmethod
    def _build_base_dossier(briefs: ResearchBriefs) -> str:
        return (
            f"【调研标的】{briefs.symbol} | 【基准日期】{briefs.as_of_date}\n\n"
            f"--- 1. 量价技术底稿 ---\n{briefs.quant_brief}\n\n"
            f"--- 2. 催化与新闻底稿 ---\n{briefs.catalyst_brief}\n\n"
            f"--- 3. 风险与公告底稿 ---\n{briefs.risk_brief}\n"
        )

    def run_debate(
        self, briefs: ResearchBriefs, user_focus: str = "", *,
        dynamic: bool = False, purpose: str = "完整研报",
        listener: DebateListener | None = None, run: ResearchRun | None = None,
    ) -> DebateTranscript:
        if dynamic:
            return self._run_dynamic(briefs, user_focus, purpose, listener, run or ResearchRun())
        transcript = DebateTranscript(
            symbol=briefs.symbol,
            as_of_date=briefs.as_of_date,
            rounds=self.max_rounds,
        )
        dossier = self._build_base_dossier(briefs)
        focus_line = f"\n用户补充关注点：{user_focus}" if user_focus else ""

        history_blocks: list[str] = []

        for r in range(1, self.max_rounds + 1):
            # --- 1. 多头发言 ---
            if r == 1:
                bull_prompt = (
                    f"以下是前期三路研究员整理的事实底稿：\n{dossier}{focus_line}\n\n"
                    f"现在开始第 {r} 轮辩论，请你作为【多头主理人】率先立论，阐述看多逻辑与核心依据（务必引用证据编号）。"
                )
            else:
                prev_history = "\n\n".join(history_blocks)
                bull_prompt = (
                    f"【前期调研底稿】\n{dossier}\n\n"
                    f"【此前辩论实录】\n{prev_history}\n\n"
                    f"现在进入第 {r} 轮辩论，请你作为【多头主理人】针对空头上一轮的质疑进行正面反驳与补充论证（务必引用证据编号）。"
                )

            bull_reply = self.bull_agent.run(bull_prompt)
            b_valid, b_invalid = self._validate_citations(bull_reply)
            transcript.turns.append(
                DebateTurn(
                    round_num=r,
                    speaker="Bull",
                    content=bull_reply,
                    cited_ids=b_valid,
                    invalid_ids=b_invalid,
                )
            )
            history_blocks.append(f"[第{r}轮-多头]:\n{bull_reply}")

            # --- 2. 空头质询与反驳 ---
            prev_history = "\n\n".join(history_blocks)
            bear_prompt = (
                f"【前期调研底稿】\n{dossier}{focus_line}\n\n"
                f"【此前辩论实录】\n{prev_history}\n\n"
                f"现在进入第 {r} 轮辩论，请你作为【空头质询官】针对多头刚刚的陈述进行拆解、反驳与风险揭示（务必引用证据编号）。"
            )

            bear_reply = self.bear_agent.run(bear_prompt)
            r_valid, r_invalid = self._validate_citations(bear_reply)
            transcript.turns.append(
                DebateTurn(
                    round_num=r,
                    speaker="Bear",
                    content=bear_reply,
                    cited_ids=r_valid,
                    invalid_ids=r_invalid,
                )
            )
            history_blocks.append(f"[第{r}轮-空头]:\n{bear_reply}")

        return transcript

    def _run_dynamic(self, briefs, focus, purpose, listener, run) -> DebateTranscript:
        transcript = DebateTranscript(symbol=briefs.symbol, as_of_date=briefs.as_of_date, rounds=0)
        with run.lock:
            run.check()
            run.transcript = transcript
        dossier = self._build_base_dossier(briefs)
        next_action = ""
        emit = listener or (lambda event: None)
        try:
            with run.scope(run.reserve):
                for number in range(1, self.max_rounds + 1):
                    for speaker, agent, role in (
                        ("Bull", self.bull_agent, BULL_SYSTEM_PROMPT),
                        ("Bear", self.bear_agent, BEAR_SYSTEM_PROMPT),
                    ):
                        task = (
                            f"{dossier}\n研究目的：{purpose}\n原始关注点和条件：{focus}\n"
                            f"{transcript.to_markdown()}\n本轮任务：{next_action or '提出并检验最重要的理由'}"
                        )
                        self._speak(transcript, number, speaker, agent, role, task, emit, run)
                    with run.lock:
                        transcript.rounds = transcript.completed_rounds()
                    decision = self._decide(dossier, purpose, focus, transcript)
                    transcript.unresolved = decision.unresolved
                    if decision.issue and decision.impact and decision.action:
                        if decision.issue not in transcript.unresolved:
                            transcript.unresolved.append(decision.issue)
                    else:
                        transcript.stop_reason = decision.reason
                        break
                    if number == self.max_rounds:
                        transcript.stop_reason = "达到轮数上限"
                        break
                    if run.remaining(run.reserve) < 15:
                        transcript.stop_reason = "剩余时间留给裁决"
                        break
                    next_action = f"{decision.issue}。{decision.action}"
                    emit(DebateEvent("continue", number, text=next_action))
        except (TimeoutError, BudgetExceeded):
            transcript.stop_reason = "辩论时间已用尽"
        except (Exception, KeyboardInterrupt):
            transcript.stop_reason = "辩论生成中断"
        if transcript.stop_reason in {"辩论时间已用尽", "辩论生成中断"}:
            transcript.unresolved.append("本轮交锋或续辩判断未完成，请结合已保存的原话查看待核实事项。")
        emit(DebateEvent("stop", transcript.rounds, text=transcript.stop_reason or "辩论结束"))
        return transcript

    def _speak(self, transcript, number, speaker, agent, role, task, emit, run) -> None:
        run.check()
        turn = DebateTurn(round_num=number, speaker=speaker, content="", completed=False)
        with run.lock:
            run.check()
            transcript.turns.append(turn)
            emit(DebateEvent("start", number, speaker))
        try:
            prep = agent.run(
                f"{task}\n仅准备这次发言所需材料。需要核实时调用工具，返回事实和证据编号。"
            )
            if agent.last_run.status != "completed":
                raise RuntimeError("辩手补证未完成")
            messages = [
                {"role": "system", "content": role + "\n直接输出公开发言正文，沿用已有证据。"},
                {"role": "user", "content": f"{task}\n补证材料：{prep}\n{self.store.dump_all_prompt_text()}"},
            ]
            with closing(self.llm.stream_invoke(messages)) as stream:
                for chunk in stream:
                    with run.lock:
                        run.check()
                        turn.content += chunk
                        emit(DebateEvent("text", number, speaker, chunk))
            stats = getattr(self.llm, "last_call_stats", None)
            finish = str(getattr(stats, "finish_reason", "")).lower()
            if not turn.content.strip() or finish in {"length", "max_tokens", "max_output_tokens"}:
                raise RuntimeError("公开发言未完整生成")
            with run.lock:
                run.check()
                turn.completed = True
        finally:
            with run.lock:
                turn.cited_ids, turn.invalid_ids = self._validate_citations(turn.content)
                status = "end" if turn.completed else "interrupted"
                note = f"待核查引用：{', '.join(turn.invalid_ids)}" if turn.invalid_ids else ""
                emit(DebateEvent(status, number, speaker, note))

    def _decide(self, dossier, purpose, focus, transcript) -> RoundDecision:
        prompt = (
            f"研究目的：{purpose}。原始条件：{focus}\n{dossier}\n{transcript.to_markdown()}\n"
            f"补查证据：{self.store.dump_all_prompt_text()}\n"
            "判断是否有反驳尚待回应，且回应会改变结论、下一轮有具体办法推进。"
            "短线只缺用户风险上限时停辩。研报以核心结论、置信度和主要风险为准。"
            "未披露信息和仅重复转载的事实不能支持续辩。检查最后一句反证。"
            "仅输出 JSON，字段 issue、impact、action、reason 是字符串，unresolved 是字符串数组。"
            "只有三项续辩条件齐备时填写 issue、impact、action，否则留空并给出停辩原因。"
            "即使停辩也必须列出重要的未决反驳。"
        )
        return RoundDecision.model_validate_json(model_json(self.llm, prompt))


class RoundDecision(BaseModel):
    issue: str = ""
    impact: str = ""
    action: str = ""
    reason: str = "暂无可推进的关键争点"
    unresolved: list[str] = Field(default_factory=list)


def model_json(llm, prompt: str) -> str:
    response = llm.invoke([{"role": "user", "content": prompt}]).content
    start, end = response.find("{"), response.rfind("}")
    return response[start:end + 1] if 0 <= start < end else response
