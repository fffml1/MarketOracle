from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
import os

# 加载项目根目录下的 .env
load_dotenv()

from hello_agents import HelloAgentsLLM
from hello_agents.tools import ToolRegistry

from .debate_arena import DebateArena, DebateTranscript
from .debate_stream import DebateEvent, DebateListener, ResearchRun
from .judge_agent import CIOJudgeAgent
from .research_team import ResearchBriefs, ResearchTeam
from ..schemas.report import FinalResearchReport
from ..tools.announcement_rag import AnnouncementSearchTool, FinancialMetricTool, enabled
from ..tools.hello_adapter import EvidenceStore, build_role_registries

# 复用你第二步验证过的 Mock 构造器；去掉参数即切换为真实 akshare 数据源
from .run_step2_demo import make_mock_tools


def run_stock_deep_research(
    symbol: str,
    as_of_date: str,
    user_focus: str = "关注短期波段买卖点与筹码压力",
    use_mock: bool = True,
    output_dir: str = "logs/runs",
    *,
    listener: DebateListener | None = None,
) -> FinalResearchReport:
    # 1. 配置模型（实战中可配置两个实例：轻量模型跑调研，强推理模型跑辩论与CIO裁决）
    research_llm = HelloAgentsLLM(
        model=os.getenv("MARKETORACLE_MODEL", "deepseek-chat"),  # 默认兜底 deepseek-chat
        api_key=os.getenv("MARKETORACLE_MODEL_KEY"),
        base_url=os.getenv("MARKETORACLE_MODEL_URL"),
    )
    reasoning_llm = HelloAgentsLLM(
        model=os.getenv("MARKETORACLE_MODEL", "deepseek-chat"),  # 默认兜底 deepseek-chat
        api_key=os.getenv("MARKETORACLE_MODEL_KEY"),
        base_url=os.getenv("MARKETORACLE_MODEL_URL"),
    )

    # 2. 初始化证据仓库与角色隔离工具注册表
    store = EvidenceStore()
    tech_tool, news_tool, risk_tool = make_mock_tools() if use_mock else (None, None, None)
    rag_tools = (FinancialMetricTool(store, symbol, as_of_date, mock=use_mock),
                 AnnouncementSearchTool(store, symbol, as_of_date, mock=use_mock)) if enabled("RESEARCH", default=use_mock) else ()
    registries = build_role_registries(
        store=store, as_of_date=as_of_date, tech_tool=tech_tool, news_tool=news_tool, risk_tool=risk_tool,
        bound_symbol=symbol, rag_tools=rag_tools,
    )

    try:
        report, transcript = conduct_research(
            symbol, as_of_date, user_focus, store, registries, research_llm, reasoning_llm,
            listener=listener,
        )
    except ResearchIncomplete as exc:
        _save_trace(output_dir, symbol, as_of_date, None, exc.transcript, store)
        raise
    _save_trace(output_dir, symbol, as_of_date, report, transcript, store)
    return report


def _save_trace(output_dir, symbol, as_of_date, report, transcript, store) -> None:
    suffix = "" if report is not None else "_incomplete"
    out_path = Path(output_dir) / f"{symbol}_{as_of_date}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_bundle = {
        "report": report.model_dump() if report is not None else None,
        "debate": transcript.model_dump(),
        "evidences": {k: v.model_dump() for k, v in store.items.items()},
    }
    out_path.write_text(json.dumps(trace_bundle, ensure_ascii=False, indent=2), encoding="utf-8")

class ResearchIncomplete(RuntimeError):
    def __init__(self, message: str, transcript: DebateTranscript) -> None:
        super().__init__(message)
        self.transcript = transcript


def interrupt_research(run, symbol, as_of_date, listener) -> ResearchIncomplete:
    with run.lock:
        run.close()
        transcript = (
            run.transcript.model_copy(deep=True) if run.transcript is not None else
            DebateTranscript(symbol=symbol, as_of_date=as_of_date, rounds=0)
        )
        transcript.rounds = transcript.completed_rounds()
        transcript.stop_reason = "研究超时或已取消"
        if listener:
            if any(not turn.completed for turn in transcript.turns):
                listener(DebateEvent("interrupted"))
            listener(DebateEvent("stop", transcript.rounds, text=transcript.stop_reason))
        return ResearchIncomplete("本次研究未完成。已保留取得的证据和发言。", transcript)


def conduct_research(
    symbol: str,
    as_of_date: str,
    user_focus: str,
    store: EvidenceStore,
    registries: dict[str, ToolRegistry],
    research_llm: HelloAgentsLLM,
    reasoning_llm: HelloAgentsLLM,
    *,
    listener: DebateListener | None = None,
    run: ResearchRun | None = None,
) -> tuple[FinalResearchReport, DebateTranscript]:
    if run is None:
        run = ResearchRun()

        def emit(event):
            with run.lock:
                if not run.closed and listener:
                    listener(event)

        future = run.start(lambda: conduct_research(
            symbol, as_of_date, user_focus, store, registries, research_llm, reasoning_llm,
            listener=emit, run=run,
        ))
        try:
            return future.result(timeout=max(0, run.remaining()))
        except (TimeoutError, KeyboardInterrupt) as exc:
            raise interrupt_research(run, symbol, as_of_date, listener) from exc
        finally:
            run.close()
    briefs, transcript = research_debate(
        symbol, as_of_date, user_focus, store, registries, research_llm, reasoning_llm,
        purpose="完整研报", listener=listener, run=run,
    )
    if not transcript.rounds:
        raise ResearchIncomplete("完整研报未生成。尚未完成一轮多空交锋。", transcript)
    judge = CIOJudgeAgent(llm=reasoning_llm, store=store)
    try:
        with run.scope():
            report = judge.adjudicate(briefs=briefs, transcript=transcript, user_focus=user_focus)
            run.check()
    except Exception as exc:
        raise ResearchIncomplete("裁决未完成。辩论实录已保留。", transcript) from exc
    return report, transcript


def research_debate(
    symbol, as_of_date, user_focus, store, registries, research_llm, reasoning_llm,
    *, purpose: str, listener: DebateListener | None, run: ResearchRun,
    debate_focus: Callable[[], str] | None = None,
) -> tuple[ResearchBriefs, DebateTranscript]:
    team = ResearchTeam(llm=research_llm, registries=registries)
    market_evidence = "\n\n".join(
        bundle.to_prompt_text()
        for bundle in store.bundles
        if bundle.tool_name == "get_market_technical"
        and bundle.symbol == symbol
        and bundle.as_of_date == as_of_date
        and bundle.evidences
    )
    try:
        with run.scope(run.reserve):
            briefs = team.run_all(
                symbol=symbol, as_of_date=as_of_date, user_focus=user_focus, market_evidence=market_evidence
            )
    except Exception as exc:
        transcript = DebateTranscript(symbol=symbol, as_of_date=as_of_date, rounds=0, stop_reason="调研未完成")
        raise ResearchIncomplete("调研未完成，尚未开辩。", transcript) from exc

    arena = DebateArena(
        llm=reasoning_llm,
        debater_registry=registries["debater"],
        store=store,
        max_rounds=3,
    )
    transcript = arena.run_debate(
        briefs, debate_focus() if debate_focus else user_focus,
        dynamic=True, purpose=purpose, listener=listener, run=run,
    )
    return briefs, transcript


if __name__ == "__main__":
    final_report = run_stock_deep_research(
        symbol="300750.SZ",
        as_of_date="2026-11-20",
        use_mock=False,  # 先用 Mock 验证第三步跑通，再改为 False 测真实 A 股行情
    )
    print(final_report.to_markdown())
