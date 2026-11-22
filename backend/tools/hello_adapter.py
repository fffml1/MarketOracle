from __future__ import annotations

from threading import Lock
from contextlib import nullcontext
from typing import Any

from hello_agents.tools import Tool, ToolParameter, ToolRegistry, ToolResponse

from ..schemas.evidence import EvidenceBundle, EvidenceItem
from ..agents.debate_stream import current_run
from .base_evidence_tool import BaseEvidenceTool
from .market_data import MarketTechnicalTool
from .news_catalyst import NewsCatalystTool
from .risk_notice import RiskAndNoticeTool


class EvidenceStore:
    """单次研究任务的共享证据仓库：按 evidence_id 索引，供辩论校验与 CIO 裁决使用"""

    def __init__(self) -> None:
        self.items: dict[str, EvidenceItem] = {}
        self.bundles: list[EvidenceBundle] = []

    def record(self, bundle: EvidenceBundle) -> None:
        self.bundles.append(bundle)
        for item in bundle.evidences:
            self.items[item.evidence_id] = item

    def get_valid_ids(self) -> set[str]:
        return set(self.items.keys())

    def dump_all_prompt_text(self) -> str:
        if not self.bundles:
            return "（暂无已采集证据）"
        return "\n\n".join(b.to_prompt_text() for b in self.bundles)


class HelloEvidenceTool(Tool):
    """将 BaseEvidenceTool 适配为 HelloAgents 可调度的标准 Tool"""

    def __init__(
        self,
        inner_tool: BaseEvidenceTool,
        store: EvidenceStore,
        default_as_of_date: str,
        description: str,
        has_lookback: bool = True,
        *,
        bound_symbol: str | None = None,
    ) -> None:
        super().__init__(name=inner_tool.tool_name, description=description)
        self._inner = inner_tool
        self._store = store
        self._default_as_of_date = default_as_of_date
        self._has_lookback = has_lookback
        self._bound_symbol = bound_symbol
        self._successful_calls: dict[int | None, EvidenceBundle] = {}
        self._call_lock = Lock()

    def get_parameters(self) -> list[ToolParameter]:
        params = [
            ToolParameter(
                name="symbol",
                type="string",
                description="股票代码，如 300750.SZ 或 600519.SH",
                required=True,
            ),
            ToolParameter(
                name="as_of_date",
                type="string",
                description=f"研究基准日 YYYY-MM-DD，默认 {self._default_as_of_date}",
                required=False,
            ),
        ]
        if self._has_lookback:
            params.append(
                ToolParameter(
                    name="lookback_days",
                    type="integer",
                    description="向前回溯天数（如 14 或 30）",
                    required=False,
                )
            )
        return params

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        symbol = str(parameters.get("symbol") or "").strip()
        if not symbol:
            return ToolResponse.error(
                code="INVALID_PARAMS",
                message="缺少必填参数 symbol",
            )

        as_of_date = str(parameters.get("as_of_date") or self._default_as_of_date).strip()

        if self._bound_symbol is not None and (
            symbol != self._bound_symbol or as_of_date != self._default_as_of_date
        ):
            return ToolResponse.error(
                code="OUTSIDE_SESSION_SCOPE",
                message=f"当前会话只研究 {self._bound_symbol}，基准日 {self._default_as_of_date}",
            )

        kwargs: dict[str, Any] = {"symbol": symbol, "as_of_date": as_of_date}
        lookback = None
        try:
            if self._has_lookback and parameters.get("lookback_days") is not None:
                lookback = int(parameters["lookback_days"])
                if self._bound_symbol is not None and lookback <= 0:
                    return ToolResponse.error(code="INVALID_PARAMS", message="回溯天数必须为正整数")
                kwargs["lookback_days"] = lookback
            if self._bound_symbol is not None and isinstance(self._inner, MarketTechnicalTool):
                lookback = lookback or self._inner.DEFAULT_LOOKBACK_DAYS
            with self._call_lock:
                bundle = self._successful_calls.get(lookback)
                if bundle is None:
                    bundle = self._inner.execute(**kwargs)
                    run = current_run.get()
                    with run.lock if run else nullcontext():
                        if run:
                            run.check()
                        self._store.record(bundle)
                        if self._bound_symbol is not None and bundle.status != "error" and bundle.evidences:
                            self._successful_calls[lookback] = bundle
        except Exception as exc:
            return ToolResponse.error(
                code="EXECUTION_ERROR",
                message=f"工具 {self.name} 执行异常: {exc}",
            )

        prompt_text = bundle.to_prompt_text()
        payload_data = bundle.model_dump()

        if bundle.status == "error" and not bundle.evidences:
            return ToolResponse.error(
                code="DATA_FETCH_ERROR",
                message=bundle.error_message or bundle.degrade_reason or prompt_text,
            )

        if bundle.status == "partial" or bundle.is_degraded:
            return ToolResponse.partial(
                text=prompt_text,
                data=payload_data,
            )

        return ToolResponse.success(
            text=prompt_text,
            data=payload_data,
        )


def build_role_registries(
    store: EvidenceStore,
    as_of_date: str,
    tech_tool: MarketTechnicalTool | None = None,
    news_tool: NewsCatalystTool | None = None,
    risk_tool: RiskAndNoticeTool | None = None,
    *,
    bound_symbol: str | None = None,
    rag_tools: tuple[Tool, ...] = (),
) -> dict[str, ToolRegistry]:
    """为不同角色构建隔离的 ToolRegistry，实现工具权限最小化"""

    t_tech = HelloEvidenceTool(
        inner_tool=tech_tool or MarketTechnicalTool(),
        store=store,
        default_as_of_date=as_of_date,
        description="获取股票截至基准日的日线行情、均线排列、区间涨跌幅、60日最大回撤与量比指标。",
        has_lookback=True,
        bound_symbol=bound_symbol,
    )
    t_news = HelloEvidenceTool(
        inner_tool=news_tool or NewsCatalystTool(),
        store=store,
        default_as_of_date=as_of_date,
        description="检索股票近期新闻、行业政策与产业链催化事件（返回带 EV_NEWS_xxx 编号的证据）。",
        has_lookback=True,
        bound_symbol=bound_symbol,
    )
    t_risk = HelloEvidenceTool(
        inner_tool=risk_tool or RiskAndNoticeTool(),
        store=store,
        default_as_of_date=as_of_date,
        description="排查大股东减持、质押、限售解禁、监管问询、诉讼及业绩亏损等风险公告（返回 EV_RISK_xxx）。",
        has_lookback=True,
        bound_symbol=bound_symbol,
    )

    quant_reg = ToolRegistry()
    quant_reg.register_tool(t_tech)

    catalyst_reg = ToolRegistry()
    catalyst_reg.register_tool(t_news)

    risk_reg = ToolRegistry()
    risk_reg.register_tool(t_risk)

    for tool in rag_tools:
        if tool.name == "get_financial_metric":
            quant_reg.register_tool(tool)
        elif tool.name == "search_announcements":
            catalyst_reg.register_tool(tool)
            risk_reg.register_tool(tool)

    debater_reg = ToolRegistry()
    debater_reg.register_tool(t_tech)
    debater_reg.register_tool(t_news)
    debater_reg.register_tool(t_risk)

    return {
        "quant": quant_reg,
        "catalyst": catalyst_reg,
        "risk": risk_reg,
        "debater": debater_reg,
    }
