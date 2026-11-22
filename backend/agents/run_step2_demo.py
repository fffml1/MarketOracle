from __future__ import annotations

from dotenv import load_dotenv
import os

# 加载项目根目录下的 .env
load_dotenv()

import pandas as pd
from hello_agents import HelloAgentsLLM

from .debate_arena import DebateArena
from .research_team import ResearchTeam
from ..tools.hello_adapter import EvidenceStore, build_role_registries
from ..tools.market_data import MarketTechnicalTool
from ..tools.news_catalyst import NewsCatalystTool
from ..tools.risk_notice import RiskAndNoticeTool


def make_mock_tools():
    """构造离线模拟数据源，用于快速调试 Agent 调研与辩论逻辑"""

    def fake_kline(symbol: str, as_of_date: str, lookback_days: int) -> pd.DataFrame:
        dates = pd.bdate_range(end=as_of_date, periods=65)
        closes = [180 + i * 0.6 if i < 55 else 213 - (i - 55) * 1.1 for i in range(len(dates))]
        return pd.DataFrame(
            {
                "date": [d.strftime("%Y-%m-%d") for d in dates],
                "open": [c - 0.5 for c in closes],
                "high": [c + 1.5 for c in closes],
                "low": [c - 1.5 for c in closes],
                "close": closes,
                "volume": [100000 + (i % 5) * 15000 for i in range(len(dates))],
            }
        )

    def fake_news(symbol: str, as_of_date: str, lookback_days: int):
        return [
            {
                "title": "公司新一代高能量密度电池斩获海外头部车企定点",
                "summary": "预计明年上半年开始批量供货，有望提升海外高毛利业务占比。",
                "published_at": "2026-11-18",
                "source": "证券时报",
            },
            {
                "title": "行业储能装机需求超预期，四季度排产维持高位",
                "summary": "产业链调研显示头部电芯厂11月排产环比持平略增。",
                "published_at": "2026-11-15",
                "source": "财联社",
            },
        ]

    def fake_risk(symbol: str, as_of_date: str, lookback_days: int):
        return [
            {
                "title": "关于部分持股5%以上股东减持计划时间过半的进展公告",
                "summary": "相关股东已累计减持0.45%股份，剩余减持额度仍在窗口期内，对短期筹码形成一定压制。",
                "published_at": "2026-11-12",
                "source": "巨潮资讯网",
                "risk_tags": ["减持"],
            }
        ]

    return (
        MarketTechnicalTool(kline_fetcher=fake_kline, source_name="mock_kline"),
        NewsCatalystTool(news_fetcher=fake_news),
        RiskAndNoticeTool(risk_fetcher=fake_risk),
    )


def main() -> None:
    symbol = "300750.SZ"
    as_of_date = "2026-11-20"

    # 配置你的 LLM（支持 OpenAI 兼容接口，如 DeepSeek / Qwen）
    llm = HelloAgentsLLM(
        model=os.getenv("MARKETORACLE_MODEL", "deepseek-chat"),  # 默认兜底 deepseek-chat
        api_key=os.getenv("MARKETORACLE_MODEL_KEY"),
        base_url=os.getenv("MARKETORACLE_MODEL_URL"),
    )

    store = EvidenceStore()
    tech_tool, news_tool, risk_tool = make_mock_tools()
    registries = build_role_registries(
        store=store,
        as_of_date=as_of_date,
        tech_tool=tech_tool,
        news_tool=news_tool,
        risk_tool=risk_tool,
    )

    # 1. 三路独立调研
    team = ResearchTeam(llm=llm, registries=registries)
    briefs = team.run_all(symbol=symbol, as_of_date=as_of_date, user_focus="关注短期波段买卖点与筹码压力")

    # 2. 多空 2 轮辩论
    arena = DebateArena(llm=llm, debater_registry=registries["debater"], store=store, max_rounds=2)
    transcript = arena.run_debate(briefs=briefs, user_focus="关注短期波段买卖点与筹码压力")

    print(transcript.to_markdown())
    print("=== 全量沉淀证据 ID ===", sorted(store.get_valid_ids()))


if __name__ == "__main__":
    main()