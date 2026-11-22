import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hello_agents.core.llm_response import LLMToolResponse, ToolCall

from backend.agents.assistant import StockChatAssistant
from backend.agents.run_step2_demo import make_mock_tools
from backend.session import ResearchSession


def request_market(call_id, **overrides):
    params = {"symbol": "300750.SZ", **overrides}
    return LLMToolResponse(
        None, [ToolCall(call_id, "get_market_technical", json.dumps(params))], "test"
    )


def call_tool(name, **arguments):
    return LLMToolResponse(None, [ToolCall(name, name, json.dumps(arguments))], "test")


def answer(text):
    return LLMToolResponse(text, [], "test")


class StockChatTests(unittest.TestCase):
    def setUp(self):
        self.session = ResearchSession("300750.SZ", "2026-11-20")
        market, _, _ = make_mock_tools()
        self.fetch = Mock(wraps=market._kline_fetcher)
        market._kline_fetcher = self.fetch
        self.llm = Mock(model="test")
        tokenizer = patch("hello_agents.context.token_counter.TokenCounter._get_encoding", return_value=None)
        tokenizer.start()
        self.addCleanup(tokenizer.stop)
        self.assistant = StockChatAssistant(self.session, self.llm, market, use_mock=True)

    def test_follow_up_reuses_evidence_without_tool_call(self):
        self.llm.invoke_with_tools.side_effect = [
            request_market("first"),
            answer("量比 1.23 [EV_TECH_001]"),
            answer("含当日在内的最近20个交易日均量 [EV_TECH_001]"),
        ]
        first = self.assistant.ask("最近走势和成交量怎么样？")
        self.assertIn("小白模式", self.llm.invoke_with_tools.call_args.kwargs["messages"][0]["content"])
        self.assistant.set_mode("专业")
        follow_up = self.assistant.ask("刚才的成交量是跟哪几天比？")
        self.assertIn("[EV_TECH_001]", first)
        self.assertIn("[EV_TECH_001]", follow_up)
        self.assertEqual(self.llm.invoke_with_tools.call_count, 3)
        self.fetch.assert_called_once_with("300750.SZ", "2026-11-20", 180)
        messages = self.llm.invoke_with_tools.call_args.kwargs["messages"]
        self.assertIn("专业模式", messages[0]["content"])
        context = messages[-1]["content"]
        self.assertIn('"vol_ma20": 130000.0', context)
        self.assertIn('"vol_ratio_20d": 1.23', context)
        self.assertIn('"vol_ma20_start_date": "2026-10-26"', context)
        self.assertTrue(any("最近走势和成交量怎么样" in str(m) for m in messages[:-1]))
        self.assertFalse(ResearchSession("300750.SZ", "2026-11-20").evidence.items)

    def test_stream_answer_replaces_tool_preview_and_keeps_session_evidence(self):
        market_request = request_market("stock")
        market_request.content = "先取行情"
        final_text = "## 行情\n\n量比 1.23 [EV_TECH_001]"

        class StreamingModel:
            model = "test"

            async def astream_invoke_with_tools(self, messages, tools, **kwargs):
                if messages[-1]["role"] != "tool":
                    yield market_request.content
                    yield market_request
                    return
                yield "## 行情\n\n"
                yield "量比 1.23 [EV_TECH_001]"
                yield answer(final_text)

        self.assistant.agent.llm = StreamingModel()
        publish = Mock()
        result = asyncio.run(self.assistant.stream_answer("最近走势？", publish))

        self.assertEqual(result, final_text)
        self.assertEqual([event.args[0] for event in publish.call_args_list],
                         ["answer_delta", "answer_reset", "phase", "answer_delta", "answer_delta"])
        self.assertIn("EV_TECH_001", self.session.evidence.items)
        self.fetch.assert_called_once()

    def test_deep_research_reuses_market_evidence_and_keeps_report_for_follow_up(self):
        draft = {
            "headline_verdict": "震荡整理 [EV_TECH_001]",
            "rating": "中性观望",
            "confidence_score": 55,
            "confidence_rationale": "模拟数据",
            "executive_summary": "先观察走势",
            "debate_synthesis": {
                "bull_strongest_argument": "产业进展 [EV_NEWS_001]",
                "bear_strongest_argument": "股东减持 [EV_RISK_001]",
            },
            "action_plan": {
                "stance": "观望等待确认",
                "entry_zone": "等待确认",
                "stop_loss_rule": "趋势转弱时重新评估",
                "position_advice": "观察",
            },
        }
        symbol = self.session.symbol
        self.llm.invoke_with_tools.side_effect = [
            request_market("chat-market"), answer("走势 [EV_TECH_001]"),
            call_tool("run_stock_deep_research", focus="成交量和减持"),
            request_market("research-market"), call_tool("Finish", answer="震荡 [EV_TECH_001]"),
            call_tool("search_news_catalyst", symbol=symbol), call_tool("Finish", answer="产业进展 [EV_NEWS_001]"),
            call_tool("check_risk_and_notices", symbol=symbol), call_tool("Finish", answer="减持计划 [EV_RISK_001]"),
            call_tool("Finish", answer="多头补证 [EV_NEWS_001]"),
            call_tool("Finish", answer="空头补证 [EV_RISK_001]"),
            answer("研报摘要"), answer("空头担心股东减持 [EV_RISK_001]"),
        ]
        self.llm.invoke.side_effect = [
            answer('{"reason":"分歧已记录"}'),
            answer(json.dumps(draft, ensure_ascii=False)),
            answer('{"passed":true,"issues":[],"summary":"核查通过"}'),
        ]
        self.llm.stream_invoke.side_effect = [
            (chunk for chunk in ["产业进展 ", "[EV_NEWS_001]"]),
            (chunk for chunk in ["股东减持 ", "[EV_RISK_001]"]),
        ]
        self.llm.last_call_stats = SimpleNamespace(finish_reason="stop")
        self.assistant.ask("最近走势怎么样？")
        self.assertEqual(self.assistant.ask("给我一份完整研报"), "研报摘要")
        self.assertEqual(self.assistant.last_report.symbol, symbol)
        self.assertEqual(self.assistant.last_report.as_of_date, self.session.as_of_date)
        self.assertEqual(self.assistant.last_report_transcript.rounds, 1)
        self.fetch.assert_called_once()
        quant_messages = self.llm.invoke_with_tools.call_args_list[3].kwargs["messages"]
        self.assertIn("沿用下列会话行情证据", quant_messages[-1]["content"])
        self.assertIn("EV_TECH_001", quant_messages[-1]["content"])
        self.assertIn("EV_NEWS_001", self.session.evidence.items)
        self.assertIn("EV_RISK_001", self.session.evidence.items)
        self.assertIn("EV_RISK_001", self.assistant.ask("空头最担心什么？"))
        context = self.llm.invoke_with_tools.call_args.kwargs["messages"][-1]["content"]
        self.assertIn('最新研报：{', context)
        self.assertIn('股东减持 [EV_RISK_001]', context)

    def test_short_entry_can_debate_before_risk_percent_and_reuse_the_result(self):
        symbol = self.session.symbol
        self.llm.invoke_with_tools.side_effect = [
            call_tool("assess_stock_entry", focus="短线", entry_price=100),
            request_market("short-market"), call_tool("Finish", answer="量价 [EV_TECH_001]"),
            call_tool("search_news_catalyst", symbol=symbol), call_tool("Finish", answer="催化 [EV_NEWS_001]"),
            call_tool("check_risk_and_notices", symbol=symbol), call_tool("Finish", answer="风险 [EV_RISK_001]"),
            call_tool("Finish", answer="多头材料 [EV_TECH_001]"),
            call_tool("Finish", answer="空头材料 [EV_TECH_001]"),
            answer("多空理由"),
            call_tool("assess_stock_entry", update_risk_only=True, risk_percent=3),
            answer("风险参考价 97 元"),
        ]
        self.llm.stream_invoke.side_effect = [
            (chunk for chunk in ["96 元失效 [EV_TECH_001]"]),
            (chunk for chunk in ["认可条件 [EV_TECH_001]"]),
        ]
        self.llm.last_call_stats = SimpleNamespace(finish_reason="stop")
        self.llm.invoke.side_effect = [
            answer('{"reason":"个人风险上限待补"}'),
            answer(json.dumps({
                "bull": "理由 [EV_TECH_001]", "bear": "反驳 [EV_TECH_001]",
                "condition": "96 元失效", "price": 96, "agreed": True,
                "evidence_ids": ["EV_TECH_001"], "reconsider_when": "交易条件更新",
            })),
        ]
        self.assertEqual(self.assistant.ask("假设 100 元入场，先听短线多空理由"), "多空理由")
        initial = self.assistant.last_entry
        self.assertIsNone(initial.risk_pct)
        self.assertEqual(initial.transcript.rounds, 1)
        self.assertEqual(self.assistant.ask("我最多接受跌 3%，其余条件不变"), "风险参考价 97 元")
        self.assertIs(self.assistant.last_entry.transcript, initial.transcript)
        self.assertIn("97.0000", self.assistant.last_entry.render())
        self.assertIn("超出风险范围", self.assistant.last_entry.render())
        self.fetch.assert_called_once()
        self.assertEqual(self.llm.stream_invoke.call_count, 2)

    def test_repeated_request_reuses_same_evidence(self):
        self.llm.invoke_with_tools.side_effect = [
            request_market("first"), answer("第一轮"),
            request_market("second", lookback_days=180), answer("第二轮"),
        ]
        self.assistant.ask("查行情")
        self.assistant.ask("再查一次")
        self.assertEqual(self.llm.invoke_with_tools.call_count, 4)
        self.fetch.assert_called_once()
        self.assertEqual(list(self.session.evidence.items), ["EV_TECH_001"])
        self.assertEqual(len(self.session.evidence.bundles), 1)

    def test_out_of_scope_request_does_not_fetch(self):
        self.llm.invoke_with_tools.side_effect = [
            request_market("wrong_symbol", symbol="600519.SH"), answer("标的超出范围"),
            request_market("wrong_date", as_of_date="2026-11-21"), answer("日期超出范围"),
        ]
        self.assistant.ask("看看另一只股票")
        self.assistant.ask("查更晚的行情")
        self.fetch.assert_not_called()
        self.assertFalse(self.session.evidence.bundles)

    def test_failed_fetch_can_be_retried(self):
        short_bars = self.fetch("300750.SZ", "2026-11-20", 180).tail(39)
        self.fetch.reset_mock()
        self.fetch.side_effect = [RuntimeError("offline"), short_bars]
        self.llm.invoke_with_tools.side_effect = [
            request_market("first"), answer("取数失败"),
            request_market("retry"), answer("已取数"),
        ]
        self.assistant.ask("查行情")
        self.assertFalse(self.session.evidence.items)
        self.assistant.ask("重试")
        self.assertEqual(self.fetch.call_count, 2)
        item = self.session.evidence.items["EV_TECH_001"]
        self.assertIn("最近 39 个交易日最大回撤", item.summary)
        self.assertIsNone(item.payload["ma60"])
