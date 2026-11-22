import asyncio
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from hello_agents.tools import ToolRegistry

from backend.agents.assistant import DeepResearchTool, EntryAssessmentTool, StockChatAssistant
from backend.agents.debate_arena import DebateArena, DebateTranscript, DebateTurn
from backend.agents.debate_stream import DebateEvent, ResearchRun
from backend.agents.entry_assessment import EntryAssessment, EntryFinding
from backend.agents.research_team import ResearchBriefs
from backend.agents.run_full_pipeline import ResearchIncomplete, conduct_research
from backend.cli import show_debate
from backend.session import ResearchSession
from backend.tools.hello_adapter import EvidenceStore


SYMBOL = "300750.SZ"
AS_OF = "2026-11-20"
BRIEFS = ResearchBriefs(SYMBOL, AS_OF, "量价", "催化", "风险")


class LiveDebateTests(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.llm.last_call_stats = SimpleNamespace(finish_reason="stop")
        self.prep = SimpleNamespace(
            last_run=SimpleNamespace(status="completed"), run=MagicMock(return_value="补证材料")
        )
        self.store = EvidenceStore()
        self.registry = ToolRegistry()
        self.events = []
        patcher = patch("backend.agents.debate_arena.ReActAgent", return_value=self.prep)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.arena = DebateArena(self.llm, self.registry, self.store, max_rounds=3)

    def test_public_text_is_visible_before_speech_finishes_and_matches_record(self):
        output = StringIO()

        def speech(_messages):
            yield "第一段"
            self.assertTrue(output.getvalue().endswith("第一段"))
            yield "第二段"

        self.llm.stream_invoke.side_effect = speech
        self.llm.invoke.return_value.content = '{"reason":"首轮已说清"}'
        with redirect_stdout(output):
            transcript = self.arena.run_debate(BRIEFS, dynamic=True, listener=show_debate)
        self.assertEqual(transcript.rounds, 1)
        self.assertEqual(transcript.stop_reason, "首轮已说清")
        self.assertEqual([turn.content for turn in transcript.turns], ["第一段第二段"] * 2)
        self.assertEqual(output.getvalue().count("第一段第二段"), 2)
        self.assertNotIn("补证材料", output.getvalue())

    def test_continues_with_original_conditions_and_keeps_final_objection(self):
        self.llm.stream_invoke.side_effect = lambda _: (chunk for chunk in ["公开正文"])
        self.llm.invoke.side_effect = [
            SimpleNamespace(content='{"issue":"旧订单","impact":"催化","action":"核对披露日"}'),
            SimpleNamespace(content='{"issue":"客户变更","impact":"主要风险","action":"查原公告"}'),
            SimpleNamespace(content='{"issue":"金额反证","impact":"判断","action":"核对补充公告"}'),
        ]
        transcript = self.arena.run_debate(
            BRIEFS, "拟100元入场", dynamic=True, listener=self.events.append,
        )
        self.assertEqual(transcript.rounds, 3)
        self.assertIn("金额反证", transcript.unresolved)
        self.assertIn("核对披露日", self.prep.run.call_args_list[2].args[0])
        self.assertIn("拟100元入场", self.prep.run.call_args_list[2].args[0])
        self.assertEqual([e.kind for e in self.events].count("continue"), 2)

    def test_timeout_retains_partial_text_without_completing_round(self):
        def speech(_messages):
            yield "已显示的内容"
            raise TimeoutError

        self.llm.stream_invoke.side_effect = speech
        transcript = self.arena.run_debate(BRIEFS, dynamic=True, listener=self.events.append)
        self.assertEqual(transcript.rounds, 0)
        self.assertEqual(transcript.turns[0].content, "已显示的内容")
        self.assertFalse(transcript.turns[0].completed)
        self.assertEqual([e.kind for e in self.events], ["start", "text", "interrupted", "stop"])

    def test_old_transcript_and_fixed_debate(self):
        old = DebateTranscript.model_validate({
            "symbol": SYMBOL, "as_of_date": AS_OF, "rounds": 2,
            "turns": [{"round_num": 1, "speaker": "Bull", "content": "原文"}],
        })
        self.assertIn("已完成 0 轮", old.to_markdown())
        self.assertIn("未记录", old.to_markdown())
        arena = DebateArena(self.llm, self.registry, self.store, max_rounds=2)
        self.assertEqual(arena.run_debate(BRIEFS).completed_rounds(), 2)
        self.llm.stream_invoke.assert_not_called()

    @patch("backend.agents.run_full_pipeline.CIOJudgeAgent")
    @patch("backend.agents.run_full_pipeline.ResearchTeam")
    def test_report_judge_timeout_keeps_completed_debate(self, team, judge):
        team.return_value.run_all.return_value = BRIEFS
        self.llm.stream_invoke.side_effect = lambda _: (chunk for chunk in ["发言"])
        self.llm.invoke.return_value.content = '{"reason":"无需续辩"}'
        judge.return_value.adjudicate.side_effect = TimeoutError
        with self.assertRaises(ResearchIncomplete) as raised:
            conduct_research(
                SYMBOL, AS_OF, "", self.store, {"debater": self.registry}, self.llm, self.llm,
                listener=self.events.append,
            )
        self.assertEqual(raised.exception.transcript.rounds, 1)
        self.assertEqual(self.events[-1].kind, "stop")


class AssistantDebateTests(unittest.TestCase):
    def setUp(self):
        self.assistant = StockChatAssistant(ResearchSession(SYMBOL, AS_OF), MagicMock(), use_mock=True)

    @patch("backend.agents.assistant.research_debate")
    def test_risk_only_update_reuses_entry_and_transcript(self, research):
        transcript = DebateTranscript(symbol=SYMBOL, as_of_date=AS_OF, rounds=1)
        finding = EntryFinding(bull="理由", bear="反驳", condition="96 元失效", reconsider_when="公告更新")
        self.assistant.last_entry = EntryAssessment(
            "", Decimal(100), "用户假设", None, transcript, finding, Decimal(96),
        )
        tool = EntryAssessmentTool(self.assistant)
        self.assertIn("个人风险上限待确认", self.assistant.last_entry.render())
        lower = tool.run({"update_risk_only": True, "risk_percent": 3})
        upper = tool.run({"update_risk_only": True, "risk_percent": 5})
        self.assertIn("97.0000", lower.text)
        self.assertIn("超出风险范围", lower.text)
        self.assertIn("95.0000", upper.text)
        self.assertIn("公告更新", upper.text)
        self.assertIs(self.assistant.last_entry.transcript, transcript)
        research.assert_not_called()


class AssistantAsyncDebateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assistant = StockChatAssistant(ResearchSession(SYMBOL, AS_OF), MagicMock(), use_mock=True)

    @patch("backend.agents.assistant.conduct_research")
    async def test_report_command_and_natural_language_tool_share_progress(self, conduct):
        transcript = DebateTranscript(symbol=SYMBOL, as_of_date=AS_OF, rounds=1)
        report = MagicMock()
        report.to_markdown.return_value = "研报"
        output = StringIO()
        self.assistant.listener = show_debate

        def research(*_args, listener, **_kwargs):
            listener(DebateEvent("start", 1, "Bull"))
            listener(DebateEvent("text", 1, "Bull", "正在生成的正文"))
            self.assertTrue(output.getvalue().endswith("正在生成的正文"))
            return report, transcript

        conduct.side_effect = research
        with redirect_stdout(output):
            self.assertIs(self.assistant.research("命令关注点"), report)
            response = await DeepResearchTool(self.assistant).arun({"focus": "自然语言关注点"})
        self.assertEqual(response.text, "研报")
        self.assertEqual(output.getvalue().count("正在生成的正文"), 2)

    async def test_cancelled_tool_preserves_partial_record_and_rejects_late_output(self):
        entered, release, finished = Event(), Event(), Event()
        events = []
        self.assistant.listener = events.append
        tool = DeepResearchTool(self.assistant)

        def research(_parameters, run: ResearchRun):
            run.transcript = DebateTranscript(
                symbol=SYMBOL, as_of_date=AS_OF, rounds=0,
                turns=[DebateTurn(round_num=1, speaker="Bull", content="片段", completed=False)],
            )
            entered.set()
            try:
                release.wait(2)
                self.assistant._emit(run, DebateEvent("text", 1, "Bull", "过期片段"))
            finally:
                finished.set()

        with patch.object(tool, "execute_research", side_effect=research):
            task = asyncio.create_task(tool.arun({}))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            self.assertTrue(await asyncio.to_thread(finished.wait, 2))
        self.assertEqual(self.assistant.last_attempt_transcript.turns[0].content, "片段")
        self.assertFalse(any(e.text == "过期片段" for e in events))
