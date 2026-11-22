from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from backend.agents.assistant import StockChatAssistant
from backend.agents.debate_stream import ResearchRun
from backend.agents.run_full_pipeline import conduct_research, run_stock_deep_research
from backend.rag.ingest import DIMENSIONS, MODEL, metadata
from backend.rag.query import AnnouncementQueries, RagError
from backend.rag.source import Announcement
from backend.session import ResearchSession
from backend.tests.test_stock_chat import answer, call_tool
from backend.tools.announcement_rag import AnnouncementSearchTool, FinancialMetricTool
from backend.tools.hello_adapter import EvidenceStore


SYMBOL = "000858.SZ"
EARLY = "2025-11-01"
LATE = "2026-05-01"
METRIC = {"period": "2025年三季度", "metric": "营业收入", "scope": "本期"}
ENDPOINT = "https://example.org/compatible-mode/v1/embeddings"


def table(block, identifier, current, cumulative=None, page=1):
    fields = []
    for header, value in (("本报告期", current), ("年初至报告期末", cumulative)):
        if value is not None:
            fields.append({"role": "数值", "unit": "元", "header": {"text": header, "page": page},
                           "value": {"text": value, "page": page}, "unit_source": {"page": page}})
    return {**block, "id": identifier, "kind": "表格行", "section": "主要会计数据",
            "page_start": page, "page_end": page, "metric": {"text": "营业收入（元）"}, "fields": fields}


class AnnouncementRagTests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.index = Path(workspace.name) / "index.csv"
        self.output = self.index.parent / "rag" / "wuliangye"
        (self.output / "chroma").mkdir(parents=True)
        (self.output / "chroma" / "chroma.sqlite3").touch()
        sources = [Announcement(identifier, "000858", "五粮液", category, date, title,
                                f"https://example.org/{identifier}.pdf", "unused.pdf", identifier)
                   for identifier, category, date, title in (
                       ("old", "定期报告", "2025-10-31", "2025年第三季度报告（更新前）"),
                       ("new", "定期报告", "2026-04-30", "2025年第三季度报告（更新后）"),
                       ("notice1", "补充更正", "2026-04-30", "会计差错更正公告"),
                       ("notice2", "补充更正", "2026-05-01", "另一份会计差错更正公告"),
                       ("future", "补充更正", "2026-05-02", "后续公告"))]
        with self.index.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=asdict(sources[0]))
            writer.writeheader()
            writer.writerows(asdict(source) for source in sources)
        self.blocks = {source.id: {
            "id": source.id, "announcement_id": source.id, "stock_code": source.code, "period": source.period,
            "category": source.category, "version": source.version, "kind": "正文", "page_start": 1, "page_end": 1,
            "sha256": source.sha256, "source_url": source.url, "parser": "test", "rule_version": "1",
            "disclosed_on": source.date, "original_text": "归类口径调整。" + "原" * 600,
        } for source in sources}
        self.blocks.update({block["id"]: block for block in (
            table(self.blocks["old"], "old-table", "8,174,336,700.05", "60,945,321,083.57"),
            table(self.blocks["new"], "new-table", "7,127,755,606.32"),
            table(self.blocks["new"], "new-table-page2", "7127755606.32", page=2))})
        self.collection = Mock()
        self.opener = Mock(return_value=self.collection)
        self.embed = Mock(return_value=[[0.5] * DIMENSIONS])
        self.service = AnnouncementQueries(self.index, opener=self.opener, embed=self.embed)
        self.write_snapshot()
        env = patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fixture-key", "MARKETORACLE_EMBEDDING_URL": ENDPOINT,
                                      "MARKETORACLE_RAG_CHAT_METRIC": "1", "MARKETORACLE_RAG_CHAT_SEARCH": "0",
                                      "MARKETORACLE_RAG_RESEARCH": "0"})
        env.start()
        self.addCleanup(env.stop)
        tokenizer = patch("hello_agents.context.token_counter.TokenCounter._get_encoding", return_value=None)
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def write_snapshot(self):
        data = "\n".join(json.dumps(block, ensure_ascii=False) for block in self.blocks.values()).encode()
        digest = sha256(data).hexdigest()
        count = len(self.blocks)
        (self.output / "fragments.jsonl").write_bytes(data)
        (self.output / "report.json").write_text(json.dumps({"passed": True, "fragment_file_sha256": digest,
                                                           "source_checks": {"fragment_citations_checked": count}}))
        config = {"fragment_file_sha256": digest, "model": MODEL, "dimensions": DIMENSIONS, "endpoint": ENDPOINT}
        fingerprint = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.ready = {**config, "complete": True, "count": count, "collection": f"wuliangye-{fingerprint[:40]}"}
        (self.output / "chroma_ready.json").write_text(json.dumps(self.ready))
        self.collection.metadata = config
        self.collection.count.return_value = count

    def hits(self, *ids):
        self.collection.query.return_value = {"ids": [list(ids)],
                                              "metadatas": [[metadata(self.blocks[key]) for key in ids]]}

    def test_parallel_dates_and_cache_do_not_use_embeddings(self):
        def ask(day):
            store = EvidenceStore()
            tool = FinancialMetricTool(store, SYMBOL, day, service=self.service)
            first = tool.run(METRIC)
            self.assertEqual(first.text, tool.run({**METRIC, "period": "2025年第三季度"}).text)
            self.assertEqual(len(store.bundles), 1)
            return store.items["EV_FUND_001"].payload

        with ThreadPoolExecutor(max_workers=2) as executor:
            early, late = executor.map(ask, (EARLY, LATE))
        self.assertEqual((early["announcement_id"], early["value"]), ("old", "8,174,336,700.05"))
        self.assertEqual((late["announcement_id"], late["value"], late["pages"]), ("new", "7,127,755,606.32", [1, 2]))
        self.assertEqual(late["fragment_ids"], ["new-table", "new-table-page2"])
        self.embed.assert_not_called()
        self.opener.assert_not_called()

    def test_financial_errors_do_not_write_or_fall_back(self):
        self.blocks["conflict"] = table(self.blocks["new"], "conflict", "1", page=3)
        self.write_snapshot()
        store = EvidenceStore()
        for symbol, day, parameters, kind in (
            ("300750.SZ", LATE, METRIC, "NOT_COVERED"), (SYMBOL, "2025-10-01", METRIC, "NO_REPORT"),
            (SYMBOL, LATE, {**METRIC, "scope": "累计"}, "NO_VERIFIED_VALUE"),
            (SYMBOL, LATE, METRIC, "CONFLICTING_VALUES"),
            (SYMBOL, EARLY, {**METRIC, "as_of": LATE}, "INVALID_PARAMS"),
        ):
            with self.subTest(kind=kind):
                response = FinancialMetricTool(store, symbol, day, service=self.service).run(parameters)
                self.assertEqual(response.error_info["code"], kind)
        self.assertEqual(store.items, {})

    def test_chat_uses_new_scope_and_retains_compact_page_evidence(self):
        llm = Mock(model="test")
        llm.invoke_with_tools.side_effect = [
            call_tool("get_financial_metric", **METRIC), answer("单季 [EV_FUND_001]"),
            call_tool("get_financial_metric", **{**METRIC, "scope": "累计"}), answer("累计 [EV_FUND_002]"),
            answer("第 1 物理页 [EV_FUND_002]"), answer("单季还是前三季度累计？"),
        ]
        session = ResearchSession(SYMBOL, EARLY)
        with patch("backend.agents.assistant.FinancialMetricTool",
                   side_effect=lambda *args, **kwargs: FinancialMetricTool(*args, **kwargs, service=self.service)):
            assistant = StockChatAssistant(session, llm)
        with patch.object(assistant, "research") as research:
            for question in ("五粮液2025年第三季度单季营收多少？", "那前三季度累计呢？", "原文第几页？"):
                assistant.ask(question)
            self.assertIn("累计？", assistant.ask("2024年三季报营收呢？"))
            research.assert_not_called()
        self.assertEqual(llm.invoke_with_tools.call_count, 6)
        self.assertEqual(session.evidence.items["EV_FUND_002"].payload["value"], "60,945,321,083.57")
        context = llm.invoke_with_tools.call_args_list[4].kwargs["messages"][-1]["content"]
        self.assertIn("PDF 物理页", context)
        self.assertIn("https://example.org/old.pdf", context)
        self.assertNotIn('"payload"', context)
        self.assertIn("先问单季还是前三季度累计", assistant.instructions)

    def test_search_prefilters_versions_and_keeps_independent_notices(self):
        self.hits("new", "notice1", "notice1")
        store = EvidenceStore()
        tool = AnnouncementSearchTool(store, SYMBOL, LATE, service=self.service)
        self.assertIn("仅作证据", tool.run({"query": "更正原因"}).text)
        where = self.collection.query.call_args.kwargs["where"]["$and"]
        self.assertEqual(where, [{"stock_code": "000858"}, {"announcement_id": {"$in": ["new", "notice1", "notice2"]}},
                                 {"disclosed_on": {"$lte": 20260501}}, {"kind": "正文"}])
        self.assertEqual(self.embed.call_args.args[2], ["更正原因"])
        self.assertEqual(self.embed.call_args.kwargs["timeout"], 10)
        self.assertEqual(len(store.items), 2)
        self.assertEqual(len(store.items["EV_NOTICE_002"].payload["excerpt"]), 500)
        tool.run({"query": "另一个检索短句"})
        self.assertEqual(len(store.bundles), 1)
        self.hits("old")
        self.assertEqual(tool.run({"query": "排除过期版本"}).error_info["code"], "HIT_INVALID")
        self.hits("future")
        self.assertEqual(tool.run({"query": "排除未来公告"}).error_info["code"], "HIT_INVALID")
        self.assertEqual(len(store.items), 2)

    def test_search_faults_are_distinct_from_no_hits_and_exact_still_works(self):
        def search():
            return self.service.search(SYMBOL, LATE, "更正原因")

        self.hits()
        self.assertEqual(search(), [])
        for target, injected, kind in ((self.opener, ValueError("missing"), "COLLECTION_UNAVAILABLE"),
                                       (self.embed, TimeoutError("fixture-key"), "EMBEDDING_TIMEOUT")):
            target.side_effect = injected
            try:
                with self.assertRaises(RagError) as raised:
                    search()
            finally:
                target.side_effect = None
            self.assertEqual(raised.exception.code, kind)
            self.assertNotIn("fixture-key", str(raised.exception))
        with patch.object(self.embed, "return_value", [[0.5]]), self.assertRaises(RagError) as raised:
            search()
        self.assertEqual(raised.exception.code, "EMBEDDING_INVALID")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}), self.assertRaises(RagError) as raised:
            search()
        self.assertEqual(raised.exception.code, "EMBEDDING_CONFIG")
        (self.output / "chroma_ready.json").write_text(json.dumps({**self.ready, "fragment_file_sha256": "wrong"}))
        with self.assertRaises(RagError) as raised:
            search()
        self.assertEqual(raised.exception.code, "INDEX_INVALID")
        self.assertEqual(self.service.metric(SYMBOL, LATE, **METRIC)["status"], "找到")

    def test_search_reassembles_wrapped_lines_without_crossing_source_boundaries(self):
        self.blocks["notice1"].update(section="更正原因", original_text="更正原因")
        continuation = {**self.blocks["notice1"], "id": "notice1-line1", "original_text": "公司调整部分"}
        tail = {**continuation, "id": "notice1-line2", "original_text": "收入确认相关核算。"}
        next_page = {**continuation, "id": "notice1-page2", "page_start": 2, "page_end": 2, "original_text": "另一页"}
        ordered = {}
        for key, block in self.blocks.items():
            ordered[key] = block
            if key == "notice1":
                ordered.update((extra["id"], extra) for extra in (continuation, tail, next_page))
        self.blocks = ordered
        self.write_snapshot()
        store = EvidenceStore()
        tool = AnnouncementSearchTool(store, SYMBOL, LATE, service=self.service)
        for hits in (("notice1", "notice1-line1", "notice1-line2"), ("notice1-line2", "notice1")):
            self.hits(*hits)
            self.assertIn("公司调整部分\n收入确认相关核算。", tool.run({"query": "更正原因"}).text)
        self.assertEqual(list(store.items), ["EV_NOTICE_001"])
        self.assertEqual(store.items["EV_NOTICE_001"].payload["fragment_ids"], ["notice1-line1", "notice1-line2"])

    def test_cancelled_lookup_does_not_record_late_evidence(self):
        store = EvidenceStore()
        entered, release = Event(), Event()
        self.hits("notice1")

        def embed(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError()
            return [[0.5] * DIMENSIONS]

        self.embed.side_effect = embed
        tool = AnnouncementSearchTool(store, SYMBOL, LATE, service=self.service)
        run = ResearchRun()

        def pending():
            with run.scope():
                return tool.run({"query": "更正原因"})

        future = run.start(pending)
        try:
            self.assertTrue(entered.wait(3))
            run.close()
        finally:
            release.set()
        self.assertEqual(future.result(timeout=3).error_info["code"], "RESEARCH_CANCELLED")
        self.assertEqual(store.items, {})

    def test_standalone_mock_research_registries_share_evidence_without_real_queries(self):
        llm = Mock(model="test", last_call_stats=SimpleNamespace(finish_reason="stop"))
        llm.invoke_with_tools.side_effect = [
            call_tool("get_financial_metric", **METRIC), call_tool("Finish", answer="模拟财报 [EV_FUND_001]"),
            call_tool("search_announcements", query="更正原因"), call_tool("Finish", answer="模拟原文 [EV_NOTICE_001]"),
            call_tool("search_announcements", query="更正风险"), call_tool("Finish", answer="模拟风险 [EV_NOTICE_001]"),
            call_tool("Finish", answer="沿用模拟证据 [EV_FUND_001]"),
            call_tool("Finish", answer="沿用模拟证据 [EV_NOTICE_001]"),
        ]
        draft = {
            "headline_verdict": "观察 [EV_FUND_001]", "rating": "中性观望", "confidence_score": 55,
            "confidence_rationale": "模拟", "executive_summary": "模拟财报与更正 [EV_NOTICE_001]",
            "debate_synthesis": {"bull_strongest_argument": "模拟 [EV_FUND_001]",
                                 "bear_strongest_argument": "模拟 [EV_NOTICE_001]"},
            "action_plan": {"stance": "观望等待确认", "entry_zone": "等待确认", "stop_loss_rule": "重估",
                            "position_advice": "观察"},
        }
        llm.invoke.side_effect = [answer('{"reason":"分歧已记录"}'), answer(json.dumps(draft)),
                                  answer('{"passed":true,"issues":[],"summary":"核查通过"}')]
        llm.stream_invoke.side_effect = [(chunk for chunk in ["模拟财报 [EV_FUND_001]"]),
                                         (chunk for chunk in ["模拟更正 [EV_NOTICE_001]"])]
        with (patch.dict(os.environ, {"MARKETORACLE_RAG_RESEARCH": "1"}),
              patch("backend.agents.run_full_pipeline.HelloAgentsLLM", return_value=llm),
              patch("backend.agents.run_full_pipeline._save_trace") as save,
              patch("backend.agents.run_full_pipeline.conduct_research") as conduct,
              patch("backend.tools.announcement_rag.queries.version") as read_real):
            def collect(symbol, day, focus, store, registries, *llms, **kwargs):
                search = registries["catalyst"].get_tool("search_announcements")
                self.assertIs(search, registries["risk"].get_tool("search_announcements"))
                self.assertIsNone(registries["debater"].get_tool("search_announcements"))
                return conduct_research(symbol, day, focus, store, registries, *llms, **kwargs)

            conduct.side_effect = collect
            report = run_stock_deep_research(SYMBOL, LATE, use_mock=True)
        self.assertEqual(report.cited_evidence_ids, ["EV_FUND_001", "EV_NOTICE_001"])
        store = save.call_args.args[-1]
        self.assertEqual(store.get_valid_ids(), set(report.cited_evidence_ids))
        self.assertTrue(all(item.source == "simulated_announcement" for item in store.items.values()))
        read_real.assert_not_called()
