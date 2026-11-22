from http.client import HTTPConnection
import json
from threading import Thread
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from backend.agents.debate_stream import DebateEvent
from backend.web import ResearchServer


class WebResearchTests(unittest.TestCase):
    def setUp(self):
        self.server = ResearchServer(port=0)
        self.worker = Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        self.addCleanup(self.connection.close)

    def test_research_stream_and_session_reuse(self):
        request = {
            "symbol": "300750.SZ", "asOf": "2026-11-20",
            "session": str(uuid4()), "mock": True, "mode": "小白",
            "action": "ask", "question": "先听多空理由",
        }
        assistant = MagicMock(last_report=None, last_entry=None)
        assistant.session.evidence.items = {}

        async def stream_answer(question, publish):
            assistant.listener(DebateEvent("start", 1, "Bull"))
            assistant.listener(DebateEvent("text", 1, "Bull", "公开正文"))
            publish("answer_delta", text="**")
            publish("answer_delta", text=question)
            return question

        assistant.stream_answer.side_effect = stream_answer
        with patch("backend.web.HelloAgentsLLM"), patch("backend.web.StockChatAssistant", return_value=assistant) as factory:
            for question in ["先听多空理由", "最多承受 3%"]:
                request["question"] = question
                self.connection.request(
                    "POST", "/api/research", json.dumps(request),
                    {"Content-Type": "application/json"},
                )
                response = self.connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.readline())["kind"], "start")
                self.assertEqual(json.loads(response.readline())["text"], "公开正文")
                self.assertEqual(json.loads(response.readline()), {"type": "answer_delta", "text": "**"})
                self.assertEqual(json.loads(response.readline()), {"type": "answer_delta", "text": question})
                self.assertEqual(json.loads(response.readline())["answer"], question)
                self.assertFalse(response.read())
            factory.assert_called_once()

    def test_external_browser_origin_is_rejected(self):
        self.connection.request(
            "POST", "/api/research", "{}",
            {"Content-Type": "application/json", "Origin": "https://example.org"},
        )
        response = self.connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
