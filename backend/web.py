"""Local HTTP bridge for the MarketOracle frontend."""

import asyncio
from dataclasses import asdict
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from threading import Lock
from urllib.parse import urlsplit

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

from .agents.assistant import ANSWER_STYLES, StockChatAssistant
from .agents.run_full_pipeline import ResearchIncomplete
from .session import ResearchSession


class ResearchServer(ThreadingHTTPServer):
    def __init__(self, port=8765):
        super().__init__(("127.0.0.1", port), ResearchHandler)
        self.sessions = {}
        self.lock = Lock()


class ResearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
            self.send_error(403)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_error(415)
            return
        if self.path != "/api/research":
            self.send_error(404)
            return
        try:
            size = int(self.headers["Content-Length"])
            if not 0 < size <= 20000:
                raise ValueError()
            request = json.loads(self.rfile.read(size))
            symbol, as_of, token = request["symbol"], request["asOf"], request["session"]
            mode, question, action, mock = (
                request["mode"], request["question"], request["action"], request["mock"]
            )
            if (not re.fullmatch(r"[0-9]{6}(?:\.(?:SH|SZ|BJ))?", symbol)
                or date.fromisoformat(as_of).isoformat() != as_of
                or not re.fullmatch(r"[a-f0-9-]{36}", token)
                or mode not in ANSWER_STYLES
                or not isinstance(question, str) or len(question) > 2000
                or action not in {"ask", "report"} or (action == "ask" and not question.strip())
                or type(mock) is not bool):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            self.send_error(400)
            return

        key = token, symbol, as_of, mock
        with self.server.lock:
            if key not in self.server.sessions:
                if len(self.server.sessions) >= 12:
                    for old, (assistant, lock) in self.server.sessions.items():
                        if not lock.locked():
                            break
                    else:
                        self.send_error(503)
                        return
                    self.server.sessions.pop(old)
                    assistant.llm.close()
                try:
                    llm = HelloAgentsLLM(
                        model=os.getenv("MARKETORACLE_MODEL", "deepseek-chat"),
                        api_key=os.getenv("MARKETORACLE_MODEL_KEY"),
                        base_url=os.getenv("MARKETORACLE_MODEL_URL"),
                    )
                except Exception:
                    self.send_error(503, "Model configuration unavailable")
                    return
                self.server.sessions[key] = (
                    StockChatAssistant(ResearchSession(symbol, as_of), llm, use_mock=mock), Lock()
                )
            assistant, lock = self.server.sessions[key]
            if not lock.acquire(blocking=False):
                self.send_error(409)
                return

        def send(event_type, **data):
            line = json.dumps({"type": event_type, **data}, ensure_ascii=False) + "\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            assistant.listener = lambda event: send("debate", **asdict(event))
            assistant.set_mode(mode)
            incomplete = False
            try:
                if action == "report":
                    report = assistant.research(question)
                    answer = report.executive_summary if mode == "小白" else report.to_markdown()
                else:
                    answer = asyncio.run(assistant.stream_answer(question, send))
            except ResearchIncomplete as exc:
                answer = str(exc)
                incomplete = True
            except (ConnectionError, OSError):
                return
            except Exception:
                send("error", message="研究请求失败。请检查本地模型配置或稍后重试。")
                return
            send(
                "result", answer=answer, incomplete=incomplete,
                evidence=[item.model_dump(mode="json") for item in assistant.session.evidence.items.values()],
                report=assistant.last_report.model_dump(mode="json") if assistant.last_report else None,
                entry=assistant.last_entry.render() if assistant.last_entry else None,
            )
        except (ConnectionError, OSError):
            pass
        finally:
            assistant.listener = None
            lock.release()


if __name__ == "__main__":
    load_dotenv()
    with ResearchServer() as server:
        print("MarketOracle API · http://127.0.0.1:8765", flush=True)
        server.serve_forever()
