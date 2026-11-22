from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock, Thread
from time import monotonic
from typing import Callable

from hello_agents.core.budget import RunBudget


@dataclass(frozen=True)
class DebateEvent:
    kind: str
    round_num: int = 0
    speaker: str = ""
    text: str = ""


DebateListener = Callable[[DebateEvent], None]


class ResearchRun:
    def __init__(self, seconds: float = 175, reserve: float = 35) -> None:
        self.deadline = monotonic() + seconds
        self.reserve = reserve
        self.closed = False
        self.lock = RLock()
        self.transcript = None

    def remaining(self, reserve: float = 0) -> float:
        return self.deadline - monotonic() - reserve

    def check(self) -> None:
        if self.closed or self.remaining() <= 0:
            raise TimeoutError("研究时间已用尽")

    @contextmanager
    def scope(self, reserve: float = 0):
        remaining = self.remaining(reserve)
        if self.closed or remaining <= 0:
            raise TimeoutError("研究时间已用尽")
        token = current_run.set(self)
        try:
            with RunBudget(max_seconds=remaining).scope():
                yield
        finally:
            current_run.reset(token)

    def close(self) -> None:
        with self.lock:
            self.closed = True

    def start(self, operation) -> Future:
        future = Future()

        def execute():
            if future.set_running_or_notify_cancel():
                try:
                    self.check()
                    future.set_result(operation())
                except BaseException as exc:
                    future.set_exception(exc)

        Thread(target=execute, name="marketoracle-research", daemon=True).start()
        return future


current_run: ContextVar[ResearchRun | None] = ContextVar("marketoracle_research_run", default=None)
