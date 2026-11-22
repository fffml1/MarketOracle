import argparse
from datetime import date
import os
import re

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

from .agents.assistant import ANSWER_STYLES, StockChatAssistant
from .agents.debate_stream import DebateEvent
from .agents.run_full_pipeline import ResearchIncomplete
from .session import ResearchSession


def show_debate(event: DebateEvent) -> None:
    if event.kind == "start":
        speaker = "多头" if event.speaker == "Bull" else "空头"
        print(f"\n第 {event.round_num} 轮 · {speaker}（准备发言）", flush=True)
    elif event.kind == "text":
        print(event.text, end="", flush=True)
    elif event.kind in {"end", "interrupted"}:
        print(flush=True)
        if event.kind == "interrupted":
            print("本次发言未完成。", flush=True)
        if event.text:
            print(event.text, flush=True)
    elif event.kind == "continue":
        print(f"下一轮核实：{event.text}", flush=True)
    elif event.kind == "stop":
        print(f"已完成 {event.round_num} 轮。{event.text}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MarketOracle 单股问答与深度研报")
    parser.add_argument("--symbol", required=True, help="六位 A 股代码，可带 .SH / .SZ / .BJ")
    parser.add_argument("--as-of", required=True, help="研究基准日 YYYY-MM-DD")
    parser.add_argument("--mock-data", action="store_true", help="使用模拟行情、新闻和风险资料")
    args = parser.parse_args()
    symbol = args.symbol.strip().upper()
    if not re.fullmatch(r"[0-9]{6}(?:\.(?:SH|SZ|BJ))?", symbol):
        parser.error("--symbol 需要六位股票代码，可附交易所后缀")
    try:
        as_of = date.fromisoformat(args.as_of).isoformat()
    except ValueError:
        parser.error("--as-of 需要有效的 YYYY-MM-DD 日期")
    if as_of != args.as_of:
        parser.error("--as-of 需要 YYYY-MM-DD 格式")

    load_dotenv()
    data_label = "模拟行情和资讯" if args.mock_data else "真实行情和资讯"
    print(f"{symbol} | 基准日 {as_of} | {data_label} | 小白模式")
    print("输入 /专业 或 /小白 切换风格，输入 /研报 [关注点] 生成完整研报，输入 exit 退出。")
    with HelloAgentsLLM(
        model=os.getenv("MARKETORACLE_MODEL", "deepseek-chat"),
        api_key=os.getenv("MARKETORACLE_MODEL_KEY"),
        base_url=os.getenv("MARKETORACLE_MODEL_URL"),
    ) as llm:
        assistant = StockChatAssistant(
            ResearchSession(symbol, as_of), llm, use_mock=args.mock_data, listener=show_debate,
        )
        while True:
            try:
                question = input("你> ").strip()
                if question.lower() == "exit":
                    break
                if question.startswith("/") and question[1:] in ANSWER_STYLES:
                    mode = question[1:]
                    assistant.set_mode(mode)
                    print(f"已切换到{mode}模式。")
                    continue
                if question:
                    try:
                        command, _, focus = question.partition(" ")
                        if command == "/研报":
                            print("正在调研、辩论并生成研报。", flush=True)
                            report = assistant.research(focus.strip())
                            if args.mock_data:
                                print("本报告使用模拟数据。")
                            if assistant.mode == "小白":
                                print(assistant.ask("请用白话概括刚生成的研报，说清多空分歧和主要风险。"))
                            else:
                                print(report.to_markdown())
                        else:
                            print(assistant.ask(question))
                    except ResearchIncomplete as exc:
                        print(str(exc))
                        for item in list(assistant.session.evidence.items.values())[:4]:
                            print(f"[{item.evidence_id}] {item.summary}")
                    except Exception:
                        print("本轮回答生成失败。可检查模型配置或连接后重试。")
            except (EOFError, KeyboardInterrupt):
                print()
                break


if __name__ == "__main__":
    main()
