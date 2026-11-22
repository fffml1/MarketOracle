from dataclasses import dataclass, replace
from decimal import Decimal

from pydantic import BaseModel, Field

from .debate_arena import DebateTranscript, model_json
from ..tools.hello_adapter import EvidenceStore


def parse_number(value, *, percentage: bool = False) -> Decimal | None:
    if value is None:
        return None
    try:
        number = Decimal(str(value))
        valid = number.is_finite() and (0 <= number <= 100 if percentage else number > 0)
    except (ValueError, ArithmeticError):
        valid = False
    if not valid:
        label = "可承受跌幅应在 0% 到 100% 之间" if percentage else "拟入场价应为有效正数"
        raise ValueError(label)
    return number


def latest_close(store: EvidenceStore, symbol: str, as_of: str) -> tuple[Decimal | None, str]:
    candidates = []
    for bundle in store.bundles:
        if bundle.tool_name != "get_market_technical" or bundle.symbol != symbol or bundle.as_of_date != as_of:
            continue
        for item in bundle.evidences:
            trade_date = item.payload.get("latest_trade_date")
            if isinstance(trade_date, str) and trade_date <= as_of:
                try:
                    price = parse_number(item.payload.get("close"))
                except ValueError:
                    continue
                if price is not None:
                    candidates.append((trade_date, price, item))
    if not candidates:
        return None, "缺少可用收盘价"
    trade_date, price, item = max(candidates, key=lambda row: row[0])
    return price, f"{trade_date} 收盘价，来源 {item.source} [{item.evidence_id}]"


class EntryFinding(BaseModel):
    bull: str
    bear: str
    condition: str
    price: Decimal | None = None
    agreed: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    reconsider_when: str


@dataclass(frozen=True)
class EntryAssessment:
    focus: str
    entry: Decimal | None
    price_source: str
    risk_pct: Decimal | None
    transcript: DebateTranscript
    finding: EntryFinding
    invalidation_price: Decimal | None

    def with_risk(self, percent: Decimal) -> "EntryAssessment":
        return replace(self, risk_pct=percent)

    def render(self) -> str:
        finding = self.finding
        lines = [
            f"{self.transcript.symbol} | 基准日 {self.transcript.as_of_date} | 短线评估",
            f"入场参考 {self.entry} 元（{self.price_source}）" if self.entry is not None else self.price_source,
            f"多头理由：{finding.bull}", f"空头反驳：{finding.bear}",
            f"失效条件：{finding.condition}",
        ]
        if self.risk_pct is None:
            lines.append("个人风险上限待确认。可继续补充从拟入场价起最多承受多少跌幅。")
        if self.entry is not None and self.risk_pct is not None:
            line = self.entry * (1 - self.risk_pct / 100)
            lines.append(f"可承受跌幅 {self.risk_pct:g}%，风险参考价 {line:.4f} 元。")
            if self.invalidation_price is not None and self.invalidation_price <= self.entry:
                if self.invalidation_price < line:
                    lines.append("价格约束超出风险范围。个人风险线会先于理由失效位触及。")
                else:
                    lines.append("该失效位处于个人价格风险范围内。其他入场条件仍需逐项核对。")
        if self.entry is not None and self.invalidation_price is not None:
            distance = (self.entry - self.invalidation_price) / self.entry * 100
            if distance >= 0:
                lines.append(f"距失效位 {self.invalidation_price:g} 元的跌幅为 {distance:.2f}%。")
            else:
                lines.append(f"失效位 {self.invalidation_price:g} 元高于拟入场价，需重新核对入场假设。")
        else:
            lines.append("现有条件无法完成失效位的价格距离比较。")
        lines.append(f"值得重评时机：{finding.reconsider_when}")
        lines.extend(f"未决：{issue}" for issue in self.transcript.unresolved)
        return "\n".join(lines)


def assess_entry(llm, store, focus, entry, source, risk_pct, transcript) -> EntryAssessment:
    prompt = (
        f"用户原始交易条件：{focus}\n入场参考：{entry}（{source}）\n"
        f"证据：\n{store.dump_all_prompt_text()}\n{transcript.to_markdown()}\n"
        "用已完成交锋和原始证据提炼短线判断，报告未决分歧。只输出 JSON。"
        "字段 bull、bear、condition、reconsider_when 为短文字，关键事实附证据编号。"
        "price 为数字或 null，agreed 为布尔值，evidence_ids 为支持价格失效位的证据编号数组。"
        "只有双方明确认可且原始证据支持的价格失效位才填写 price 并设 agreed 为 true。"
        "非价格条件和有争议的价格保留在 condition 内。"
    )
    finding = EntryFinding.model_validate_json(model_json(llm, prompt))
    ids = set(finding.evidence_ids)
    price = None
    if finding.agreed and ids and ids <= store.get_valid_ids():
        if ids <= {eid for turn in transcript.turns if turn.completed for eid in turn.cited_ids}:
            price = parse_number(finding.price)
    return EntryAssessment(focus, entry, source, risk_pct, transcript, finding, price)
