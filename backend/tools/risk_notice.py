from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..schemas.evidence import EvidenceBundle, EvidenceCategory, EvidenceItem
from .base_evidence_tool import BaseEvidenceTool
from .news_catalyst import _extract_date_str, _normalize_records

# 风险/公告 fetcher 契约：输入 (symbol, as_of_date, lookback_days)，返回包含 title, summary/content, published_at, source, risk_type(可选) 的列表或 DataFrame
RiskFetcher = Callable[[str, str, int], list[dict[str, Any]] | pd.DataFrame]

DEFAULT_RISK_KEYWORDS: tuple[str, ...] = (
    "减持",
    "质押",
    "解禁",
    "问询",
    "立案",
    "处罚",
    "诉讼",
    "仲裁",
    "亏损",
    "下滑",
    "风险提示",
    "冻结",
    "违规",
    "退市",
)


def default_akshare_risk_fetcher(symbol: str, as_of_date: str, lookback_days: int = 30) -> list[dict[str, Any]]:
    """
    默认公告/风险检索兜底实现（复用个股资讯接口做关键词过滤，实际项目中可替换为正式公告/减持/解禁数据源）。
    """
    import akshare as ak

    code = symbol.split(".")[0].strip()
    df = ak.stock_news_em(symbol=code)
    if df is None or df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        title = str(row.get("新闻标题") or row.get("title") or "").strip()
        content = str(row.get("新闻内容") or row.get("content") or row.get("summary") or "").strip()
        text_blob = f"{title} {content}"
        matched_tags = [kw for kw in DEFAULT_RISK_KEYWORDS if kw in text_blob]
        if matched_tags:
            rows.append(
                {
                    "title": title,
                    "summary": content,
                    "published_at": row.get("发布时间") or row.get("published_at"),
                    "source": str(row.get("文章来源") or row.get("source") or "akshare_news_risk_filter").strip(),
                    "risk_tags": matched_tags,
                }
            )
    return rows


class RiskAndNoticeTool(BaseEvidenceTool):
    """公告与风险排查工具（聚焦减持、质押、解禁、问询、业绩风险等，输出 EV_RISK_xxx）"""

    tool_name: str = "check_risk_and_notices"
    id_prefix: str = "EV_RISK"
    default_category: EvidenceCategory = EvidenceCategory.RISK_EVENT

    def __init__(
        self,
        risk_fetcher: RiskFetcher | None = None,
        risk_keywords: tuple[str, ...] = DEFAULT_RISK_KEYWORDS,
        max_items: int = 8,
        summary_max_len: int = 240,
    ) -> None:
        super().__init__()
        self._risk_fetcher: RiskFetcher = risk_fetcher or default_akshare_risk_fetcher
        self._risk_keywords = risk_keywords
        self._max_items = max_items
        self._summary_max_len = summary_max_len

    def execute(
        self,
        symbol: str,
        as_of_date: str | None = None,
        lookback_days: int = 30,
    ) -> EvidenceBundle:
        target_date = self.normalize_as_of_date(as_of_date)
        start_date = (pd.to_datetime(target_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        try:
            raw_data = self._risk_fetcher(symbol, target_date, lookback_days)
        except Exception as exc:
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="error",
                is_degraded=True,
                degrade_reason=f"风险/公告接口调用异常: {exc}",
                error_message=str(exc),
            )

        rows = _normalize_records(raw_data)
        if not rows:
            # 没查到风险条目不一定是坏事，但也提示“未检索到显性风险记录”，避免 Agent 误以为“零风险”
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="partial",
                is_degraded=True,
                degrade_reason=f"近 {lookback_days} 天（{start_date} ~ {target_date}）未检出高危公告/风险条目（需注意数据源覆盖范围）",
            )

        seen_titles: set[str] = set()
        evidences: list[EvidenceItem] = []

        for row in rows:
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            norm_title = title.lower()
            if norm_title in seen_titles:
                continue

            pub_date = _extract_date_str(row.get("published_at"))
            if pub_date is not None and (pub_date > target_date or pub_date < start_date):
                continue

            raw_summary = str(row.get("summary") or row.get("content") or title).strip().replace("\n", " ")
            tags = row.get("risk_tags")
            if not isinstance(tags, list) or not tags:
                blob = f"{title} {raw_summary}"
                tags = [kw for kw in self._risk_keywords if kw in blob]

            # 若 fetcher 没做过预筛且没有任何风险关键词命中，跳过普通噪音
            if not tags and self._risk_fetcher is default_akshare_risk_fetcher:
                continue

            seen_titles.add(norm_title)

            short_summary = (
                raw_summary[: self._summary_max_len].rstrip() + "..."
                if len(raw_summary) > self._summary_max_len
                else raw_summary
            )
            source = str(row.get("source") or "risk_notice_source").strip()
            date_label = pub_date or "日期未知"
            tag_str = ",".join(str(t) for t in tags) if tags else "常规风险关注"

            ev_item = self.build_evidence(
                symbol=symbol,
                title=title,
                summary=f"[{date_label} | 标签: {tag_str}] {short_summary}",
                source=source,
                published_at=pub_date,
                payload={
                    "title": title,
                    "full_summary": raw_summary,
                    "published_at": pub_date,
                    "source": source,
                    "risk_tags": tags,
                },
                is_degraded=(pub_date is None),
                degrade_reason="缺少明确公告/披露日期" if pub_date is None else None,
            )
            evidences.append(ev_item)

            if len(evidences) >= self._max_items:
                break

        if not evidences:
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="partial",
                is_degraded=True,
                degrade_reason=f"近 {lookback_days} 天内未筛出明确风险关键词条目",
            )

        return self.build_bundle(
            symbol=symbol,
            as_of_date=target_date,
            evidences=evidences,
            status="ok",
            is_degraded=any(e.is_degraded for e in evidences),
            degrade_reason="部分风险条目缺少明确日期" if any(e.is_degraded for e in evidences) else None,
        )