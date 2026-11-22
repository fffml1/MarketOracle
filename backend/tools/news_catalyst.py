from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..schemas.evidence import EvidenceBundle, EvidenceCategory, EvidenceItem
from .base_evidence_tool import BaseEvidenceTool

# 新闻 fetcher 契约：输入 (symbol, as_of_date, lookback_days)，返回包含 title, summary/content, published_at, source 的列表或 DataFrame
NewsFetcher = Callable[[str, str, int], list[dict[str, Any]] | pd.DataFrame]


def _normalize_records(raw: list[dict[str, Any]] | pd.DataFrame | None) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return []
        return raw.to_dict(orient="records")
    return list(raw)


def _extract_date_str(val: Any) -> str | None:
    """将常见日期/时间戳格式统一提取为 YYYY-MM-DD"""
    if val is None or val == "":
        return None
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def default_akshare_news_fetcher(symbol: str, as_of_date: str, lookback_days: int = 14) -> list[dict[str, Any]]:
    """
    默认 A 股个股新闻获取器（基于 akshare stock_news_em）。
    注意：该接口通常返回近期的个股新闻，调用方仍需按时间窗口二次过滤。
    """
    import akshare as ak

    code = symbol.split(".")[0].strip()
    df = ak.stock_news_em(symbol=code)
    if df is None or df.empty:
        return []

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        title = str(row.get("新闻标题") or row.get("title") or "").strip()
        content = str(row.get("新闻内容") or row.get("content") or row.get("summary") or "").strip()
        pub_time = row.get("发布时间") or row.get("published_at")
        source = str(row.get("文章来源") or row.get("source") or "akshare_news_em").strip()
        records.append(
            {
                "title": title,
                "summary": content,
                "published_at": pub_time,
                "source": source,
            }
        )
    return records


class NewsCatalystTool(BaseEvidenceTool):
    """近期新闻与催化检索工具（输出带 EV_NEWS_xxx 编号的 EvidenceBundle）"""

    tool_name: str = "search_news_catalyst"
    id_prefix: str = "EV_NEWS"
    default_category: EvidenceCategory = EvidenceCategory.NEWS_CATALYST

    def __init__(
        self,
        news_fetcher: NewsFetcher | None = None,
        max_items: int = 8,
        summary_max_len: int = 220,
    ) -> None:
        super().__init__()
        self._news_fetcher: NewsFetcher = news_fetcher or default_akshare_news_fetcher
        self._max_items = max_items
        self._summary_max_len = summary_max_len

    def execute(
        self,
        symbol: str,
        as_of_date: str | None = None,
        lookback_days: int = 14,
    ) -> EvidenceBundle:
        target_date = self.normalize_as_of_date(as_of_date)
        start_date = (pd.to_datetime(target_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        try:
            raw_data = self._news_fetcher(symbol, target_date, lookback_days)
        except Exception as exc:
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="error",
                is_degraded=True,
                degrade_reason=f"新闻接口调用异常: {exc}",
                error_message=str(exc),
            )

        rows = _normalize_records(raw_data)
        if not rows:
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="partial",
                is_degraded=True,
                degrade_reason=f"近 {lookback_days} 天（{start_date} ~ {target_date}）未检索到有效新闻",
            )

        seen_titles: set[str] = set()
        evidences: list[EvidenceItem] = []
        missing_date_count = 0

        for row in rows:
            title = str(row.get("title") or "").strip()
            if not title:
                continue

            norm_title = title.lower()
            if norm_title in seen_titles:
                continue

            pub_date = _extract_date_str(row.get("published_at"))
            if pub_date is None:
                missing_date_count += 1
                # 无明确日期的条目若开启严格回测，可酌情降级标记
                item_degraded = True
                item_deg_reason = "缺少明确发布日期，时间锚点待核实"
            else:
                # 严格时间窗口过滤：剔除未来数据与超出回看窗口的旧闻
                if pub_date > target_date or pub_date < start_date:
                    continue
                item_degraded = False
                item_deg_reason = None

            seen_titles.add(norm_title)

            raw_summary = str(row.get("summary") or row.get("content") or title).strip().replace("\n", " ")
            if len(raw_summary) > self._summary_max_len:
                short_summary = raw_summary[: self._summary_max_len].rstrip() + "..."
            else:
                short_summary = raw_summary

            source = str(row.get("source") or "news_source").strip()
            date_label = pub_date or "日期未知"

            ev_item = self.build_evidence(
                symbol=symbol,
                title=title,
                summary=f"[{date_label}] {short_summary}",
                source=source,
                published_at=pub_date,
                payload={
                    "title": title,
                    "full_summary": raw_summary,
                    "published_at": pub_date,
                    "source": source,
                },
                is_degraded=item_degraded,
                degrade_reason=item_deg_reason,
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
                degrade_reason=f"在 {start_date} ~ {target_date} 时间窗口内无匹配新闻",
            )

        bundle_degraded = missing_date_count > 0
        bundle_deg_reason = (
            f"部分新闻缺少明确发布日期（共 {missing_date_count} 条），已做降级标注"
            if bundle_degraded
            else None
        )

        return self.build_bundle(
            symbol=symbol,
            as_of_date=target_date,
            evidences=evidences,
            status="ok",
            is_degraded=bundle_degraded,
            degrade_reason=bundle_deg_reason,
        )