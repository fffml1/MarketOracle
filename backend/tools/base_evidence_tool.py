from __future__ import annotations

from datetime import date, datetime
from itertools import count
from typing import Any

from ..schemas.evidence import EvidenceBundle, EvidenceCategory, EvidenceItem


class BaseEvidenceTool:
    """证据类工具基类：负责生成唯一 evidence_id 与统一组装 EvidenceBundle"""

    tool_name: str = "base_evidence_tool"
    id_prefix: str = "EV_GEN"
    default_category: EvidenceCategory = EvidenceCategory.TECHNICAL

    def __init__(self) -> None:
        self._seq = count(1)

    def next_evidence_id(self) -> str:
        return f"{self.id_prefix}_{next(self._seq):03d}"

    @staticmethod
    def normalize_as_of_date(as_of_date: str | None) -> str:
        if not as_of_date:
            return date.today().isoformat()
        # 校验格式是否为 YYYY-MM-DD
        datetime.strptime(as_of_date, "%Y-%m-%d")
        return as_of_date

    def build_evidence(
        self,
        *,
        symbol: str,
        title: str,
        summary: str,
        source: str,
        category: EvidenceCategory | None = None,
        payload: dict[str, Any] | None = None,
        published_at: str | None = None,
        is_degraded: bool = False,
        degrade_reason: str | None = None,
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=self.next_evidence_id(),
            category=category or self.default_category,
            symbol=symbol,
            title=title,
            summary=summary,
            payload=payload or {},
            source=source,
            published_at=published_at,
            is_degraded=is_degraded,
            degrade_reason=degrade_reason,
        )

    def build_bundle(
        self,
        *,
        symbol: str,
        as_of_date: str,
        evidences: list[EvidenceItem],
        status: str = "ok",
        is_degraded: bool = False,
        degrade_reason: str | None = None,
        error_message: str | None = None,
    ) -> EvidenceBundle:
        any_item_degraded = any(e.is_degraded for e in evidences)
        final_degraded = is_degraded or any_item_degraded
        final_status = status
        if final_degraded and final_status == "ok":
            final_status = "partial"

        return EvidenceBundle(
            status=final_status,  # type: ignore[arg-type]
            tool_name=self.tool_name,
            symbol=symbol,
            as_of_date=as_of_date,
            evidences=evidences,
            is_degraded=final_degraded,
            degrade_reason=degrade_reason,
            error_message=error_message,
        )