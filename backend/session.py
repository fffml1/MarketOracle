from dataclasses import dataclass, field

from .tools.hello_adapter import EvidenceStore


@dataclass(frozen=True)
class ResearchSession:
    symbol: str
    as_of_date: str
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
