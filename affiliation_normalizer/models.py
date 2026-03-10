"""Data models for affiliation normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MatchStatus = Literal["matched", "ambiguous", "not_found"]


@dataclass(frozen=True)
class AliasHit:
    """Alias evidence for a candidate institution."""

    alias: str
    alias_type: str
    policy: str


@dataclass(frozen=True)
class MatchResult:
    """Normalized affiliation match output."""

    status: MatchStatus
    reason: str
    canonical_id: str | None = None
    canonical_name: str | None = None
    standardized_name: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    ror_id: str | None = None
    grid_id: str | None = None
    openalex_id: str | None = None
    confidence: float = 0.0
    matched_aliases: tuple[AliasHit, ...] = field(default_factory=tuple)
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        return self.status == "matched"
