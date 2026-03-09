"""Runtime affiliation matcher."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import AliasHit, MatchResult

NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_RULES_PATH = Path(__file__).with_name("data") / "rules.json"


@dataclass(frozen=True)
class _AliasRule:
    alias: str
    alias_norm: str
    canonical_id: str
    alias_type: str
    policy: str
    token_count: int
    char_count: int


class AffiliationNormalizer:
    """Normalize affiliation strings to canonical institutions."""

    def __init__(self, rules: dict) -> None:
        self._institutions: dict[str, dict[str, str]] = rules.get("institutions", {})
        raw_rules = rules.get("alias_rules", [])
        self._precedence_pairs = [
            ((r.get("preferred") or "").strip(), (r.get("demoted") or "").strip())
            for r in rules.get("precedence_rules", [])
            if (r.get("preferred") or "").strip() and (r.get("demoted") or "").strip()
        ]

        self._alias_rules: list[_AliasRule] = []
        for row in raw_rules:
            alias_norm = (row.get("alias_norm") or "").strip()
            if not alias_norm:
                continue
            self._alias_rules.append(
                _AliasRule(
                    alias=(row.get("alias") or "").strip(),
                    alias_norm=alias_norm,
                    canonical_id=(row.get("canonical_id") or "").strip(),
                    alias_type=(row.get("alias_type") or "").strip(),
                    policy=(row.get("policy") or "allow").strip(),
                    token_count=len(alias_norm.split()),
                    char_count=len(alias_norm),
                )
            )

        self._alias_rules.sort(key=lambda r: (r.token_count, r.char_count), reverse=True)
        self._token_index: dict[str, list[_AliasRule]] = defaultdict(list)
        for rule in self._alias_rules:
            first_token = rule.alias_norm.split(" ", 1)[0]
            self._token_index[first_token].append(rule)

    @classmethod
    def from_rules_json(cls, path: str | Path = DEFAULT_RULES_PATH) -> "AffiliationNormalizer":
        rules = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(rules=rules)

    def match(self, affiliation_text: str) -> MatchResult:
        text = normalize_text(affiliation_text)
        if not text:
            return MatchResult(status="not_found", reason="empty_input")

        wrapped = f" {text} "
        tokens = set(TOKEN_RE.findall(text))

        allow_hits: list[_AliasRule] = []
        review_hits: list[_AliasRule] = []
        seen = set()

        for token in tokens:
            for rule in self._token_index.get(token, []):
                key = (rule.alias_norm, rule.canonical_id, rule.policy)
                if key in seen:
                    continue
                if f" {rule.alias_norm} " not in wrapped:
                    continue
                seen.add(key)
                if rule.policy == "allow":
                    allow_hits.append(rule)
                else:
                    review_hits.append(rule)

        if not allow_hits:
            if review_hits:
                candidates = tuple(sorted({r.canonical_id for r in review_hits if r.canonical_id}))
                return MatchResult(
                    status="not_found",
                    reason="review_only_match",
                    candidate_ids=candidates,
                )
            return MatchResult(status="not_found", reason="no_match")

        candidate_hits: dict[str, list[_AliasRule]] = defaultdict(list)
        for hit in allow_hits:
            candidate_hits[hit.canonical_id].append(hit)

        candidate_ids = set(candidate_hits)
        self._apply_precedence(candidate_ids)

        if len(candidate_ids) == 1:
            only_id = next(iter(candidate_ids))
            return self._build_matched_result(
                canonical_id=only_id,
                hits=candidate_hits[only_id],
                reason="precedence_or_direct_match",
                low_confidence=len(candidate_hits) > 1,
            )

        # Specificity tie-break: unique longest alias wins.
        winner = self._unique_longest_alias_winner(candidate_ids, candidate_hits)
        if winner is not None:
            return self._build_matched_result(
                canonical_id=winner,
                hits=candidate_hits[winner],
                reason="longest_alias_tiebreak",
                low_confidence=True,
            )

        return MatchResult(
            status="ambiguous",
            reason="multiple_candidates",
            candidate_ids=tuple(sorted(candidate_ids)),
            matched_aliases=tuple(
                sorted(
                    (
                        AliasHit(alias=h.alias, alias_type=h.alias_type, policy=h.policy)
                        for h in allow_hits
                    ),
                    key=lambda x: (x.alias, x.alias_type),
                )
            ),
        )

    def _apply_precedence(self, candidate_ids: set[str]) -> None:
        for preferred, demoted in self._precedence_pairs:
            if preferred in candidate_ids and demoted in candidate_ids:
                candidate_ids.discard(demoted)

    @staticmethod
    def _unique_longest_alias_winner(
        candidate_ids: set[str],
        candidate_hits: dict[str, list[_AliasRule]],
    ) -> str | None:
        scored: list[tuple[int, int, str]] = []
        for cid in sorted(candidate_ids):
            hits = candidate_hits.get(cid, [])
            if not hits:
                continue
            best_token_len = max(h.token_count for h in hits)
            best_char_len = max(h.char_count for h in hits)
            scored.append((best_token_len, best_char_len, cid))

        if not scored:
            return None

        scored.sort(reverse=True)
        top = scored[0]
        ties = [row for row in scored if (row[0], row[1]) == (top[0], top[1])]
        if len(ties) == 1:
            return ties[0][2]
        return None

    def _build_matched_result(
        self,
        canonical_id: str,
        hits: list[_AliasRule],
        reason: str,
        low_confidence: bool,
    ) -> MatchResult:
        inst = self._institutions.get(canonical_id, {})
        canonical_name = inst.get("canonical_name")
        city = inst.get("city")
        state = inst.get("state")
        standardized_name = canonical_name
        if canonical_name and city and state:
            standardized_name = f"{canonical_name}, {city}, {state}"

        confidence = 0.95
        if low_confidence:
            confidence = 0.85
        if any(h.alias_type == "acronym" for h in hits):
            confidence = min(confidence, 0.8)

        return MatchResult(
            status="matched",
            reason=reason,
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            standardized_name=standardized_name,
            city=city,
            state=state,
            country=inst.get("country"),
            ror_id=inst.get("ror_id"),
            openalex_id=inst.get("openalex_id"),
            confidence=confidence,
            matched_aliases=tuple(
                sorted(
                    (
                        AliasHit(alias=h.alias, alias_type=h.alias_type, policy=h.policy)
                        for h in hits
                    ),
                    key=lambda x: (x.alias, x.alias_type),
                )
            ),
            candidate_ids=(canonical_id,),
        )


_DEFAULT_NORMALIZER: AffiliationNormalizer | None = None


def normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    folded = folded.encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().replace("&", " and ")
    folded = NON_ALNUM_RE.sub(" ", folded)
    return WHITESPACE_RE.sub(" ", folded).strip()


def default_normalizer() -> AffiliationNormalizer:
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = AffiliationNormalizer.from_rules_json(DEFAULT_RULES_PATH)
    return _DEFAULT_NORMALIZER


def match_affiliation(affiliation_text: str) -> MatchResult:
    """Match an affiliation string using bundled default rules."""
    return default_normalizer().match(affiliation_text)
