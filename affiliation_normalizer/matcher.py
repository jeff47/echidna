"""Runtime affiliation matcher."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .models import AliasHit, MatchResult

NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
GRID_ID_RE = re.compile(r"grid\.[a-z0-9]+\.[a-z0-9]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
EMAIL_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$")
IN_WORD_APOSTROPHE_RE = re.compile(r"(?<=\w)'(?=\w)")

DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
APOSTROPHE_CHARS = "'\u2018\u2019\u2032\u02bc"
DASH_TRANSLATION = str.maketrans({ch: " " for ch in DASH_CHARS})
APOSTROPHE_TRANSLATION = str.maketrans({ch: "'" for ch in APOSTROPHE_CHARS})

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

        self._ror_index: dict[str, set[str]] = defaultdict(set)
        for canonical_id, inst in self._institutions.items():
            ror_norm = normalize_ror(inst.get("ror_id") or "")
            if ror_norm:
                self._ror_index[ror_norm].add(canonical_id)

        self._grid_index: dict[str, set[str]] = defaultdict(set)
        for canonical_id, inst in self._institutions.items():
            grid_norm = normalize_grid(inst.get("grid_id") or "")
            if grid_norm:
                self._grid_index[grid_norm].add(canonical_id)

        self._email_domain_index: dict[str, set[str]] = defaultdict(set)
        for canonical_id, inst in self._institutions.items():
            for domain in parse_email_domains(inst.get("email_domains") or ""):
                self._email_domain_index[domain].add(canonical_id)

    @classmethod
    def from_rules_json(cls, path: str | Path = DEFAULT_RULES_PATH) -> "AffiliationNormalizer":
        rules = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(rules=rules)

    def match(self, affiliation_text: str) -> MatchResult:
        email_domains = extract_email_domains(affiliation_text)
        email_result: MatchResult | None = None
        if email_domains:
            email_result = self._match_email_domains(email_domains)
            if email_result.status == "matched":
                return email_result

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
            if email_result is not None and email_result.status == "ambiguous":
                return email_result

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

        if len(candidate_ids) > 1 and email_domains:
            email_result = email_result or self._match_email_domains(email_domains)
            if email_result.status == "ambiguous":
                overlap = candidate_ids.intersection(set(email_result.candidate_ids))
                self._apply_precedence(overlap)
                if len(overlap) == 1:
                    only_id = next(iter(overlap))
                    return self._build_matched_result(
                        canonical_id=only_id,
                        hits=candidate_hits[only_id],
                        reason="email_domain_disambiguation",
                        low_confidence=False,
                    )
                if len(overlap) > 1:
                    candidate_ids = overlap

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

    def match_ror(self, ror_id: str) -> MatchResult:
        ror_norm = normalize_ror(ror_id)
        if not ror_norm:
            return MatchResult(status="not_found", reason="empty_ror")

        candidate_ids = self._ror_index.get(ror_norm, set())
        if not candidate_ids:
            return MatchResult(status="not_found", reason="no_ror_match")
        if len(candidate_ids) == 1:
            only_id = next(iter(candidate_ids))
            return self._build_matched_result(
                canonical_id=only_id,
                hits=[],
                reason="ror_match",
                low_confidence=False,
            )

        return MatchResult(
            status="ambiguous",
            reason="ror_multiple_candidates",
            candidate_ids=tuple(sorted(candidate_ids)),
        )

    def match_grid(self, grid_id: str) -> MatchResult:
        grid_norm = normalize_grid(grid_id)
        if not grid_norm:
            return MatchResult(status="not_found", reason="empty_grid")

        candidate_ids = self._grid_index.get(grid_norm, set())
        if not candidate_ids:
            return MatchResult(status="not_found", reason="no_grid_match")
        if len(candidate_ids) == 1:
            only_id = next(iter(candidate_ids))
            return self._build_matched_result(
                canonical_id=only_id,
                hits=[],
                reason="grid_match",
                low_confidence=False,
            )

        return MatchResult(
            status="ambiguous",
            reason="grid_multiple_candidates",
            candidate_ids=tuple(sorted(candidate_ids)),
        )

    def match_email_domain(self, email_domain: str) -> MatchResult:
        domain_norm = normalize_email_domain(email_domain)
        if not domain_norm:
            return MatchResult(status="not_found", reason="empty_email_domain")

        return self._match_email_domains({domain_norm})

    def match_record(
        self,
        *,
        affiliation_text: str = "",
        ror_id: str = "",
        grid_id: str = "",
        email: str = "",
    ) -> MatchResult:
        ror_norm = normalize_ror(ror_id)
        if ror_norm:
            ror_result = self.match_ror(ror_norm)
            if ror_result.status != "not_found":
                return ror_result

        grid_norm = normalize_grid(grid_id)
        if grid_norm:
            grid_result = self.match_grid(grid_norm)
            if grid_result.status != "not_found":
                return grid_result

        domains = set(parse_email_domains(email))
        domains.update(extract_email_domains(email))
        if domains:
            email_result = self._match_email_domains(domains)
            if email_result.status == "matched":
                return email_result
            if email_result.status == "ambiguous":
                if affiliation_text.strip():
                    text_result = self.match(affiliation_text)
                    if text_result.status == "matched" and text_result.canonical_id in set(email_result.candidate_ids):
                        return MatchResult(
                            status="matched",
                            reason="email_domain_disambiguation",
                            canonical_id=text_result.canonical_id,
                            canonical_name=text_result.canonical_name,
                            standardized_name=text_result.standardized_name,
                            city=text_result.city,
                            state=text_result.state,
                            country=text_result.country,
                            ror_id=text_result.ror_id,
                            grid_id=text_result.grid_id,
                            openalex_id=text_result.openalex_id,
                            confidence=text_result.confidence,
                            matched_aliases=text_result.matched_aliases,
                            candidate_ids=text_result.candidate_ids,
                        )
                return email_result

        if affiliation_text.strip():
            return self.match(affiliation_text)
        return MatchResult(status="not_found", reason="empty_input")

    def _match_email_domains(self, domains: set[str]) -> MatchResult:
        candidate_ids = self._candidate_ids_from_email_domains(domains)
        self._apply_precedence(candidate_ids)
        if not candidate_ids:
            return MatchResult(status="not_found", reason="no_email_domain_match")
        if len(candidate_ids) == 1:
            only_id = next(iter(candidate_ids))
            return self._build_matched_result(
                canonical_id=only_id,
                hits=[],
                reason="email_domain_match",
                low_confidence=False,
            )

        return MatchResult(
            status="ambiguous",
            reason="email_domain_multiple_candidates",
            candidate_ids=tuple(sorted(candidate_ids)),
        )

    def _candidate_ids_from_email_domains(self, domains: set[str]) -> set[str]:
        candidate_ids: set[str] = set()
        for domain in domains:
            for suffix in domain_suffixes(domain):
                candidate_ids.update(self._email_domain_index.get(suffix, set()))
        return candidate_ids

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
            grid_id=inst.get("grid_id"),
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
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.translate(DASH_TRANSLATION)
    folded = folded.translate(APOSTROPHE_TRANSLATION)
    folded = IN_WORD_APOSTROPHE_RE.sub("", folded)
    folded = unicodedata.normalize("NFKD", folded)
    folded = folded.encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().replace("&", " and ")
    folded = folded.replace("'", " ")
    folded = NON_ALNUM_RE.sub(" ", folded)
    return WHITESPACE_RE.sub(" ", folded).strip()


def normalize_ror(ror_id: str) -> str:
    raw = ror_id.strip().lower()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc in {"ror.org", "www.ror.org"}:
        raw = parsed.path

    raw = raw.strip().strip("/")
    if raw.startswith("https://ror.org/"):
        raw = raw[len("https://ror.org/") :]
    elif raw.startswith("http://ror.org/"):
        raw = raw[len("http://ror.org/") :]
    elif raw.startswith("www.ror.org/"):
        raw = raw[len("www.ror.org/") :]
    elif raw.startswith("ror.org/"):
        raw = raw[len("ror.org/") :]
    return raw.strip("/")


def normalize_grid(grid_id: str) -> str:
    raw = grid_id.strip().lower()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        raw = f"{parsed.netloc}{parsed.path}"

    match = GRID_ID_RE.search(raw)
    if not match:
        return ""
    return match.group(0)


def normalize_email_domain(domain: str) -> str:
    raw = domain.strip().lower()
    if not raw:
        return ""

    if raw.startswith("mailto:"):
        raw = raw[len("mailto:") :]
    raw = raw.lstrip("@")
    if "@" in raw:
        raw = raw.rsplit("@", 1)[-1]
    raw = raw.strip(".")
    if not raw or ".." in raw:
        return ""

    if not EMAIL_DOMAIN_RE.fullmatch(raw):
        return ""
    return raw


def parse_email_domains(value: str) -> tuple[str, ...]:
    domains: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[|,;]", value or ""):
        normalized = normalize_email_domain(token)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        domains.append(normalized)
    return tuple(domains)


def extract_email_domains(text: str) -> set[str]:
    out: set[str] = set()
    for match in EMAIL_RE.finditer(text):
        normalized = normalize_email_domain(match.group(1))
        if normalized:
            out.add(normalized)
    return out


def domain_suffixes(domain: str) -> tuple[str, ...]:
    labels = [label for label in domain.split(".") if label]
    out: list[str] = []
    for idx in range(0, len(labels) - 1):
        suffix = ".".join(labels[idx:])
        if "." in suffix:
            out.append(suffix)
    return tuple(out)


def default_normalizer() -> AffiliationNormalizer:
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = AffiliationNormalizer.from_rules_json(DEFAULT_RULES_PATH)
    return _DEFAULT_NORMALIZER


def match_affiliation(affiliation_text: str) -> MatchResult:
    """Match an affiliation string using bundled default rules."""
    return default_normalizer().match(affiliation_text)


def match_ror(ror_id: str) -> MatchResult:
    """Match a ROR identifier using bundled default rules."""
    return default_normalizer().match_ror(ror_id)


def match_grid(grid_id: str) -> MatchResult:
    """Match a GRID identifier using bundled default rules."""
    return default_normalizer().match_grid(grid_id)


def match_email_domain(email_domain: str) -> MatchResult:
    """Match an email domain using bundled default rules."""
    return default_normalizer().match_email_domain(email_domain)


def match_record(
    *,
    affiliation_text: str = "",
    ror_id: str = "",
    grid_id: str = "",
    email: str = "",
) -> MatchResult:
    """Match with explicit priority: ROR/GRID, then email domain, then affiliation text."""
    return default_normalizer().match_record(
        affiliation_text=affiliation_text,
        ror_id=ror_id,
        grid_id=grid_id,
        email=email,
    )
