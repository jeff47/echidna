from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.models import Author, AuthorMatch, Citation, Cluster, ReportRow, TargetName


NON_ALPHA_NUM = re.compile(r"[^a-z0-9]+")
FOUR_DIGIT_YEAR = re.compile(r"(19|20)\d{2}")
EXCLUDED_TYPE_KEYWORDS = (
    "preprint",
    "editorial",
    "comment",
    "letter",
    "news",
    "newspaper article",
    "published erratum",
    "retraction of publication",
    "retracted publication",
    "expression of concern",
    "corrected and republished article",
    "patient education handout",
)
UNIT_PREFIX_PATTERN = re.compile(
    r"^(department|departments|division|center|centre|laboratory|lab|program|section|unit|faculty)\b",
    flags=re.IGNORECASE,
)
TRAILING_PUNCT = re.compile(r"[.;,\s]+$")
INSTITUTION_HINTS = (
    "university",
    "hospital",
    "institute",
    "college",
    "school",
    "clinic",
    "health",
)
SECONDARY_INSTITUTION_HINTS = (
    "medicine",
    "foundation",
)
UNIT_WORDS = (
    "department",
    "departments",
    "division",
    "center",
    "centre",
    "laboratory",
    "lab",
    "program",
    "section",
    "unit",
)


def normalize_token(value: str) -> str:
    return NON_ALPHA_NUM.sub("", value.lower())


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def extract_initials(given: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z]+", given) if p]
    return "".join(p[0].lower() for p in parts)


def parse_pmids(raw_text: str) -> list[str]:
    candidates = re.findall(r"\d+", raw_text)
    seen: set[str] = set()
    ordered: list[str] = []
    for pmid in candidates:
        if pmid not in seen:
            seen.add(pmid)
            ordered.append(pmid)
    return ordered


def parse_target_name(raw_name: str) -> TargetName:
    tokens = [t for t in raw_name.strip().split() if t]
    if len(tokens) < 2:
        raise ValueError("Author name must include at least given name and surname")
    surname = normalize_token(tokens[-1])
    given = normalize_text(" ".join(tokens[:-1]))
    initials = extract_initials(given)
    return TargetName(raw=raw_name.strip(), surname=surname, given=given, initials=initials)


def parse_year(pub_date: str) -> int | None:
    match = FOUR_DIGIT_YEAR.search(pub_date)
    if not match:
        return None
    return int(match.group(0))


def normalize_publication_type(value: str) -> str:
    return normalize_text(value)


def parse_type_terms(raw_text: str) -> list[str]:
    terms = [normalize_publication_type(part) for part in re.split(r"[,\n;]+", raw_text) if part.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def default_excluded_type_terms() -> list[str]:
    return list(EXCLUDED_TYPE_KEYWORDS)


def citation_matches_excluded_type(citation: Citation, excluded_terms: list[str]) -> tuple[bool, list[str]]:
    normalized_types = [normalize_publication_type(value) for value in citation.publication_types]
    matched: list[str] = []
    for pub_type in normalized_types:
        for term in excluded_terms:
            if term and term in pub_type:
                matched.append(pub_type)
                break
    seen: set[str] = set()
    deduped: list[str] = []
    for value in matched:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return bool(deduped), deduped


def is_review_article(citation: Citation) -> bool:
    for pub_type in citation.publication_types:
        if "review" in normalize_publication_type(pub_type):
            return True
    return False


def match_author(author: Author, target: TargetName) -> tuple[bool, str | None]:
    if normalize_token(author.last_name) != target.surname:
        return False, None

    author_given = normalize_text(author.fore_name)
    author_initials = normalize_token(author.initials).lower()

    if author_given and target.given:
        # Prefix matching handles cases like "John Michael" vs "John M"
        author_first = author_given.split(" ")[0]
        target_first = target.given.split(" ")[0]
        if author_first == target_first:
            return True, "given"
        # Precision-first: if a full given name is present and does not match,
        # do not fall back to initials (prevents Jeffrey vs James merges).
        author_first_token = normalize_token(author_first)
        if len(author_first_token) > 1:
            return False, None

    if author_initials and target.initials and author_initials.startswith(target.initials[:1]):
        return True, "initials"

    return False, None


def affiliation_fingerprint(affiliation: str) -> str:
    institutions = _extract_institution_names(affiliation)
    if not institutions:
        return "no-affiliation"
    return "|".join(normalize_text(name) for name in institutions)[:120]


def _extract_institution_names(affiliation: str) -> list[str]:
    pieces = [piece.strip() for piece in affiliation.split(";") if piece.strip()]
    names: list[str] = []
    seen_keys: set[str] = set()
    for piece in pieces:
        candidates = _institution_candidates(piece)
        if not candidates:
            continue
        strong = [c for c in candidates if c[0] >= 3]
        selected = strong or [max(candidates, key=lambda item: item[0])]
        for _, key, label in selected:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            names.append(label)
    return names


def _institution_candidates(piece: str) -> list[tuple[int, str, str]]:
    clauses = [clause.strip() for clause in piece.split(",") if clause.strip()]
    candidates: list[tuple[int, str, str]] = []
    for clause in clauses:
        display = _clean_clause_label(clause)
        normalized = normalize_text(display)
        if not normalized:
            continue
        if UNIT_PREFIX_PATTERN.match(normalized):
            continue
        score = 0
        if any(hint in normalized for hint in INSTITUTION_HINTS):
            score += 3
        if any(hint in normalized for hint in SECONDARY_INSTITUTION_HINTS):
            score += 1
        if any(word in normalized for word in UNIT_WORDS):
            score -= 2
        if score <= 0:
            continue
        candidates.append((score, normalized, display))
    return candidates


def _clean_clause_label(clause: str) -> str:
    return TRAILING_PUNCT.sub("", " ".join(clause.split()))


def build_clusters(citations: list[Citation], target: TargetName) -> tuple[list[Cluster], list[AuthorMatch]]:
    cluster_mentions: dict[str, list[AuthorMatch]] = defaultdict(list)
    matches: list[AuthorMatch] = []

    for citation in citations:
        for author in citation.authors:
            matched, method = match_author(author, target)
            if not matched or method is None:
                continue
            aff_key = affiliation_fingerprint(author.affiliation)
            cluster_id = f"{normalize_token(author.last_name)}|{extract_initials(author.fore_name)}|{aff_key}"
            if aff_key == "no-affiliation":
                # Keep no-affiliation candidates independent so users can include/exclude per citation.
                cluster_id = f"{cluster_id}|pmid:{citation.pmid}"
            match = AuthorMatch(
                pmid=citation.pmid,
                position=author.position,
                method=method,
                author=author,
                cluster_id=cluster_id,
            )
            matches.append(match)
            cluster_mentions[cluster_id].append(match)

    clusters: list[Cluster] = []
    for cluster_id, cluster_matches in cluster_mentions.items():
        years = sorted(
            {
                citation.print_year
                for citation in citations
                for cm in cluster_matches
                if citation.pmid == cm.pmid and citation.print_year is not None
            }
        )
        affiliations = _cluster_affiliation_labels(cluster_matches)
        exemplar = cluster_matches[0].author
        label = f"{exemplar.last_name}, {exemplar.fore_name or exemplar.initials}"
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                label=label,
                mention_count=len(cluster_matches),
                years=years,
                affiliations=affiliations or ["(no affiliation listed)"],
            )
        )

    clusters.sort(key=lambda c: c.mention_count, reverse=True)
    return clusters, matches


def _cluster_affiliation_labels(matches: list[AuthorMatch]) -> list[str]:
    seen: set[str] = set()
    labels: list[str] = []
    for match in matches:
        for name in _extract_institution_names(match.author.affiliation):
            key = normalize_text(name)
            if not key or key in seen:
                continue
            seen.add(key)
            labels.append(name)
    labels.sort(key=str.lower)
    return labels


def citation_in_window(citation: Citation, start_year: int, end_year: int) -> bool:
    if citation.print_year is None:
        return False
    return start_year <= citation.print_year <= end_year


@dataclass(slots=True)
class AnalysisResult:
    rows: list[ReportRow]
    uncertain_indices: list[int]


def analyze_citations(
    citations: list[Citation],
    matches: list[AuthorMatch],
    selected_cluster_ids: set[str],
    start_year: int,
    end_year: int,
) -> AnalysisResult:
    by_pmid: dict[str, list[AuthorMatch]] = defaultdict(list)
    for match in matches:
        by_pmid[match.pmid].append(match)

    rows: list[ReportRow] = []
    uncertain_indices: list[int] = []

    for citation in citations:
        if not citation_in_window(citation, start_year, end_year):
            continue

        matched_mentions = by_pmid.get(citation.pmid, [])
        selected_mentions = [m for m in matched_mentions if m.cluster_id in selected_cluster_ids]
        if not selected_mentions:
            continue

        reasons: list[str] = []

        if len(matched_mentions) > 1:
            reasons.append("multiple matching authors with same/similar name")

        if any(m.method == "initials" for m in selected_mentions):
            reasons.append("matched by initials fallback")

        if citation.source_for_roles == "pubmed":
            reasons.append("PMC missing; co-first/co-senior not detectable")

        first_positions = {1} | citation.co_first_positions
        last_position = len(citation.authors)
        senior_positions = {last_position} | citation.co_senior_positions
        selected_positions = {m.position for m in selected_mentions}

        counted_first = bool(first_positions & selected_positions)
        counted_senior = bool(senior_positions & selected_positions)
        review_article = is_review_article(citation)
        counted_review_senior = review_article and counted_senior

        row = ReportRow(
            citation=citation,
            counted_overall=not review_article,
            counted_first=counted_first,
            counted_senior=counted_senior,
            counted_review_senior=counted_review_senior,
            is_review=review_article,
            uncertainty_reasons=reasons,
            include=not reasons,
        )
        rows.append(row)
        if reasons:
            uncertain_indices.append(len(rows) - 1)

    rows.sort(
        key=lambda r: (
            r.citation.print_year is None,
            -(r.citation.print_year or 0),
            r.citation.journal.lower(),
        )
    )

    return AnalysisResult(rows=rows, uncertain_indices=uncertain_indices)


def format_summary(
    rows: list[ReportRow],
    start_year: int,
    end_year: int,
) -> str:
    included = [r for r in rows if r.include]
    overall_included = [r for r in included if r.counted_overall]
    total = len(overall_included)
    first_senior = [r for r in overall_included if r.counted_first or r.counted_senior]
    review_senior_total = sum(1 for r in included if r.counted_review_senior)
    return (
        f"Peer-reviewed Publications ({start_year}-{end_year}): {total} total, "
        f"{len(first_senior)} 1st/Sr author. "
        f"In addition: {review_senior_total} review(s) as Sr author"
    )
