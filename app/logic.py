from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from app.models import Author, AuthorMatch, Citation, Cluster, ReportRow, TargetName


NON_ALPHA_NUM = re.compile(r"[^a-z0-9]+")
FOUR_DIGIT_YEAR = re.compile(r"(19|20)\d{2}")
ORCID_CANONICAL = re.compile(r"(\d{4})-?(\d{4})-?(\d{4})-?(\d{3}[\dXx])")
ORCID_COMPACT = re.compile(r"^\d{15}[\dXx]$")
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
AMPERSAND = re.compile(r"\s*&\s*")
APOSTROPHE_VARIANTS = re.compile(r"[’`´]")
AFFILIATION_PIECE_SPLIT = re.compile(r"[;|]+")
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
GENERIC_INSTITUTION_TOKENS = {
    "and",
    "at",
    "center",
    "centre",
    "college",
    "department",
    "departments",
    "division",
    "faculty",
    "for",
    "health",
    "hospital",
    "in",
    "institute",
    "institutes",
    "lab",
    "laboratory",
    "medical",
    "medicine",
    "of",
    "program",
    "school",
    "state",
    "system",
    "the",
    "university",
    "unit",
}
SHORT_DISTINCTIVE_TOKENS = {"nih", "nyu", "mit", "ucla", "ucsf"}
UC_CAMPUS_ALIASES = {
    "berkeley": "Berkeley",
    "davis": "Davis",
    "irvine": "Irvine",
    "los angeles": "Los Angeles",
    "merced": "Merced",
    "riverside": "Riverside",
    "san diego": "San Diego",
    "san francisco": "San Francisco",
    "santa barbara": "Santa Barbara",
    "santa cruz": "Santa Cruz",
}


def normalize_token(value: str) -> str:
    return NON_ALPHA_NUM.sub("", value.lower())


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def extract_initials(given: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z]+", given) if p]
    return "".join(p[0].lower() for p in parts)


def _normalized_name_tokens(tokens: list[str]) -> list[str]:
    cleaned = [normalize_token(token) for token in tokens]
    return [token for token in cleaned if token]


def _looks_like_initial_token(token: str) -> bool:
    cleaned = normalize_token(token)
    if not cleaned or len(cleaned) > 4:
        return False
    if len(cleaned) == 1:
        return True
    return token.isupper() or "." in token or len(cleaned) <= 2

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
    raw = raw_name.strip()
    if not raw:
        raise ValueError("Author name must include at least given name and surname")

    if "," in raw:
        surname_part, given_part = raw.split(",", 1)
        surname = normalize_token(surname_part)
        given_tokens = [t for t in re.split(r"[,\s]+", given_part.strip()) if t]
        cleaned_given_tokens = _normalized_name_tokens(given_tokens)
        if surname and cleaned_given_tokens:
            if all(_looks_like_initial_token(token) for token in given_tokens):
                initials = "".join(normalize_token(token) for token in given_tokens)
                return TargetName(
                    raw=raw,
                    surname=surname,
                    given=" ".join(cleaned_given_tokens),
                    initials=initials,
                    style="initials",
                )
            given = " ".join(cleaned_given_tokens)
            initials = extract_initials(given)
            return TargetName(raw=raw, surname=surname, given=given, initials=initials, style="full")

    tokens = [t for t in raw.split() if t]
    if len(tokens) < 2:
        raise ValueError("Author name must include at least given name and surname")

    first = tokens[0]
    rest = tokens[1:]
    if all(_looks_like_initial_token(token) for token in rest):
        initials = "".join(normalize_token(token) for token in rest)
        return TargetName(
            raw=raw,
            surname=normalize_token(first),
            given=" ".join(_normalized_name_tokens(rest)),
            initials=initials,
            style="initials",
        )

    surname = normalize_token(tokens[-1])
    given_tokens = tokens[:-1]
    if all(_looks_like_initial_token(token) for token in given_tokens):
        initials = "".join(normalize_token(token) for token in given_tokens)
        return TargetName(
            raw=raw,
            surname=surname,
            given=" ".join(_normalized_name_tokens(given_tokens)),
            initials=initials,
            style="initials",
        )

    given = " ".join(_normalized_name_tokens(given_tokens))
    initials = extract_initials(given)
    return TargetName(raw=raw, surname=surname, given=given, initials=initials, style="full")


def normalize_orcid(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    lowered = raw.lower()
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if lowered.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break

    canonical = ORCID_CANONICAL.search(raw)
    if canonical is not None:
        return f"{canonical.group(1)}-{canonical.group(2)}-{canonical.group(3)}-{canonical.group(4).upper()}"

    compact = re.sub(r"[^0-9Xx]", "", raw)
    if ORCID_COMPACT.fullmatch(compact):
        compact = compact.upper()
        return f"{compact[0:4]}-{compact[4:8]}-{compact[8:12]}-{compact[12:16]}"

    return ""


def parse_target_orcid(raw_orcid: str) -> str:
    normalized = normalize_orcid(raw_orcid)
    if raw_orcid.strip() and not normalized:
        raise ValueError("ORCID must be in the form 0000-0000-0000-0000 (or an ORCID URL)")
    return normalized


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


def is_preprint_article(citation: Citation) -> bool:
    for pub_type in citation.publication_types:
        if "preprint" in normalize_publication_type(pub_type):
            return True
    return False


def is_review_article(citation: Citation) -> bool:
    for pub_type in citation.publication_types:
        if "review" in normalize_publication_type(pub_type):
            return True
    return False


def match_author(
    author: Author, target: TargetName, target_orcid: str = ""
) -> tuple[bool, Literal["orcid", "given", "initials"] | None]:
    if target_orcid:
        author_orcid = normalize_orcid(author.orcid)
        if author_orcid:
            if author_orcid == target_orcid:
                return True, "orcid"
            return False, None

    if normalize_token(author.last_name) != target.surname:
        return False, None

    author_given = normalize_text(author.fore_name)
    author_initials = normalize_token(author.initials).lower() or extract_initials(author.fore_name)

    if target.style == "initials":
        if author_initials and target.initials:
            if author_initials.startswith(target.initials) or target.initials.startswith(author_initials):
                return True, "initials"
        return False, None

    if author_given and target.given:
        # Compare normalized first tokens to tolerate punctuation in user input.
        author_first = author_given.split(" ")[0]
        target_first = target.given.split(" ")[0]
        author_first_token = normalize_token(author_first)
        target_first_token = normalize_token(target_first)
        if author_first_token and author_first_token == target_first_token:
            return True, "given"
        # Precision-first: if a full given name is present and does not match,
        # do not fall back to initials (prevents Jeffrey vs James merges).
        if len(author_first_token) > 1:
            return False, None

    if author_initials and target.initials and author_initials.startswith(target.initials[:1]):
        return True, "initials"

    return False, None


def affiliation_fingerprint(affiliation: str) -> str:
    institutions = _extract_institution_names(affiliation)
    keys = sorted({_institution_key(name) for name in institutions if _institution_key(name)})
    if not keys:
        return "no-affiliation"
    return "|".join(keys)[:120]


def _extract_institution_names(affiliation: str) -> list[str]:
    pieces = [piece.strip() for piece in AFFILIATION_PIECE_SPLIT.split(affiliation) if piece.strip()]
    uc_campus_hint = _uc_campus_from_clauses(pieces)
    names: list[str] = []
    seen_keys: set[str] = set()
    for piece in pieces:
        candidates = _institution_candidates(piece, uc_campus_hint=uc_campus_hint)
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


def _institution_candidates(piece: str, *, uc_campus_hint: str = "") -> list[tuple[int, str, str]]:
    clauses = [clause.strip() for clause in piece.split(",") if clause.strip()]
    candidates: list[tuple[int, str, str]] = []
    for clause in clauses:
        display = _expand_institution_clause(clause, clauses, uc_campus_hint=uc_campus_hint)
        normalized = normalize_text(display)
        if not normalized:
            continue
        key = _institution_key(display)
        if not key or UNIT_PREFIX_PATTERN.match(normalized):
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
        candidates.append((score, key, display))
    return candidates


def _expand_institution_clause(clause: str, clauses: list[str], *, uc_campus_hint: str = "") -> str:
    display = _normalize_institution_label(clause)
    normalized = normalize_text(display)
    if normalized in {"university of california", "university of california system"}:
        campus = _uc_campus_from_clauses(clauses) or uc_campus_hint
        if campus:
            return f"University of California, {campus}"
    return display


def _uc_campus_from_clauses(clauses: list[str]) -> str:
    for clause in clauses:
        normalized_clause = normalize_text(_normalize_institution_label(clause))
        if not normalized_clause:
            continue
        for alias, campus in UC_CAMPUS_ALIASES.items():
            if alias in normalized_clause:
                return campus
    return ""


def _institution_key(value: str) -> str:
    normalized = normalize_text(value)
    normalized = AMPERSAND.sub(" and ", normalized)
    normalized = APOSTROPHE_VARIANTS.sub("'", normalized)
    return normalize_token(normalized)


def _institution_distinctive_tokens(value: str) -> set[str]:
    normalized = normalize_text(value)
    normalized = AMPERSAND.sub(" and ", normalized)
    normalized = APOSTROPHE_VARIANTS.sub("'", normalized)
    raw_tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    distinctive: set[str] = set()
    for token in raw_tokens:
        if token in GENERIC_INSTITUTION_TOKENS:
            continue
        if token.isdigit():
            continue
        if len(token) < 3 and token not in SHORT_DISTINCTIVE_TOKENS:
            continue
        distinctive.add(token)
    return distinctive


def _has_distinctive_institution_overlap(left: str, right: str) -> bool:
    left_tokens = _institution_distinctive_tokens(left)
    right_tokens = _institution_distinctive_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return bool(left_tokens & right_tokens)


def _clean_clause_label(clause: str) -> str:
    return TRAILING_PUNCT.sub("", " ".join(clause.split()))


def _normalize_institution_label(clause: str) -> str:
    cleaned = APOSTROPHE_VARIANTS.sub("'", clause)
    cleaned = AMPERSAND.sub(" and ", cleaned)
    return _clean_clause_label(cleaned)


def build_clusters(
    citations: list[Citation],
    target: TargetName,
    target_orcid: str = "",
) -> tuple[list[Cluster], list[AuthorMatch]]:
    cluster_mentions: dict[str, list[AuthorMatch]] = defaultdict(list)
    matches: list[AuthorMatch] = []

    for citation in citations:
        citation_matches: list[AuthorMatch] = []
        for author in citation.authors:
            matched, method = match_author(author, target, target_orcid=target_orcid)
            if not matched or method is None:
                continue

            author_orcid = normalize_orcid(author.orcid)
            if method == "orcid" and author_orcid:
                cluster_id = f"orcid:{author_orcid}"
            else:
                aff_key = affiliation_fingerprint(author.affiliation)
                cluster_id = f"{normalize_token(author.last_name)}|{extract_initials(author.fore_name)}|{aff_key}"
                if aff_key == "no-affiliation":
                    # Keep no-affiliation candidates independent so users can include/exclude per citation.
                    cluster_id = f"{cluster_id}|pmid:{citation.pmid}"

            citation_matches.append(
                AuthorMatch(
                    pmid=citation.pmid,
                    position=author.position,
                    method=method,
                    author=author,
                    cluster_id=cluster_id,
                )
            )

        if target_orcid:
            orcid_matches = [match for match in citation_matches if match.method == "orcid"]
            if orcid_matches:
                citation_matches = orcid_matches

        for match in citation_matches:
            matches.append(match)
            cluster_mentions[match.cluster_id].append(match)

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
    labels: dict[str, str] = {}
    for match in matches:
        raw_affiliation = (match.author.affiliation or "").strip()
        if not raw_affiliation:
            continue
        names = _extract_institution_names(raw_affiliation)
        if not names:
            names = [
                _normalize_institution_label(part)
                for part in AFFILIATION_PIECE_SPLIT.split(raw_affiliation)
                if _normalize_institution_label(part)
            ]
        for name in names:
            key = _institution_key(name)
            if not key:
                key = normalize_token(name)
            if not key or key in seen:
                continue
            seen.add(key)
            labels[key] = _normalize_institution_label(name)
    return sorted(labels.values(), key=str.lower)


def citation_in_window(citation: Citation, start_year: int | None, end_year: int | None) -> bool:
    if start_year is None or end_year is None:
        return True
    if citation.print_year is None:
        return False
    return start_year <= citation.print_year <= end_year


@dataclass(slots=True)
class AnalysisResult:
    rows: list[ReportRow]
    uncertain_indices: list[int]


def _pmid_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _normalize_doi_for_match(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def _superseded_preprint_targets(citations: list[Citation]) -> dict[str, list[str]]:
    by_pmid = {citation.pmid: citation for citation in citations}
    non_preprint_by_doi: dict[str, set[str]] = defaultdict(set)
    reverse_update_links: dict[str, set[str]] = defaultdict(set)

    for citation in citations:
        if not is_preprint_article(citation):
            doi = _normalize_doi_for_match(citation.doi)
            if doi:
                non_preprint_by_doi[doi].add(citation.pmid)
        if is_preprint_article(citation):
            continue
        for prior_pmid in citation.update_of_pmids:
            reverse_update_links[prior_pmid].add(citation.pmid)

    superseded: dict[str, list[str]] = {}
    for citation in citations:
        if not is_preprint_article(citation):
            continue
        targets: set[str] = set()
        for updated_pmid in citation.update_in_pmids:
            linked = by_pmid.get(updated_pmid)
            if linked is not None and is_preprint_article(linked):
                continue
            targets.add(updated_pmid)
        targets.update(reverse_update_links.get(citation.pmid, set()))

        doi = _normalize_doi_for_match(citation.doi)
        if doi:
            targets.update(non_preprint_by_doi.get(doi, set()))

        targets.discard(citation.pmid)
        if not targets:
            continue
        superseded[citation.pmid] = sorted(targets, key=_pmid_sort_key)

    return superseded


def analyze_citations(
    citations: list[Citation],
    matches: list[AuthorMatch],
    selected_cluster_ids: set[str],
    start_year: int | None,
    end_year: int | None,
    forced_include_pmids: set[str] | None = None,
    excluded_pmids: set[str] | None = None,
) -> AnalysisResult:
    if forced_include_pmids is None:
        forced_include_pmids = set()
    if excluded_pmids is None:
        excluded_pmids = set()
    by_pmid: dict[str, list[AuthorMatch]] = defaultdict(list)
    for match in matches:
        by_pmid[match.pmid].append(match)

    rows: list[ReportRow] = []
    uncertain_indices: list[int] = []
    superseded_preprints = _superseded_preprint_targets(citations)

    for citation in citations:
        force_included = citation.pmid in forced_include_pmids
        if citation.pmid in excluded_pmids and not force_included:
            continue
        if not citation_in_window(citation, start_year, end_year) and not force_included:
            continue

        matched_mentions = by_pmid.get(citation.pmid, [])
        selected_mentions = [m for m in matched_mentions if m.cluster_id in selected_cluster_ids]
        if not selected_mentions and not force_included:
            continue
        # Forced-excluded citations can be explicitly restored at step 1.
        if force_included:
            selected_mentions = matched_mentions

        reasons: list[str] = []

        has_definitive_orcid = any(m.method == "orcid" for m in selected_mentions)

        if len(matched_mentions) > 1 and not has_definitive_orcid:
            reasons.append("multiple matching authors with same/similar name")

        if any(m.method == "initials" for m in selected_mentions) and not has_definitive_orcid:
            reasons.append("matched by initials fallback")

        if citation.source_for_roles == "pubmed":
            reasons.append("PMC missing; co-first/co-senior not detectable")

        first_positions = {1} | citation.co_first_positions
        last_position = len(citation.authors)
        senior_positions = {last_position} | citation.co_senior_positions
        selected_positions = {m.position for m in selected_mentions}

        counted_first = bool(first_positions & selected_positions)
        counted_senior = bool(senior_positions & selected_positions)
        preprint_article = is_preprint_article(citation)
        review_article = is_review_article(citation)
        counted_review_senior = review_article and counted_senior
        superseded_by = list(superseded_preprints.get(citation.pmid, []))
        superseded_preprint = preprint_article and bool(superseded_by)

        row = ReportRow(
            citation=citation,
            counted_overall=not (review_article or preprint_article),
            counted_first=counted_first,
            counted_senior=counted_senior,
            counted_review_senior=counted_review_senior,
            is_review=review_article,
            is_preprint=preprint_article,
            uncertainty_reasons=reasons,
            include=not reasons and not superseded_preprint,
            forced_include=force_included,
            matched_positions=set(selected_positions),
            preprint_superseded_by=superseded_by,
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
    # Recompute after sorting so review indices map to the rendered row order.
    uncertain_indices = [idx for idx, row in enumerate(rows) if row.uncertainty_reasons]

    return AnalysisResult(rows=rows, uncertain_indices=uncertain_indices)


def format_summary(
    rows: list[ReportRow],
    start_year: int | None,
    end_year: int | None,
) -> str:
    window_label = f"{start_year}-{end_year}" if start_year is not None and end_year is not None else "all years"
    included = [r for r in rows if r.include]
    overall_included = [r for r in included if r.counted_overall]
    total = len(overall_included)
    first_senior = [r for r in overall_included if r.counted_first or r.counted_senior]
    review_senior_total = sum(1 for r in included if r.counted_review_senior)
    preprint_total = sum(1 for r in included if r.is_preprint)
    detail = _first_senior_detail(first_senior)
    return (
        f"Peer-reviewed Publications ({window_label}): {total} total, "
        f"{len(first_senior)} 1st/Sr author ({detail}). "
        f"In addition: {review_senior_total} review(s) as Sr author; "
        f"{preprint_total} preprint(s) (separate subtotal, not included in peer-reviewed total)."
    )


def _first_senior_detail(rows: list[ReportRow]) -> str:
    if not rows:
        return "none"

    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        year = row.citation.print_year
        if year is not None:
            grouped[row.citation.journal].append(year)
        else:
            grouped[row.citation.journal].append(-1)

    parts: list[str] = []
    # Sort by descending count then journal name for deterministic output.
    for journal, years in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0].lower())):
        count = len(years)
        year_counts: dict[int, int] = defaultdict(int)
        unknown_year_count = 0
        for year in years:
            if year >= 0:
                year_counts[year] += 1
            else:
                unknown_year_count += 1

        year_tokens: list[str] = []
        for year in sorted(year_counts):
            year_count = year_counts[year]
            if year_count > 1:
                year_tokens.append(f"{year_count}x{year}")
            else:
                year_tokens.append(str(year))
        if unknown_year_count > 0:
            if unknown_year_count > 1:
                year_tokens.append(f"{unknown_year_count}xn/a")
            else:
                year_tokens.append("n/a")

        if year_tokens:
            year_text = ", ".join(year_tokens)
            parts.append(f"{count} {journal} {year_text}")
        else:
            parts.append(f"{count} {journal}")
    return "; ".join(parts)
