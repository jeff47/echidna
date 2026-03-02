from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.models import Author, AuthorMatch, Citation, Cluster, ReportRow, TargetName


NON_ALPHA_NUM = re.compile(r"[^a-z0-9]+")
FOUR_DIGIT_YEAR = re.compile(r"(19|20)\d{2}")


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

    if author_initials and target.initials and author_initials.startswith(target.initials[:1]):
        return True, "initials"

    return False, None


def affiliation_fingerprint(affiliation: str) -> str:
    cleaned = normalize_text(affiliation)
    if not cleaned:
        return "no-affiliation"
    return cleaned[:80]


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
        affiliations = sorted({m.author.affiliation for m in cluster_matches if m.author.affiliation})
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

        row = ReportRow(
            citation=citation,
            counted_overall=True,
            counted_first=counted_first,
            counted_senior=counted_senior,
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
    total = len(included)
    first_senior = [r for r in included if r.counted_first or r.counted_senior]

    by_journal_year: dict[str, int] = defaultdict(int)
    for row in first_senior:
        key = f"{row.citation.journal} {row.citation.print_year or 'n/a'}"
        by_journal_year[key] += 1

    fragments = [f"{count} {label}" for label, count in sorted(by_journal_year.items(), key=lambda x: (-x[1], x[0]))]
    detail = "; ".join(fragments) if fragments else "none"

    return (
        f"Publications ({start_year}-{end_year}): {total} total, "
        f"{len(first_senior)} 1st/Sr author ({detail})"
    )
