from __future__ import annotations

import html
import io
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from app.logic import (
    AnalysisResult,
    analyze_citations,
    build_clusters,
    citation_in_window,
    citation_matches_excluded_type,
    default_excluded_type_terms,
    format_summary,
    parse_type_terms,
    parse_pmids,
    parse_target_name,
)
from app.models import AuthorMatch, Citation, Cluster
from app.ncbi import NcbiClient, _normalize_affiliation_text, split_chunks

logging.basicConfig(
    level=os.getenv("ECHIDNA_LOG_LEVEL", os.getenv("PMCPARSER_LOG_LEVEL", "INFO")).upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ECHIDNA")
templates = Jinja2Templates(directory="templates")
client = NcbiClient()
RUNS_DIR = Path(os.getenv("ECHIDNA_RUNS_DIR", os.getenv("PMCPARSER_RUNS_DIR", "/tmp/echidna-runs")))


@dataclass(slots=True)
class RunState:
    author_name: str
    start_year: int | None
    end_year: int | None
    peer_reviewed_only: bool
    excluded_type_terms: list[str]
    citations: list[Citation]
    clusters: list[Cluster]
    matches: list[AuthorMatch]
    excluded_pmids: set[str]
    excluded_reasons: dict[str, str]


RUNS: dict[str, RunState] = {}
RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
YEAR_OPTION_MIN = 1950


def _to_author_list(citation: Citation) -> str:
    return ", ".join(f"{a.last_name} {a.initials}".strip() for a in citation.authors)


def _to_author_list_html(citation: Citation, highlight_positions: set[int]) -> str:
    parts: list[str] = []
    for author in citation.authors:
        name = f"{author.last_name} {author.initials}".strip()
        escaped = html.escape(name)
        if author.position in highlight_positions:
            parts.append(f"<strong>{escaped}</strong>")
        else:
            parts.append(escaped)
    return ", ".join(parts)


def _year_options() -> list[int]:
    current = date.today().year
    return list(range(current, YEAR_OPTION_MIN - 1, -1))


def _year_window_label(start_year: int | None, end_year: int | None) -> str:
    if start_year is None or end_year is None:
        return "all years"
    return f"{start_year}-{end_year}"


def _default_correction_mode(row: Any) -> str:
    if not row.include:
        return "exclude"
    if row.is_review and row.counted_review_senior:
        return "first_senior_review"
    if row.is_review:
        return "coauthor_review"
    if row.counted_first or row.counted_senior:
        return "first_senior_author"
    if row.counted_overall:
        return "coauthor"
    return "exclude"


def _counted_as_category(row: Any) -> str:
    if row.is_review:
        return "1st/Sr review" if (row.counted_senior or row.counted_review_senior) else "coauthor review"
    return "1st/Sr author" if (row.counted_first or row.counted_senior) else "coauthor"


def _apply_correction_mode(row: Any, mode: str) -> None:
    mode_aliases = {
        "overall": "coauthor",
        "senior": "first_senior_author",
        "review": "coauthor_review",
        "review_senior": "first_senior_review",
    }
    mode = mode_aliases.get(mode, mode)
    if mode == "exclude":
        row.include = False
        row.counted_overall = False
        row.counted_first = False
        row.counted_senior = False
        row.counted_review_senior = False
        row.is_review = False
        return
    if mode == "coauthor":
        row.include = True
        row.counted_overall = True
        row.counted_first = False
        row.counted_senior = False
        row.counted_review_senior = False
        row.is_review = False
        return
    if mode == "first_senior_author":
        row.include = True
        row.counted_overall = True
        row.counted_first = False
        row.counted_senior = True
        row.counted_review_senior = False
        row.is_review = False
        return
    if mode == "coauthor_review":
        row.include = True
        row.counted_overall = False
        row.counted_first = False
        row.counted_senior = False
        row.counted_review_senior = False
        row.is_review = True
        return
    if mode == "first_senior_review":
        row.include = True
        row.counted_overall = False
        row.counted_first = False
        row.counted_senior = True
        row.counted_review_senior = True
        row.is_review = True
        return


def _with_row_render_fields(analysis: AnalysisResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(analysis.rows):
        counted_as = _counted_as_category(row)
        counted_as_report = counted_as
        notes = list(row.citation.notes) + list(row.uncertainty_reasons)
        if row.forced_include:
            notes.append("Forced include override from Step 1")
        rows.append(
            {
                "idx": idx,
                "pmid": row.citation.pmid,
                "pmcid": row.citation.pmcid,
                "pmid_link": f"https://pubmed.ncbi.nlm.nih.gov/{row.citation.pmid}/",
                "pmcid_link": f"https://pmc.ncbi.nlm.nih.gov/articles/{row.citation.pmcid}/" if row.citation.pmcid else None,
                "title": row.citation.title,
                "journal": row.citation.journal,
                "print_date": row.citation.print_date,
                "authors": _to_author_list(row.citation),
                "authors_html": _to_author_list_html(row.citation, row.matched_positions),
                "publication_types": ", ".join(row.citation.publication_types) if row.citation.publication_types else "(unknown)",
                "counted_as": counted_as,
                "counted_as_report": counted_as_report,
                "correction_default": _default_correction_mode(row),
                "role_source": row.citation.source_for_roles,
                "notes": "; ".join(notes),
                "include": row.include,
                "status_label": (
                    "included (review separate)"
                    if row.include and not row.counted_overall and row.is_review
                    else ("included" if row.include else "excluded")
                ),
            }
        )
    return rows


def _parse_correction_entries(correction_entries: list[str]) -> dict[str, str]:
    corrections_by_pmid: dict[str, str] = {}
    for entry in correction_entries:
        if "::" not in entry:
            continue
        pmid, mode = entry.split("::", 1)
        pmid = pmid.strip()
        mode = mode.strip()
        if not pmid:
            continue
        corrections_by_pmid[pmid] = mode
    return corrections_by_pmid


def _rebuild_analysis_with_overrides(
    *,
    run: RunState,
    selected_cluster_ids: set[str],
    selected_pmids: set[str],
    selected_excluded_pmids: set[str],
    pmid_filter_enabled: int,
    accepted_uncertain_pmids: set[str],
    correction_entries: list[str],
) -> tuple[AnalysisResult, list[AuthorMatch]]:
    filtered_matches = _filter_matches_for_analysis(
        run=run,
        selected_cluster_ids=selected_cluster_ids,
        selected_pmids=selected_pmids,
        pmid_filter_enabled=pmid_filter_enabled,
        forced_excluded_pmids=selected_excluded_pmids,
    )
    analysis = analyze_citations(
        citations=run.citations,
        matches=filtered_matches,
        selected_cluster_ids=selected_cluster_ids,
        start_year=run.start_year,
        end_year=run.end_year,
        forced_include_pmids=selected_excluded_pmids,
        excluded_pmids=run.excluded_pmids,
    )

    for row in analysis.rows:
        if row.uncertainty_reasons:
            row.include = row.citation.pmid in accepted_uncertain_pmids

    corrections_by_pmid = _parse_correction_entries(correction_entries)
    for row in analysis.rows:
        mode = corrections_by_pmid.get(row.citation.pmid)
        if not mode:
            continue
        _apply_correction_mode(row, mode)

    return analysis, filtered_matches


def _xlsx_response_for_analysis(*, run: RunState, analysis: AnalysisResult) -> Response:
    try:
        from openpyxl import Workbook
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Excel export dependency missing: {exc}") from exc

    row_data = _with_row_render_fields(analysis)
    excluded_rows = [row for row in row_data if not row["include"]]
    included_pubmed_rows = [row for row in row_data if row["include"] and row["role_source"] != "pmc"]
    included_pmc_rows = [row for row in row_data if row["include"] and row["role_source"] == "pmc"]
    report_rows = excluded_rows + included_pubmed_rows + included_pmc_rows

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Publications"
    sheet.append(
        [
            "PMID/PMCID",
            "PubMed Link",
            "PMC Link",
            "Title",
            "Journal",
            "Date",
            "Authors",
            "Publication Type",
            "Counted as",
        ]
    )

    for row in report_rows:
        pmid_pmcid = f"PMID {row['pmid']}"
        if row["pmcid"]:
            pmid_pmcid = f"{pmid_pmcid} | {row['pmcid']}"
        sheet.append(
            [
                pmid_pmcid,
                "PubMed",
                "PMC" if row["pmcid_link"] else "",
                row["title"],
                row["journal"],
                row["print_date"],
                row["authors"],
                row["publication_types"],
                row["counted_as_report"],
            ]
        )
        current_row = sheet.max_row
        pubmed_cell = sheet.cell(row=current_row, column=2)
        pubmed_cell.hyperlink = str(row["pmid_link"])
        pubmed_cell.style = "Hyperlink"
        if row["pmcid_link"]:
            pmc_cell = sheet.cell(row=current_row, column=3)
            pmc_cell.hyperlink = str(row["pmcid_link"])
            pmc_cell.style = "Hyperlink"

    sheet.freeze_panes = "A2"
    column_widths = {
        "A": 22,
        "B": 14,
        "C": 10,
        "D": 85,
        "E": 24,
        "F": 14,
        "G": 56,
        "H": 24,
        "I": 20,
    }
    for col, width in column_widths.items():
        sheet.column_dimensions[col].width = width

    if run.start_year is None or run.end_year is None:
        filename = "echidna_all_years.xlsx"
    else:
        filename = f"echidna_{run.start_year}_{run.end_year}.xlsx"
    output = io.BytesIO()
    workbook.save(output)
    return Response(
        content=output.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _missing_affiliation_reason(citation: Citation, affiliation: str) -> str:
    if affiliation.strip():
        return ""

    if not citation.pmcid:
        return "No PMCID mapping, and PubMed author entry has no affiliation."

    if citation.source_for_roles == "pubmed":
        for note in citation.notes:
            if note.startswith("PMC fetch/parse failed"):
                return f"{note}; PubMed author entry has no affiliation."
        return "PMC unavailable, and PubMed author entry has no affiliation."

    if any("Author metadata sourced from PMC" in note for note in citation.notes):
        return "PMC author entry for this contributor does not include affiliation text."

    if any("PMC author metadata incomplete" in note for note in citation.notes):
        return "PMC author metadata incomplete and PubMed fallback also has no affiliation."

    return "Affiliation missing in parsed PMC/PubMed author metadata."


def _build_citation_selection_rows(
    *,
    citations: list[Citation],
    matches: list[AuthorMatch],
    all_matches: list[AuthorMatch] | None = None,
    cluster_label_by_id: dict[str, str],
    start_year: int | None,
    end_year: int | None,
) -> list[dict[str, object]]:
    if all_matches is None:
        all_matches = matches

    citation_by_pmid = {citation.pmid: citation for citation in citations}
    all_positions_by_pmid: dict[str, set[int]] = {}
    all_methods_by_pmid: dict[str, set[str]] = {}
    for match in all_matches:
        all_positions_by_pmid.setdefault(match.pmid, set()).add(match.position)
        all_methods_by_pmid.setdefault(match.pmid, set()).add(match.method)
    by_pmid: dict[str, dict[str, object]] = {}

    for match in matches:
        citation = citation_by_pmid.get(match.pmid)
        if citation is None:
            continue
        row = by_pmid.setdefault(
            match.pmid,
            {
                "pmid": citation.pmid,
                "pmid_link": f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/",
                "pmcid": citation.pmcid,
                "pmcid_link": f"https://pmc.ncbi.nlm.nih.gov/articles/{citation.pmcid}/" if citation.pmcid else None,
                "year": citation.print_year,
                "journal": citation.journal,
                "title": citation.title,
                "source_for_roles": citation.source_for_roles,
                "publication_types": ", ".join(citation.publication_types) if citation.publication_types else "(unknown)",
                "cluster_labels": set(),
                "affiliations": set(),
                "positions": set(),
                "notes": set(citation.notes),
            },
        )
        row["cluster_labels"].add(cluster_label_by_id.get(match.cluster_id, match.cluster_id))
        affiliation = (match.author.affiliation or "").strip()
        if affiliation:
            row["affiliations"].add(affiliation)
        row["positions"].add(match.position)

    rendered: list[dict[str, object]] = []
    for row in by_pmid.values():
        pmid = str(row["pmid"])
        all_positions = all_positions_by_pmid.get(pmid, set())
        all_methods = all_methods_by_pmid.get(pmid, set())
        has_no_affiliation = not bool(row["affiliations"])
        has_multiple_matches = len(all_positions) > 1
        has_initials_match = "initials" in all_methods
        review_reasons: list[str] = []
        if has_no_affiliation:
            review_reasons.append("No affiliation listed for matched author")
        if has_multiple_matches:
            review_reasons.append("Multiple matched author positions in citation")
        if has_initials_match:
            review_reasons.append("Matched by initials fallback")
        rendered.append(
            {
                "pmid": row["pmid"],
                "pmid_link": row["pmid_link"],
                "pmcid": row["pmcid"],
                "pmcid_link": row["pmcid_link"],
                "year": row["year"],
                "journal": row["journal"],
                "title": row["title"],
                "source_for_roles": row["source_for_roles"],
                "publication_types": row["publication_types"],
                "cluster_labels": " | ".join(sorted(row["cluster_labels"])),
                "affiliations": " | ".join(sorted(row["affiliations"])) if row["affiliations"] else "(no affiliation listed)",
                "authors_html": _to_author_list_html(citation_by_pmid[pmid], set(row["positions"])),
                "notes": "; ".join(sorted(row["notes"])),
                "needs_review": bool(review_reasons),
                "review_reason": "; ".join(review_reasons),
                "in_window": bool(citation_in_window(citation_by_pmid[row["pmid"]], start_year, end_year)),
            }
        )

    rendered.sort(key=lambda row: (row["year"] is None, -(row["year"] or 0), str(row["pmid"])))
    return rendered


def _build_cluster_citation_rows(
    *,
    citations: list[Citation],
    matches: list[AuthorMatch],
) -> dict[str, list[dict[str, object]]]:
    citation_by_pmid = {citation.pmid: citation for citation in citations}
    by_cluster: dict[str, dict[str, dict[str, object]]] = {}

    for match in matches:
        citation = citation_by_pmid.get(match.pmid)
        if citation is None:
            continue
        cluster_rows = by_cluster.setdefault(match.cluster_id, {})
        cluster_rows.setdefault(
            match.pmid,
            {
                "pmid": citation.pmid,
                "pmid_link": f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/",
                "year": citation.print_year,
                "journal": citation.journal,
                "title": citation.title,
            },
        )

    rendered: dict[str, list[dict[str, object]]] = {}
    for cluster_id, rows_by_pmid in by_cluster.items():
        rows = list(rows_by_pmid.values())
        rows.sort(key=lambda row: (row["year"] is None, -(row["year"] or 0), str(row["pmid"])))
        rendered[cluster_id] = rows
    return rendered


def _sanitize_citation_affiliations(citations: list[Citation]) -> int:
    changed = 0
    for citation in citations:
        for author in citation.authors:
            original = author.affiliation or ""
            cleaned = _normalize_affiliation_text(original)
            if cleaned != original:
                author.affiliation = cleaned
                changed += 1
    return changed


def _read_uploaded_text(upload: UploadFile | None) -> str:
    if upload is None:
        return ""
    try:
        raw = upload.file.read()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read uploaded PMID file: filename=%r", upload.filename)
        return ""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="ignore")


def _build_out_of_window_rows(
    *,
    citations: list[Citation],
    matches: list[AuthorMatch],
    start_year: int | None,
    end_year: int | None,
    forced_include_pmids: set[str] | None = None,
) -> list[dict[str, object]]:
    if start_year is None or end_year is None:
        return []
    if forced_include_pmids is None:
        forced_include_pmids = set()
    citation_by_pmid = {citation.pmid: citation for citation in citations}
    selected_pmids = {match.pmid for match in matches}
    rows: list[dict[str, object]] = []
    for pmid in selected_pmids:
        if pmid in forced_include_pmids:
            continue
        citation = citation_by_pmid.get(pmid)
        if citation is None:
            continue
        if citation_in_window(citation, start_year, end_year):
            continue
        rows.append(
            {
                "pmid": citation.pmid,
                "pmid_link": f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/",
                "pmcid": citation.pmcid,
                "pmcid_link": f"https://pmc.ncbi.nlm.nih.gov/articles/{citation.pmcid}/" if citation.pmcid else None,
                "year": citation.print_year,
                "journal": citation.journal,
                "title": citation.title,
            }
        )
    rows.sort(key=lambda row: (row["year"] is None, -(row["year"] or 0), str(row["pmid"])))
    return rows


def _build_excluded_citation_rows(
    *,
    citations: list[Citation],
    excluded_pmids: set[str],
    excluded_reasons: dict[str, str],
    matches: list[AuthorMatch],
) -> list[dict[str, object]]:
    citation_by_pmid = {citation.pmid: citation for citation in citations}
    matched_pmids = {match.pmid for match in matches}
    rows: list[dict[str, object]] = []
    for pmid in sorted(excluded_pmids):
        if pmid not in matched_pmids:
            continue
        citation = citation_by_pmid.get(pmid)
        if citation is None:
            continue
        rows.append(
            {
                "pmid": citation.pmid,
                "pmid_link": f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/",
                "pmcid": citation.pmcid,
                "pmcid_link": f"https://pmc.ncbi.nlm.nih.gov/articles/{citation.pmcid}/" if citation.pmcid else None,
                "year": citation.print_year,
                "journal": citation.journal,
                "title": citation.title,
                "reason": excluded_reasons.get(citation.pmid, "Excluded by pre-disambiguation filter"),
            }
        )
    rows.sort(key=lambda row: (row["year"] is None, -(row["year"] or 0), str(row["pmid"])))
    return rows


def _filter_matches_for_analysis(
    *,
    run: RunState,
    selected_cluster_ids: set[str],
    selected_pmids: set[str],
    pmid_filter_enabled: int,
    forced_excluded_pmids: set[str],
) -> list[AuthorMatch]:
    return [
        match
        for match in run.matches
        if (
            (match.cluster_id in selected_cluster_ids or match.pmid in forced_excluded_pmids)
            and (pmid_filter_enabled == 0 or match.pmid in selected_pmids)
            and (match.pmid not in run.excluded_pmids or match.pmid in forced_excluded_pmids)
        )
    ]


def _save_run(run_id: str, run: RunState) -> None:
    RUNS[run_id] = run
    _save_run_to_disk(run_id, run)


def _run_file(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _save_run_to_disk(run_id: str, run: RunState) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        logger.warning("Refused to persist run with invalid id format: %r", run_id)
        return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target = _run_file(run_id)
    temp = target.with_suffix(".json.tmp")
    payload = _serialize_run(run)
    temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temp.replace(target)


def _load_run_from_disk(run_id: str) -> RunState | None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return None
    target = _run_file(run_id)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read run-state file for run_id=%s", run_id)
        return None
    try:
        return _deserialize_run(payload)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to parse run-state payload for run_id=%s", run_id)
        return None


def _serialize_run(run: RunState) -> dict[str, Any]:
    return {
        "author_name": run.author_name,
        "start_year": run.start_year,
        "end_year": run.end_year,
        "peer_reviewed_only": run.peer_reviewed_only,
        "excluded_type_terms": list(run.excluded_type_terms),
        "citations": [_serialize_citation(citation) for citation in run.citations],
        "clusters": [_serialize_cluster(cluster) for cluster in run.clusters],
        "matches": [_serialize_match(match) for match in run.matches],
        "excluded_pmids": sorted(run.excluded_pmids),
        "excluded_reasons": dict(run.excluded_reasons),
    }


def _deserialize_run(payload: dict[str, Any]) -> RunState:
    start_raw = payload.get("start_year")
    end_raw = payload.get("end_year")
    return RunState(
        author_name=str(payload["author_name"]),
        start_year=(int(start_raw) if start_raw is not None else None),
        end_year=(int(end_raw) if end_raw is not None else None),
        peer_reviewed_only=bool(payload.get("peer_reviewed_only", False)),
        excluded_type_terms=[str(v) for v in payload.get("excluded_type_terms", [])],
        citations=[_deserialize_citation(item) for item in payload.get("citations", [])],
        clusters=[_deserialize_cluster(item) for item in payload.get("clusters", [])],
        matches=[_deserialize_match(item) for item in payload.get("matches", [])],
        excluded_pmids={str(v) for v in payload.get("excluded_pmids", [])},
        excluded_reasons={str(k): str(v) for k, v in dict(payload.get("excluded_reasons", {})).items()},
    )


def _serialize_author(author: Any) -> dict[str, Any]:
    return {
        "position": int(author.position),
        "last_name": str(author.last_name),
        "fore_name": str(author.fore_name),
        "initials": str(author.initials),
        "affiliation": str(author.affiliation),
    }


def _deserialize_author(payload: dict[str, Any]) -> Any:
    from app.models import Author

    return Author(
        position=int(payload["position"]),
        last_name=str(payload.get("last_name", "")),
        fore_name=str(payload.get("fore_name", "")),
        initials=str(payload.get("initials", "")),
        affiliation=str(payload.get("affiliation", "")),
    )


def _serialize_citation(citation: Citation) -> dict[str, Any]:
    return {
        "pmid": citation.pmid,
        "pmcid": citation.pmcid,
        "title": citation.title,
        "journal": citation.journal,
        "print_year": citation.print_year,
        "print_date": citation.print_date,
        "authors": [_serialize_author(author) for author in citation.authors],
        "publication_types": list(citation.publication_types),
        "source_for_roles": citation.source_for_roles,
        "co_first_positions": sorted(citation.co_first_positions),
        "co_senior_positions": sorted(citation.co_senior_positions),
        "notes": list(citation.notes),
    }


def _deserialize_citation(payload: dict[str, Any]) -> Citation:
    return Citation(
        pmid=str(payload.get("pmid", "")),
        pmcid=payload.get("pmcid"),
        title=str(payload.get("title", "")),
        journal=str(payload.get("journal", "")),
        print_year=payload.get("print_year"),
        print_date=str(payload.get("print_date", "")),
        authors=[_deserialize_author(item) for item in payload.get("authors", [])],
        publication_types=[str(v) for v in payload.get("publication_types", [])],
        source_for_roles=str(payload.get("source_for_roles", "pubmed")),  # type: ignore[arg-type]
        co_first_positions={int(v) for v in payload.get("co_first_positions", [])},
        co_senior_positions={int(v) for v in payload.get("co_senior_positions", [])},
        notes=[str(v) for v in payload.get("notes", [])],
    )


def _serialize_cluster(cluster: Cluster) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "label": cluster.label,
        "mention_count": cluster.mention_count,
        "years": list(cluster.years),
        "affiliations": list(cluster.affiliations),
    }


def _deserialize_cluster(payload: dict[str, Any]) -> Cluster:
    return Cluster(
        cluster_id=str(payload.get("cluster_id", "")),
        label=str(payload.get("label", "")),
        mention_count=int(payload.get("mention_count", 0)),
        years=[int(v) for v in payload.get("years", [])],
        affiliations=[str(v) for v in payload.get("affiliations", [])],
    )


def _serialize_match(match: AuthorMatch) -> dict[str, Any]:
    return {
        "pmid": match.pmid,
        "position": match.position,
        "method": match.method,
        "author": _serialize_author(match.author),
        "cluster_id": match.cluster_id,
    }


def _deserialize_match(payload: dict[str, Any]) -> AuthorMatch:
    return AuthorMatch(
        pmid=str(payload.get("pmid", "")),
        position=int(payload.get("position", 0)),
        method=str(payload.get("method", "initials")),  # type: ignore[arg-type]
        author=_deserialize_author(payload.get("author", {})),
        cluster_id=str(payload.get("cluster_id", "")),
    )


def _load_run(run_id: str) -> RunState:
    run = RUNS.get(run_id)
    if run is not None:
        return run
    run = _load_run_from_disk(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found or expired")
    RUNS[run_id] = run
    return run


def _render_analysis_result(
    request: Request,
    *,
    run_id: str,
    selected_cluster_ids: set[str],
    selected_pmids: set[str],
    selected_excluded_pmids: set[str],
    pmid_filter_enabled: int,
    analysis: AnalysisResult,
    start_year: int | None,
    end_year: int | None,
    force_report: bool = False,
    out_of_window_rows: list[dict[str, object]] | None = None,
) -> HTMLResponse:
    if analysis.uncertain_indices and not force_report:
        logger.info(
            "Analysis requires manual review for %d citation(s)",
            len(analysis.uncertain_indices),
        )
        row_data = _with_row_render_fields(analysis)
        uncertain_rows = [row_data[i] for i in analysis.uncertain_indices]
        return templates.TemplateResponse(
            request,
            "review_uncertain.html",
            {
                "run_id": run_id,
                "selected_cluster_ids": sorted(selected_cluster_ids),
                "selected_pmids": sorted(selected_pmids),
                "selected_excluded_pmids": sorted(selected_excluded_pmids),
                "pmid_filter_enabled": pmid_filter_enabled,
                "rows": uncertain_rows,
            },
        )

    summary = format_summary(analysis.rows, start_year, end_year)
    row_data = _with_row_render_fields(analysis)
    accepted_uncertain_pmids = sorted(
        {
            row.citation.pmid
            for row in analysis.rows
            if row.uncertainty_reasons and row.include
        }
    )
    # Final report grouping:
    # 1) excluded rows
    # 2) included rows where PMC metadata was unavailable (PubMed fallback)
    # 3) included rows where PMC author contribution metadata was available
    excluded_rows = [row for row in row_data if not row["include"]]
    included_pubmed_rows = [row for row in row_data if row["include"] and row["role_source"] != "pmc"]
    included_pmc_rows = [row for row in row_data if row["include"] and row["role_source"] == "pmc"]
    report_rows = excluded_rows + included_pubmed_rows + included_pmc_rows
    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "summary": summary,
            "rows": report_rows,
            "out_of_window_rows": out_of_window_rows or [],
            "start_year": start_year,
            "end_year": end_year,
            "year_window_label": _year_window_label(start_year, end_year),
            "run_id": run_id,
            "selected_cluster_ids": sorted(selected_cluster_ids),
            "selected_pmids": sorted(selected_pmids),
            "selected_excluded_pmids": sorted(selected_excluded_pmids),
            "pmid_filter_enabled": pmid_filter_enabled,
            "accepted_uncertain_pmids": accepted_uncertain_pmids,
        },
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    year = date.today().year
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "author_name": "",
            "start_year": year - 5,
            "end_year": year,
            "pmids": "",
            "peer_reviewed_only": True,
            "all_years": False,
            "extra_excluded_types": "",
            "year_options": _year_options(),
            "error": None,
        },
    )


@app.post("/disambiguate", response_class=HTMLResponse)
def disambiguate(
    request: Request,
    author_name: str = Form(...),
    pmids: str = Form(default=""),
    pmid_file: UploadFile | None = File(default=None),
    start_year: int = Form(...),
    end_year: int = Form(...),
    all_years: str | None = Form(default=None),
    peer_reviewed_only: str | None = Form(default=None),
    extra_excluded_types: str = Form(default=""),
) -> HTMLResponse:
    all_years_enabled = all_years is not None
    peer_reviewed_only_enabled = peer_reviewed_only is not None
    excluded_type_terms = default_excluded_type_terms()
    excluded_type_terms.extend(parse_type_terms(extra_excluded_types))
    logger.info(
        "Disambiguation request received: author=%r, start_year=%s, end_year=%s, all_years=%s, peer_reviewed_only=%s",
        author_name,
        start_year,
        end_year,
        all_years_enabled,
        peer_reviewed_only_enabled,
    )
    try:
        target = parse_target_name(author_name)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "author_name": author_name,
                "start_year": start_year,
                "end_year": end_year,
                "pmids": pmids,
                "peer_reviewed_only": peer_reviewed_only_enabled,
                "all_years": all_years_enabled,
                "extra_excluded_types": extra_excluded_types,
                "year_options": _year_options(),
                "error": str(exc),
            },
            status_code=400,
        )

    pmid_file_text = _read_uploaded_text(pmid_file)
    combined_pmids_text = "\n".join(part for part in (pmids, pmid_file_text) if part.strip())
    parsed_pmids = parse_pmids(combined_pmids_text)
    logger.info("Parsed %d PMID(s) from input", len(parsed_pmids))
    if not parsed_pmids:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "author_name": author_name,
                "start_year": start_year,
                "end_year": end_year,
                "pmids": pmids,
                "peer_reviewed_only": peer_reviewed_only_enabled,
                "all_years": all_years_enabled,
                "extra_excluded_types": extra_excluded_types,
                "year_options": _year_options(),
                "error": "No PMIDs found in text input or uploaded file",
            },
            status_code=400,
        )

    if not all_years_enabled and start_year > end_year:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "author_name": author_name,
                "start_year": start_year,
                "end_year": end_year,
                "pmids": pmids,
                "peer_reviewed_only": peer_reviewed_only_enabled,
                "all_years": all_years_enabled,
                "extra_excluded_types": extra_excluded_types,
                "year_options": _year_options(),
                "error": "Start Year must be less than or equal to End Year",
            },
            status_code=400,
        )

    citations: list[Citation] = []
    errors: list[str] = []
    for chunk in split_chunks(parsed_pmids):
        try:
            citations.extend(client.fetch_citations(chunk))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            logger.exception("Chunk fetch failed for %d PMID(s)", len(chunk))
    cleaned_affiliations = _sanitize_citation_affiliations(citations)
    if cleaned_affiliations:
        logger.info("Post-fetch affiliation sanitation updated %d author affiliation value(s)", cleaned_affiliations)

    window_start = None if all_years_enabled else start_year
    window_end = None if all_years_enabled else end_year

    in_window_citations = [citation for citation in citations if citation_in_window(citation, window_start, window_end)]
    in_window_pmids = {citation.pmid for citation in in_window_citations}
    excluded_out_of_window = 0 if all_years_enabled else (len(citations) - len(in_window_citations))
    excluded_pmids: set[str] = {citation.pmid for citation in citations if citation.pmid not in in_window_pmids}
    excluded_reasons: dict[str, str] = {
        citation.pmid: f"Outside year window {_year_window_label(window_start, window_end)}"
        for citation in citations
        if citation.pmid not in in_window_pmids
    }
    excluded_by_type = 0
    disambiguation_citations = in_window_citations
    if peer_reviewed_only_enabled:
        type_filtered: list[Citation] = []
        for citation in in_window_citations:
            is_excluded, matched_types = citation_matches_excluded_type(citation, excluded_type_terms)
            if is_excluded:
                excluded_by_type += 1
                citation.notes.append(f"Excluded by publication type filter: {', '.join(matched_types)}")
                excluded_pmids.add(citation.pmid)
                excluded_reasons[citation.pmid] = f"Excluded by publication type: {', '.join(matched_types)}"
                continue
            type_filtered.append(citation)
        disambiguation_citations = type_filtered
    logger.info(
        "Window/type filters before disambiguation: in_window=%d, excluded_out_of_window=%d, excluded_by_type=%d",
        len(disambiguation_citations),
        excluded_out_of_window,
        excluded_by_type,
    )

    clusters, eligible_matches = build_clusters(disambiguation_citations, target)
    _, all_matches = build_clusters(citations, target)
    citation_by_pmid = {citation.pmid: citation for citation in disambiguation_citations}
    cluster_label_by_id = {cluster.cluster_id: cluster.label for cluster in clusters}
    cluster_citation_rows = _build_cluster_citation_rows(citations=disambiguation_citations, matches=eligible_matches)
    missing_affiliation_rows: list[dict[str, object]] = []
    for match in eligible_matches:
        citation = citation_by_pmid.get(match.pmid)
        if citation is None:
            continue
        affiliation = match.author.affiliation or ""
        reason = _missing_affiliation_reason(citation, affiliation)
        row = {
            "cluster_label": cluster_label_by_id.get(match.cluster_id, match.cluster_id),
            "pmid": citation.pmid,
            "pmid_link": f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}/",
            "pmcid": citation.pmcid,
            "pmcid_link": f"https://pmc.ncbi.nlm.nih.gov/articles/{citation.pmcid}/" if citation.pmcid else None,
            "year": citation.print_year,
            "source_for_roles": citation.source_for_roles,
            "affiliation": affiliation or "(no affiliation listed)",
            "missing_reason": reason,
            "notes": "; ".join(citation.notes),
        }
        if reason:
            missing_affiliation_rows.append(row)
    missing_affiliation_rows.sort(key=lambda row: (row["year"] is None, -(row["year"] or 0), str(row["pmid"])))
    excluded_citation_rows = _build_excluded_citation_rows(
        citations=citations,
        excluded_pmids=excluded_pmids,
        excluded_reasons=excluded_reasons,
        matches=all_matches,
    )
    logger.info(
        "Disambiguation computed: citations=%d, matches=%d, clusters=%d, errors=%d",
        len(citations),
        len(all_matches),
        len(clusters),
        len(errors),
    )

    run_id = uuid.uuid4().hex
    _save_run(
        run_id,
        RunState(
            author_name=author_name,
            start_year=window_start,
            end_year=window_end,
            peer_reviewed_only=peer_reviewed_only_enabled,
            excluded_type_terms=excluded_type_terms,
            citations=citations,
            clusters=clusters,
            matches=all_matches,
            excluded_pmids=excluded_pmids,
            excluded_reasons=excluded_reasons,
        ),
    )

    return templates.TemplateResponse(
        request,
        "disambiguate.html",
        {
            "run_id": run_id,
            "author_name": author_name,
            "start_year": window_start,
            "end_year": window_end,
            "year_window_label": _year_window_label(window_start, window_end),
            "clusters": clusters,
            "cluster_citation_rows": cluster_citation_rows,
            "errors": errors,
            "missing_affiliation_rows": missing_affiliation_rows,
            "excluded_citation_rows": excluded_citation_rows,
            "selected_excluded_pmids": [],
            "excluded_out_of_window": excluded_out_of_window,
            "excluded_by_type": excluded_by_type,
            "peer_reviewed_only": peer_reviewed_only_enabled,
            "extra_excluded_types": extra_excluded_types,
        },
    )


@app.post("/citation-select", response_class=HTMLResponse)
def citation_select(
    request: Request,
    run_id: str = Form(...),
    selected_cluster_ids: list[str] = Form(default=[]),
    selected_excluded_pmids: list[str] = Form(default=[]),
) -> HTMLResponse:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    forced_excluded_pmids = set(selected_excluded_pmids)
    if not selected:
        eligible_citations = [citation for citation in run.citations if citation.pmid not in run.excluded_pmids]
        eligible_matches = [match for match in run.matches if match.pmid not in run.excluded_pmids]
        cluster_citation_rows = _build_cluster_citation_rows(citations=eligible_citations, matches=eligible_matches)
        excluded_citation_rows = _build_excluded_citation_rows(
            citations=run.citations,
            excluded_pmids=run.excluded_pmids,
            excluded_reasons=run.excluded_reasons,
            matches=run.matches,
        )
        return templates.TemplateResponse(
            request,
            "disambiguate.html",
            {
                "run_id": run_id,
                "author_name": run.author_name,
                "start_year": run.start_year,
                "end_year": run.end_year,
                "year_window_label": _year_window_label(run.start_year, run.end_year),
                "clusters": run.clusters,
                "cluster_citation_rows": cluster_citation_rows,
                "errors": ["Select at least one author identity cluster."],
                "missing_affiliation_rows": [],
                "excluded_citation_rows": excluded_citation_rows,
                "selected_excluded_pmids": sorted(forced_excluded_pmids),
                "excluded_out_of_window": 0,
                "excluded_by_type": 0,
            },
            status_code=400,
        )

    cluster_label_by_id = {cluster.cluster_id: cluster.label for cluster in run.clusters}
    selected_matches = _filter_matches_for_analysis(
        run=run,
        selected_cluster_ids=selected,
        selected_pmids=set(),
        pmid_filter_enabled=0,
        forced_excluded_pmids=forced_excluded_pmids,
    )
    citation_selection_rows = _build_citation_selection_rows(
        citations=run.citations,
        matches=selected_matches,
        all_matches=run.matches,
        cluster_label_by_id=cluster_label_by_id,
        start_year=run.start_year,
        end_year=run.end_year,
    )
    needs_review = any(bool(row["needs_review"]) for row in citation_selection_rows)
    logger.info(
        "Citation-selection gate: run_id=%s, selected_clusters=%d, matched_citations=%d, needs_review=%s",
        run_id,
        len(selected),
        len(citation_selection_rows),
        needs_review,
    )

    if not needs_review:
        analysis = analyze_citations(
            citations=run.citations,
            matches=selected_matches,
            selected_cluster_ids=selected,
            start_year=run.start_year,
            end_year=run.end_year,
            forced_include_pmids=forced_excluded_pmids,
            excluded_pmids=run.excluded_pmids,
        )
        out_of_window_rows = _build_out_of_window_rows(
            citations=run.citations,
            matches=selected_matches,
            start_year=run.start_year,
            end_year=run.end_year,
            forced_include_pmids=forced_excluded_pmids,
        )
        return _render_analysis_result(
            request,
            run_id=run_id,
            selected_cluster_ids=selected,
            selected_pmids=set(),
            selected_excluded_pmids=forced_excluded_pmids,
            pmid_filter_enabled=0,
            analysis=analysis,
            start_year=run.start_year,
            end_year=run.end_year,
            out_of_window_rows=out_of_window_rows,
        )

    return templates.TemplateResponse(
        request,
        "citation_select.html",
        {
            "run_id": run_id,
            "author_name": run.author_name,
            "start_year": run.start_year,
            "end_year": run.end_year,
            "year_window_label": _year_window_label(run.start_year, run.end_year),
            "selected_cluster_ids": sorted(selected),
            "selected_excluded_pmids": sorted(forced_excluded_pmids),
            "citation_selection_rows": [row for row in citation_selection_rows if bool(row["needs_review"])],
            "auto_included_pmids": sorted(
                {
                    str(row["pmid"])
                    for row in citation_selection_rows
                    if not bool(row["needs_review"])
                }
            ),
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    request: Request,
    run_id: str = Form(...),
    selected_cluster_ids: list[str] = Form(default=[]),
    selected_pmids: list[str] = Form(default=[]),
    selected_excluded_pmids: list[str] = Form(default=[]),
    pmid_filter_enabled: int = Form(default=0),
) -> HTMLResponse:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    selected_pmid_set = set(selected_pmids)
    forced_excluded_pmids = set(selected_excluded_pmids)
    logger.info(
        "Analyze request: run_id=%s, selected_clusters=%d, selected_pmids=%d, forced_excluded=%d, pmid_filter_enabled=%d",
        run_id,
        len(selected),
        len(selected_pmid_set),
        len(forced_excluded_pmids),
        pmid_filter_enabled,
    )

    filtered_matches = _filter_matches_for_analysis(
        run=run,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        pmid_filter_enabled=pmid_filter_enabled,
        forced_excluded_pmids=forced_excluded_pmids,
    )

    analysis = analyze_citations(
        citations=run.citations,
        matches=filtered_matches,
        selected_cluster_ids=selected,
        start_year=run.start_year,
        end_year=run.end_year,
        forced_include_pmids=forced_excluded_pmids,
        excluded_pmids=run.excluded_pmids,
    )
    out_of_window_rows = _build_out_of_window_rows(
        citations=run.citations,
        matches=filtered_matches,
        start_year=run.start_year,
        end_year=run.end_year,
        forced_include_pmids=forced_excluded_pmids,
    )
    return _render_analysis_result(
        request,
        run_id=run_id,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        selected_excluded_pmids=forced_excluded_pmids,
        pmid_filter_enabled=pmid_filter_enabled,
        analysis=analysis,
        start_year=run.start_year,
        end_year=run.end_year,
        out_of_window_rows=out_of_window_rows,
    )


@app.post("/finalize", response_class=HTMLResponse)
def finalize(
    request: Request,
    run_id: str = Form(...),
    selected_cluster_ids: list[str] = Form(default=[]),
    selected_pmids: list[str] = Form(default=[]),
    selected_excluded_pmids: list[str] = Form(default=[]),
    pmid_filter_enabled: int = Form(default=0),
    accepted_row_indices: list[int] = Form(default=[]),
) -> HTMLResponse:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    selected_pmid_set = set(selected_pmids)
    forced_excluded_pmids = set(selected_excluded_pmids)
    accepted = set(accepted_row_indices)
    logger.info(
        "Finalize request: run_id=%s, selected_clusters=%d, selected_pmids=%d, forced_excluded=%d, pmid_filter_enabled=%d, accepted_uncertain=%d",
        run_id,
        len(selected),
        len(selected_pmid_set),
        len(forced_excluded_pmids),
        pmid_filter_enabled,
        len(accepted),
    )

    filtered_matches = _filter_matches_for_analysis(
        run=run,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        pmid_filter_enabled=pmid_filter_enabled,
        forced_excluded_pmids=forced_excluded_pmids,
    )

    analysis = analyze_citations(
        citations=run.citations,
        matches=filtered_matches,
        selected_cluster_ids=selected,
        start_year=run.start_year,
        end_year=run.end_year,
        forced_include_pmids=forced_excluded_pmids,
        excluded_pmids=run.excluded_pmids,
    )

    for idx in analysis.uncertain_indices:
        analysis.rows[idx].include = idx in accepted
    out_of_window_rows = _build_out_of_window_rows(
        citations=run.citations,
        matches=filtered_matches,
        start_year=run.start_year,
        end_year=run.end_year,
        forced_include_pmids=forced_excluded_pmids,
    )
    return _render_analysis_result(
        request,
        run_id=run_id,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        selected_excluded_pmids=forced_excluded_pmids,
        pmid_filter_enabled=pmid_filter_enabled,
        analysis=analysis,
        start_year=run.start_year,
        end_year=run.end_year,
        force_report=True,
        out_of_window_rows=out_of_window_rows,
    )


@app.post("/report-correct", response_class=HTMLResponse)
def report_correct(
    request: Request,
    run_id: str = Form(...),
    selected_cluster_ids: list[str] = Form(default=[]),
    selected_pmids: list[str] = Form(default=[]),
    selected_excluded_pmids: list[str] = Form(default=[]),
    pmid_filter_enabled: int = Form(default=0),
    accepted_uncertain_pmids: list[str] = Form(default=[]),
    correction_entries: list[str] = Form(default=[]),
    download: str | None = Form(default=None),
) -> Response:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    selected_pmid_set = set(selected_pmids)
    forced_excluded_pmids = set(selected_excluded_pmids)
    accepted_uncertain_set = set(accepted_uncertain_pmids)
    logger.info(
        "Report-correct request: run_id=%s, selected_clusters=%d, selected_pmids=%d, forced_excluded=%d, corrections=%d",
        run_id,
        len(selected),
        len(selected_pmid_set),
        len(forced_excluded_pmids),
        len(correction_entries),
    )

    analysis, filtered_matches = _rebuild_analysis_with_overrides(
        run=run,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        selected_excluded_pmids=forced_excluded_pmids,
        pmid_filter_enabled=pmid_filter_enabled,
        accepted_uncertain_pmids=accepted_uncertain_set,
        correction_entries=correction_entries,
    )
    if (download or "").strip().lower() == "xlsx":
        return _xlsx_response_for_analysis(run=run, analysis=analysis)

    out_of_window_rows = _build_out_of_window_rows(
        citations=run.citations,
        matches=filtered_matches,
        start_year=run.start_year,
        end_year=run.end_year,
        forced_include_pmids=forced_excluded_pmids,
    )
    return _render_analysis_result(
        request,
        run_id=run_id,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        selected_excluded_pmids=forced_excluded_pmids,
        pmid_filter_enabled=pmid_filter_enabled,
        analysis=analysis,
        start_year=run.start_year,
        end_year=run.end_year,
        force_report=True,
        out_of_window_rows=out_of_window_rows,
    )
