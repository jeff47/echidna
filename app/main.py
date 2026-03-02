from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import date

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.logic import (
    AnalysisResult,
    analyze_citations,
    build_clusters,
    format_summary,
    parse_pmids,
    parse_target_name,
)
from app.models import AuthorMatch, Citation, Cluster
from app.ncbi import NcbiClient, split_chunks

logging.basicConfig(
    level=os.getenv("PMCPARSER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Publication Counter")
templates = Jinja2Templates(directory="templates")
client = NcbiClient()


@dataclass(slots=True)
class RunState:
    author_name: str
    start_year: int
    end_year: int
    citations: list[Citation]
    clusters: list[Cluster]
    matches: list[AuthorMatch]


RUNS: dict[str, RunState] = {}


def _to_author_list(citation: Citation) -> str:
    return ", ".join(f"{a.last_name} {a.initials}".strip() for a in citation.authors)


def _with_row_render_fields(analysis: AnalysisResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(analysis.rows):
        cited_as: list[str] = ["overall"]
        if row.counted_first:
            cited_as.append("first")
        if row.counted_senior:
            cited_as.append("senior")
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
                "counted_as": ", ".join(cited_as),
                "role_source": row.citation.source_for_roles,
                "notes": "; ".join(row.citation.notes + row.uncertainty_reasons),
                "include": row.include,
            }
        )
    return rows


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
    cluster_label_by_id: dict[str, str],
) -> list[dict[str, object]]:
    citation_by_pmid = {citation.pmid: citation for citation in citations}
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
        has_no_affiliation = not bool(row["affiliations"])
        has_multiple_matches = len(row["positions"]) > 1
        review_reasons: list[str] = []
        if has_no_affiliation:
            review_reasons.append("No affiliation listed for matched author")
        if has_multiple_matches:
            review_reasons.append("Multiple matched author positions in citation")
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
                "cluster_labels": " | ".join(sorted(row["cluster_labels"])),
                "affiliations": " | ".join(sorted(row["affiliations"])) if row["affiliations"] else "(no affiliation listed)",
                "notes": "; ".join(sorted(row["notes"])),
                "needs_review": bool(review_reasons),
                "review_reason": "; ".join(review_reasons),
            }
        )

    rendered.sort(key=lambda row: (row["year"] is None, -(row["year"] or 0), str(row["pmid"])))
    return rendered


def _load_run(run_id: str) -> RunState:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found or expired")
    return run


def _render_analysis_result(
    request: Request,
    *,
    run_id: str,
    selected_cluster_ids: set[str],
    selected_pmids: set[str],
    pmid_filter_enabled: int,
    analysis: AnalysisResult,
    start_year: int,
    end_year: int,
    force_report: bool = False,
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
                "pmid_filter_enabled": pmid_filter_enabled,
                "rows": uncertain_rows,
            },
        )

    summary = format_summary(analysis.rows, start_year, end_year)
    return templates.TemplateResponse(
        request,
        "report.html",
        {
            "summary": summary,
            "rows": _with_row_render_fields(analysis),
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
            "error": None,
        },
    )


@app.post("/disambiguate", response_class=HTMLResponse)
def disambiguate(
    request: Request,
    author_name: str = Form(...),
    pmids: str = Form(...),
    start_year: int = Form(...),
    end_year: int = Form(...),
) -> HTMLResponse:
    logger.info(
        "Disambiguation request received: author=%r, start_year=%d, end_year=%d",
        author_name,
        start_year,
        end_year,
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
                "error": str(exc),
            },
            status_code=400,
        )

    parsed_pmids = parse_pmids(pmids)
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
                "error": "No PMIDs found in input",
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

    clusters, matches = build_clusters(citations, target)
    citation_by_pmid = {citation.pmid: citation for citation in citations}
    cluster_label_by_id = {cluster.cluster_id: cluster.label for cluster in clusters}
    missing_affiliation_rows: list[dict[str, object]] = []
    for match in matches:
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
    logger.info(
        "Disambiguation computed: citations=%d, matches=%d, clusters=%d, errors=%d",
        len(citations),
        len(matches),
        len(clusters),
        len(errors),
    )

    run_id = uuid.uuid4().hex
    RUNS[run_id] = RunState(
        author_name=author_name,
        start_year=start_year,
        end_year=end_year,
        citations=citations,
        clusters=clusters,
        matches=matches,
    )

    return templates.TemplateResponse(
        request,
        "disambiguate.html",
        {
            "run_id": run_id,
            "author_name": author_name,
            "start_year": start_year,
            "end_year": end_year,
            "clusters": clusters,
            "errors": errors,
            "missing_affiliation_rows": missing_affiliation_rows,
        },
    )


@app.post("/citation-select", response_class=HTMLResponse)
def citation_select(
    request: Request,
    run_id: str = Form(...),
    selected_cluster_ids: list[str] = Form(default=[]),
) -> HTMLResponse:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    if not selected:
        return templates.TemplateResponse(
            request,
            "disambiguate.html",
            {
                "run_id": run_id,
                "author_name": run.author_name,
                "start_year": run.start_year,
                "end_year": run.end_year,
                "clusters": run.clusters,
                "errors": ["Select at least one author identity cluster."],
                "missing_affiliation_rows": [],
            },
            status_code=400,
        )

    cluster_label_by_id = {cluster.cluster_id: cluster.label for cluster in run.clusters}
    selected_matches = [match for match in run.matches if match.cluster_id in selected]
    citation_selection_rows = _build_citation_selection_rows(
        citations=run.citations,
        matches=selected_matches,
        cluster_label_by_id=cluster_label_by_id,
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
        )
        return _render_analysis_result(
            request,
            run_id=run_id,
            selected_cluster_ids=selected,
            selected_pmids=set(),
            pmid_filter_enabled=0,
            analysis=analysis,
            start_year=run.start_year,
            end_year=run.end_year,
        )

    return templates.TemplateResponse(
        request,
        "citation_select.html",
        {
            "run_id": run_id,
            "author_name": run.author_name,
            "start_year": run.start_year,
            "end_year": run.end_year,
            "selected_cluster_ids": sorted(selected),
            "citation_selection_rows": citation_selection_rows,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    request: Request,
    run_id: str = Form(...),
    selected_cluster_ids: list[str] = Form(default=[]),
    selected_pmids: list[str] = Form(default=[]),
    pmid_filter_enabled: int = Form(default=0),
) -> HTMLResponse:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    selected_pmid_set = set(selected_pmids)
    logger.info(
        "Analyze request: run_id=%s, selected_clusters=%d, selected_pmids=%d, pmid_filter_enabled=%d",
        run_id,
        len(selected),
        len(selected_pmid_set),
        pmid_filter_enabled,
    )

    filtered_matches = [
        match
        for match in run.matches
        if match.cluster_id in selected and (pmid_filter_enabled == 0 or match.pmid in selected_pmid_set)
    ]

    analysis = analyze_citations(
        citations=run.citations,
        matches=filtered_matches,
        selected_cluster_ids=selected,
        start_year=run.start_year,
        end_year=run.end_year,
    )
    return _render_analysis_result(
        request,
        run_id=run_id,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        pmid_filter_enabled=pmid_filter_enabled,
        analysis=analysis,
        start_year=run.start_year,
        end_year=run.end_year,
        force_report=True,
    )


@app.post("/finalize", response_class=HTMLResponse)
def finalize(
    request: Request,
    run_id: str = Form(...),
    selected_cluster_ids: list[str] = Form(default=[]),
    selected_pmids: list[str] = Form(default=[]),
    pmid_filter_enabled: int = Form(default=0),
    accepted_row_indices: list[int] = Form(default=[]),
) -> HTMLResponse:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    selected_pmid_set = set(selected_pmids)
    accepted = set(accepted_row_indices)
    logger.info(
        "Finalize request: run_id=%s, selected_clusters=%d, selected_pmids=%d, pmid_filter_enabled=%d, accepted_uncertain=%d",
        run_id,
        len(selected),
        len(selected_pmid_set),
        pmid_filter_enabled,
        len(accepted),
    )

    filtered_matches = [
        match
        for match in run.matches
        if match.cluster_id in selected and (pmid_filter_enabled == 0 or match.pmid in selected_pmid_set)
    ]

    analysis = analyze_citations(
        citations=run.citations,
        matches=filtered_matches,
        selected_cluster_ids=selected,
        start_year=run.start_year,
        end_year=run.end_year,
    )

    for idx in analysis.uncertain_indices:
        analysis.rows[idx].include = idx in accepted
    return _render_analysis_result(
        request,
        run_id=run_id,
        selected_cluster_ids=selected,
        selected_pmids=selected_pmid_set,
        pmid_filter_enabled=pmid_filter_enabled,
        analysis=analysis,
        start_year=run.start_year,
        end_year=run.end_year,
    )
