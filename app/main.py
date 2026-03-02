from __future__ import annotations

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
from fastapi.responses import HTMLResponse
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
    level=os.getenv("PMCPARSER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Publication Counter")
templates = Jinja2Templates(directory="templates")
client = NcbiClient()
RUNS_DIR = Path(os.getenv("PMCPARSER_RUNS_DIR", "/tmp/pmcparser-runs"))


@dataclass(slots=True)
class RunState:
    author_name: str
    start_year: int
    end_year: int
    peer_reviewed_only: bool
    excluded_type_terms: list[str]
    citations: list[Citation]
    clusters: list[Cluster]
    matches: list[AuthorMatch]


RUNS: dict[str, RunState] = {}
RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _to_author_list(citation: Citation) -> str:
    return ", ".join(f"{a.last_name} {a.initials}".strip() for a in citation.authors)


def _with_row_render_fields(analysis: AnalysisResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(analysis.rows):
        cited_as: list[str] = []
        if row.counted_overall:
            cited_as.append("overall")
        if row.counted_first:
            cited_as.append("first")
        if row.counted_senior:
            cited_as.append("senior")
        if row.counted_review_senior:
            cited_as.append("review_senior")
        if row.is_review and not row.counted_overall:
            cited_as.append("review_excluded_from_total")
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
                "publication_types": ", ".join(row.citation.publication_types) if row.citation.publication_types else "(unknown)",
                "counted_as": ", ".join(cited_as) if cited_as else "none",
                "role_source": row.citation.source_for_roles,
                "notes": "; ".join(row.citation.notes + row.uncertainty_reasons),
                "include": row.include,
                "status_label": (
                    "included (review separate)"
                    if row.include and not row.counted_overall and row.is_review
                    else ("included" if row.include else "excluded")
                ),
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
    start_year: int,
    end_year: int,
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
                "publication_types": row["publication_types"],
                "cluster_labels": " | ".join(sorted(row["cluster_labels"])),
                "affiliations": " | ".join(sorted(row["affiliations"])) if row["affiliations"] else "(no affiliation listed)",
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
    start_year: int,
    end_year: int,
) -> list[dict[str, object]]:
    citation_by_pmid = {citation.pmid: citation for citation in citations}
    selected_pmids = {match.pmid for match in matches}
    rows: list[dict[str, object]] = []
    for pmid in selected_pmids:
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
    }


def _deserialize_run(payload: dict[str, Any]) -> RunState:
    return RunState(
        author_name=str(payload["author_name"]),
        start_year=int(payload["start_year"]),
        end_year=int(payload["end_year"]),
        peer_reviewed_only=bool(payload.get("peer_reviewed_only", False)),
        excluded_type_terms=[str(v) for v in payload.get("excluded_type_terms", [])],
        citations=[_deserialize_citation(item) for item in payload.get("citations", [])],
        clusters=[_deserialize_cluster(item) for item in payload.get("clusters", [])],
        matches=[_deserialize_match(item) for item in payload.get("matches", [])],
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
    pmid_filter_enabled: int,
    analysis: AnalysisResult,
    start_year: int,
    end_year: int,
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
            "out_of_window_rows": out_of_window_rows or [],
            "start_year": start_year,
            "end_year": end_year,
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
            "extra_excluded_types": "",
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
    peer_reviewed_only: str | None = Form(default=None),
    extra_excluded_types: str = Form(default=""),
) -> HTMLResponse:
    peer_reviewed_only_enabled = peer_reviewed_only is not None
    excluded_type_terms = default_excluded_type_terms()
    excluded_type_terms.extend(parse_type_terms(extra_excluded_types))
    logger.info(
        "Disambiguation request received: author=%r, start_year=%d, end_year=%d, peer_reviewed_only=%s",
        author_name,
        start_year,
        end_year,
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
                "extra_excluded_types": extra_excluded_types,
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
                "extra_excluded_types": extra_excluded_types,
                "error": "No PMIDs found in text input or uploaded file",
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

    in_window_citations = [
        citation
        for citation in citations
        if citation_in_window(citation, start_year, end_year)
    ]
    excluded_out_of_window = len(citations) - len(in_window_citations)
    excluded_by_type = 0
    if peer_reviewed_only_enabled:
        type_filtered: list[Citation] = []
        for citation in in_window_citations:
            is_excluded, matched_types = citation_matches_excluded_type(citation, excluded_type_terms)
            if is_excluded:
                excluded_by_type += 1
                citation.notes.append(f"Excluded by publication type filter: {', '.join(matched_types)}")
                continue
            type_filtered.append(citation)
        in_window_citations = type_filtered
    logger.info(
        "Window/type filters before disambiguation: in_window=%d, excluded_out_of_window=%d, excluded_by_type=%d",
        len(in_window_citations),
        excluded_out_of_window,
        excluded_by_type,
    )

    clusters, matches = build_clusters(in_window_citations, target)
    citation_by_pmid = {citation.pmid: citation for citation in in_window_citations}
    cluster_label_by_id = {cluster.cluster_id: cluster.label for cluster in clusters}
    cluster_citation_rows = _build_cluster_citation_rows(citations=in_window_citations, matches=matches)
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
    _save_run(
        run_id,
        RunState(
            author_name=author_name,
            start_year=start_year,
            end_year=end_year,
            peer_reviewed_only=peer_reviewed_only_enabled,
            excluded_type_terms=excluded_type_terms,
            citations=in_window_citations,
            clusters=clusters,
            matches=matches,
        ),
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
            "cluster_citation_rows": cluster_citation_rows,
            "errors": errors,
            "missing_affiliation_rows": missing_affiliation_rows,
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
) -> HTMLResponse:
    run = _load_run(run_id)
    selected = set(selected_cluster_ids)
    if not selected:
        cluster_citation_rows = _build_cluster_citation_rows(citations=run.citations, matches=run.matches)
        return templates.TemplateResponse(
            request,
            "disambiguate.html",
            {
                "run_id": run_id,
                "author_name": run.author_name,
                "start_year": run.start_year,
                "end_year": run.end_year,
                "clusters": run.clusters,
                "cluster_citation_rows": cluster_citation_rows,
                "errors": ["Select at least one author identity cluster."],
                "missing_affiliation_rows": [],
                "excluded_out_of_window": 0,
                "excluded_by_type": 0,
            },
            status_code=400,
        )

    cluster_label_by_id = {cluster.cluster_id: cluster.label for cluster in run.clusters}
    selected_matches = [match for match in run.matches if match.cluster_id in selected]
    citation_selection_rows = _build_citation_selection_rows(
        citations=run.citations,
        matches=selected_matches,
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
        )
        out_of_window_rows = _build_out_of_window_rows(
            citations=run.citations,
            matches=selected_matches,
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
    out_of_window_rows = _build_out_of_window_rows(
        citations=run.citations,
        matches=filtered_matches,
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
        out_of_window_rows=out_of_window_rows,
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
    out_of_window_rows = _build_out_of_window_rows(
        citations=run.citations,
        matches=filtered_matches,
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
        out_of_window_rows=out_of_window_rows,
    )
