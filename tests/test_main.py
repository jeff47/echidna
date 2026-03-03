import pytest

pytest.importorskip("fastapi")

from app.logic import AnalysisResult
from app.main import _build_citation_selection_rows, _build_cluster_orcid_affiliation_matches, _orcid_row_fields, _with_row_render_fields
from app.models import Author, AuthorMatch, Citation, ReportRow


def _citation(pmid: str, *, year: int = 2024) -> Citation:
    return Citation(
        pmid=pmid,
        pmcid=None,
        title="Example title",
        journal="J Test",
        print_year=year,
        print_date=str(year),
        authors=[
            Author(position=1, last_name="Rice", fore_name="Jeffrey", initials="J", affiliation="Inst A"),
            Author(position=2, last_name="Rice", fore_name="James", initials="J", affiliation="Inst B"),
        ],
        source_for_roles="pmc",
    )


def test_citation_selection_marks_review_when_multiple_positions_exist_across_all_matches() -> None:
    citation = _citation("100")
    selected = AuthorMatch(
        pmid="100",
        position=1,
        method="given",
        author=citation.authors[0],
        cluster_id="cluster-a",
    )
    other = AuthorMatch(
        pmid="100",
        position=2,
        method="given",
        author=citation.authors[1],
        cluster_id="cluster-b",
    )

    rows = _build_citation_selection_rows(
        citations=[citation],
        matches=[selected],
        all_matches=[selected, other],
        cluster_label_by_id={"cluster-a": "Rice, Jeffrey", "cluster-b": "Rice, James"},
        start_year=2021,
        end_year=2026,
    )

    assert len(rows) == 1
    assert rows[0]["needs_review"] is True
    assert "Multiple matched author positions in citation" in str(rows[0]["review_reason"])


def test_citation_selection_marks_review_for_initials_fallback() -> None:
    citation = _citation("101")
    selected = AuthorMatch(
        pmid="101",
        position=1,
        method="initials",
        author=citation.authors[0],
        cluster_id="cluster-a",
    )

    rows = _build_citation_selection_rows(
        citations=[citation],
        matches=[selected],
        all_matches=[selected],
        cluster_label_by_id={"cluster-a": "Rice, Jeffrey"},
        start_year=2021,
        end_year=2026,
    )

    assert len(rows) == 1
    assert rows[0]["needs_review"] is True
    assert "Matched by initials fallback" in str(rows[0]["review_reason"])


def test_citation_selection_no_review_when_single_given_match_with_affiliation() -> None:
    citation = _citation("102")
    selected = AuthorMatch(
        pmid="102",
        position=1,
        method="given",
        author=citation.authors[0],
        cluster_id="cluster-a",
    )

    rows = _build_citation_selection_rows(
        citations=[citation],
        matches=[selected],
        all_matches=[selected],
        cluster_label_by_id={"cluster-a": "Rice, Jeffrey"},
        start_year=2021,
        end_year=2026,
    )

    assert len(rows) == 1
    assert rows[0]["needs_review"] is False
    assert rows[0]["review_reason"] == ""


def test_cluster_orcid_affiliation_matches_dedupes_institutions_and_keeps_strongest_level() -> None:
    cluster_rows = {
        "cluster-a": [
            {
                "orcid_affiliation_verified": True,
                "orcid_affiliation_institution": "Brigham and Women's Hospital",
                "orcid_affiliation_level": "contains",
            },
            {
                "orcid_affiliation_verified": True,
                "orcid_affiliation_institution": "Harvard Medical School",
                "orcid_affiliation_level": "possible",
            },
            {
                "orcid_affiliation_verified": True,
                "orcid_affiliation_institution": "Brigham and Women's Hospital",
                "orcid_affiliation_level": "likely",
            },
            {
                "orcid_affiliation_verified": False,
                "orcid_affiliation_institution": "Ignored Institute",
                "orcid_affiliation_level": "verified",
            },
        ]
    }

    matches = _build_cluster_orcid_affiliation_matches(cluster_rows)

    assert matches["cluster-a"] == [
        {
            "key": "brighamandwomenshospital",
            "name": "Brigham and Women's Hospital",
            "level": "likely",
        },
        {
            "key": "harvardmedicalschool",
            "name": "Harvard Medical School",
            "level": "possible",
        },
    ]


def test_orcid_row_fields_hides_citation_badge_for_affiliation_only_match() -> None:
    fields = _orcid_row_fields(
        pmid="200",
        target_orcid="0000-0001-2345-6789",
        orcid_identifier_matches={},
        orcid_affiliation_matches={
            "200": {
                "level": "contains",
                "label": "affiliation contains institution (Brigham and Women's Hospital)",
                "institution": "Brigham and Women's Hospital",
            }
        },
    )

    assert fields["orcid_badge_show"] is False
    assert fields["orcid_affiliation_verified"] is True
    assert fields["orcid_affiliation_level"] == "contains"


def test_with_row_render_fields_adds_note_when_missing_from_orcid_works_list() -> None:
    citation = _citation("200")
    analysis = AnalysisResult(
        rows=[
            ReportRow(
                citation=citation,
                counted_overall=True,
                counted_first=False,
                counted_senior=False,
                counted_review_senior=False,
                is_review=False,
                uncertainty_reasons=[],
                include=True,
                forced_include=False,
                matched_positions={1},
            )
        ],
        uncertain_indices=[],
    )

    rows = _with_row_render_fields(
        analysis,
        orcid_identifier_matches={},
        orcid_affiliation_matches={
            "200": {
                "level": "likely",
                "label": "institution match (Inst A)",
                "institution": "Inst A",
            }
        },
        target_orcid="0000-0001-2345-6789",
        orcid_identifier_checked=True,
    )

    assert "No match from ORCiD" in str(rows[0]["notes"])


def test_with_row_render_fields_skips_missing_orcid_note_when_identifier_check_unavailable() -> None:
    citation = _citation("201")
    analysis = AnalysisResult(
        rows=[
            ReportRow(
                citation=citation,
                counted_overall=True,
                counted_first=False,
                counted_senior=False,
                counted_review_senior=False,
                is_review=False,
                uncertainty_reasons=[],
                include=True,
                forced_include=False,
                matched_positions={1},
            )
        ],
        uncertain_indices=[],
    )

    rows = _with_row_render_fields(
        analysis,
        orcid_identifier_matches={},
        orcid_affiliation_matches={},
        target_orcid="0000-0001-2345-6789",
        orcid_identifier_checked=False,
    )

    assert "No match from ORCiD" not in str(rows[0]["notes"])
