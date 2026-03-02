import pytest

pytest.importorskip("fastapi")

from app.main import _build_citation_selection_rows
from app.models import Author, AuthorMatch, Citation


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
