from typing import cast

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.logic import AnalysisResult
from app.main import app as fastapi_app
from app.main import (
    RunState,
    _apply_uncertain_selection_and_corrections,
    _build_cluster_affiliation_display_rows,
    _build_citation_selection_rows,
    _build_cluster_citation_rows,
    _build_cluster_orcid_affiliation_matches,
    _build_out_of_window_rows,
    _orcid_row_fields,
    _partition_excluded_citation_rows,
    _selected_default_excluded_type_terms,
    _with_row_render_fields,
    _xlsx_download_filename,
)
from app.models import Author, AuthorMatch, Citation, Cluster, ReportRow


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


def test_cluster_affiliation_display_rows_ignores_generic_contains_overlap() -> None:
    clusters = [
        Cluster(
            cluster_id="cluster-a",
            label="Rice, Jeffrey",
            mention_count=1,
            years=[2024],
            affiliations=["Department of Medicine, University School of Medicine"],
        )
    ]
    cluster_orcid_affiliation_matches = {
        "cluster-a": [
            {
                "key": "harvardmedicalschool",
                "name": "Harvard Medical School",
                "level": "likely",
            }
        ]
    }

    rows_by_cluster = _build_cluster_affiliation_display_rows(
        clusters=clusters,
        cluster_orcid_affiliation_matches=cluster_orcid_affiliation_matches,
        target_orcid="0000-0001-2345-6789",
    )
    rows = rows_by_cluster["cluster-a"]

    assert len(rows) == 1
    assert rows[0]["matched"] is False
    assert rows[0]["match_level"] == ""


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


def test_with_row_render_fields_adds_superseded_preprint_note() -> None:
    citation = _citation("299")
    citation.publication_types = ["Preprint"]
    analysis = AnalysisResult(
        rows=[
            ReportRow(
                citation=citation,
                counted_overall=False,
                counted_first=False,
                counted_senior=False,
                counted_review_senior=False,
                is_review=False,
                is_preprint=True,
                uncertainty_reasons=[],
                include=False,
                forced_include=False,
                matched_positions={1},
                preprint_superseded_by=["38821936"],
            )
        ],
        uncertain_indices=[],
    )

    rows = _with_row_render_fields(
        analysis,
        orcid_identifier_matches={},
        orcid_affiliation_matches={},
        target_orcid="",
        orcid_identifier_checked=False,
    )

    assert "Preprint skipped: superseded by peer-reviewed PMID(s): 38821936" in str(rows[0]["notes"])


def test_with_row_render_fields_includes_source_badges() -> None:
    citation = _citation("202")
    citation.source_tags = {"user", "litsense"}
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
        target_orcid="",
        orcid_identifier_checked=False,
    )

    source_badges = cast(list[dict[str, object]], rows[0]["source_badges"])
    labels = [str(item["label"]) for item in source_badges]
    assert labels == ["user", "litsense"]


def test_apply_uncertain_selection_and_corrections_applies_mode_only_to_included_rows() -> None:
    uncertain_included = ReportRow(
        citation=_citation("401"),
        counted_overall=False,
        counted_first=False,
        counted_senior=False,
        counted_review_senior=False,
        is_review=False,
        uncertainty_reasons=["matched by initials fallback"],
        include=False,
    )
    uncertain_excluded = ReportRow(
        citation=_citation("402"),
        counted_overall=False,
        counted_first=False,
        counted_senior=False,
        counted_review_senior=False,
        is_review=False,
        uncertainty_reasons=["multiple matching authors"],
        include=False,
    )
    analysis = AnalysisResult(rows=[uncertain_included, uncertain_excluded], uncertain_indices=[0, 1])

    _apply_uncertain_selection_and_corrections(
        analysis=analysis,
        accepted_row_indices={0},
        correction_entries=["401::first_senior_author", "402::coauthor"],
    )

    assert analysis.rows[0].include is True
    assert analysis.rows[0].counted_overall is True
    assert analysis.rows[0].counted_senior is True
    assert analysis.rows[0].is_review is False

    # Excluded uncertain rows remain excluded regardless of dropdown value.
    assert analysis.rows[1].include is False
    assert analysis.rows[1].counted_overall is False


def test_build_cluster_citation_rows_includes_source_badges() -> None:
    citation = _citation("203")
    citation.source_tags = {"litsense"}
    match = AuthorMatch(
        pmid="203",
        position=1,
        method="given",
        author=citation.authors[0],
        cluster_id="cluster-a",
    )

    rows = _build_cluster_citation_rows(citations=[citation], matches=[match])
    cluster_rows = rows["cluster-a"]
    source_badges = cast(list[dict[str, object]], cluster_rows[0]["source_badges"])
    labels = [str(item["label"]) for item in source_badges]

    assert labels == ["litsense"]


def test_partition_excluded_citation_rows_splits_type_and_window() -> None:
    rows: list[dict[str, object]] = [
        {"pmid": "1", "reason": "Excluded by publication type: Editorial"},
        {"pmid": "2", "reason": "Outside year window 2020-2026"},
        {"pmid": "3", "reason": "Excluded by pre-disambiguation filter"},
    ]

    by_type, by_window, other = _partition_excluded_citation_rows(rows)

    assert [str(row["pmid"]) for row in by_type] == ["1"]
    assert [str(row["pmid"]) for row in by_window] == ["2"]
    assert [str(row["pmid"]) for row in other] == ["3"]


def test_selected_default_excluded_type_terms_preserves_default_order_and_filters_unknown() -> None:
    options = ["preprint", "editorial", "comment", "letter"]
    selected = _selected_default_excluded_type_terms(["COMMENT", "letter", "unknown"], options)

    assert selected == ["comment", "letter"]


def test_build_out_of_window_rows_preserves_user_source_badge() -> None:
    citation = _citation("204", year=2018)
    citation.source_tags = {"user"}
    match = AuthorMatch(
        pmid="204",
        position=1,
        method="given",
        author=citation.authors[0],
        cluster_id="cluster-a",
    )

    rows = _build_out_of_window_rows(
        citations=[citation],
        matches=[match],
        start_year=2020,
        end_year=2026,
    )

    assert len(rows) == 1
    source_badges = cast(list[dict[str, object]], rows[0]["source_badges"])
    labels = [str(item["label"]) for item in source_badges]
    assert labels == ["user"]


def test_orcid_search_endpoint_returns_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main as main_module

    def fake_search_profiles(author_name: str, limit: int = 12) -> list[dict[str, str]]:
        assert author_name == "Jeffrey Rice"
        assert limit == 3
        return [
            {
                "orcid": "0000-0002-1825-0097",
                "display_name": "Jeffrey Rice",
                "institution": "NIH",
                "country": "US",
                "record_url": "https://orcid.org/0000-0002-1825-0097",
            }
        ]

    monkeypatch.setattr(main_module.orcid_client, "search_profiles", fake_search_profiles)

    client = TestClient(fastapi_app)
    response = client.get("/orcid-search", params={"author_name": "Jeffrey Rice", "limit": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["author_name"] == "Jeffrey Rice"
    assert payload["matches"][0]["orcid"] == "0000-0002-1825-0097"


def test_disambiguate_merges_litsense_only_pmids_into_verification_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main as main_module

    citation_111 = _citation("111")
    citation_222 = _citation("222")
    citation_by_pmid = {
        "111": citation_111,
        "222": citation_222,
    }
    fetch_batches: list[list[str]] = []
    captured_run: dict[str, RunState] = {}

    def fake_fetch_citations(pmids: list[str]) -> list[Citation]:
        fetch_batches.append(list(pmids))
        return [citation_by_pmid[pmid] for pmid in pmids if pmid in citation_by_pmid]

    monkeypatch.setattr(main_module.client, "fetch_citations", fake_fetch_citations)
    monkeypatch.setattr(
        main_module.client,
        "fetch_litsense_pmids_by_query",
        lambda *, seed_pmid, author_name, max_pages=10: {"111", "222"},
    )

    def fake_save_run(run_id: str, run: RunState) -> None:
        captured_run["state"] = run

    monkeypatch.setattr(main_module, "_save_run", fake_save_run)

    client = TestClient(fastapi_app)
    response = client.post(
        "/disambiguate",
        data={
            "author_name": "Jeffrey Rice",
            "author_orcid": "",
            "pmids": "111",
            "start_year": "2020",
            "end_year": "2026",
            "all_years": "on",
        },
    )

    assert response.status_code == 200
    assert fetch_batches == [["111"], ["222"]]

    run = captured_run["state"]
    pmids = {citation.pmid for citation in run.citations}
    assert pmids == {"111", "222"}

    tags_by_pmid = {citation.pmid: citation.source_tags for citation in run.citations}
    assert tags_by_pmid["111"] == {"user", "litsense"}
    assert tags_by_pmid["222"] == {"litsense"}


@pytest.mark.parametrize(
    ("include_preprints_field", "expect_preprint_excluded"),
    [
        (None, True),
        ("on", False),
    ],
)
def test_disambiguate_include_preprints_toggle_controls_preprint_exclusion(
    monkeypatch: pytest.MonkeyPatch,
    include_preprints_field: str | None,
    expect_preprint_excluded: bool,
) -> None:
    from app import main as main_module

    captured_run: dict[str, RunState] = {}
    citation_111 = _citation("111")

    monkeypatch.setattr(main_module.client, "fetch_citations", lambda pmids: [citation_111])
    monkeypatch.setattr(
        main_module.client,
        "fetch_litsense_pmids_by_query",
        lambda *, seed_pmid, author_name, max_pages=10: {"111"},
    )
    monkeypatch.setattr(main_module, "_save_run", lambda run_id, run: captured_run.setdefault("state", run))

    form_data: dict[str, str] = {
        "author_name": "Jeffrey Rice",
        "author_orcid": "",
        "pmids": "111",
        "start_year": "2020",
        "end_year": "2026",
        "all_years": "on",
    }
    if include_preprints_field is not None:
        form_data["include_preprints"] = include_preprints_field

    client = TestClient(fastapi_app)
    response = client.post("/disambiguate", data=form_data)

    assert response.status_code == 200
    excluded_type_terms = captured_run["state"].excluded_type_terms
    assert ("preprint" in excluded_type_terms) is expect_preprint_excluded


def _run_state(author_name: str, start_year: int | None, end_year: int | None) -> RunState:
    return RunState(
        author_name=author_name,
        target_orcid="",
        start_year=start_year,
        end_year=end_year,
        excluded_type_terms=[],
        citations=[],
        clusters=[],
        matches=[],
        excluded_pmids=set(),
        excluded_reasons={},
        orcid_identifier_matches={},
        orcid_affiliation_matches={},
        orcid_sync_error=None,
    )


def test_xlsx_download_filename_uses_surname_plus_initials_for_full_name() -> None:
    run = _run_state("Jeffrey S Rice", 2021, 2026)
    assert _xlsx_download_filename(run) == "ricejs_publications_2021_2026.xlsx"


def test_xlsx_download_filename_uses_surname_plus_initials_for_pubmed_style_name() -> None:
    run = _run_state("Rice JS", 2021, 2026)
    assert _xlsx_download_filename(run) == "ricejs_publications_2021_2026.xlsx"


def test_xlsx_download_filename_all_years_suffix_when_year_filter_disabled() -> None:
    run = _run_state("Rice JS", None, None)
    assert _xlsx_download_filename(run) == "ricejs_publications_all_years.xlsx"


def test_xlsx_download_filename_falls_back_when_name_cannot_be_parsed() -> None:
    run = _run_state("Cher", 2021, 2026)
    assert _xlsx_download_filename(run) == "cher_publications_2021_2026.xlsx"
