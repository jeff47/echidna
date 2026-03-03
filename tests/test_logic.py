import pytest

from app.logic import (
    analyze_citations,
    build_clusters,
    citation_in_window,
    citation_matches_excluded_type,
    format_summary,
    match_author,
    parse_pmids,
    parse_target_name,
    parse_target_orcid,
)
from app.models import Author, Citation


def _citation(
    pmid: str,
    year: int,
    authors: list[Author],
    *,
    source: str = "pmc",
    co_first: set[int] | None = None,
    co_senior: set[int] | None = None,
    publication_types: list[str] | None = None,
) -> Citation:
    return Citation(
        pmid=pmid,
        pmcid=f"PMC{pmid}",
        title=f"Title {pmid}",
        journal="J Immunol",
        print_year=year,
        print_date=str(year),
        authors=authors,
        source_for_roles=source,
        publication_types=publication_types or [],
        co_first_positions=co_first or set(),
        co_senior_positions=co_senior or set(),
    )


def _author(position: int, last: str, fore: str, initials: str, aff: str, *, orcid: str = "") -> Author:
    return Author(position=position, last_name=last, fore_name=fore, initials=initials, affiliation=aff, orcid=orcid)






def test_parse_target_name_accepts_surname_initial_formats() -> None:
    t1 = parse_target_name("Rice J")
    assert t1.surname == "rice"
    assert t1.initials == "j"
    assert t1.style == "initials"

    t2 = parse_target_name("Rice JS")
    assert t2.surname == "rice"
    assert t2.initials == "js"
    assert t2.style == "initials"


@pytest.mark.parametrize(
    ("raw", "style", "surname", "given", "initials"),
    [
        ("Rice, Jeffrey", "full", "rice", "jeffrey", "j"),
        ("Rice, Jeffrey,", "full", "rice", "jeffrey", "j"),
        ("Rice, J", "initials", "rice", "j", "j"),
        ("Rice, J.", "initials", "rice", "j", "j"),
        ("RiCe, jS", "initials", "rice", "js", "js"),
    ],
)
def test_parse_target_name_accepts_comma_forms_and_mixed_case(
    raw: str,
    style: str,
    surname: str,
    given: str,
    initials: str,
) -> None:
    parsed = parse_target_name(raw)

    assert parsed.style == style
    assert parsed.surname == surname
    assert parsed.given == given
    assert parsed.initials == initials


def test_initials_style_matches_surname_plus_initials() -> None:
    target = parse_target_name("Rice JS")
    author = _author(1, "Rice", "Jeffrey Stephen", "JS", "Inst")
    matched, method = match_author(author, target)

    assert matched is True
    assert method == "initials"

def test_initials_style_matches_when_record_has_only_first_initial() -> None:
    target = parse_target_name("Sage PT")
    author = _author(1, "Sage", "Peter", "P", "Inst")
    matched, method = match_author(author, target)

    assert matched is True
    assert method == "initials"


def test_full_name_style_ignores_input_punctuation_for_match() -> None:
    target = parse_target_name("Rice, Jeffrey,")
    author = _author(1, "Rice", "Jeffrey", "J", "Inst")
    matched, method = match_author(author, target)

    assert matched is True
    assert method == "given"


def test_parse_target_orcid_accepts_normalized_and_url_forms() -> None:
    assert parse_target_orcid("0000-0002-1825-0097") == "0000-0002-1825-0097"
    assert parse_target_orcid("https://orcid.org/0000-0002-1825-0097") == "0000-0002-1825-0097"
    assert parse_target_orcid("0000000218250097") == "0000-0002-1825-0097"


def test_parse_target_orcid_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        parse_target_orcid("not-an-orcid")


def test_match_author_orcid_is_definitive_when_available() -> None:
    target = parse_target_name("Jeffrey Rice")
    good = _author(1, "Rice", "James", "J", "Inst", orcid="0000-0002-1825-0097")
    matched, method = match_author(good, target, target_orcid="0000-0002-1825-0097")
    assert matched is True
    assert method == "orcid"

    bad = _author(1, "Rice", "Jeffrey", "J", "Inst", orcid="0000-0002-1825-0098")
    matched, method = match_author(bad, target, target_orcid="0000-0002-1825-0097")
    assert matched is False
    assert method is None


def test_build_clusters_prefers_orcid_matches_for_citation() -> None:
    target = parse_target_name("Jane Doe")
    citation = _citation(
        "orcid-1",
        2025,
        [
            _author(1, "Doe", "Jane", "J", "Inst A", orcid="0000-0002-1825-0097"),
            _author(2, "Doe", "Jane", "J", "Inst B"),
        ],
    )
    _, matches = build_clusters([citation], target, target_orcid="0000-0002-1825-0097")

    assert len(matches) == 1
    assert matches[0].method == "orcid"

def test_parse_pmids_deduplicates_and_extracts_digits() -> None:
    raw = "PMID: 12345\n12345\nfoo 9999 bar\nnoid"
    assert parse_pmids(raw) == ["12345", "9999"]


def test_build_clusters_groups_by_affiliation_fingerprint() -> None:
    target = parse_target_name("Jane Q Doe")
    citations = [
        _citation(
            "1",
            2023,
            [
                _author(1, "Doe", "Jane Q", "JQ", "Inst A"),
                _author(2, "Roe", "John", "J", "Inst B"),
            ],
        ),
        _citation(
            "2",
            2024,
            [
                _author(1, "Doe", "Jane Q", "JQ", "Inst C"),
                _author(2, "Roe", "John", "J", "Inst B"),
            ],
        ),
    ]

    clusters, matches = build_clusters(citations, target)

    assert len(matches) == 2
    assert len(clusters) == 2


def test_build_clusters_does_not_group_no_affiliation_mentions() -> None:
    target = parse_target_name("Jane Q Doe")
    citations = [
        _citation("101", 2023, [_author(1, "Doe", "Jane Q", "JQ", "")]),
        _citation("102", 2024, [_author(1, "Doe", "Jane Q", "JQ", "")]),
    ]
    clusters, matches = build_clusters(citations, target)

    assert len(matches) == 2
    assert len(clusters) == 2
    assert all(cluster.mention_count == 1 for cluster in clusters)


def test_build_clusters_groups_same_institute_across_different_departments() -> None:
    target = parse_target_name("Laura Donlin")
    citations = [
        _citation(
            "201",
            2023,
            [
                _author(
                    1,
                    "Donlin",
                    "Laura",
                    "L",
                    "Department of Rheumatology, Hospital for Special Surgery, New York, NY",
                )
            ],
        ),
        _citation(
            "202",
            2021,
            [
                _author(
                    1,
                    "Donlin",
                    "Laura",
                    "L",
                    "Derfner Foundation Precision Medicine Laboratory, Hospital for Special Surgery, New York, NY",
                )
            ],
        ),
    ]
    clusters, matches = build_clusters(citations, target)

    assert len(matches) == 2
    assert len(clusters) == 1
    assert clusters[0].mention_count == 2
    assert "Hospital for Special Surgery" in clusters[0].affiliations


def test_cluster_affiliation_labels_dedupe_noisy_variants() -> None:
    target = parse_target_name("Peter Sage")
    citations = [
        _citation(
            "301",
            2024,
            [
                _author(
                    1,
                    "Sage",
                    "Peter T.",
                    "PT",
                    "Transplantation Research Center, Renal Division, Brigham and Women's Hospital, Harvard Medical School, Boston, MA, USA",
                )
            ],
        ),
        _citation(
            "302",
            2025,
            [
                _author(
                    1,
                    "Sage",
                    "Peter T.",
                    "PT",
                    "Transplant Research Center, Renal Division, Brigham and Women's Hospital, Harvard Medical School, Boston, MA USA",
                )
            ],
        ),
    ]
    clusters, _ = build_clusters(citations, target)

    assert len(clusters) == 1
    assert "Brigham and Women's Hospital" in clusters[0].affiliations
    assert "Harvard Medical School" in clusters[0].affiliations
    assert len(clusters[0].affiliations) <= 3


def test_match_author_does_not_fallback_to_initials_for_given_name_mismatch() -> None:
    target = parse_target_name("Jeffrey Rice")
    matched, method = match_author(
        _author(1, "Rice", "James", "J", "Inst A"),
        target,
    )
    assert matched is False
    assert method is None


def test_analyze_citations_marks_first_and_senior_from_pmc_flags() -> None:
    target = parse_target_name("Jane Doe")
    citation = _citation(
        "10",
        2024,
        [
            _author(1, "Doe", "Jane", "J", "Inst A"),
            _author(2, "Smith", "Alan", "A", "Inst B"),
            _author(3, "Brown", "Chris", "C", "Inst C"),
        ],
        co_first={2},
        co_senior={2},
    )
    clusters, matches = build_clusters([citation], target)

    result = analyze_citations(
        citations=[citation],
        matches=matches,
        selected_cluster_ids={clusters[0].cluster_id},
        start_year=2021,
        end_year=2026,
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.counted_first is True
    assert row.counted_senior is False


def test_uncertain_when_multiple_name_matches_or_pubmed_only() -> None:
    target = parse_target_name("Jane Doe")
    citation = _citation(
        "20",
        2025,
        [
            _author(1, "Doe", "Jane", "J", "Inst A"),
            _author(2, "Doe", "James", "J", "Inst X"),
            _author(3, "Lee", "Ann", "A", "Inst Y"),
        ],
        source="pubmed",
    )
    clusters, matches = build_clusters([citation], target)

    result = analyze_citations(
        citations=[citation],
        matches=matches,
        selected_cluster_ids={clusters[0].cluster_id},
        start_year=2021,
        end_year=2026,
    )

    assert result.uncertain_indices == [0]
    assert result.rows[0].include is False
    assert "PMC missing" in " ".join(result.rows[0].uncertainty_reasons)


def test_analyze_citations_forced_include_overrides_exclusion_and_window() -> None:
    target = parse_target_name("Jane Doe")
    citation = _citation(
        "2099",
        2020,
        [
            _author(1, "Doe", "Jane", "J", "Example Hospital"),
            _author(2, "Lee", "Ann", "A", "Example Hospital"),
        ],
        source="pmc",
    )
    clusters, matches = build_clusters([citation], target)
    result = analyze_citations(
        citations=[citation],
        matches=matches,
        selected_cluster_ids={clusters[0].cluster_id},
        start_year=2021,
        end_year=2026,
        forced_include_pmids={"2099"},
        excluded_pmids={"2099"},
    )

    assert len(result.rows) == 1
    assert result.rows[0].citation.pmid == "2099"
    assert result.rows[0].forced_include is True


def test_uncertain_indices_follow_sorted_row_order() -> None:
    target = parse_target_name("Jane Doe")
    uncertain = _citation(
        "21",
        2022,
        [
            _author(1, "Doe", "Jane", "J", "Example Hospital"),
            _author(2, "Lee", "Ann", "A", "Example Hospital"),
        ],
        source="pubmed",
    )
    certain = _citation(
        "22",
        2025,
        [
            _author(1, "Doe", "Jane", "J", "Example Hospital"),
            _author(2, "Lee", "Ann", "A", "Example Hospital"),
        ],
        source="pmc",
    )
    clusters, matches = build_clusters([uncertain, certain], target)
    result = analyze_citations(
        citations=[uncertain, certain],
        matches=matches,
        selected_cluster_ids={clusters[0].cluster_id},
        start_year=2021,
        end_year=2026,
    )

    assert [row.citation.pmid for row in result.rows] == ["22", "21"]
    assert result.uncertain_indices == [1]


def test_format_summary_counts_only_included_rows() -> None:
    target = parse_target_name("Jane Doe")
    citation1 = _citation(
        "30",
        2023,
        [_author(1, "Doe", "Jane", "J", "Example Hospital"), _author(2, "Smith", "A", "A", "Example Hospital")],
    )
    citation2 = _citation(
        "31",
        2023,
        [_author(1, "Smith", "A", "A", "Example Hospital"), _author(2, "Doe", "Jane", "J", "Example Hospital")],
    )

    clusters, matches = build_clusters([citation1, citation2], target)
    result = analyze_citations(
        citations=[citation1, citation2],
        matches=matches,
        selected_cluster_ids={clusters[0].cluster_id},
        start_year=2021,
        end_year=2026,
    )

    # Force one exclusion to verify summary behavior.
    assert len(result.rows) == 2
    result.rows[1].include = False
    summary = format_summary(result.rows, 2021, 2026)
    assert "1 total" in summary


def test_citation_matches_excluded_type_detects_preprint_and_editorial() -> None:
    citation = _citation(
        "40",
        2024,
        [_author(1, "Doe", "Jane", "J", "Inst A")],
        publication_types=["Preprint", "Journal Article"],
    )
    matched, matched_types = citation_matches_excluded_type(citation, ["preprint", "editorial"])
    assert matched is True
    assert "preprint" in matched_types


def test_review_senior_is_separate_from_overall_total() -> None:
    target = parse_target_name("Jane Doe")
    citation = _citation(
        "41",
        2024,
        [_author(1, "Smith", "Amy", "A", "Inst"), _author(2, "Doe", "Jane", "J", "Inst A")],
        publication_types=["Review"],
    )
    clusters, matches = build_clusters([citation], target)
    result = analyze_citations(
        citations=[citation],
        matches=matches,
        selected_cluster_ids={clusters[0].cluster_id},
        start_year=2021,
        end_year=2026,
    )
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.counted_overall is False
    assert row.counted_review_senior is True
    summary = format_summary(result.rows, 2021, 2026)
    assert "0 total" in summary
    assert "1 review(s) as Sr author" in summary


def test_format_summary_includes_first_senior_journal_year_subtotals() -> None:
    rows = [
        # Included first/senior rows
        _row_summary("Nat Immunol", 2025, counted_overall=True, counted_first=True, counted_senior=False),
        _row_summary("Nat Immunol", 2026, counted_overall=True, counted_first=False, counted_senior=True),
        _row_summary("JCI Insight", 2022, counted_overall=True, counted_first=True, counted_senior=False),
        _row_summary("JCI Insight", 2025, counted_overall=True, counted_first=False, counted_senior=True),
        # Included but not first/senior
        _row_summary("Cell", 2024, counted_overall=True, counted_first=False, counted_senior=False),
    ]
    summary = format_summary(rows, 2021, 2026)
    assert "5 total, 4 1st/Sr author" in summary
    assert "2 JCI Insight 2022, 2025" in summary
    assert "2 Nat Immunol 2025, 2026" in summary


def test_citation_in_window_all_years_includes_missing_year() -> None:
    citation = _citation("w1", 2024, [])
    citation.print_year = None
    assert citation_in_window(citation, None, None) is True


def test_format_summary_all_years_label() -> None:
    rows = [
        _row_summary("Nat Immunol", 2025, counted_overall=True, counted_first=False, counted_senior=False),
    ]
    summary = format_summary(rows, None, None)
    assert "Peer-reviewed Publications (all years):" in summary


def _row_summary(
    journal: str,
    year: int,
    *,
    counted_overall: bool,
    counted_first: bool,
    counted_senior: bool,
):
    from app.models import Citation, ReportRow

    citation = Citation(
        pmid=f"{journal}-{year}",
        pmcid=None,
        title=f"{journal} {year}",
        journal=journal,
        print_year=year,
        print_date=str(year),
        authors=[],
        source_for_roles="pubmed",
    )
    return ReportRow(
        citation=citation,
        counted_overall=counted_overall,
        counted_first=counted_first,
        counted_senior=counted_senior,
        include=True,
    )
