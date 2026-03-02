from app.logic import analyze_citations, build_clusters, format_summary, parse_pmids, parse_target_name
from app.models import Author, Citation


def _citation(
    pmid: str,
    year: int,
    authors: list[Author],
    *,
    source: str = "pmc",
    co_first: set[int] | None = None,
    co_senior: set[int] | None = None,
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
        co_first_positions=co_first or set(),
        co_senior_positions=co_senior or set(),
    )


def _author(position: int, last: str, fore: str, initials: str, aff: str) -> Author:
    return Author(position=position, last_name=last, fore_name=fore, initials=initials, affiliation=aff)


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


def test_format_summary_counts_only_included_rows() -> None:
    target = parse_target_name("Jane Doe")
    citation1 = _citation(
        "30",
        2023,
        [_author(1, "Doe", "Jane", "J", "Inst A"), _author(2, "Smith", "A", "A", "Inst")],
    )
    citation2 = _citation(
        "31",
        2023,
        [_author(1, "Smith", "A", "A", "Inst"), _author(2, "Doe", "Jane", "J", "Inst A")],
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
    result.rows[1].include = False
    summary = format_summary(result.rows, 2021, 2026)
    assert "1 total" in summary
