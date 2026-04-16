import pytest
from typing import Literal

from app.logic import (
    _affiliation_match_candidates,
    _extract_institution_names,
    _extract_affiliation_record_signals,
    affiliation_fingerprint,
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
    source: Literal["pmc", "pubmed"] = "pmc",
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


def test_full_name_style_matches_compound_surname_suffix_from_metadata() -> None:
    target = parse_target_name("Wu, Hsin-Jung")
    author = _author(1, "Joyce Wu", "Hsin-Jung", "HJ", "Inst")
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


def test_build_clusters_unifies_distinct_and_combined_affiliation_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_institution_names(affiliation: str, *, us_system_context: set[str] | None = None) -> list[str]:
        names: list[str] = []
        if "brigham" in affiliation.lower():
            names.append("Brigham and Women's Hospital")
        if "harvard" in affiliation.lower():
            names.append("Harvard Medical School")
        return names

    monkeypatch.setattr("app.logic._extract_institution_names", fake_extract_institution_names)

    target = parse_target_name("Peter T Sage")
    citations = [
        _citation(
            "301",
            2024,
            [
                Author(
                    position=1,
                    last_name="Sage",
                    fore_name="Peter T",
                    initials="PT",
                    affiliation=(
                        "Transplantation Research Center, Brigham and Women's Hospital; "
                        "Harvard Medical School"
                    ),
                    affiliation_blocks=[
                        "Transplantation Research Center, Brigham and Women's Hospital",
                        "Harvard Medical School",
                    ],
                )
            ],
        ),
        _citation(
            "302",
            2025,
            [
                Author(
                    position=1,
                    last_name="Sage",
                    fore_name="Peter T",
                    initials="PT",
                    affiliation=(
                        "Transplantation Research Center, Brigham and Women's Hospital; "
                        "Harvard Medical School"
                    ),
                )
            ],
        ),
    ]

    clusters, matches = build_clusters(citations, target)

    assert len(matches) == 2
    assert len(clusters) == 1
    assert clusters[0].affiliations == ["Brigham and Women's Hospital", "Harvard Medical School"]


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


def test_build_clusters_merges_subset_affiliation_keys_for_same_name() -> None:
    target = parse_target_name("Peter Sage")
    citations = [
        _citation(
            "901",
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
            "902",
            2025,
            [
                _author(
                    1,
                    "Sage",
                    "Peter T.",
                    "PT",
                    "Department of Medicine, Transplantation Research Center, Division of Renal Medicine, Brigham and Women's Hospital, Boston, MA USA",
                )
            ],
        ),
    ]
    clusters, matches = build_clusters(citations, target)

    assert len(matches) == 2
    assert len(clusters) == 1
    assert clusters[0].mention_count == 2
    assert {match.cluster_id for match in matches} == {clusters[0].cluster_id}


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
                    "Transplant Research Center, Renal Division, Brigham & Women's Hospital, Harvard Medical School, Boston, MA USA",
                )
            ],
        ),
    ]
    clusters, _ = build_clusters(citations, target)

    assert len(clusters) == 1
    assert "Brigham and Women's Hospital" in clusters[0].affiliations
    assert "Harvard University" not in clusters[0].affiliations
    assert len(clusters[0].affiliations) <= 2
    assert not any("&" in label for label in clusters[0].affiliations)


def test_affiliation_fingerprint_collapses_ampersand_and_apostrophe_variants() -> None:
    left = affiliation_fingerprint(
        "Transplantation Research Center, Renal Division, Brigham & Women's Hospital, Harvard Medical School, Boston, MA, USA"
    )
    right = affiliation_fingerprint(
        "Transplantation Research Center, Renal Division, Brigham and Women`s Hospital, Harvard Medical School, Boston, MA USA"
    )
    assert left == right


def test_affiliation_fingerprint_splits_multi_institution_and_clauses() -> None:
    value = affiliation_fingerprint(
        "Transplantation Research Center, Division of Renal Medicine, Brigham and Women’s Hospital and Harvard Medical School, Boston, MA, USA"
    )
    assert value == "brighamandwomenshospital|harvarduniversity"


def test_affiliation_match_candidates_filters_unit_and_geo_fragments() -> None:
    candidates = _affiliation_match_candidates(
        "Transplantation Research Center, Division of Renal Medicine, Brigham and Women's Hospital and Harvard Medical School, Boston, MA, USA"
    )

    assert "Renal Division" not in candidates
    assert "Boston" not in candidates
    assert "MA" not in candidates
    assert "USA" not in candidates
    assert "Transplantation Research Center" not in candidates
    assert "Women's Hospital" not in candidates
    assert "Brigham and Women's Hospital" not in candidates
    assert "Harvard Medical School" not in candidates
    assert "Harvard Medical School, Boston, MA, USA" in candidates


def test_affiliation_match_candidates_does_not_split_on_commas() -> None:
    candidates = _affiliation_match_candidates(
        "Department of Immunology, Yale School of Medicine, New Haven, CT, USA"
    )
    assert candidates == ["Department of Immunology, Yale School of Medicine, New Haven, CT, USA"]


def test_affiliation_fingerprint_handles_unicode_dash_separating_institutions() -> None:
    value = affiliation_fingerprint(
        "Department of Immunology, Blavatnik Institute, Harvard Medical School, Boston, MA;; Evergrande Center for Immunologic Diseases, Harvard Medical School–Brigham and Women's Hospital, Boston, MA;"
    )
    assert value == "brighamandwomenshospital|harvarduniversity"


def test_affiliation_fingerprint_prefers_ror_identifier_over_text() -> None:
    value = affiliation_fingerprint(
        "https://ror.org/04b6nzv94 Department of Immunology, Harvard Medical School, Boston, MA, USA"
    )
    assert value == "brighamandwomenshospital"


def test_affiliation_fingerprint_uses_grid_identifier_for_matching() -> None:
    value = affiliation_fingerprint(
        "grid.62560.37 Department of Medicine, Unknown Institute, Boston, MA, USA"
    )
    assert value == "brighamandwomenshospital"


def test_affiliation_fingerprint_recovers_noisy_grid_suffix_from_concatenated_identifiers() -> None:
    value = affiliation_fingerprint(
        "https://ror.org/04b6nzv94 grid.62560.370000 0004 0378 8294 Department of Medicine, "
        "Transplantation Research Center, Division of Renal Medicine, Brigham and Women's Hospital, Boston, MA USA"
    )
    assert value == "brighamandwomenshospital"


def test_extract_affiliation_record_signals_normalizes_spaced_ror_url() -> None:
    ror_ids, grid_ids, emails = _extract_affiliation_record_signals(
        "https://ror.org/03 vek6s52grid.38142. Transplantation Research Center, Brigham and Women's Hospital"
    )
    assert ror_ids == ["https://ror.org/03vek6s52"]
    assert grid_ids == []
    assert emails == []


def test_cluster_affiliation_labels_keep_uc_davis_from_split_university_clause() -> None:
    target = parse_target_name("Kevin O'Connor")
    citations = [
        _citation(
            "33788616",
            2021,
            [
                _author(
                    1,
                    "O'Connor",
                    "Kevin",
                    "K",
                    "Department of Neurobiology, Physiology and Behavior, University of California, Davis, CA 95616, USA",
                )
            ],
        )
    ]
    clusters, _ = build_clusters(citations, target)

    assert len(clusters) == 1
    assert "University of California, Davis" in clusters[0].affiliations


def test_cluster_affiliation_labels_ignores_dangling_and_fragment() -> None:
    target = parse_target_name("Peter Sage")
    citations = [
        _citation(
            "888",
            2025,
            [
                _author(
                    1,
                    "Sage",
                    "Peter",
                    "P",
                    "Transplantation Research Center, Division of Renal Medicine, Department of Medicine; and",
                )
            ],
        )
    ]
    clusters, _ = build_clusters(citations, target)

    assert len(clusters) == 1
    assert "and" not in [value.lower() for value in clusters[0].affiliations]


def test_cluster_affiliation_labels_infer_uc_davis_from_city_plus_system() -> None:
    target = parse_target_name("Kevin O'Connor")
    citations = [
        _citation(
            "39963949",
            2025,
            [
                _author(
                    1,
                    "O'Connor",
                    "Kevin",
                    "K",
                    "Davis, California 95618 | Department of Neurobiology, Physiology and Behavior | University of California System",
                )
            ],
        )
    ]
    clusters, _ = build_clusters(citations, target)

    assert len(clusters) == 1
    assert "University of California, Davis" in clusters[0].affiliations


def test_extract_institution_names_does_not_infer_uc_from_city_only() -> None:
    assert _extract_institution_names(
        "Department of Gastroenterology, Rady Children's Hospital San Diego, San Diego, CA, USA."
    ) == []
    assert _extract_institution_names("Sharp Memorial Hospital, San Diego, California, USA.") == []
    assert _extract_institution_names("Biohub, San Francisco, San Francisco, CA, USA.") == []


def test_extract_institution_names_infers_ucsb_when_uc_system_is_explicit() -> None:
    assert _extract_institution_names(
        "Neuroscience Research Institute, University of California, Santa Barbara, CA 93106."
    ) == ["University of California, Santa Barbara"]


def test_extract_institution_names_infers_other_us_university_system_campuses() -> None:
    tx_names = _extract_institution_names(
        "Department of Biology, University of Texas at Austin, Austin, TX, USA."
    )
    assert tx_names and "University of Texas" in tx_names[0] and "Austin" in tx_names[0]

    nc_names = _extract_institution_names(
        "Department of Medicine, University of North Carolina at Chapel Hill, Chapel Hill, NC, USA."
    )
    assert nc_names and "University of North Carolina" in nc_names[0] and "Chapel Hill" in nc_names[0]

    co_names = _extract_institution_names(
        "Department of Immunology, University of Colorado Anschutz Medical Campus, Aurora, CO, USA."
    )
    assert co_names and "University of Colorado" in co_names[0] and "Anschutz" in co_names[0]


def test_extract_institution_names_infers_university_of_state_fallback() -> None:
    mi_names = _extract_institution_names(
        "Animal Phenotyping Core, University of Michigan, Ann Arbor, MI 48109, USA."
    )
    assert mi_names == ["University of Michigan"]

    tn_names = _extract_institution_names(
        "Advanced Microscopy and Imaging Center, College of Arts and Sciences, University of Tennessee, Knoxville, Tennessee, USA."
    )
    assert tn_names == ["University of Tennessee"]

    la_names = _extract_institution_names(
        "Department of Biology, University of Louisiana at Lafayette, Lafayette, Louisiana, USA."
    )
    assert la_names == ["University of Louisiana, Lafayette"]

    al_names = _extract_institution_names(
        "Alabama Life Research Institute, The University of Alabama, Tuscaloosa, Alabama, USA."
    )
    assert al_names == ["University of Alabama"]

    md_med_names = _extract_institution_names(
        "Center for Advanced Microbiome Research and Innovation, Institute for Genome Sciences, University of Maryland School of Medicine, Baltimore, MD, USA."
    )
    assert md_med_names == ["University of Maryland School of Medicine"]

    ms_med_center_names = _extract_institution_names(
        "Center for Immunology and Microbial Research, University of Mississippi Medical Center, Jackson, MS, USA."
    )
    assert ms_med_center_names == ["University of Mississippi Medical Center"]

    mn_med_school_names = _extract_institution_names(
        "Center for Immunology, Department of Laboratory Medicine and Pathology, University of Minnesota Medical School, Minneapolis, MN 55455, USA."
    )
    assert mn_med_school_names == ["University of Minnesota Medical School"]

    va_school_of_med_names = _extract_institution_names(
        "Center for Membrane and Cell Physiology, Department of Molecular Physiology and Biological Physics, University of Virginia School of Medicine, Charlottesville, VA, USA."
    )
    assert va_school_of_med_names == ["University of Virginia School of Medicine"]

    ne_lincoln_names = _extract_institution_names(
        "Department of Animal Science, University of Nebraska-Lincoln, 3940 Fair St, Lincoln, NE 68503, United States."
    )
    assert ne_lincoln_names == ["University of Nebraska, Lincoln"]

    mo_umkc_names = _extract_institution_names(
        "Department of Allergy and Immunology, Children's Mercy Kansas City, University of Missouri-Kansas City, Kansas City, MO, USA."
    )
    assert mo_umkc_names == ["University of Missouri, Kansas City"]

    assert _extract_institution_names("University of North Carolina Project Malawi, Lilongwe, Malawi.") == []


def test_extract_institution_names_literal_fallback_captures_selected_institutions() -> None:
    assert _extract_institution_names("Florida Atlantic University, Boca Raton, FL, USA.") == ["Florida Atlantic University"]
    assert _extract_institution_names("Alabama State University, Montgomery, AL, USA.") == ["Alabama State University"]
    assert _extract_institution_names(
        "Albany College of Pharmacy and Health Sciences, Albany, NY, USA."
    ) == ["Albany College of Pharmacy and Health Sciences"]
    assert _extract_institution_names("Albany Medical Center, Albany, NY, USA.") == ["Albany Medical Center"]
    assert _extract_institution_names(
        "Arkansas Children's Research Institute, Little Rock, AR, USA."
    ) == ["Arkansas Children's Research Institute"]
    assert _extract_institution_names(
        "Augusta University School of Medicine, Augusta, GA, USA."
    ) == ["Augusta University School of Medicine"]
    assert _extract_institution_names(
        "Center for Global Infectious Disease Research, Seattle Children's Research Institute , Seattle, Washington, USA."
    ) == ["Seattle Children's Research Institute"]
    assert _extract_institution_names("Department of Biology, Texas A&M University, College Station, TX, USA.") == [
        "Texas A&M"
    ]
    assert _extract_institution_names("Department of Biology, Texas A and M University, College Station, TX, USA.") == [
        "Texas A&M"
    ]
    assert _extract_institution_names(
        "Department of Biostatistics, Harvard T.H. Chan School of Public Health, Boston, MA, USA."
    ) == ["Harvard T.H. Chan School of Public Health"]
    assert _extract_institution_names(
        "Department of Epidemiology, School of Public Health, Boston, MA, USA."
    ) == []
    assert _extract_institution_names("Harvey Mudd College, Claremont, CA, USA.") == ["Harvey Mudd College"]
    assert _extract_institution_names("University of Houston, Houston, TX, USA.") == ["University of Houston"]

    # Canonical normalizer matches still take precedence over literal fallback.
    assert _extract_institution_names("Arizona State University, Tempe, AZ, USA.") == ["Arizona State University-Tempe"]

    # Keep precision-first behavior for project site strings.
    assert _extract_institution_names("University of North Carolina Project Malawi, Lilongwe, Malawi.") == []


def test_extract_institution_names_rejects_multi_assignment_narrative_blocks() -> None:
    narrative = (
        "Adam M. Whalen, Alexander Furuya, Jessica Contreras, Asa Radix, and Dustin T. Duncan are with the "
        "Department of Epidemiology, Mailman School of Public Health, Columbia University, New York, NY. "
        "Asa Radix is also with the Callen-Lorde Community Health Center, New York, NY. "
        "John A. Schneider is with the Department of Public Health Sciences, University of Chicago, Chicago, IL. "
        "Sahnah Lim and Chau Trinh-Shevrin are with the Department of Population Health, Grossman School of "
        "Medicine, New York University, New York, NY, and the Callen-Lorde Community Health Center, New York, NY."
    )
    assert _extract_institution_names(narrative) == []


def test_extract_institution_names_allows_single_assignment_narrative_line() -> None:
    single_assignment = (
        "John A. Schneider is with the Department of Public Health Sciences, University of Chicago, Chicago, IL."
    )
    assert _extract_institution_names(single_assignment) == ["University of Chicago"]


def test_cluster_affiliation_labels_keep_company_name_without_city_state() -> None:
    target = parse_target_name("Kevin O'Connor")
    citations = [
        _citation(
            "401",
            2024,
            [
                _author(
                    1,
                    "O'Connor",
                    "Kevin",
                    "K",
                    "Bioplastech Ltd., Dublin, Ireland.",
                )
            ],
        ),
        _citation(
            "402",
            2025,
            [
                _author(
                    1,
                    "O'Connor",
                    "Kevin C.",
                    "KC",
                    "Department of Neurology Yale University School of Medicine New Haven Connecticut 06520 USA",
                )
            ],
        ),
    ]
    clusters, _ = build_clusters(citations, target)

    labels = " | ".join(value for cluster in clusters for value in cluster.affiliations)
    assert "Bioplastech Ltd, Ireland" in labels
    assert "Dublin" not in labels
    assert "Yale University" in labels


def test_cluster_affiliation_labels_ignore_location_only_and_contribution_note_text() -> None:
    target = parse_target_name("Peter Sage")
    citations = [
        _citation(
            "501",
            2025,
            [
                _author(
                    1,
                    "Sage",
                    "Peter",
                    "P",
                    "These authors contributed equally",
                )
            ],
        ),
        _citation(
            "502",
            2025,
            [
                _author(
                    1,
                    "Sage",
                    "Peter",
                    "P",
                    "Boston, MA, USA",
                )
            ],
        ),
        _citation(
            "503",
            2025,
            [
                _author(
                    1,
                    "Sage",
                    "Peter",
                    "P",
                    "Transplantation Research Center, Division of Renal Medicine, Department of Medicine",
                )
            ],
        ),
    ]
    clusters, _ = build_clusters(citations, target)

    assert len(clusters) == 3
    assert all(cluster.affiliations == ["(no affiliation listed)"] for cluster in clusters)


def test_extract_institution_names_ignores_contribution_note_but_keeps_institution() -> None:
    names = _extract_institution_names(
        "These authors contributed equally. Brigham and Women's Hospital, Boston, MA, USA."
    )
    assert names == ["Brigham and Women's Hospital"]


def test_extract_institution_names_appends_non_us_trailing_country() -> None:
    india = _extract_institution_names(
        "12. Department of Pulmonary Medicine, Christian Medical College, Vellore, Tamil Nadu, India."
    )
    assert india == ["Christian Medical College, India"]

    greece = _extract_institution_names(
        "2nd Department of Internal Medicine, HIV Unit, Medical School, Hippokration General Hospital, "
        "National and Kapodistrian University of Athens, Athens, Greece."
    )
    assert greece == ["University of Athens, Greece"]

    south_africa = _extract_institution_names(
        "6. DSI-NRF Centre of Excellence for Biomedical Tuberculosis Research, South African Medical Research Council "
        "Centre for Tuberculosis Research, Division of Molecular Biology and Human Genetics, Faculty of Medicine and "
        "Health Sciences, Stellenbosch University, South Africa."
    )
    assert south_africa == ["Stellenbosch University, South Africa"]


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


def test_preprint_is_separate_from_overall_total() -> None:
    target = parse_target_name("Jane Doe")
    citation = _citation(
        "42",
        2024,
        [_author(1, "Doe", "Jane", "J", "Inst A"), _author(2, "Smith", "Amy", "A", "Inst")],
        publication_types=["Preprint"],
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
    assert row.include is True
    assert row.is_preprint is True
    assert row.counted_overall is False
    summary = format_summary(result.rows, 2021, 2026)
    assert "0 total" in summary
    assert "1 preprint(s)" in summary


def test_superseded_preprint_is_skipped_when_peer_reviewed_version_is_present() -> None:
    target = parse_target_name("Jane Doe")
    preprint = _citation(
        "37066336",
        2023,
        [_author(1, "Doe", "Jane", "J", "Inst A"), _author(2, "Smith", "Amy", "A", "Inst")],
        publication_types=["Preprint"],
    )
    preprint.update_in_pmids = {"38821936"}
    published = _citation(
        "38821936",
        2024,
        [_author(1, "Doe", "Jane", "J", "Inst A"), _author(2, "Smith", "Amy", "A", "Inst")],
        publication_types=["Journal Article"],
    )
    published.update_of_pmids = {"37066336"}
    clusters, matches = build_clusters([preprint, published], target)
    result = analyze_citations(
        citations=[preprint, published],
        matches=matches,
        selected_cluster_ids={cluster.cluster_id for cluster in clusters},
        start_year=2021,
        end_year=2026,
    )
    row_by_pmid = {row.citation.pmid: row for row in result.rows}
    preprint_row = row_by_pmid["37066336"]
    published_row = row_by_pmid["38821936"]

    assert preprint_row.is_preprint is True
    assert preprint_row.include is False
    assert preprint_row.preprint_superseded_by == ["38821936"]
    assert published_row.include is True
    assert published_row.counted_overall is True
    summary = format_summary(result.rows, 2021, 2026)
    assert "1 total" in summary
    assert "0 preprint(s)" in summary
    assert "Excluded as superseded by a peer-reviewed version: 1 preprint(s)." in summary


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


def test_format_summary_uses_nx_year_for_duplicate_journal_years() -> None:
    rows = [
        _row_summary("Sci Immunol", 2023, counted_overall=True, counted_first=True, counted_senior=False),
        _row_summary("Sci Immunol", 2023, counted_overall=True, counted_first=False, counted_senior=True),
        _row_summary("Sci Immunol", 2024, counted_overall=True, counted_first=True, counted_senior=False),
        _row_summary("Sci Immunol", 2025, counted_overall=True, counted_first=False, counted_senior=True),
        _row_summary("Sci Immunol", 2025, counted_overall=True, counted_first=True, counted_senior=False),
    ]

    summary = format_summary(rows, 2021, 2026)
    assert "5 Sci Immunol 2x2023, 2024, 2x2025" in summary


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
