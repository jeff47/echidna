from app.models import Author, Citation
from app.orcid import (
    OrcidClient,
    OrcidOrganization,
    _build_profile_search_queries,
    _extract_affiliation_organizations,
    _extract_basic_search_matches,
    _extract_expanded_search_matches,
    _extract_work_identifiers,
)


def _citation(*, pmid: str, pmcid: str | None, doi: str | None) -> Citation:
    return Citation(
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        title=f"Title {pmid}",
        journal="J Test",
        print_year=2025,
        print_date="2025",
        authors=[Author(position=1, last_name="Rice", fore_name="Jeffrey", initials="J", affiliation="Inst")],
        publication_types=["Journal Article"],
        source_for_roles="pubmed",
    )


def test_extract_work_identifiers_collects_doi_pmid_pmcid() -> None:
    payload = {
        "group": [
            {
                "external-ids": {
                    "external-id": [
                        {"external-id-type": "doi", "external-id-value": "https://doi.org/10.1000/ABC.1"},
                        {"external-id-type": "pmid", "external-id-value": "12345"},
                        {"external-id-type": "pmcid", "external-id-value": "PMC54321"},
                    ]
                }
            }
        ]
    }

    identifiers = _extract_work_identifiers(payload)

    assert "10.1000/abc.1" in identifiers["doi"]
    assert "12345" in identifiers["pmid"]
    assert "PMC54321" in identifiers["pmcid"]


def test_match_citations_matches_by_any_identifier() -> None:
    class FakeOrcidClient(OrcidClient):
        def fetch_work_identifiers(self, target_orcid: str) -> dict[str, set[str]]:
            _ = target_orcid
            return {
                "doi": {"10.1000/a.1"},
                "pmid": {"222"},
                "pmcid": {"PMC333"},
            }

    client = FakeOrcidClient(client_id="id", client_secret="secret")

    citations = [
        _citation(pmid="111", pmcid=None, doi="10.1000/A.1"),
        _citation(pmid="222", pmcid=None, doi=None),
        _citation(pmid="444", pmcid="333", doi=None),
        _citation(pmid="555", pmcid=None, doi=None),
    ]

    matches = client.match_citations("0000-0001-2345-6789", citations)

    assert matches["111"] == ["doi"]
    assert matches["222"] == ["pmid"]
    assert matches["444"] == ["pmcid"]
    assert "555" not in matches


def test_extract_affiliation_organizations_reads_activity_summaries() -> None:
    payload = {
        "activities-summary": {
            "employments": {
                "affiliation-group": [
                    {
                        "summaries": [
                            {
                                "employment-summary": {
                                    "organization": {
                                        "name": "Brigham & Women's Hospital",
                                        "address": {"city": "Boston", "country": "US"},
                                        "disambiguated-organization": {
                                            "disambiguated-organization-identifier": "https://ror.org/03zjqec80",
                                            "disambiguation-source": "ROR",
                                        },
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }

    organizations = _extract_affiliation_organizations(payload)

    assert len(organizations) == 1
    assert organizations[0].name == "Brigham & Women's Hospital"
    assert organizations[0].country == "US"
    assert organizations[0].identifier == "03zjqec80"


def test_match_affiliations_supports_likely_contains_and_possible_levels() -> None:
    class FakeOrcidClient(OrcidClient):
        def fetch_affiliation_organizations(self, target_orcid: str) -> list[OrcidOrganization]:
            _ = target_orcid
            return _extract_affiliation_organizations(
                {
                    "activities-summary": {
                        "employments": {
                            "affiliation-group": [
                                {
                                    "summaries": [
                                        {
                                            "employment-summary": {
                                                "organization": {
                                                    "name": "Brigham and Women's Hospital",
                                                    "address": {"country": "US"},
                                                    "disambiguated-organization": {
                                                        "disambiguated-organization-identifier": "https://ror.org/03zjqec80"
                                                    },
                                                }
                                            }
                                        },
                                        {
                                            "employment-summary": {
                                                "organization": {
                                                    "name": "Harvard Medical School",
                                                    "address": {"country": "US"},
                                                }
                                            }
                                        },
                                    ]
                                }
                            ]
                        }
                    }
                }
            )

    client = FakeOrcidClient(client_id="id", client_secret="secret")
    matches = client.match_affiliations(
        "0000-0001-2345-6789",
        {
            "100": ["Renal Division, Brigham & Women's Hospital, Boston, MA, USA"],
            "101": ["Harvard Med School, Boston, MA"],
            "102": ["Some Other Institute"],
            "103": ["Mass General Brigham/Brigham and Women's Hospital, Boston, MA, USA"],
        },
    )

    assert matches["100"]["level"] == "likely"
    assert matches["100"]["institution"] == "Brigham and Women's Hospital"
    assert "institution match" in matches["100"]["label"]
    assert matches["101"]["level"] == "possible"
    assert matches["101"]["institution"] == "Harvard Medical School"
    assert "name similarity" in matches["101"]["label"]
    assert matches["103"]["level"] == "contains"
    assert matches["103"]["institution"] == "Brigham and Women's Hospital"
    assert "affiliation contains institution" in matches["103"]["label"]
    assert "102" not in matches


def test_extract_expanded_search_matches_parses_profiles() -> None:
    payload = {
        "expanded-result": [
            {
                "orcid-id": "https://orcid.org/0000-0002-1825-0097",
                "given-names": "Jeffrey",
                "family-names": "Rice",
                "institution-name": "NIH",
                "country": "US",
            },
            {
                "orcid-id": "0000-0002-1825-0097",  # duplicate should dedupe
                "given-names": "Jeff",
                "family-names": "Rice",
            },
        ]
    }

    matches = _extract_expanded_search_matches(payload)

    assert len(matches) == 1
    assert matches[0]["orcid"] == "0000-0002-1825-0097"
    assert matches[0]["display_name"] == "Jeffrey Rice"
    assert matches[0]["institution"] == "NIH"
    assert matches[0]["country"] == "US"


def test_extract_expanded_search_matches_handles_list_institutions() -> None:
    payload = {
        "expanded-result": [
            {
                "orcid-id": "0000-0002-1825-0097",
                "given-names": "Peter",
                "family-names": "Sage",
                "institution-name": ["Brigham and Women's Hospital", "Harvard Medical School"],
            }
        ]
    }

    matches = _extract_expanded_search_matches(payload)

    assert len(matches) == 1
    assert matches[0]["institution"] == "Brigham and Women's Hospital; Harvard Medical School"


def test_extract_basic_search_matches_parses_orcid_identifier() -> None:
    payload = {
        "result": [
            {"orcid-identifier": {"path": "0000-0002-1825-0097"}},
            {"orcid-identifier": {"uri": "https://orcid.org/0000-0002-1825-0098"}},
        ]
    }

    matches = _extract_basic_search_matches(payload)

    assert [row["orcid"] for row in matches] == ["0000-0002-1825-0097", "0000-0002-1825-0098"]


def test_search_profiles_falls_back_to_basic_search_when_expanded_empty() -> None:
    class FakeOrcidClient(OrcidClient):
        def _search_payload(self, *, path: str, query: str, rows: int, headers: dict[str, str]) -> dict[str, object]:
            _ = query
            _ = rows
            _ = headers
            if path == "/expanded-search":
                return {"expanded-result": []}
            return {"result": [{"orcid-identifier": {"path": "0000-0002-1825-0097"}}]}

    client = FakeOrcidClient(client_id="", client_secret="")
    matches = client.search_profiles("Jeffrey Rice", limit=5)

    assert len(matches) == 1
    assert matches[0]["orcid"] == "0000-0002-1825-0097"


def test_build_profile_search_queries_prioritizes_fielded_name_queries() -> None:
    queries = _build_profile_search_queries("Peter Sage")

    assert queries
    assert queries[0] == 'given-and-family-names:"peter sage"'
    assert "family-name:sage AND given-names:peter*" in queries
    assert queries[-1] == "Peter Sage"


def test_build_profile_search_queries_supports_initials_format() -> None:
    queries = _build_profile_search_queries("Rice JS")

    assert "family-name:rice AND given-names:j*" in queries
    assert "family-name:rice" in queries


def test_search_profiles_uses_multiple_queries_and_dedupes() -> None:
    called: list[tuple[str, str]] = []

    class FakeOrcidClient(OrcidClient):
        def _search_payload(self, *, path: str, query: str, rows: int, headers: dict[str, str]) -> dict[str, object]:
            _ = rows
            _ = headers
            called.append((path, query))
            if path == "/expanded-search":
                if query == 'given-and-family-names:"peter sage"':
                    return {"expanded-result": [{"orcid-id": "0000-0002-7287-0326", "given-names": "Peter", "family-names": "Sage"}]}
                if query == "family-name:sage AND given-names:peter*":
                    return {"expanded-result": [{"orcid-id": "0000-0002-7287-0326", "given-names": "Peter", "family-names": "Sage"}]}
                return {"expanded-result": []}
            return {"result": []}

    client = FakeOrcidClient(client_id="", client_secret="")
    matches = client.search_profiles("Peter Sage", limit=10)

    assert [match["orcid"] for match in matches] == ["0000-0002-7287-0326"]
    assert called
    assert called[0] == ("/expanded-search", 'given-and-family-names:"peter sage"')
