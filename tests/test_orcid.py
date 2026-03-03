from app.models import Author, Citation
from app.orcid import OrcidClient, OrcidOrganization, _extract_affiliation_organizations, _extract_work_identifiers


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
