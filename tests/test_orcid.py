from app.models import Author, Citation
from app.orcid import OrcidClient, _extract_work_identifiers


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
