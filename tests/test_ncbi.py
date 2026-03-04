import httpx

import app.ncbi as ncbi_module
from app.ncbi import (
    LITSENSE_AUTHOR_URL,
    NcbiClient,
    _augment_params,
    _build_litsense_author_query,
    _extract_litsense_pmids,
    _extract_pmc_article_doi,
    _idconv_records,
    _retry_delay_seconds,
)


def test_idconv_records_old_format() -> None:
    payload = {"records": [{"pmid": "1", "pmcid": "PMC1"}]}
    records = _idconv_records(payload)
    assert records == [{"pmid": "1", "pmcid": "PMC1"}]


def test_idconv_records_new_format() -> None:
    payload = {"articles": [{"pmid": "2", "pmcid": "PMC2"}]}
    records = _idconv_records(payload)
    assert records == [{"pmid": "2", "pmcid": "PMC2"}]


def test_idconv_records_nested_response_format() -> None:
    payload = {"response": {"records": [{"pmid": "3", "pmcid": "PMC3"}]}}
    records = _idconv_records(payload)
    assert records == [{"pmid": "3", "pmcid": "PMC3"}]


def test_augment_params_adds_api_key_and_eutils_fields() -> None:
    params = _augment_params(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed"},
        api_key="abc123",
        tool="ECHIDNA",
        email="ops@example.org",
    )
    assert params["api_key"] == "abc123"
    assert params["tool"] == "ECHIDNA"
    assert params["email"] == "ops@example.org"


def test_retry_delay_prefers_retry_after_header() -> None:
    response = httpx.Response(429, headers={"Retry-After": "2"})
    assert _retry_delay_seconds(response, attempt=0) == 2.0


def test_retry_delay_uses_backoff_when_header_missing() -> None:
    response = httpx.Response(429)
    assert _retry_delay_seconds(response, attempt=2) == 2.4




def test_parse_pubmed_author_orcid_from_identifier() -> None:
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>555</PMID>
          <Article>
            <ArticleTitle>ORCID test.</ArticleTitle>
            <Journal>
              <ISOAbbreviation>J Test</ISOAbbreviation>
              <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
            </Journal>
            <AuthorList>
              <Author>
                <LastName>Doe</LastName>
                <ForeName>Jane</ForeName>
                <Initials>J</Initials>
                <Identifier Source="ORCID">https://orcid.org/0000-0002-1825-0097</Identifier>
              </Author>
            </AuthorList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    client = NcbiClient()
    citations = client._parse_pubmed(xml)

    assert len(citations) == 1
    assert len(citations[0].authors) == 1
    assert citations[0].authors[0].orcid == "0000-0002-1825-0097"

def test_parse_pubmed_title_with_embedded_tags() -> None:
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>39164478</PMID>
          <Article>
            <ArticleTitle>T<sub>reg</sub> responses in disease.</ArticleTitle>
            <Journal>
              <ISOAbbreviation>J Test</ISOAbbreviation>
              <JournalIssue>
                <PubDate>
                  <Year>2024</Year>
                </PubDate>
              </JournalIssue>
            </Journal>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    client = NcbiClient()
    citations = client._parse_pubmed(xml)

    assert len(citations) == 1
    assert citations[0].title == "Treg responses in disease."


def test_parse_pubmed_extracts_doi_from_article_id() -> None:
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>999</PMID>
          <Article>
            <ArticleTitle>DOI Test</ArticleTitle>
            <Journal>
              <ISOAbbreviation>J Test</ISOAbbreviation>
              <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
            </Journal>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="doi">https://doi.org/10.1000/test-doi.1</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    client = NcbiClient()
    citations = client._parse_pubmed(xml)

    assert len(citations) == 1
    assert citations[0].doi == "10.1000/test-doi.1"


def test_extract_pmc_article_doi_from_article_meta() -> None:
    xml = """
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="doi">doi:10.2000/example.2</article-id>
        </article-meta>
      </front>
    </article>
    """

    assert _extract_pmc_article_doi(xml) == "10.2000/example.2"


def test_extract_litsense_pmids_from_nested_payload() -> None:
    payload = {
        "count": 2,
        "results": [
            {"pmid": 12345, "score": 0.9},
            {"entry": {"pmids": ["67890", "111213"]}},
        ],
        "metadata": {"ignored": "value"},
    }

    pmids = _extract_litsense_pmids(payload)

    assert pmids == {"12345", "67890", "111213"}


def test_build_litsense_author_query_uses_last_name_and_first_initial() -> None:
    query = _build_litsense_author_query(seed_pmid="39164478", author_name="Jeffrey Stephen Rice")

    assert query == "39164478 Rice J"


def test_fetch_litsense_pmids_by_query_uses_percent_encoded_spaces(monkeypatch) -> None:
    captured: dict[str, object] = {}
    client = NcbiClient()

    def fake_fetch(*, max_pages: int, params: dict[str, str] | None = None, initial_url: str | None = None) -> set[str]:
        captured["max_pages"] = max_pages
        captured["params"] = params
        captured["initial_url"] = initial_url
        return {"1"}

    monkeypatch.setattr(client, "_fetch_litsense_pmids", fake_fetch)

    result = client.fetch_litsense_pmids_by_query(seed_pmid="39164478", author_name="Jeffrey Rice", max_pages=3)

    assert result == {"1"}
    assert captured["max_pages"] == 3
    assert captured["params"] is None
    assert captured["initial_url"] == f"{LITSENSE_AUTHOR_URL}?query=39164478%20Rice%20J"


def test_fetch_litsense_pmids_by_orcid_retries_after_429(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
            self.calls += 1
            request = httpx.Request("GET", url, params=params)
            if self.calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, request=request)
            return httpx.Response(200, json={"results": [{"pmid": "12345"}]}, request=request)

    fake_http_client = FakeClient()
    sleep_calls: list[float] = []

    monkeypatch.setattr(ncbi_module.httpx, "Client", lambda *args, **kwargs: fake_http_client)
    monkeypatch.setattr(ncbi_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    client = NcbiClient()
    monkeypatch.setattr(client, "_throttle", lambda: None)

    pmids = client.fetch_litsense_pmids_by_orcid(target_orcid="0000-0002-1825-0097", max_pages=1)

    assert pmids == {"12345"}
    assert fake_http_client.calls == 2
    assert sleep_calls == [0.5]
