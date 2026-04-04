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
    _parse_pmc_xml_by_pmcid,
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


def test_parse_pubmed_extracts_update_relationship_pmids() -> None:
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>37066336</PMID>
          <Article>
            <ArticleTitle>Preprint test</ArticleTitle>
            <Journal>
              <ISOAbbreviation>bioRxiv</ISOAbbreviation>
              <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
            </Journal>
          </Article>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="UpdateIn">
              <PMID Version="1">38821936</PMID>
            </CommentsCorrections>
            <CommentsCorrections RefType="ErratumIn">
              <PMID Version="1">39999999</PMID>
            </CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation>
      </PubmedArticle>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>38821936</PMID>
          <Article>
            <ArticleTitle>Published test</ArticleTitle>
            <Journal>
              <ISOAbbreviation>Nat Commun</ISOAbbreviation>
              <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
            </Journal>
          </Article>
          <CommentsCorrectionsList>
            <CommentsCorrections RefType="UpdateOf">
              <PMID Version="1">37066336</PMID>
            </CommentsCorrections>
          </CommentsCorrectionsList>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    client = NcbiClient()
    citations = client._parse_pubmed(xml)
    citation_by_pmid = {citation.pmid: citation for citation in citations}

    assert citation_by_pmid["37066336"].update_in_pmids == {"38821936"}
    assert citation_by_pmid["37066336"].update_of_pmids == set()
    assert citation_by_pmid["38821936"].update_of_pmids == {"37066336"}
    assert citation_by_pmid["38821936"].update_in_pmids == set()


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


def test_parse_pmc_xml_by_pmcid_splits_batched_article_xml() -> None:
    xml = """
    <pmc-articleset>
      <article>
        <front>
          <article-meta>
            <article-id pub-id-type="pmc">111</article-id>
          </article-meta>
        </front>
      </article>
      <article>
        <front>
          <article-meta>
            <article-id pub-id-type="pmcid">PMC222</article-id>
          </article-meta>
        </front>
      </article>
    </pmc-articleset>
    """

    parsed = _parse_pmc_xml_by_pmcid(xml, requested_pmcids=["PMC111", "PMC222"])

    assert sorted(parsed) == ["PMC111", "PMC222"]
    assert "<article-id pub-id-type=\"pmc\">111</article-id>" in parsed["PMC111"]
    assert "<article-id pub-id-type=\"pmcid\">PMC222</article-id>" in parsed["PMC222"]


def test_fetch_citations_batches_pmc_fetches(monkeypatch) -> None:
    pubmed_xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>100</PMID>
          <Article>
            <ArticleTitle>First title</ArticleTitle>
            <Journal>
              <ISOAbbreviation>J Test</ISOAbbreviation>
              <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
            </Journal>
            <AuthorList>
              <Author>
                <LastName>Rice</LastName>
                <ForeName>Jeffrey</ForeName>
                <Initials>J</Initials>
              </Author>
            </AuthorList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>200</PMID>
          <Article>
            <ArticleTitle>Second title</ArticleTitle>
            <Journal>
              <ISOAbbreviation>J Test</ISOAbbreviation>
              <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
            </Journal>
            <AuthorList>
              <Author>
                <LastName>Rice</LastName>
                <ForeName>Jeffrey</ForeName>
                <Initials>J</Initials>
              </Author>
            </AuthorList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    pmc_xml = """
    <pmc-articleset>
      <article>
        <front>
          <article-meta>
            <article-id pub-id-type="pmc">111</article-id>
            <contrib-group>
              <contrib contrib-type="author">
                <name>
                  <surname>Rice</surname>
                  <given-names>Jeffrey</given-names>
                </name>
              </contrib>
            </contrib-group>
          </article-meta>
        </front>
      </article>
      <article>
        <front>
          <article-meta>
            <article-id pub-id-type="pmc">222</article-id>
            <contrib-group>
              <contrib contrib-type="author">
                <name>
                  <surname>Rice</surname>
                  <given-names>Jeffrey</given-names>
                </name>
              </contrib>
            </contrib-group>
          </article-meta>
        </front>
      </article>
    </pmc-articleset>
    """
    requested_params: list[dict[str, str]] = []

    def fake_request(url: str, params: dict[str, str]) -> httpx.Response:
        request = httpx.Request("GET", url, params=params)
        requested_params.append(params)
        if params.get("db") == "pubmed":
            return httpx.Response(200, text=pubmed_xml, request=request)
        if params.get("db") == "pmc":
            return httpx.Response(200, text=pmc_xml, request=request)
        return httpx.Response(
            200,
            json={"records": [{"pmid": "100", "pmcid": "PMC111"}, {"pmid": "200", "pmcid": "PMC222"}]},
            request=request,
        )

    client = NcbiClient()
    monkeypatch.setattr(client, "_request", fake_request)

    citations = client.fetch_citations(["100", "200"])

    assert [citation.pmcid for citation in citations] == ["PMC111", "PMC222"]
    assert [citation.source_for_roles for citation in citations] == ["pmc", "pmc"]
    assert [params["id"] for params in requested_params if params.get("db") == "pmc"] == ["111,222"]


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
