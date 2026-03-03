import httpx

from app.ncbi import (
    NcbiClient,
    _augment_params,
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
