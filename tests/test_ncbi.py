import httpx

from app.ncbi import _augment_params, _idconv_records, _retry_delay_seconds


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
        tool="pmcparser",
        email="ops@example.org",
    )
    assert params["api_key"] == "abc123"
    assert params["tool"] == "pmcparser"
    assert params["email"] == "ops@example.org"


def test_retry_delay_prefers_retry_after_header() -> None:
    response = httpx.Response(429, headers={"Retry-After": "2"})
    assert _retry_delay_seconds(response, attempt=0) == 2.0


def test_retry_delay_uses_backoff_when_header_missing() -> None:
    response = httpx.Response(429)
    assert _retry_delay_seconds(response, attempt=2) == 2.4
