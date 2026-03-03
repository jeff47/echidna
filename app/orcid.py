from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from app.models import Citation

DOI_URL_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", flags=re.IGNORECASE)
NON_DIGIT = re.compile(r"\D+")
NON_ALNUM = re.compile(r"[^A-Z0-9]+")


class OrcidClient:
    def __init__(
        self,
        timeout: float = 20.0,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_url: str | None = None,
        api_base_url: str | None = None,
    ) -> None:
        self.timeout = timeout
        configured_client_id = client_id if client_id is not None else os.getenv("ORCID_CLIENT_ID", "")
        configured_client_secret = client_secret if client_secret is not None else os.getenv("ORCID_CLIENT_SECRET", "")
        configured_token_url = token_url if token_url is not None else os.getenv("ORCID_TOKEN_URL", "https://orcid.org/oauth/token")
        configured_api_base_url = api_base_url if api_base_url is not None else os.getenv("ORCID_API_BASE_URL", "https://pub.orcid.org/v3.0")

        self.client_id = configured_client_id.strip()
        self.client_secret = configured_client_secret.strip()
        self.token_url = configured_token_url.strip()
        self.api_base_url = configured_api_base_url.strip().rstrip("/")
        self._cached_token: str = ""
        self._token_expires_at: float = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def match_citations(self, target_orcid: str, citations: list[Citation]) -> dict[str, list[str]]:
        if not target_orcid.strip():
            return {}
        identifier_index = self.fetch_work_identifiers(target_orcid)
        matches_by_pmid: dict[str, list[str]] = {}
        for citation in citations:
            match_types: list[str] = []
            if citation.doi and _normalize_doi(citation.doi) in identifier_index["doi"]:
                match_types.append("doi")
            if citation.pmid and _normalize_pmid(citation.pmid) in identifier_index["pmid"]:
                match_types.append("pmid")
            if citation.pmcid and _normalize_pmcid(citation.pmcid) in identifier_index["pmcid"]:
                match_types.append("pmcid")
            if match_types:
                matches_by_pmid[citation.pmid] = sorted(set(match_types))
        return matches_by_pmid

    def fetch_work_identifiers(self, target_orcid: str) -> dict[str, set[str]]:
        if not self.is_configured:
            raise RuntimeError("ORCiD API client credentials are not configured")

        token = self._get_access_token()
        url = f"{self.api_base_url}/{target_orcid}/works"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {"doi": set(), "pmid": set(), "pmcid": set()}
        return _extract_work_identifiers(payload)

    def _get_access_token(self) -> str:
        now = time.monotonic()
        if self._cached_token and now < self._token_expires_at - 30:
            return self._cached_token

        if not self.client_id or not self.client_secret:
            raise RuntimeError("ORCiD API client credentials are not configured")

        form_data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": "/read-public",
        }
        headers = {"Accept": "application/json"}
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.post(self.token_url, data=form_data, headers=headers)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected ORCiD token response")
        token = str(payload.get("access_token", "")).strip()
        if not token:
            raise RuntimeError("ORCiD token response missing access_token")
        expires_in_raw = payload.get("expires_in", 3600)
        try:
            expires_in = int(expires_in_raw)
        except (TypeError, ValueError):
            expires_in = 3600

        self._cached_token = token
        self._token_expires_at = now + max(expires_in, 60)
        return token


def _extract_work_identifiers(payload: dict[str, Any]) -> dict[str, set[str]]:
    identifiers: dict[str, set[str]] = {
        "doi": set(),
        "pmid": set(),
        "pmcid": set(),
    }

    groups = payload.get("group", [])
    if not isinstance(groups, list):
        return identifiers

    for group in groups:
        if not isinstance(group, dict):
            continue
        _collect_external_ids(group.get("external-ids"), identifiers)

        summaries = group.get("work-summary", [])
        if isinstance(summaries, list):
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                _collect_external_ids(summary.get("external-ids"), identifiers)

    return identifiers


def _collect_external_ids(external_ids_node: Any, identifiers: dict[str, set[str]]) -> None:
    if not isinstance(external_ids_node, dict):
        return
    items = external_ids_node.get("external-id", [])
    if not isinstance(items, list):
        return

    for item in items:
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("external-id-type", "")).strip().lower()
        raw_value = _external_id_value(item)

        if raw_type in {"doi"}:
            normalized = _normalize_doi(raw_value)
            if normalized:
                identifiers["doi"].add(normalized)
            continue

        if raw_type in {"pmid", "pubmed", "pubmed-id"}:
            normalized = _normalize_pmid(raw_value)
            if normalized:
                identifiers["pmid"].add(normalized)
            continue

        if raw_type in {"pmcid", "pmc", "pmc-id", "pubmed-central", "pubmed-central-id"}:
            normalized = _normalize_pmcid(raw_value)
            if normalized:
                identifiers["pmcid"].add(normalized)
            continue


def _external_id_value(external_id: dict[str, Any]) -> str:
    normalized = external_id.get("external-id-normalized")
    if isinstance(normalized, dict):
        normalized_value = normalized.get("value")
        if isinstance(normalized_value, str) and normalized_value.strip():
            return normalized_value
    raw_value = external_id.get("external-id-value", "")
    return str(raw_value)


def _normalize_doi(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = DOI_URL_PREFIX.sub("", raw)
    raw = re.sub(r"^doi:\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.strip().strip(".,;")
    return raw.lower()


def _normalize_pmid(value: str) -> str:
    return NON_DIGIT.sub("", value)


def _normalize_pmcid(value: str) -> str:
    raw = NON_ALNUM.sub("", value.upper())
    if not raw:
        return ""
    if raw.startswith("PMC"):
        return raw
    if raw.isdigit():
        return f"PMC{raw}"
    return raw
