from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx

from app.logic import _extract_institution_names, _institution_key, normalize_text
from app.models import Citation

DOI_URL_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", flags=re.IGNORECASE)
NON_DIGIT = re.compile(r"\D+")
NON_ALNUM = re.compile(r"[^A-Z0-9]+")
NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class OrcidOrganization:
    name: str
    key: str
    signature: str
    country: str
    identifier: str


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

    def match_affiliations(
        self,
        target_orcid: str,
        affiliations_by_pmid: dict[str, list[str]],
    ) -> dict[str, dict[str, str]]:
        if not target_orcid.strip() or not affiliations_by_pmid:
            return {}

        organizations = self.fetch_affiliation_organizations(target_orcid)
        if not organizations:
            return {}

        orgs_by_key: dict[str, list[OrcidOrganization]] = {}
        for org in organizations:
            orgs_by_key.setdefault(org.key, []).append(org)

        matches_by_pmid: dict[str, dict[str, str]] = {}
        for pmid, affiliations in affiliations_by_pmid.items():
            match = _match_affiliation_set(affiliations, organizations, orgs_by_key)
            if match is None:
                continue
            matches_by_pmid[pmid] = match
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

    def fetch_affiliation_organizations(self, target_orcid: str) -> list[OrcidOrganization]:
        if not self.is_configured:
            raise RuntimeError("ORCiD API client credentials are not configured")

        token = self._get_access_token()
        url = f"{self.api_base_url}/{target_orcid}/activities"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return []
        return _extract_affiliation_organizations(payload)

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


def _extract_affiliation_organizations(payload: dict[str, Any]) -> list[OrcidOrganization]:
    organizations: list[OrcidOrganization] = []
    seen: set[tuple[str, str, str]] = set()

    for org in _iter_organizations(payload):
        raw_name = str(org.get("name", "")).strip()
        if not raw_name:
            continue
        key = _institution_key(raw_name)
        if not key:
            continue

        address = org.get("address")
        country = ""
        if isinstance(address, dict):
            country = str(address.get("country", "")).strip().upper()

        disambiguated = org.get("disambiguated-organization")
        identifier = ""
        if isinstance(disambiguated, dict):
            identifier = _normalize_org_identifier(str(disambiguated.get("disambiguated-organization-identifier", "")))

        dedupe_key = (key, country, identifier)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        organizations.append(
            OrcidOrganization(
                name=_clean_org_name(raw_name),
                key=key,
                signature=_org_signature(raw_name),
                country=country,
                identifier=identifier,
            )
        )

    return organizations


def _iter_organizations(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        org = node.get("organization")
        if isinstance(org, dict):
            found.append(org)
        for value in node.values():
            found.extend(_iter_organizations(value))
        return found
    if isinstance(node, list):
        for item in node:
            found.extend(_iter_organizations(item))
    return found


def _match_affiliation_set(
    affiliations: list[str],
    organizations: list[OrcidOrganization],
    orgs_by_key: dict[str, list[OrcidOrganization]],
) -> dict[str, str] | None:
    best_possible: tuple[float, OrcidOrganization] | None = None

    for affiliation in affiliations:
        raw_affiliation = (affiliation or "").strip()
        if not raw_affiliation:
            continue

        institution_names = _extract_institution_names(raw_affiliation)
        if not institution_names:
            institution_names = [raw_affiliation]

        for institution_name in institution_names:
            inst_key = _institution_key(institution_name)
            if not inst_key:
                continue

            candidates = orgs_by_key.get(inst_key, [])
            if candidates:
                with_identifier = [candidate for candidate in candidates if candidate.identifier]
                if with_identifier and any(_identifier_in_affiliation(raw_affiliation, candidate.identifier) for candidate in with_identifier):
                    chosen = with_identifier[0]
                    return {
                        "level": "verified",
                        "label": f"organization identifier ({chosen.name})",
                        "institution": chosen.name,
                    }

                chosen = candidates[0]
                return {
                    "level": "likely",
                    "label": f"institution match ({chosen.name})",
                    "institution": chosen.name,
                }

            # Weaker match: publication affiliation contains an ORCiD institution string
            # inside a compound affiliation (e.g., "Harvard ... - Brigham and Women's").
            contains_candidates = [
                org
                for org in organizations
                if len(org.key) >= 10 and (org.key in inst_key or inst_key in org.key)
            ]
            if contains_candidates:
                chosen = max(contains_candidates, key=lambda org: len(org.key))
                return {
                    "level": "contains",
                    "label": f"affiliation contains institution ({chosen.name})",
                    "institution": chosen.name,
                }

            inst_signature = _org_signature(institution_name)
            if not inst_signature:
                continue
            for org in organizations:
                ratio = SequenceMatcher(a=inst_signature, b=org.signature).ratio()
                if ratio < 0.9:
                    continue
                if best_possible is None or ratio > best_possible[0]:
                    best_possible = (ratio, org)

    if best_possible is None:
        return None
    return {
        "level": "possible",
        "label": f"name similarity ({best_possible[1].name})",
        "institution": best_possible[1].name,
    }


def _identifier_in_affiliation(affiliation: str, identifier: str) -> bool:
    if not affiliation or not identifier:
        return False
    normalized_affiliation = NON_WORD.sub("", affiliation.lower())
    normalized_identifier = NON_WORD.sub("", identifier.lower())
    if not normalized_identifier:
        return False
    return normalized_identifier in normalized_affiliation


def _normalize_org_identifier(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = raw.rstrip("/")
    lower = raw.lower()
    if "ror.org/" in lower:
        raw = raw[lower.rfind("ror.org/") + len("ror.org/") :]
    return NON_WORD.sub("", raw.lower())


def _clean_org_name(name: str) -> str:
    return " ".join(name.strip().split())


def _org_signature(name: str) -> str:
    value = normalize_text(name)
    value = value.replace("&", " and ")
    value = NON_WORD.sub(" ", value)
    return " ".join(value.split())


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
