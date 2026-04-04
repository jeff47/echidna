from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.logic import (
    _extract_institution_names,
    _has_distinctive_institution_overlap,
    _institution_key,
    normalize_orcid,
    normalize_text,
    parse_target_name,
)
from app.models import Citation
from app.perf import perf_logging_enabled

DOI_URL_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", flags=re.IGNORECASE)
NON_DIGIT = re.compile(r"\D+")
NON_ALNUM = re.compile(r"[^A-Z0-9]+")
NON_WORD = re.compile(r"[^a-z0-9]+")

logger = logging.getLogger(__name__)


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
        max_retries: int = 4,
        min_interval_seconds: float = 0.2,
    ) -> None:
        self.timeout = timeout
        configured_client_id = client_id if client_id is not None else os.getenv("ORCID_CLIENT_ID", "")
        configured_client_secret = client_secret if client_secret is not None else os.getenv("ORCID_CLIENT_SECRET", "")
        configured_token_url = token_url if token_url is not None else os.getenv("ORCID_TOKEN_URL", "https://orcid.org/oauth/token")
        configured_api_base_url = api_base_url if api_base_url is not None else os.getenv("ORCID_API_BASE_URL", "https://pub.orcid.org/v3.0")
        configured_min_interval = os.getenv("ORCID_MIN_INTERVAL_SECONDS", str(min_interval_seconds))

        self.client_id = configured_client_id.strip()
        self.client_secret = configured_client_secret.strip()
        self.token_url = configured_token_url.strip()
        self.api_base_url = configured_api_base_url.strip().rstrip("/")
        self.max_retries = max_retries
        try:
            parsed_min_interval = float(configured_min_interval)
        except ValueError:
            parsed_min_interval = min_interval_seconds
        self.min_interval_seconds = max(0.0, parsed_min_interval)
        self._next_request_at = 0.0
        self._cached_token: str = ""
        self._token_expires_at: float = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def match_citations(self, target_orcid: str, citations: list[Citation]) -> dict[str, list[str]]:
        if not target_orcid.strip():
            return {}
        perf_enabled = perf_logging_enabled()
        started_at = time.perf_counter() if perf_enabled else 0.0
        stage_started_at = time.perf_counter() if perf_enabled else 0.0
        identifier_index = self.fetch_work_identifiers(target_orcid)
        fetch_seconds = time.perf_counter() - stage_started_at if perf_enabled else 0.0
        stage_started_at = time.perf_counter() if perf_enabled else 0.0
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
        if perf_enabled:
            logger.info(
                (
                    "perf orcid.match_citations citations=%d matched_pmids=%d "
                    "fetch_seconds=%.3f local_match_seconds=%.3f total_seconds=%.3f"
                ),
                len(citations),
                len(matches_by_pmid),
                fetch_seconds,
                time.perf_counter() - stage_started_at,
                time.perf_counter() - started_at,
            )
        return matches_by_pmid

    def match_affiliations(
        self,
        target_orcid: str,
        affiliations_by_pmid: dict[str, list[str]],
    ) -> dict[str, dict[str, str]]:
        if not target_orcid.strip() or not affiliations_by_pmid:
            return {}

        perf_enabled = perf_logging_enabled()
        started_at = time.perf_counter() if perf_enabled else 0.0
        stage_started_at = time.perf_counter() if perf_enabled else 0.0
        organizations = self.fetch_affiliation_organizations(target_orcid)
        fetch_seconds = time.perf_counter() - stage_started_at if perf_enabled else 0.0
        if not organizations:
            if perf_enabled:
                logger.info(
                    (
                        "perf orcid.match_affiliations pmids=%d organizations=0 matched_pmids=0 "
                        "fetch_seconds=%.3f local_match_seconds=0.000 total_seconds=%.3f"
                    ),
                    len(affiliations_by_pmid),
                    fetch_seconds,
                    time.perf_counter() - started_at,
                )
            return {}

        orgs_by_key: dict[str, list[OrcidOrganization]] = {}
        for org in organizations:
            orgs_by_key.setdefault(org.key, []).append(org)

        stage_started_at = time.perf_counter() if perf_enabled else 0.0
        matches_by_pmid: dict[str, dict[str, str]] = {}
        for pmid, affiliations in affiliations_by_pmid.items():
            match = _match_affiliation_set(affiliations, organizations, orgs_by_key)
            if match is None:
                continue
            matches_by_pmid[pmid] = match
        if perf_enabled:
            logger.info(
                (
                    "perf orcid.match_affiliations pmids=%d organizations=%d matched_pmids=%d "
                    "fetch_seconds=%.3f local_match_seconds=%.3f total_seconds=%.3f"
                ),
                len(affiliations_by_pmid),
                len(organizations),
                len(matches_by_pmid),
                fetch_seconds,
                time.perf_counter() - stage_started_at,
                time.perf_counter() - started_at,
            )
        return matches_by_pmid

    def search_profiles(self, author_name: str, limit: int = 12) -> list[dict[str, str]]:
        query = author_name.strip()
        if not query:
            return []

        perf_enabled = perf_logging_enabled()
        started_at = time.perf_counter() if perf_enabled else 0.0
        expanded_requests = 0
        basic_requests = 0

        rows = max(1, min(limit, 50))
        headers = {"Accept": "application/json"}
        token = self._optional_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        search_queries = _build_profile_search_queries(query)
        deduped: list[dict[str, str]] = []
        seen_orcids: set[str] = set()

        for candidate_query in search_queries:
            expanded_payload = self._search_payload(path="/expanded-search", query=candidate_query, rows=rows, headers=headers)
            expanded_requests += 1
            expanded_matches = _extract_expanded_search_matches(expanded_payload)
            for match in expanded_matches:
                orcid = match.get("orcid", "")
                if not orcid or orcid in seen_orcids:
                    continue
                seen_orcids.add(orcid)
                deduped.append(match)
                if len(deduped) >= rows:
                    if perf_enabled:
                        logger.info(
                            (
                                "perf orcid.search_profiles queries=%d expanded_requests=%d "
                                "basic_requests=%d returned=%d total_seconds=%.3f"
                            ),
                            len(search_queries),
                            expanded_requests,
                            basic_requests,
                            len(deduped[:rows]),
                            time.perf_counter() - started_at,
                        )
                    return deduped[:rows]

        if deduped:
            if perf_enabled:
                logger.info(
                    (
                        "perf orcid.search_profiles queries=%d expanded_requests=%d "
                        "basic_requests=%d returned=%d total_seconds=%.3f"
                    ),
                    len(search_queries),
                    expanded_requests,
                    basic_requests,
                    len(deduped[:rows]),
                    time.perf_counter() - started_at,
                )
            return deduped[:rows]

        for candidate_query in search_queries:
            basic_payload = self._search_payload(path="/search", query=candidate_query, rows=rows, headers=headers)
            basic_requests += 1
            basic_matches = _extract_basic_search_matches(basic_payload)
            for match in basic_matches:
                orcid = match.get("orcid", "")
                if not orcid or orcid in seen_orcids:
                    continue
                seen_orcids.add(orcid)
                deduped.append(match)
                if len(deduped) >= rows:
                    if perf_enabled:
                        logger.info(
                            (
                                "perf orcid.search_profiles queries=%d expanded_requests=%d "
                                "basic_requests=%d returned=%d total_seconds=%.3f"
                            ),
                            len(search_queries),
                            expanded_requests,
                            basic_requests,
                            len(deduped[:rows]),
                            time.perf_counter() - started_at,
                        )
                    return deduped[:rows]
        if perf_enabled:
            logger.info(
                (
                    "perf orcid.search_profiles queries=%d expanded_requests=%d "
                    "basic_requests=%d returned=%d total_seconds=%.3f"
                ),
                len(search_queries),
                expanded_requests,
                basic_requests,
                len(deduped[:rows]),
                time.perf_counter() - started_at,
            )
        return deduped[:rows]

    def fetch_work_identifiers(self, target_orcid: str) -> dict[str, set[str]]:
        if not self.is_configured:
            raise RuntimeError("ORCiD API client credentials are not configured")

        token = self._get_access_token()
        url = f"{self.api_base_url}/{target_orcid}/works"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        response = self._request("GET", url, headers=headers)
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
        response = self._request("GET", url, headers=headers)
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
        response = self._request("POST", self.token_url, headers=headers, data=form_data)
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

    def _optional_access_token(self) -> str:
        if not self.is_configured:
            return ""
        try:
            return self._get_access_token()
        except Exception:  # noqa: BLE001
            return ""

    def _search_payload(self, *, path: str, query: str, rows: int, headers: dict[str, str]) -> dict[str, Any]:
        url = f"{self.api_base_url}{path}"
        params = {
            "q": query,
            "start": "0",
            "rows": str(rows),
        }
        response = self._request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return {}

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
    ) -> httpx.Response:
        last_response: httpx.Response | None = None
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for attempt in range(self.max_retries + 1):
                self._throttle()
                response = client.request(method, url, headers=headers, params=params, data=data)
                last_response = response
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    delay_seconds = _retry_delay_seconds(response, attempt)
                    logger.warning(
                        "ORCiD request retry: status=%d method=%s url=%s attempt=%d/%d wait=%.2fs",
                        response.status_code,
                        method,
                        url,
                        attempt + 1,
                        self.max_retries + 1,
                        delay_seconds,
                    )
                    time.sleep(delay_seconds)
                    continue
                response.raise_for_status()
                return response
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("ORCiD request failed without response")

    def _throttle(self) -> None:
        now = time.monotonic()
        if now < self._next_request_at:
            time.sleep(self._next_request_at - now)
        self._next_request_at = time.monotonic() + self.min_interval_seconds


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return max(0.5, float(retry_after))
        except ValueError:
            retry_at = parsedate_to_datetime(retry_after)
            now = time.time()
            return max(0.5, retry_at.timestamp() - now)
    # Exponential backoff with cap.
    return min(8.0, 0.6 * (2**attempt))


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



def _extract_expanded_search_matches(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_results = payload.get("expanded-result", [])
    if not isinstance(raw_results, list):
        return []

    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        orcid = _extract_orcid_from_node(item.get("orcid-id")) or _extract_orcid_from_node(item.get("orcid-identifier"))
        if not orcid or orcid in seen:
            continue
        seen.add(orcid)

        given = str(item.get("given-names", "")).strip()
        family = str(item.get("family-names", "")).strip()
        credit = str(item.get("credit-name", "")).strip()
        display_name = " ".join(part for part in (given, family) if part).strip() or credit

        institution = _expanded_institution_text(item.get("institution-name")) or _expanded_institution_text(
            item.get("current-institution-affiliation-name")
        )
        country = str(item.get("country", "")).strip()
        matches.append(
            {
                "orcid": orcid,
                "display_name": display_name,
                "institution": institution,
                "country": country,
                "record_url": f"https://orcid.org/{orcid}",
            }
        )
    return matches


def _extract_basic_search_matches(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_results = payload.get("result", [])
    if not isinstance(raw_results, list):
        return []

    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        orcid = _extract_orcid_from_node(item.get("orcid-identifier")) or _extract_orcid_from_node(item.get("orcid-id"))
        if not orcid or orcid in seen:
            continue
        seen.add(orcid)
        matches.append(
            {
                "orcid": orcid,
                "display_name": "",
                "institution": "",
                "country": "",
                "record_url": f"https://orcid.org/{orcid}",
            }
        )
    return matches


def _extract_orcid_from_node(node: Any) -> str:
    if isinstance(node, dict):
        path_value = str(node.get("path", "")).strip()
        if path_value:
            normalized = normalize_orcid(path_value)
            if normalized:
                return normalized
        uri_value = str(node.get("uri", "")).strip()
        if uri_value:
            normalized = normalize_orcid(uri_value)
            if normalized:
                return normalized
        return ""
    if isinstance(node, str):
        return normalize_orcid(node)
    return ""


def _expanded_institution_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if str(part).strip()]
        return "; ".join(parts)
    return ""


def _build_profile_search_queries(author_name: str) -> list[str]:
    raw = author_name.strip()
    if not raw:
        return []

    queries: list[str] = []
    try:
        target = parse_target_name(raw)
    except ValueError:
        target = None

    if target is not None:
        surname = _orcid_query_term(target.surname)
        given = _orcid_query_term(target.given.split()[0] if target.given else "")
        first_initial = _orcid_query_term(target.initials[:1] if target.initials else "")
        if surname and given:
            queries.append(f'given-and-family-names:"{given} {surname}"')
            queries.append(f"family-name:{surname} AND given-names:{given}*")
        if surname and first_initial:
            queries.append(f"family-name:{surname} AND given-names:{first_initial}*")
        if surname:
            queries.append(f"family-name:{surname}")

    escaped_raw = _orcid_query_phrase(raw)
    if escaped_raw:
        queries.append(f'given-and-family-names:"{escaped_raw}"')
        queries.append(f'credit-name:"{escaped_raw}"')
        queries.append(raw)

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


def _orcid_query_term(value: str) -> str:
    token = normalize_text(value).replace(" ", "")
    return re.sub(r"[^a-z0-9]", "", token)


def _orcid_query_phrase(value: str) -> str:
    phrase = " ".join(value.split()).strip()
    return phrase.replace('"', "")

def _extract_affiliation_organizations(payload: dict[str, Any]) -> list[OrcidOrganization]:
    organizations: list[OrcidOrganization] = []
    seen: set[tuple[str, str, str]] = set()

    for org in _iter_organizations(payload):
        raw_name = str(org.get("name", "")).strip()
        if not raw_name:
            continue
        canonical_names = _extract_institution_names(raw_name)
        display_name = _clean_org_name(raw_name)
        match_basis = canonical_names[0] if canonical_names else display_name
        key = _institution_key(match_basis)
        if not key:
            key = _institution_key(display_name)
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
                name=display_name,
                key=key,
                signature=_org_signature(display_name),
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
            institution_names = _fallback_institution_phrases(raw_affiliation)

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
                if len(org.key) >= 10
                and (org.key in inst_key or inst_key in org.key)
                and _has_distinctive_institution_overlap(institution_name, org.name)
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
                if not _has_distinctive_institution_overlap(institution_name, org.name):
                    continue
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


def _fallback_institution_phrases(affiliation: str) -> list[str]:
    phrases = [part.strip() for part in re.split(r"[;|]", affiliation) if part.strip()]
    if not phrases:
        phrases = [affiliation]

    out: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        comma_chunks = [chunk.strip() for chunk in phrase.split(",") if chunk.strip()]
        for value in ([phrase] + comma_chunks):
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(value)
    return out or [affiliation]


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
