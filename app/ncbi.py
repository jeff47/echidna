from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from typing import Any, Iterable
from urllib.parse import quote
from xml.etree import ElementTree as ET

import httpx

from app.logic import normalize_orcid, parse_target_name, parse_year
from app.models import Author, Citation
from app.perf import perf_logging_enabled

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
LITSENSE_AUTHOR_URL = "https://www.ncbi.nlm.nih.gov/research/litsense-api/api/author/"
XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"
PMC_XML_FETCH_CHUNK_SIZE = 50

EQUAL_PATTERNS = (
    "contributed equally",
    "equal contribution",
    "equal contributions",
    "equally contributed",
    "contributed in equal measure",
    "co-first",
    "co first",
)
SENIOR_PATTERNS = (
    "co-senior",
    "co senior",
    "co-senior authorship",
    "co senior authorship",
    "co-last",
    "co last",
    "co-last authorship",
    "co last authorship",
    "share senior authorship",
    "shared senior authorship",
    "jointly supervised",
    "joint supervision",
    "jointly led",
    "jointly directed",
    "senior author",
    "supervised the work",
)
AFFILIATION_HINT_PATTERNS = (
    "department",
    "division",
    "institute",
    "university",
    "school",
    "college",
    "hospital",
    "laboratory",
    "center",
    "centre",
    "nih",
    "national institutes of health",
)
AFFILIATION_FOOTNOTE_TARGET_WORDS = (
    "department",
    "departments",
    "division",
    "institute",
    "university",
    "school",
    "college",
    "hospital",
    "laboratory",
    "lab",
    "center",
    "centre",
    "faculty",
    "program",
    "unit",
)
SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")
DOI_URL_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", flags=re.IGNORECASE)
PERSON_TOKEN = re.compile(r"[^a-z0-9]+")

logger = logging.getLogger(__name__)


class NcbiClient:
    def __init__(self, timeout: float = 30.0, api_key: str | None = None, max_retries: int = 4) -> None:
        self.timeout = timeout
        configured_api_key = api_key if api_key is not None else os.getenv("NCBI_API_KEY", "")
        self.api_key = configured_api_key.strip() or None
        self.email = os.getenv("NCBI_EMAIL", "").strip() or None
        self.tool = os.getenv("NCBI_TOOL", "ECHIDNA").strip() or "ECHIDNA"
        self.max_retries = max_retries
        self.min_interval_seconds = 0.11 if self.api_key else 0.34
        self._next_request_at = 0.0
        self._throttle_lock = threading.Lock()

    def fetch_citations(self, pmids: list[str]) -> list[Citation]:
        if not pmids:
            return []

        perf_enabled = perf_logging_enabled()
        started_at = time.perf_counter() if perf_enabled else 0.0
        pubmed_fetch_seconds = 0.0
        pmcid_map_seconds = 0.0
        pmc_fetch_seconds = 0.0
        pmc_parse_seconds = 0.0
        pmc_request_count = 0
        pmc_fetch_attempts = 0
        pmc_fetch_successes = 0
        pmc_fetch_failures = 0

        logger.info("Fetching PubMed citations for %d PMID(s)", len(pmids))
        stage_started_at = time.perf_counter() if perf_enabled else 0.0
        pubmed_xml = self._fetch_pubmed_xml(pmids)
        citations = self._parse_pubmed(pubmed_xml)
        if perf_enabled:
            pubmed_fetch_seconds = time.perf_counter() - stage_started_at
        logger.info("Parsed %d citation(s) from PubMed", len(citations))
        try:
            stage_started_at = time.perf_counter() if perf_enabled else 0.0
            pmcid_map = self._fetch_pmcid_map(pmids)
            if perf_enabled:
                pmcid_map_seconds = time.perf_counter() - stage_started_at
            logger.info("Mapped %d/%d PMID(s) to PMCID", len(pmcid_map), len(pmids))
        except Exception as exc:  # noqa: BLE001
            pmcid_map = {}
            if perf_enabled:
                pmcid_map_seconds = time.perf_counter() - stage_started_at
            logger.warning("PMCID mapping request failed: %s", exc)
            for citation in citations:
                citation.notes.append(f"PMCID mapping failed: {exc}")

        for citation in citations:
            citation.pmcid = pmcid_map.get(citation.pmid)

        pmc_xml_by_pmcid: dict[str, str] = {}
        pmcids = [citation.pmcid for citation in citations if citation.pmcid]
        if pmcids:
            stage_started_at = time.perf_counter() if perf_enabled else 0.0
            pmc_xml_by_pmcid, pmc_request_count = self._fetch_pmc_xml_map(pmcids)
            if perf_enabled:
                pmc_fetch_seconds = time.perf_counter() - stage_started_at

        for citation in citations:
            if not citation.pmcid:
                citation.notes.append("No PMCID mapping")
                logger.debug("PMID %s has no PMCID mapping", citation.pmid)
                continue
            pmc_fetch_attempts += 1
            try:
                pubmed_authors = list(citation.authors)
                pmc_xml = pmc_xml_by_pmcid.get(citation.pmcid)
                if not pmc_xml:
                    raise RuntimeError(f"PMC XML unavailable for {citation.pmcid}")

                stage_started_at = time.perf_counter() if perf_enabled else 0.0
                pmc_doi = _extract_pmc_article_doi(pmc_xml)
                if pmc_doi and not citation.doi:
                    citation.doi = pmc_doi
                co_first, co_senior, pmc_authors, pmc_count = self._extract_pmc_enrichment(pmc_xml)
                if perf_enabled:
                    pmc_parse_seconds += time.perf_counter() - stage_started_at
                if pmc_count and pmc_count != len(citation.authors):
                    citation.notes.append(
                        f"PMC author count mismatch (PMC {pmc_count} vs PubMed {len(citation.authors)})"
                    )
                    logger.debug(
                        "PMID %s author count mismatch (PMC=%d, PubMed=%d)",
                        citation.pmid,
                        pmc_count,
                        len(citation.authors),
                    )
                citation.co_first_positions = co_first
                citation.co_senior_positions = co_senior
                used_pmc_authors = self._replace_authors_from_pmc(citation, pmc_authors)
                if used_pmc_authors:
                    citation.notes.append("Author metadata sourced from PMC")
                    filled = self._fill_missing_affiliations_from_pubmed(citation, pubmed_authors)
                    if filled:
                        citation.notes.append(f"Filled {filled} affiliation(s) from PubMed fallback")
                else:
                    pmc_affiliations = {
                        author.position: _author_affiliation_blocks(author)
                        for author in pmc_authors
                        if _author_affiliation_blocks(author)
                    }
                    filled = self._fill_missing_affiliations(citation, pmc_affiliations)
                    if filled:
                        citation.notes.append(f"Filled {filled} affiliation(s) from PMC")
                    citation.notes.append("PMC author metadata incomplete; kept PubMed author list")
                citation.source_for_roles = "pmc"
                pmc_fetch_successes += 1
                logger.debug(
                    "PMID %s enriched from PMC: co_first=%s, co_senior=%s, pmc_authors=%d, used_pmc_authors=%s, filled_affiliations=%d",
                    citation.pmid,
                    sorted(co_first),
                    sorted(co_senior),
                    len(pmc_authors),
                    used_pmc_authors,
                    filled,
                )
            except Exception as exc:  # noqa: BLE001
                citation.source_for_roles = "pubmed"
                citation.notes.append(f"PMC fetch/parse failed: {exc}")
                pmc_fetch_failures += 1
                logger.warning("PMID %s PMC fetch/parse failed: %s", citation.pmid, exc)

            if not citation.doi:
                citation.notes.append("DOI not found in PubMed/PMC metadata")

        if perf_enabled:
            logger.info(
                (
                    "perf ncbi.fetch_citations pmids=%d citations=%d pmc_attempts=%d "
                    "pmc_successes=%d pmc_failures=%d pmc_request_count=%d pubmed_fetch_seconds=%.3f "
                    "pmcid_map_seconds=%.3f pmc_fetch_seconds=%.3f pmc_parse_seconds=%.3f "
                    "min_interval_seconds=%.3f total_seconds=%.3f"
                ),
                len(pmids),
                len(citations),
                pmc_fetch_attempts,
                pmc_fetch_successes,
                pmc_fetch_failures,
                pmc_request_count,
                pubmed_fetch_seconds,
                pmcid_map_seconds,
                pmc_fetch_seconds,
                pmc_parse_seconds,
                self.min_interval_seconds,
                time.perf_counter() - started_at,
            )

        return citations

    def _fetch_pubmed_xml(self, pmids: list[str]) -> str:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        url = f"{EUTILS_BASE}/efetch.fcgi"
        response = self._request(url, params=params)
        return response.text

    def _fetch_pmcid_map(self, pmids: list[str]) -> dict[str, str]:
        params = {"ids": ",".join(pmids), "format": "json"}
        response = self._request(IDCONV_URL, params=params)
        payload = response.json()

        mapping: dict[str, str] = {}
        for record in _idconv_records(payload):
            pmid = record.get("pmid")
            pmcid = record.get("pmcid")
            if pmid and pmcid:
                mapping[str(pmid)] = str(pmcid)
        return mapping

    def _fetch_pmc_xml(self, pmcid: str) -> str:
        return self._fetch_pmc_xml_batch([pmcid])

    def _fetch_pmc_xml_map(self, pmcids: list[str]) -> tuple[dict[str, str], int]:
        pmc_xml_by_pmcid: dict[str, str] = {}
        request_count = 0
        for chunk in split_chunks(_dedupe_preserve_order(pmcids), chunk_size=PMC_XML_FETCH_CHUNK_SIZE):
            request_count += 1
            try:
                batch_xml = self._fetch_pmc_xml_batch(chunk)
                parsed_xml = _parse_pmc_xml_by_pmcid(batch_xml, requested_pmcids=chunk)
                pmc_xml_by_pmcid.update(parsed_xml)
            except Exception as exc:  # noqa: BLE001
                logger.warning("PMC batch fetch/parse failed for %d PMCID(s): %s", len(chunk), exc)

            missing_pmcids = [pmcid for pmcid in chunk if pmcid not in pmc_xml_by_pmcid]
            for pmcid in missing_pmcids:
                request_count += 1
                try:
                    pmc_xml_by_pmcid[pmcid] = self._fetch_pmc_xml(pmcid)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PMC single-article fallback failed for %s: %s", pmcid, exc)
        return pmc_xml_by_pmcid, request_count

    def _fetch_pmc_xml_batch(self, pmcids: list[str]) -> str:
        pmcid_clean = ",".join(_pmc_query_id(pmcid) for pmcid in pmcids if _pmc_query_id(pmcid))
        if not pmcid_clean:
            return ""
        params = {
            "db": "pmc",
            "id": pmcid_clean,
            "retmode": "xml",
        }
        url = f"{EUTILS_BASE}/efetch.fcgi"
        response = self._request(url, params=params)
        return response.text

    def fetch_litsense_pmids_by_query(self, *, seed_pmid: str, author_name: str, max_pages: int = 10) -> set[str]:
        query = _build_litsense_author_query(seed_pmid=seed_pmid, author_name=author_name)
        if not query:
            return set()
        encoded_query = quote(query, safe="")
        url = f"{LITSENSE_AUTHOR_URL}?query={encoded_query}"
        return self._fetch_litsense_pmids(max_pages=max_pages, initial_url=url)

    def fetch_litsense_pmids_by_orcid(self, *, target_orcid: str, max_pages: int = 10) -> set[str]:
        normalized = target_orcid.strip()
        if not normalized:
            return set()
        return self._fetch_litsense_pmids(max_pages=max_pages, params={"orcid": normalized})

    def _fetch_litsense_pmids(
        self,
        *,
        max_pages: int,
        params: dict[str, str] | None = None,
        initial_url: str | None = None,
    ) -> set[str]:
        perf_enabled = perf_logging_enabled()
        started_at = time.perf_counter() if perf_enabled else 0.0
        request_count = 0

        collected: set[str] = set()
        current_url = initial_url or LITSENSE_AUTHOR_URL
        current_params: dict[str, str] | None = None
        if params:
            current_params = {key: str(value) for key, value in params.items() if str(value).strip()}
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for _ in range(max_pages):
                response: httpx.Response | None = None
                for attempt in range(self.max_retries + 1):
                    self._throttle()
                    candidate = client.get(current_url, params=current_params)
                    response = candidate
                    if candidate.status_code in (429, 500, 502, 503, 504):
                        if attempt >= self.max_retries:
                            candidate.raise_for_status()
                        delay_seconds = _retry_delay_seconds(candidate, attempt)
                        logger.warning(
                            "LitSense request retry: status=%d url=%s attempt=%d/%d wait=%.2fs",
                            candidate.status_code,
                            current_url,
                            attempt + 1,
                            self.max_retries + 1,
                            delay_seconds,
                        )
                        time.sleep(delay_seconds)
                        continue
                    candidate.raise_for_status()
                    break
                if response is None:
                    raise RuntimeError("LitSense request failed without response")
                request_count += 1
                payload = response.json()
                collected.update(_extract_litsense_pmids(payload))

                if not isinstance(payload, dict):
                    break
                next_url = str(payload.get("next", "")).strip()
                if not next_url:
                    break
                current_url = next_url
                current_params = None

        if perf_enabled:
            logger.info(
                (
                    "perf ncbi.fetch_litsense_pmids requests=%d returned_pmids=%d "
                    "max_pages=%d min_interval_seconds=%.3f total_seconds=%.3f"
                ),
                request_count,
                len(collected),
                max_pages,
                self.min_interval_seconds,
                time.perf_counter() - started_at,
            )
        return collected

    def _request(self, url: str, params: dict[str, str]) -> httpx.Response:
        request_params = _augment_params(
            url=url,
            params=params,
            api_key=self.api_key,
            tool=self.tool,
            email=self.email,
        )
        last_response: httpx.Response | None = None
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for attempt in range(self.max_retries + 1):
                self._throttle()
                response = client.get(url, params=request_params)
                last_response = response
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    delay_seconds = _retry_delay_seconds(response, attempt)
                    logger.warning(
                        "NCBI request retry: status=%d url=%s attempt=%d/%d wait=%.2fs",
                        response.status_code,
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
        raise RuntimeError("NCBI request failed without response")

    def _throttle(self) -> None:
        with self._throttle_lock:
            now = time.monotonic()
            if now < self._next_request_at:
                time.sleep(self._next_request_at - now)
            self._next_request_at = time.monotonic() + self.min_interval_seconds

    def _parse_pubmed(self, xml_text: str) -> list[Citation]:
        root = ET.fromstring(xml_text)
        citations: list[Citation] = []

        for article in root.findall(".//PubmedArticle"):
            pmid = _find_text(article, ".//MedlineCitation/PMID")
            if not pmid:
                continue

            title = _find_itertext(article, ".//Article/ArticleTitle")
            title = html.unescape(re.sub(r"\s+", " ", title)).strip()

            journal = _find_text(article, ".//Article/Journal/ISOAbbreviation") or _find_text(
                article,
                ".//Article/Journal/Title",
            )
            journal = journal.strip() or "(unknown journal)"

            pub_date = _find_pub_date(article)
            authors = _find_authors(article)
            doi = _extract_pubmed_article_doi(article)
            update_in_pmids, update_of_pmids = _extract_pubmed_update_relationships(article)

            citations.append(
                Citation(
                    pmid=pmid,
                    pmcid=None,
                    title=title,
                    journal=journal,
                    print_year=parse_year(pub_date),
                    print_date=pub_date,
                    authors=authors,
                    publication_types=_find_publication_types(article),
                    source_for_roles="pubmed",
                    doi=doi or None,
                    update_in_pmids=update_in_pmids,
                    update_of_pmids=update_of_pmids,
                )
            )

        return citations

    def _extract_pmc_enrichment(
        self,
        xml_text: str,
    ) -> tuple[set[int], set[int], list[Author], int]:
        root = ET.fromstring(xml_text)
        article_meta = root.find(".//front/article-meta")
        contribs = root.findall(".//front/article-meta/contrib-group/contrib[@contrib-type='author']")
        if not contribs:
            contribs = root.findall(".//contrib-group/contrib[@contrib-type='author']")
        aff_context = article_meta if article_meta is not None else root
        aff_by_id, aff_by_label = _extract_pmc_affiliation_maps(aff_context)
        note_by_id = _extract_pmc_note_text_by_id(root)

        fn_text = _extract_pmc_contrib_note_text(root)

        explicit_senior_ids: set[str] = set()
        equal_note_ids: set[str] = set()
        note_ids = set(fn_text.keys())
        for note_id, text in fn_text.items():
            if _contains_any_phrase(text, SENIOR_PATTERNS):
                explicit_senior_ids.add(note_id)
            elif _contains_any_phrase(text, EQUAL_PATTERNS):
                equal_note_ids.add(note_id)

        note_positions: dict[str, set[int]] = {}
        authors: list[Author] = []

        for idx, contrib in enumerate(contribs, start=1):
            ref_ids: set[str] = set()
            for xref in contrib.findall(".//xref"):
                ref_type = (xref.attrib.get("ref-type") or "").strip().lower()
                for rid in _split_reference_ids(xref.attrib.get("rid", "")):
                    if ref_type in {"fn", "corresp"} or rid in note_ids:
                        ref_ids.add(rid)
            for rid in ref_ids:
                note_positions.setdefault(rid, set()).add(idx)

            last_name = _find_text(contrib, ".//name/surname") or _find_text(contrib, ".//surname")
            fore_name = _find_text(contrib, ".//name/given-names") or _find_text(contrib, ".//given-names")
            if not last_name:
                # Fall back to collaborative/string names when person name markup is absent.
                last_name = _find_text(contrib, ".//collab") or _find_text(contrib, ".//string-name")
            affiliation_blocks = _extract_contrib_affiliation_blocks(
                contrib,
                aff_by_id=aff_by_id,
                aff_by_label=aff_by_label,
                note_by_id=note_by_id,
            )
            authors.append(
                Author(
                    position=idx,
                    last_name=last_name,
                    fore_name=fore_name,
                    initials=_initials_from_name(fore_name),
                    affiliation="; ".join(affiliation_blocks),
                    affiliation_blocks=affiliation_blocks,
                    orcid=_extract_pmc_contrib_orcid(contrib),
                )
            )

        author_count = len(contribs)
        co_first_positions: set[int] = set()
        co_senior_positions: set[int] = set()
        for note_id in explicit_senior_ids:
            co_senior_positions.update(note_positions.get(note_id, set()))

        for note_id in equal_note_ids:
            positions = sorted(note_positions.get(note_id, set()))
            if not positions:
                continue
            is_prefix = _is_prefix_positions(positions)
            is_suffix = _is_suffix_positions(positions, author_count)
            # Precision-first: if note marks all authors (both prefix and suffix),
            # skip inference rather than forcing first/senior.
            if is_prefix and is_suffix:
                continue
            if is_prefix:
                co_first_positions.update(positions)
                continue
            if is_suffix:
                co_senior_positions.update(positions)

        return co_first_positions, co_senior_positions, authors, len(contribs)

    def _fill_missing_affiliations(
        self,
        citation: Citation,
        pmc_affiliations: Mapping[int, list[str] | str],
    ) -> int:
        filled = 0
        for author in citation.authors:
            if _author_has_affiliation(author):
                continue
            raw_pmc_aff = pmc_affiliations.get(author.position)
            pmc_aff_blocks = (
                _normalize_affiliation_blocks(raw_pmc_aff)
                if isinstance(raw_pmc_aff, list)
                else _normalize_affiliation_blocks([raw_pmc_aff or ""])
            )
            if not pmc_aff_blocks:
                continue
            _set_author_affiliations(author, pmc_aff_blocks)
            filled += 1
        return filled

    def _replace_authors_from_pmc(self, citation: Citation, pmc_authors: list[Author]) -> bool:
        if not pmc_authors:
            return False
        non_empty_names = sum(1 for author in pmc_authors if author.last_name.strip())
        if non_empty_names == 0:
            return False
        citation.authors = pmc_authors
        return True

    def _fill_missing_affiliations_from_pubmed(self, citation: Citation, pubmed_authors: list[Author]) -> int:
        by_position: dict[int, Author] = {
            author.position: author for author in pubmed_authors if _author_has_affiliation(author)
        }
        filled = 0
        for author in citation.authors:
            if _author_has_affiliation(author):
                continue
            source = by_position.get(author.position)
            if source is None:
                continue
            if not _authors_likely_same_person(author, source):
                continue
            _set_author_affiliations(author, _author_affiliation_blocks(source))
            filled += 1
        return filled


def _find_text(node: ET.Element, path: str) -> str:
    target = node.find(path)
    if target is None or target.text is None:
        return ""
    return target.text


def _find_itertext(node: ET.Element, path: str) -> str:
    target = node.find(path)
    if target is None:
        return ""
    return "".join(target.itertext())


def _find_pub_date(article: ET.Element) -> str:
    candidates = [
        ".//Article/Journal/JournalIssue/PubDate",
        ".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']",
    ]

    for path in candidates:
        date_node = article.find(path)
        if date_node is None:
            continue
        year = _find_text(date_node, "Year")
        month = _find_text(date_node, "Month")
        day = _find_text(date_node, "Day")
        medline = _find_text(date_node, "MedlineDate")
        parts = [p for p in (year, month, day) if p]
        if parts:
            return "-".join(parts)
        if medline:
            return medline
    return ""


def _find_authors(article: ET.Element) -> list[Author]:
    authors: list[Author] = []
    for idx, author in enumerate(article.findall(".//Article/AuthorList/Author"), start=1):
        if author.find("CollectiveName") is not None:
            continue
        last_name = _find_text(author, "LastName")
        fore_name = _find_text(author, "ForeName")
        initials = _find_text(author, "Initials")
        affs = _normalize_affiliation_blocks(
            [
                "".join(a.itertext())
                for a in author.findall("AffiliationInfo/Affiliation")
            ]
        )
        authors.append(
            Author(
                position=idx,
                last_name=last_name,
                fore_name=fore_name,
                initials=initials,
                affiliation="; ".join(affs),
                affiliation_blocks=affs,
                orcid=_extract_pubmed_author_orcid(author),
            )
        )
    # Legacy PubMed records may expose one shared affiliation either at article-level or
    # in one author's AffiliationInfo without explicit links for the remaining authors.
    # If exactly one unique affiliation block is present for the citation, apply it to
    # missing authors.
    author_level_affiliations = {
        block
        for author in authors
        for block in _author_affiliation_blocks(author)
    }
    article_affiliations = {
        block
        for block in _normalize_affiliation_blocks(
            [
                "".join(node.itertext())
                for node in article.findall(".//Article/Affiliation")
            ]
        )
    }
    shared_affiliations = author_level_affiliations | article_affiliations
    if len(shared_affiliations) == 1:
        only_affiliation = [next(iter(shared_affiliations))]
        for parsed_author in authors:
            if not _author_has_affiliation(parsed_author):
                _set_author_affiliations(parsed_author, only_affiliation)
    return authors




def _extract_pubmed_author_orcid(author: ET.Element) -> str:
    identifier_nodes = list(author.findall("Identifier")) + list(author.findall("AuthorIdentifier"))
    for node in identifier_nodes:
        source = (node.attrib.get("Source") or node.attrib.get("source") or "").strip().lower()
        text = normalize_orcid("".join(node.itertext()))
        if not text:
            continue
        if source in {"", "orcid"}:
            return text

    for node in identifier_nodes:
        text = normalize_orcid("".join(node.itertext()))
        if text:
            return text

    return ""

def _normalize_doi(value: str) -> str:
    raw = _clean_text(value)
    if not raw:
        return ""
    raw = DOI_URL_PREFIX.sub("", raw)
    raw = re.sub(r"^doi:\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.strip().strip(".,;")
    return raw


def _extract_pubmed_article_doi(article: ET.Element) -> str:
    for node in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = (node.attrib.get("IdType") or node.attrib.get("idtype") or "").strip().lower()
        if id_type != "doi":
            continue
        doi = _normalize_doi("".join(node.itertext()))
        if doi:
            return doi

    for node in article.findall(".//Article/ELocationID"):
        id_type = (node.attrib.get("EIdType") or node.attrib.get("eidtype") or "").strip().lower()
        if id_type != "doi":
            continue
        doi = _normalize_doi("".join(node.itertext()))
        if doi:
            return doi

    return ""


def _extract_pmc_article_doi(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    for node in root.findall(".//front/article-meta/article-id"):
        pub_id_type = (node.attrib.get("pub-id-type") or "").strip().lower()
        if pub_id_type != "doi":
            continue
        doi = _normalize_doi("".join(node.itertext()))
        if doi:
            return doi
    return ""


def _parse_pmc_xml_by_pmcid(xml_text: str, *, requested_pmcids: list[str]) -> dict[str, str]:
    if not xml_text.strip():
        return {}

    root = ET.fromstring(xml_text)
    articles = list(_iter_pmc_article_nodes(root))
    if not articles:
        return {}

    xml_by_pmcid: dict[str, str] = {}
    requested = [_normalize_pmcid(value) for value in requested_pmcids if _normalize_pmcid(value)]
    if len(articles) == len(requested):
        for article, requested_pmcid in zip(articles, requested, strict=True):
            pmcid = _extract_pmc_article_pmcid(article) or requested_pmcid
            xml_by_pmcid[pmcid] = ET.tostring(article, encoding="unicode")
        return xml_by_pmcid

    for article in articles:
        pmcid = _extract_pmc_article_pmcid(article)
        if pmcid:
            xml_by_pmcid[pmcid] = ET.tostring(article, encoding="unicode")
    return xml_by_pmcid


def _iter_pmc_article_nodes(root: ET.Element) -> list[ET.Element]:
    if _local_tag_name(root.tag) == "article":
        return [root]
    return [node for node in root.iter() if _local_tag_name(node.tag) == "article"]


def _extract_pmc_article_pmcid(article: ET.Element) -> str:
    for node in article.iter():
        if _local_tag_name(node.tag) != "article-id":
            continue
        pub_id_type = (node.attrib.get("pub-id-type") or "").strip().lower()
        if pub_id_type not in {"pmc", "pmcid"}:
            continue
        pmcid = _normalize_pmcid("".join(node.itertext()))
        if pmcid:
            return pmcid
    return ""


def _local_tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_pmcid(value: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9]+", "", value).upper()
    if not raw:
        return ""
    if raw.startswith("PMC"):
        return raw
    if raw.isdigit():
        return f"PMC{raw}"
    return raw


def _pmc_query_id(value: str) -> str:
    return _normalize_pmcid(value).removeprefix("PMC")


def _extract_pubmed_update_relationships(article: ET.Element) -> tuple[set[str], set[str]]:
    update_in_pmids: set[str] = set()
    update_of_pmids: set[str] = set()

    for node in article.findall(".//CommentsCorrectionsList/CommentsCorrections"):
        ref_type = (node.attrib.get("RefType") or node.attrib.get("reftype") or "").strip().lower()
        related_pmid = _clean_text(_find_text(node, "PMID"))
        if not related_pmid:
            continue
        if ref_type == "updatein":
            update_in_pmids.add(related_pmid)
        elif ref_type == "updateof":
            update_of_pmids.add(related_pmid)

    return update_in_pmids, update_of_pmids


def _find_publication_types(article: ET.Element) -> list[str]:
    values: list[str] = []
    for node in article.findall(".//Article/PublicationTypeList/PublicationType"):
        text = _clean_text("".join(node.itertext()))
        if text:
            values.append(text)
    # Preserve order while deduplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _extract_pmc_affiliation_maps(context: ET.Element) -> tuple[dict[str, str], dict[str, str]]:
    by_id: dict[str, str] = {}
    by_label: dict[str, str] = {}
    for aff in context.findall(".//aff"):
        aff_id = (aff.attrib.get("id") or aff.attrib.get(XML_NS_ID) or "").strip()
        label = _clean_text(_find_text(aff, "label"))
        text = _normalize_affiliation_text("".join(aff.itertext()))
        if text:
            if aff_id and aff_id not in by_id:
                by_id[aff_id] = text
            if label:
                normalized_label = _normalize_label(label)
                if normalized_label and normalized_label not in by_label:
                    by_label[normalized_label] = text
    return by_id, by_label


def _extract_pmc_note_text_by_id(root: ET.Element) -> dict[str, str]:
    by_id: dict[str, str] = {}
    paths = (
        ".//front/article-meta/author-notes/fn",
        ".//fn-group/fn",
        ".//front/article-meta/author-notes/corresp",
        ".//corresp-group/corresp",
    )
    for path in paths:
        for node in root.findall(path):
            node_id = (node.attrib.get("id") or node.attrib.get(XML_NS_ID) or "").strip()
            if not node_id:
                continue
            text = _normalize_affiliation_text("".join(node.itertext()))
            if text:
                by_id[node_id] = text
    return by_id


def _extract_pmc_contrib_note_text(root: ET.Element) -> dict[str, str]:
    by_id: dict[str, str] = {}
    paths = (
        ".//front/article-meta/author-notes/fn",
        ".//fn-group/fn",
        ".//front/article-meta/author-notes/corresp",
        ".//corresp-group/corresp",
    )
    for path in paths:
        for node in root.findall(path):
            node_id = (node.attrib.get("id") or node.attrib.get(XML_NS_ID) or "").strip()
            if not node_id:
                continue
            text = " ".join("".join(node.itertext()).split()).lower()
            if text:
                by_id[node_id] = text
    return by_id


def _is_prefix_positions(positions: list[int]) -> bool:
    if not positions:
        return False
    return positions == list(range(1, positions[-1] + 1))


def _is_suffix_positions(positions: list[int], author_count: int) -> bool:
    if not positions or author_count <= 0:
        return False
    return positions == list(range(positions[0], author_count + 1))




def _extract_pmc_contrib_orcid(contrib: ET.Element) -> str:
    for node in contrib.findall(".//contrib-id"):
        contrib_type = (node.attrib.get("contrib-id-type") or "").strip().lower()
        if contrib_type != "orcid":
            continue
        value = normalize_orcid("".join(node.itertext()))
        if value:
            return value
        for attr_name, attr_value in node.attrib.items():
            if attr_name.lower().endswith("href"):
                value = normalize_orcid(attr_value)
                if value:
                    return value

    for attr_name, attr_value in contrib.attrib.items():
        if attr_name.lower().endswith("href"):
            value = normalize_orcid(attr_value)
            if value:
                return value

    return ""

def _extract_contrib_affiliation(
    contrib: ET.Element,
    *,
    aff_by_id: dict[str, str],
    aff_by_label: dict[str, str],
    note_by_id: dict[str, str],
) -> str:
    return "; ".join(
        _extract_contrib_affiliation_blocks(
            contrib,
            aff_by_id=aff_by_id,
            aff_by_label=aff_by_label,
            note_by_id=note_by_id,
        )
    )


def _extract_contrib_affiliation_blocks(
    contrib: ET.Element,
    *,
    aff_by_id: dict[str, str],
    aff_by_label: dict[str, str],
    note_by_id: dict[str, str],
) -> list[str]:
    direct_collected: list[str] = []
    rid_collected: list[str] = []

    # Some JATS records include inline affiliation text in <aff> under each contributor.
    for aff in contrib.findall("./aff"):
        text = _normalize_affiliation_text("".join(aff.itertext()))
        if text:
            direct_collected.append(text)

    aff_xrefs = contrib.findall(".//xref[@ref-type='aff']")
    has_aff_xref = bool(aff_xrefs)
    for xref in contrib.findall(".//xref[@ref-type='aff']"):
        for rid in _split_reference_ids(xref.attrib.get("rid", "")):
            if rid in aff_by_id:
                rid_collected.append(aff_by_id[rid])
    if rid_collected:
        direct_collected.extend(rid_collected)
    # Only use label-based affiliation lookup when explicit RID links are absent.
    elif has_aff_xref:
        for xref in aff_xrefs:
            for label in _split_label_tokens("".join(xref.itertext())):
                match = aff_by_label.get(_normalize_label(label))
                if match:
                    direct_collected.append(match)

    # Superscript-label fallback is only used when no xref-based affiliation was recovered.
    if not direct_collected and not has_aff_xref:
        for sup in contrib.findall(".//sup"):
            for label in _split_label_tokens("".join(sup.itertext())):
                match = aff_by_label.get(_normalize_label(label))
                if match:
                    direct_collected.append(match)

    # Precision-first: only fall back to fn/corresp note text when no direct
    # author->affiliation linkage is available.
    if direct_collected:
        return _normalize_affiliation_blocks(direct_collected)

    note_collected: list[str] = []
    for xref in contrib.findall(".//xref[@ref-type='fn']") + contrib.findall(".//xref[@ref-type='corresp']"):
        for rid in _split_reference_ids(xref.attrib.get("rid", "")):
            note_text = note_by_id.get(rid, "")
            if _looks_like_affiliation_text(note_text):
                note_collected.append(note_text)

    return _normalize_affiliation_blocks(note_collected)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _split_affiliation_blocks(value: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"[;|]+", value) if piece.strip()]


def _normalize_affiliation_blocks(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for piece in _split_affiliation_blocks(value):
            cleaned = _normalize_affiliation_text(piece)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _author_affiliation_blocks(author: Author) -> list[str]:
    explicit = _normalize_affiliation_blocks(getattr(author, "affiliation_blocks", []))
    if explicit:
        return explicit
    return _normalize_affiliation_blocks([author.affiliation or ""])


def _author_has_affiliation(author: Author) -> bool:
    return bool(_author_affiliation_blocks(author))


def _set_author_affiliations(author: Author, blocks: Iterable[str]) -> None:
    normalized_blocks = _normalize_affiliation_blocks(blocks)
    author.affiliation_blocks = normalized_blocks
    author.affiliation = "; ".join(normalized_blocks)


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _person_token(value: str) -> str:
    return PERSON_TOKEN.sub("", value.lower())


def _authors_likely_same_person(left: Author, right: Author) -> bool:
    left_last = _person_token(left.last_name)
    right_last = _person_token(right.last_name)
    if left_last and right_last and left_last != right_last:
        return False
    left_initials = (_person_token(left.initials) or _person_token(_initials_from_name(left.fore_name)))[:1]
    right_initials = (_person_token(right.initials) or _person_token(_initials_from_name(right.fore_name)))[:1]
    if left_initials and right_initials and left_initials != right_initials:
        return False
    return True


def _normalize_affiliation_text(value: str) -> str:
    cleaned = _clean_text(value)
    cleaned = cleaned.translate(SUPERSCRIPT_TRANSLATION)
    cleaned = _normalize_spaced_ror_urls(cleaned)
    # Insert boundary after the fixed-width 9-character ROR identifier when
    # publisher/source text concatenates the next token without a delimiter
    # (e.g. ".../03zjqec80HSS", ".../03vek6s52grid.38142.3c").
    cleaned = re.sub(
        r"(https?://(?:www\.)?ror\.org/[0-9a-z]{9})(?=[^\W_])",
        r"\1 ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = _strip_affiliation_identifiers(cleaned)
    cleaned = _strip_leading_numeric_marker_clusters(cleaned)
    cleaned = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", cleaned)
    cleaned = _strip_segment_prefix_markers(cleaned)
    cleaned = re.sub(
        r"((?i:\bgrid\.[a-z0-9]+\.[a-z0-9]+)\s+)(?:[a-z]|X)(?=[A-Z])\s*",
        r"\1",
        cleaned,
    )
    # Common legacy formatting artifacts: "1Department", "and2 Department".
    cleaned = re.sub(r"\band(?=\d)", "and ", cleaned, flags=re.IGNORECASE)
    # Avoid splitting digit+alpha tokens in URL paths (e.g. ROR IDs).
    cleaned = re.sub(r"(?<![A-Za-z0-9/.])(\d+)(?=[A-Za-z])", r"\1 ", cleaned)
    cleaned = _strip_numeric_footnote_markers(cleaned)
    # Remove numeric affiliation markers that are typically superscript references.
    cleaned = re.sub(r"(^|[;,\(\[]\s*|\band\s+)\d+\s+(?=[A-Za-z])", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(\d+)\s+(st|nd|rd|th)\b", r"\1\2", cleaned, flags=re.IGNORECASE)
    cleaned = _clean_text(cleaned)
    cleaned = _strip_segment_prefix_markers(cleaned)
    cleaned = _clean_text(cleaned)
    cleaned = _collapse_merged_department_markers(cleaned)
    cleaned = _strip_embedded_numbered_affiliation_tail(cleaned)
    return cleaned


def _initials_from_name(value: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z]+", value) if part]
    return "".join(part[0].upper() for part in parts)


def _split_reference_ids(value: str) -> list[str]:
    tokens = [token.strip() for token in re.split(r"[,\s;]+", value) if token.strip()]
    return tokens


def _split_label_tokens(value: str) -> list[str]:
    tokens = [token.strip() for token in re.split(r"[,\s;]+", value) if token.strip()]
    return tokens


def _normalize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


def _normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _contains_any_phrase(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_phrase(text)
    if not normalized:
        return False
    return any(_normalize_phrase(pattern) in normalized for pattern in patterns)


def _looks_like_affiliation_text(value: str) -> bool:
    lowered = value.lower()
    if _contains_any_phrase(lowered, EQUAL_PATTERNS):
        return False
    if _contains_any_phrase(lowered, SENIOR_PATTERNS):
        return False
    return any(pattern in lowered for pattern in AFFILIATION_HINT_PATTERNS)


def _strip_affiliation_identifiers(value: str) -> str:
    cleaned = value
    grid_placeholders: dict[str, str] = {}
    ror_placeholders: dict[str, str] = {}

    def _protect_grid(match: re.Match[str]) -> str:
        token = _normalize_grid_token(match.group(0))
        key = f"__AFF_GRID_{len(grid_placeholders)}__"
        grid_placeholders[key] = token
        return key

    def _protect_ror(match: re.Match[str]) -> str:
        token = match.group(0)
        key = f"__AFF_ROR_{len(ror_placeholders)}__"
        ror_placeholders[key] = token
        return key

    cleaned = re.sub(r"https?://(?:www\.)?ror\.org/[0-9a-z]{9}", _protect_ror, cleaned, flags=re.IGNORECASE)
    # Accept GRID tokens even when prefixed by numeric footnote markers
    # (e.g. "1grid.239915.50000..."), so we preserve the real GRID value.
    cleaned = re.sub(
        r"(?<![A-Za-z])grid\s*\.\s*[a-z0-9]+\.[a-z0-9]+\b",
        _protect_grid,
        cleaned,
        flags=re.IGNORECASE,
    )
    # Keep ROR/GRID identifiers so downstream record-level matching can prioritize
    # them over free-text aliases (match_record: ROR/GRID > email > text).
    cleaned = re.sub(r"\b(?:orcid|orcid id)[:\s]*\d{4}(?:[-\s]\d{4}){2,3}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4}(?:\s+\d{4}){2,3}(?=\b|[A-Za-z])", " ", cleaned)
    # Drop compact identifier fragments stuck onto institution names (e.g. zjqec80HSS).
    cleaned = re.sub(r"\b[0-9a-z]{7,}(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"\b\d+(?:\.\d+)+\b", " ", cleaned)
    cleaned = re.sub(r"(^|;)\s*[.,]+\s*", r"\1 ", cleaned)
    for key, token in ror_placeholders.items():
        cleaned = cleaned.replace(key, token)
    for key, token in grid_placeholders.items():
        cleaned = cleaned.replace(key, token)
    return cleaned


def _normalize_grid_token(value: str) -> str:
    match = re.search(r"grid\s*\.\s*([A-Za-z0-9]+)\.([A-Za-z0-9]+)", value, flags=re.IGNORECASE)
    if match is None:
        return value
    segment = match.group(1).lower()
    suffix_raw = match.group(2)
    suffix = suffix_raw.lower()
    attached_text = ""
    # Publisher feeds often concatenate extra numeric IDs after the true GRID
    # suffix (e.g. ".3c000000041936754", ".370000"). Trim those tails.
    tail_match = re.fullmatch(r"([A-Za-z0-9]{1,2})0{3,}[0-9]*(?:X)?([A-Z][A-Za-z].*)?", suffix_raw)
    if tail_match is not None:
        compact = tail_match.group(1).lower().rstrip("0")
        suffix = compact or tail_match.group(1)[0].lower()
        attached = (tail_match.group(2) or "").strip()
        if attached:
            attached_text = f" {attached}"
    return f"grid.{segment}.{suffix}{attached_text}"


def _normalize_spaced_ror_urls(value: str) -> str:
    def _collapse(match: re.Match[str]) -> str:
        prefix = match.group(1)
        spaced_id = match.group(2)
        compact_id = re.sub(r"\s+", "", spaced_id)
        return f"{prefix}{compact_id}"

    return re.sub(
        r"(https?://(?:www\.)?ror\.org/)\s*(0(?:\s*[0-9a-z]){8})",
        _collapse,
        value,
        flags=re.IGNORECASE,
    )


def _strip_segment_prefix_markers(value: str) -> str:
    # Remove single-letter segment markers (e.g. aHospital, XWeill) only at segment starts.
    # Restrict to lowercase letters and uppercase X marker to avoid stripping real acronyms (e.g. HSS).
    return re.sub(r"(^|[;,.])\s*(?:[a-z]|X)\s*(?=[A-Z])", r"\1 ", value)


def _strip_leading_numeric_marker_clusters(value: str) -> str:
    # Drop footnote-like leading numeric marker clusters at the start of the
    # affiliation or after semicolon-separated segments, e.g.:
    # "1 .38142. Transplantation Research Center, ..."
    # "[2] Department of Medicine, ..."
    cleaned = re.sub(
        r"(^|;)\s*(?:\[\s*\d+\s*\]\s*|(?:\d+\s*[.)\]:-]?\s*)+)(?=[A-Za-z])",
        r"\1 ",
        value,
    )
    return cleaned


def _strip_numeric_footnote_markers(value: str) -> str:
    target_words = "|".join(AFFILIATION_FOOTNOTE_TARGET_WORDS)
    cleaned = value
    pattern = re.compile(
        rf"(^|[;,\(\[]\s*|\band\s+)"
        rf"(?:\d+\s*[,/&]\s*)*\d+\s*(?:[.)\]:-]\s*)?"
        rf"(?=(?:{target_words})\b)",
        flags=re.IGNORECASE,
    )
    # A short fixed-point loop handles stacked markers like "1,2 Department".
    for _ in range(3):
        updated = pattern.sub(r"\1", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _strip_embedded_numbered_affiliation_tail(value: str) -> str:
    # Some legacy strings embed multiple numbered affiliations into one text blob,
    # e.g. "... Columbia University, USA. 2 Department of ... Albert Einstein ..."
    # Keep the first segment for precision-focused attribution.
    marker = re.search(
        r"([.;,])\s+\d+\s+(?=(departments?|division|institute|university|school|college|hospital|laboratory|center|centre)\b)",
        value,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return value
    trimmed = value[: marker.start()].strip()
    return trimmed.rstrip(";,")


def _collapse_merged_department_markers(value: str) -> str:
    # Precision-first cleanup for merged numbered affiliations like:
    # "Department of A, and Department of B, Institute ..."
    # Keep first department and shared institution tail.
    match = re.match(
        r"^(Department of [^,]+),\s+and Department of [^,]+,\s+(.+)$",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return value
    first_department = match.group(1).strip()
    institution_tail = match.group(2).strip()
    return f"{first_department}, {institution_tail}"


def _augment_params(
    *,
    url: str,
    params: dict[str, str],
    api_key: str | None,
    tool: str | None,
    email: str | None,
) -> dict[str, str]:
    out = {key: str(value) for key, value in params.items()}
    if api_key:
        out.setdefault("api_key", api_key)
    if url.startswith(EUTILS_BASE):
        if tool:
            out.setdefault("tool", tool)
        if email:
            out.setdefault("email", email)
    return out


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after.isdigit():
        return max(0.5, float(retry_after))
    # Exponential backoff with cap.
    return min(8.0, 0.6 * (2**attempt))


def split_chunks(values: Iterable[str], chunk_size: int = 200) -> list[list[str]]:
    chunk: list[str] = []
    chunks: list[list[str]] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= chunk_size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def _idconv_records(payload: dict) -> list[dict]:
    records = payload.get("records")
    if isinstance(records, list):
        return [r for r in records if isinstance(r, dict)]

    articles = payload.get("articles")
    if isinstance(articles, list):
        return [a for a in articles if isinstance(a, dict)]

    response = payload.get("response")
    if isinstance(response, dict):
        nested = response.get("records")
        if isinstance(nested, list):
            return [r for r in nested if isinstance(r, dict)]

    return []


def _build_litsense_author_query(*, seed_pmid: str, author_name: str) -> str:
    seed = seed_pmid.strip()
    if not seed:
        return ""
    try:
        target = parse_target_name(author_name)
    except ValueError:
        fallback = author_name.strip()
        return " ".join(part for part in (seed, fallback) if part)

    surname = target.surname.strip()
    first_initial = target.initials[:1].upper()
    if not surname or not first_initial:
        fallback = author_name.strip()
        return " ".join(part for part in (seed, fallback) if part)
    surname_formatted = f"{surname[:1].upper()}{surname[1:]}"
    return f"{seed} {surname_formatted} {first_initial}"


def _extract_litsense_pmids(payload: Any) -> set[str]:
    pmids: set[str] = set()
    _collect_litsense_pmids(payload, pmids)
    return pmids


def _collect_litsense_pmids(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).strip().lower()
            if lowered in {"pmid", "pmids"}:
                out.update(_coerce_litsense_pmids(value))
            else:
                _collect_litsense_pmids(value, out)
        return
    if isinstance(node, list):
        for item in node:
            _collect_litsense_pmids(item, out)


def _coerce_litsense_pmids(value: Any) -> set[str]:
    pmids: set[str] = set()
    if isinstance(value, int):
        if value > 0:
            pmids.add(str(value))
        return pmids
    if isinstance(value, str):
        for token in re.findall(r"\d+", value):
            if token:
                pmids.add(token)
        return pmids
    if isinstance(value, list):
        for item in value:
            pmids.update(_coerce_litsense_pmids(item))
        return pmids
    if isinstance(value, dict):
        _collect_litsense_pmids(value, pmids)
    return pmids
