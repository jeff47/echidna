from __future__ import annotations

import html
import logging
import os
import re
import time
from typing import Iterable
from xml.etree import ElementTree as ET

import httpx

from app.logic import parse_year
from app.models import Author, Citation

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"

EQUAL_PATTERNS = (
    "contributed equally",
    "equal contribution",
    "equally contributed",
    "co-first",
    "co first",
)
SENIOR_PATTERNS = (
    "co-senior",
    "co senior",
    "jointly supervised",
    "joint supervision",
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

logger = logging.getLogger(__name__)


class NcbiClient:
    def __init__(self, timeout: float = 30.0, api_key: str | None = None, max_retries: int = 4) -> None:
        self.timeout = timeout
        self.api_key = (api_key or os.getenv("NCBI_API_KEY", "")).strip() or None
        self.email = os.getenv("NCBI_EMAIL", "").strip() or None
        self.tool = os.getenv("NCBI_TOOL", "pmcparser").strip() or "pmcparser"
        self.max_retries = max_retries
        self.min_interval_seconds = 0.11 if self.api_key else 0.34
        self._next_request_at = 0.0

    def fetch_citations(self, pmids: list[str]) -> list[Citation]:
        if not pmids:
            return []

        logger.info("Fetching PubMed citations for %d PMID(s)", len(pmids))
        pubmed_xml = self._fetch_pubmed_xml(pmids)
        citations = self._parse_pubmed(pubmed_xml)
        logger.info("Parsed %d citation(s) from PubMed", len(citations))
        try:
            pmcid_map = self._fetch_pmcid_map(pmids)
            logger.info("Mapped %d/%d PMID(s) to PMCID", len(pmcid_map), len(pmids))
        except Exception as exc:  # noqa: BLE001
            pmcid_map = {}
            logger.warning("PMCID mapping request failed: %s", exc)
            for citation in citations:
                citation.notes.append(f"PMCID mapping failed: {exc}")

        for citation in citations:
            citation.pmcid = pmcid_map.get(citation.pmid)
            if not citation.pmcid:
                citation.notes.append("No PMCID mapping")
                logger.debug("PMID %s has no PMCID mapping", citation.pmid)
                continue
            try:
                pmc_xml = self._fetch_pmc_xml(citation.pmcid)
                co_first, co_senior, pmc_authors, pmc_count = self._extract_pmc_enrichment(pmc_xml)
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
                    filled = 0
                else:
                    pmc_affiliations = {
                        author.position: author.affiliation for author in pmc_authors if author.affiliation.strip()
                    }
                    filled = self._fill_missing_affiliations(citation, pmc_affiliations)
                    if filled:
                        citation.notes.append(f"Filled {filled} affiliation(s) from PMC")
                    citation.notes.append("PMC author metadata incomplete; kept PubMed author list")
                citation.source_for_roles = "pmc"
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
                logger.warning("PMID %s PMC fetch/parse failed: %s", citation.pmid, exc)

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
        pmcid_clean = pmcid.replace("PMC", "")
        params = {
            "db": "pmc",
            "id": pmcid_clean,
            "retmode": "xml",
        }
        url = f"{EUTILS_BASE}/efetch.fcgi"
        response = self._request(url, params=params)
        return response.text

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

            title = _find_text(article, ".//Article/ArticleTitle")
            title = html.unescape(re.sub(r"\s+", " ", title)).strip()

            journal = _find_text(article, ".//Article/Journal/ISOAbbreviation") or _find_text(
                article,
                ".//Article/Journal/Title",
            )
            journal = journal.strip() or "(unknown journal)"

            pub_date = _find_pub_date(article)
            authors = _find_authors(article)

            citations.append(
                Citation(
                    pmid=pmid,
                    pmcid=None,
                    title=title,
                    journal=journal,
                    print_year=parse_year(pub_date),
                    print_date=pub_date,
                    authors=authors,
                    source_for_roles="pubmed",
                )
            )

        return citations

    def _extract_pmc_enrichment(
        self,
        xml_text: str,
    ) -> tuple[set[int], set[int], list[Author], int]:
        root = ET.fromstring(xml_text)
        contribs = root.findall(".//front/article-meta/contrib-group/contrib[@contrib-type='author']")
        if not contribs:
            contribs = root.findall(".//contrib-group/contrib[@contrib-type='author']")
        aff_by_id, aff_by_label = _extract_pmc_affiliation_maps(root)
        note_by_id = _extract_pmc_note_text_by_id(root)

        fn_text: dict[str, str] = {}
        for fn in root.findall(".//front/article-meta/author-notes/fn") + root.findall(".//fn-group/fn"):
            fn_id = fn.attrib.get("id")
            if not fn_id:
                continue
            text = " ".join("".join(fn.itertext()).split()).lower()
            fn_text[fn_id] = text

        first_fn_ids: set[str] = set()
        senior_fn_ids: set[str] = set()

        for fn_id, text in fn_text.items():
            has_senior_pattern = any(p in text for p in SENIOR_PATTERNS)
            has_equal_pattern = any(p in text for p in EQUAL_PATTERNS)
            if has_senior_pattern:
                senior_fn_ids.add(fn_id)
            elif has_equal_pattern:
                first_fn_ids.add(fn_id)

        co_first_positions: set[int] = set()
        co_senior_positions: set[int] = set()
        authors: list[Author] = []

        for idx, contrib in enumerate(contribs, start=1):
            ref_ids = {
                x.attrib.get("rid", "")
                for x in contrib.findall(".//xref")
                if x.attrib.get("rid")
            }
            if first_fn_ids & ref_ids:
                co_first_positions.add(idx)
            if senior_fn_ids & ref_ids:
                co_senior_positions.add(idx)

            last_name = _find_text(contrib, ".//name/surname") or _find_text(contrib, ".//surname")
            fore_name = _find_text(contrib, ".//name/given-names") or _find_text(contrib, ".//given-names")
            if not last_name:
                # Fall back to collaborative/string names when person name markup is absent.
                last_name = _find_text(contrib, ".//collab") or _find_text(contrib, ".//string-name")
            affiliation = _extract_contrib_affiliation(
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
                    affiliation=affiliation,
                )
            )

        return co_first_positions, co_senior_positions, authors, len(contribs)

    def _fill_missing_affiliations(self, citation: Citation, pmc_affiliations: dict[int, str]) -> int:
        filled = 0
        for author in citation.authors:
            if author.affiliation.strip():
                continue
            pmc_aff = pmc_affiliations.get(author.position, "").strip()
            if not pmc_aff:
                continue
            author.affiliation = pmc_aff
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


def _find_text(node: ET.Element, path: str) -> str:
    target = node.find(path)
    if target is None or target.text is None:
        return ""
    return target.text


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
        affs = [
            " ".join("".join(a.itertext()).split())
            for a in author.findall("AffiliationInfo/Affiliation")
            if "".join(a.itertext()).strip()
        ]
        affiliation = "; ".join(affs)
        authors.append(
            Author(
                position=idx,
                last_name=last_name,
                fore_name=fore_name,
                initials=initials,
                affiliation=affiliation,
            )
        )
    # Legacy PubMed records may expose one shared affiliation either at article-level or
    # in one author's AffiliationInfo without explicit links for the remaining authors.
    # If exactly one unique affiliation is present for the citation, apply it to missing authors.
    author_level_affiliations = {
        author.affiliation.strip()
        for author in authors
        if author.affiliation.strip()
    }
    article_affiliations = {
        _clean_text("".join(node.itertext()))
        for node in article.findall(".//Article/Affiliation")
        if _clean_text("".join(node.itertext()))
    }
    shared_affiliations = author_level_affiliations | article_affiliations
    if len(shared_affiliations) == 1:
        only_affiliation = next(iter(shared_affiliations))
        for author in authors:
            if not author.affiliation.strip():
                author.affiliation = only_affiliation
    return authors


def _extract_pmc_affiliation_maps(root: ET.Element) -> tuple[dict[str, str], dict[str, str]]:
    by_id: dict[str, str] = {}
    by_label: dict[str, str] = {}
    for aff in root.findall(".//front/article-meta/aff") + root.findall(".//aff"):
        aff_id = (aff.attrib.get("id") or aff.attrib.get(XML_NS_ID) or "").strip()
        label = _clean_text(_find_text(aff, "label"))
        text = _clean_text("".join(aff.itertext()))
        if text:
            if aff_id:
                by_id[aff_id] = text
            if label:
                by_label[_normalize_label(label)] = text
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
            text = _clean_text("".join(node.itertext()))
            if text:
                by_id[node_id] = text
    return by_id


def _extract_contrib_affiliation(
    contrib: ET.Element,
    *,
    aff_by_id: dict[str, str],
    aff_by_label: dict[str, str],
    note_by_id: dict[str, str],
) -> str:
    collected: list[str] = []

    # Some JATS records include inline affiliation text in <aff> under each contributor.
    for aff in contrib.findall("./aff"):
        text = _clean_text("".join(aff.itertext()))
        if text:
            collected.append(text)

    for xref in contrib.findall(".//xref[@ref-type='aff']"):
        for rid in _split_reference_ids(xref.attrib.get("rid", "")):
            if rid in aff_by_id:
                collected.append(aff_by_id[rid])
        for label in _split_label_tokens("".join(xref.itertext())):
            match = aff_by_label.get(_normalize_label(label))
            if match:
                collected.append(match)

    for xref in contrib.findall(".//xref[@ref-type='fn']") + contrib.findall(".//xref[@ref-type='corresp']"):
        for rid in _split_reference_ids(xref.attrib.get("rid", "")):
            note_text = note_by_id.get(rid, "")
            if _looks_like_affiliation_text(note_text):
                collected.append(note_text)

    for sup in contrib.findall(".//sup"):
        for label in _split_label_tokens("".join(sup.itertext())):
            match = aff_by_label.get(_normalize_label(label))
            if match:
                collected.append(match)

    return "; ".join(_dedupe_preserve_order(collected))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _clean_text(value: str) -> str:
    return " ".join(value.split())


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


def _looks_like_affiliation_text(value: str) -> bool:
    lowered = value.lower()
    if any(pattern in lowered for pattern in EQUAL_PATTERNS):
        return False
    if any(pattern in lowered for pattern in SENIOR_PATTERNS):
        return False
    return any(pattern in lowered for pattern in AFFILIATION_HINT_PATTERNS)


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
