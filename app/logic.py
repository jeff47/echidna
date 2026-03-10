from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from affiliation_normalizer import match_affiliation, match_record

from app.models import Author, AuthorMatch, Citation, Cluster, ReportRow, TargetName


NON_ALPHA_NUM = re.compile(r"[^a-z0-9]+")
FOUR_DIGIT_YEAR = re.compile(r"(19|20)\d{2}")
ORCID_CANONICAL = re.compile(r"(\d{4})-?(\d{4})-?(\d{4})-?(\d{3}[\dXx])")
ORCID_COMPACT = re.compile(r"^\d{15}[\dXx]$")
EXCLUDED_TYPE_KEYWORDS = (
    "preprint",
    "editorial",
    "comment",
    "letter",
    "news",
    "newspaper article",
    "published erratum",
    "retraction of publication",
    "retracted publication",
    "expression of concern",
    "corrected and republished article",
    "patient education handout",
)
UNIT_PREFIX_PATTERN = re.compile(
    r"^(department|departments|division|center|centre|laboratory|lab|program|section|unit|faculty)\b",
    flags=re.IGNORECASE,
)
TRAILING_PUNCT = re.compile(r"[.;,\s]+$")
AMPERSAND = re.compile(r"\s*&\s*")
APOSTROPHE_VARIANTS = re.compile(r"[’`´]")
AFFILIATION_PIECE_SPLIT = re.compile(r"[;|]+")
AFFILIATION_AND_SPLIT = re.compile(r"\band\b", flags=re.IGNORECASE)
AFFILIATION_AND_DELIMITER_SPLIT = re.compile(r"\band\b(?!\s+[A-Za-z]+'s\b)", flags=re.IGNORECASE)
INSTITUTION_HINTS = (
    "university",
    "hospital",
    "institute",
    "college",
    "school",
    "clinic",
    "health",
)
SECONDARY_INSTITUTION_HINTS = (
    "medicine",
    "foundation",
)
UNIT_WORDS = (
    "department",
    "departments",
    "division",
    "center",
    "centre",
    "laboratory",
    "lab",
    "program",
    "section",
    "unit",
)
AFFILIATION_ANCHOR_WORDS = {
    "clinic",
    "college",
    "foundation",
    "hospital",
    "institute",
    "institutes",
    "school",
    "system",
    "university",
}
ORG_SUFFIX_WORDS = {"inc", "incorporated", "ltd", "llc", "corp", "corporation", "gmbh", "sa", "ag", "bv", "plc"}
UNIT_ONLY_WORDS = {
    "center",
    "centre",
    "department",
    "departments",
    "division",
    "faculty",
    "lab",
    "laboratory",
    "program",
    "section",
    "unit",
}
US_STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "ia", "id", "il", "in", "ks",
    "ky", "la", "ma", "md", "me", "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj", "nm", "nv",
    "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "va", "vt", "wa", "wi", "wv", "wy",
}
GEO_TOKENS = {
    "country",
    "state",
    "city",
    "county",
    "province",
    "usa",
    "us",
    "united",
    "states",
    "massachusetts",
    "boston",
}
POSTAL_CODE_TOKEN = re.compile(r"^\d{5}(?:-\d{4})?$")
AFFILIATION_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
AFFILIATION_ROR_URL_RE = re.compile(r"https?://(?:www\.)?ror\.org/[0-9a-z]{9}", flags=re.IGNORECASE)
AFFILIATION_ROR_HOST_RE = re.compile(r"\bror\.org/[0-9a-z]{9}\b", flags=re.IGNORECASE)
AFFILIATION_ROR_SPACED_URL_RE = re.compile(
    r"https?://(?:www\.)?ror\.org/\s*0(?:\s*[0-9a-z]){8}",
    flags=re.IGNORECASE,
)
AFFILIATION_ROR_SPACED_HOST_RE = re.compile(
    r"\bror\.org/\s*0(?:\s*[0-9a-z]){8}",
    flags=re.IGNORECASE,
)
AFFILIATION_ROR_PREFIX_RE = re.compile(
    r"\bror(?:\s*id)?\s*[:#-]?\s*(0(?:\s*[0-9a-z]){8})\b",
    flags=re.IGNORECASE,
)
AFFILIATION_GRID_RE = re.compile(r"\bgrid\s*\.\s*[a-z0-9]+\.[a-z0-9]+\b", flags=re.IGNORECASE)
GENERIC_INSTITUTION_TOKENS = {
    "and",
    "at",
    "center",
    "centre",
    "college",
    "department",
    "departments",
    "division",
    "faculty",
    "for",
    "health",
    "hospital",
    "in",
    "institute",
    "institutes",
    "lab",
    "laboratory",
    "medical",
    "medicine",
    "of",
    "program",
    "school",
    "state",
    "system",
    "the",
    "university",
    "unit",
}
SHORT_DISTINCTIVE_TOKENS = {"nih", "nyu", "mit", "ucla", "ucsf"}
CONJUNCTION_ONLY_FRAGMENTS = {"and", "or"}
INSTITUTION_MARKER_WORDS = (
    "center",
    "centre",
    "clinic",
    "college",
    "foundation",
    "hospital",
    "institute",
    "institutes",
    "laboratory",
    "school",
    "system",
    "university",
)
UC_CAMPUS_ALIASES = {
    "berkeley": "Berkeley",
    "davis": "Davis",
    "irvine": "Irvine",
    "los angeles": "Los Angeles",
    "merced": "Merced",
    "riverside": "Riverside",
    "san diego": "San Diego",
    "san francisco": "San Francisco",
    "santa barbara": "Santa Barbara",
    "santa cruz": "Santa Cruz",
}


def normalize_token(value: str) -> str:
    return NON_ALPHA_NUM.sub("", value.lower())


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def extract_initials(given: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z]+", given) if p]
    return "".join(p[0].lower() for p in parts)


def _normalized_name_tokens(tokens: list[str]) -> list[str]:
    cleaned = [normalize_token(token) for token in tokens]
    return [token for token in cleaned if token]


def _looks_like_initial_token(token: str) -> bool:
    cleaned = normalize_token(token)
    if not cleaned or len(cleaned) > 4:
        return False
    if len(cleaned) == 1:
        return True
    return token.isupper() or "." in token or len(cleaned) <= 2

def parse_pmids(raw_text: str) -> list[str]:
    candidates = re.findall(r"\d+", raw_text)
    seen: set[str] = set()
    ordered: list[str] = []
    for pmid in candidates:
        if pmid not in seen:
            seen.add(pmid)
            ordered.append(pmid)
    return ordered


def parse_target_name(raw_name: str) -> TargetName:
    raw = raw_name.strip()
    if not raw:
        raise ValueError("Author name must include at least given name and surname")

    if "," in raw:
        surname_part, given_part = raw.split(",", 1)
        surname = normalize_token(surname_part)
        given_tokens = [t for t in re.split(r"[,\s]+", given_part.strip()) if t]
        cleaned_given_tokens = _normalized_name_tokens(given_tokens)
        if surname and cleaned_given_tokens:
            if all(_looks_like_initial_token(token) for token in given_tokens):
                initials = "".join(normalize_token(token) for token in given_tokens)
                return TargetName(
                    raw=raw,
                    surname=surname,
                    given=" ".join(cleaned_given_tokens),
                    initials=initials,
                    style="initials",
                )
            given = " ".join(cleaned_given_tokens)
            initials = extract_initials(given)
            return TargetName(raw=raw, surname=surname, given=given, initials=initials, style="full")

    tokens = [t for t in raw.split() if t]
    if len(tokens) < 2:
        raise ValueError("Author name must include at least given name and surname")

    first = tokens[0]
    rest = tokens[1:]
    if all(_looks_like_initial_token(token) for token in rest):
        initials = "".join(normalize_token(token) for token in rest)
        return TargetName(
            raw=raw,
            surname=normalize_token(first),
            given=" ".join(_normalized_name_tokens(rest)),
            initials=initials,
            style="initials",
        )

    surname = normalize_token(tokens[-1])
    given_tokens = tokens[:-1]
    if all(_looks_like_initial_token(token) for token in given_tokens):
        initials = "".join(normalize_token(token) for token in given_tokens)
        return TargetName(
            raw=raw,
            surname=surname,
            given=" ".join(_normalized_name_tokens(given_tokens)),
            initials=initials,
            style="initials",
        )

    given = " ".join(_normalized_name_tokens(given_tokens))
    initials = extract_initials(given)
    return TargetName(raw=raw, surname=surname, given=given, initials=initials, style="full")


def normalize_orcid(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    lowered = raw.lower()
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if lowered.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break

    canonical = ORCID_CANONICAL.search(raw)
    if canonical is not None:
        return f"{canonical.group(1)}-{canonical.group(2)}-{canonical.group(3)}-{canonical.group(4).upper()}"

    compact = re.sub(r"[^0-9Xx]", "", raw)
    if ORCID_COMPACT.fullmatch(compact):
        compact = compact.upper()
        return f"{compact[0:4]}-{compact[4:8]}-{compact[8:12]}-{compact[12:16]}"

    return ""


def parse_target_orcid(raw_orcid: str) -> str:
    normalized = normalize_orcid(raw_orcid)
    if raw_orcid.strip() and not normalized:
        raise ValueError("ORCID must be in the form 0000-0000-0000-0000 (or an ORCID URL)")
    return normalized


def parse_year(pub_date: str) -> int | None:
    match = FOUR_DIGIT_YEAR.search(pub_date)
    if not match:
        return None
    return int(match.group(0))


def normalize_publication_type(value: str) -> str:
    return normalize_text(value)


def parse_type_terms(raw_text: str) -> list[str]:
    terms = [normalize_publication_type(part) for part in re.split(r"[,\n;]+", raw_text) if part.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def default_excluded_type_terms() -> list[str]:
    return list(EXCLUDED_TYPE_KEYWORDS)


def citation_matches_excluded_type(citation: Citation, excluded_terms: list[str]) -> tuple[bool, list[str]]:
    normalized_types = [normalize_publication_type(value) for value in citation.publication_types]
    matched: list[str] = []
    for pub_type in normalized_types:
        for term in excluded_terms:
            if term and term in pub_type:
                matched.append(pub_type)
                break
    seen: set[str] = set()
    deduped: list[str] = []
    for value in matched:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return bool(deduped), deduped


def is_preprint_article(citation: Citation) -> bool:
    for pub_type in citation.publication_types:
        if "preprint" in normalize_publication_type(pub_type):
            return True
    return False


def is_review_article(citation: Citation) -> bool:
    for pub_type in citation.publication_types:
        if "review" in normalize_publication_type(pub_type):
            return True
    return False


def match_author(
    author: Author, target: TargetName, target_orcid: str = ""
) -> tuple[bool, Literal["orcid", "given", "initials"] | None]:
    if target_orcid:
        author_orcid = normalize_orcid(author.orcid)
        if author_orcid:
            if author_orcid == target_orcid:
                return True, "orcid"
            return False, None

    if normalize_token(author.last_name) != target.surname:
        return False, None

    author_given = normalize_text(author.fore_name)
    author_initials = normalize_token(author.initials).lower() or extract_initials(author.fore_name)

    if target.style == "initials":
        if author_initials and target.initials:
            if author_initials.startswith(target.initials) or target.initials.startswith(author_initials):
                return True, "initials"
        return False, None

    if author_given and target.given:
        # Compare normalized first tokens to tolerate punctuation in user input.
        author_first = author_given.split(" ")[0]
        target_first = target.given.split(" ")[0]
        author_first_token = normalize_token(author_first)
        target_first_token = normalize_token(target_first)
        if author_first_token and author_first_token == target_first_token:
            return True, "given"
        # Precision-first: if a full given name is present and does not match,
        # do not fall back to initials (prevents Jeffrey vs James merges).
        if len(author_first_token) > 1:
            return False, None

    if author_initials and target.initials and author_initials.startswith(target.initials[:1]):
        return True, "initials"

    return False, None


def affiliation_fingerprint(affiliation: str) -> str:
    return affiliation_blocks_fingerprint([], fallback_affiliation=affiliation)


def affiliation_blocks_fingerprint(affiliation_blocks: list[str], *, fallback_affiliation: str = "") -> str:
    blocks = _author_affiliation_blocks(affiliation_blocks=affiliation_blocks, fallback_affiliation=fallback_affiliation)
    if not blocks:
        return "no-affiliation"

    keys: set[str] = set()
    for block in blocks:
        institutions = _extract_institution_names(block)
        keys.update(_institution_key(name) for name in institutions if _institution_key(name))
    sorted_keys = sorted(keys)
    if sorted_keys:
        return "|".join(sorted_keys)[:120]

    raw_affiliation = "; ".join(blocks)
    if not raw_affiliation:
        raw_affiliation = fallback_affiliation.strip()
    fallback = _institution_key(raw_affiliation) or normalize_token(raw_affiliation)
    return fallback[:120] if fallback else "no-affiliation"


def _author_affiliation_blocks(affiliation_blocks: list[str], *, fallback_affiliation: str = "") -> list[str]:
    cleaned_blocks = [block.strip() for block in affiliation_blocks if block and block.strip()]
    if cleaned_blocks:
        return cleaned_blocks

    raw_affiliation = fallback_affiliation.strip()
    if not raw_affiliation:
        return []
    return [piece.strip() for piece in AFFILIATION_PIECE_SPLIT.split(raw_affiliation) if piece.strip()]


def _author_record_affiliation_blocks(author: Author) -> list[str]:
    return _author_affiliation_blocks(
        affiliation_blocks=getattr(author, "affiliation_blocks", []),
        fallback_affiliation=author.affiliation or "",
    )


def _extract_institution_names(affiliation: str) -> list[str]:
    raw_affiliation = affiliation.strip()
    if not raw_affiliation:
        return []

    ror_ids, grid_ids, emails = _extract_affiliation_record_signals(raw_affiliation)
    preferred_ror = ror_ids[0] if len(ror_ids) == 1 else ""
    preferred_grid = grid_ids[0] if len(grid_ids) == 1 else ""
    preferred_email = ";".join(emails) if emails else ""

    id_names: list[str] = []
    email_names: list[str] = []
    text_names: list[str] = []
    seen_keys: set[str] = set()

    def add_match(target: list[str], result_status: str, canonical_name: str | None) -> None:
        if result_status != "matched" or not canonical_name:
            return
        label = _normalize_institution_label(canonical_name)
        key = _institution_key(label)
        if not key or key in seen_keys:
            return
        seen_keys.add(key)
        target.append(label)

    # Prefer record-level identifiers and email-domain signals before free text.
    for ror_id in ror_ids:
        result = match_record(ror_id=ror_id, affiliation_text=raw_affiliation)
        add_match(id_names, result.status, result.canonical_name)
    for grid_id in grid_ids:
        for grid_candidate in _grid_signal_candidates(grid_id):
            result = match_record(grid_id=grid_candidate, affiliation_text=raw_affiliation)
            add_match(id_names, result.status, result.canonical_name)
            if result.status == "matched":
                break
    if id_names:
        return id_names

    for email in emails:
        result = match_record(email=email, affiliation_text=raw_affiliation)
        add_match(email_names, result.status, result.canonical_name)
    if email_names:
        return email_names

    for candidate in _affiliation_match_candidates(raw_affiliation):
        if preferred_ror or preferred_grid or preferred_email:
            result = match_record(
                affiliation_text=candidate,
                ror_id=preferred_ror,
                grid_id=preferred_grid,
                email=preferred_email,
            )
        else:
            result = match_affiliation(candidate)
        add_match(text_names, result.status, result.canonical_name)

    if text_names:
        return text_names

    # Support UC system strings where campus is present elsewhere in the AD line.
    uc_campus = _uc_campus_from_text(raw_affiliation)
    if uc_campus:
        result = match_affiliation(f"University of California, {uc_campus}")
        if result.status == "matched" and result.canonical_name:
            return [_normalize_institution_label(result.canonical_name)]

    return []


def _affiliation_match_candidates(affiliation: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str, *, force: bool = False) -> None:
        normalized = _normalize_institution_label(value)
        if not normalized:
            return
        if not force and not _is_match_worthy_affiliation_candidate(normalized):
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    add(affiliation, force=True)
    # Split only on explicit affiliation delimiters (';' and '|').
    # Do not split on commas: commas are usually intra-affiliation structure
    # (department/institution/address), not reliable affiliation boundaries.
    pieces = [piece.strip() for piece in AFFILIATION_PIECE_SPLIT.split(affiliation) if piece.strip()]
    for piece in pieces:
        add(piece)
        for conjunct_segment in _conjunction_split_candidates(piece):
            add(conjunct_segment)
    return candidates


def _extract_affiliation_record_signals(value: str) -> tuple[list[str], list[str], list[str]]:
    ror_ids: list[str] = []
    grid_ids: list[str] = []
    emails: list[str] = []
    seen_ror: set[str] = set()
    seen_grid: set[str] = set()
    seen_email: set[str] = set()

    def add_ror(raw: str) -> None:
        cleaned = _normalize_ror_signal(raw)
        if not cleaned or cleaned in seen_ror:
            return
        seen_ror.add(cleaned)
        ror_ids.append(cleaned)

    def add_grid(raw: str) -> None:
        cleaned = _normalize_grid_signal(raw)
        if not cleaned or cleaned in seen_grid:
            return
        seen_grid.add(cleaned)
        grid_ids.append(cleaned)

    def add_email(raw: str) -> None:
        cleaned = raw.strip()
        if not cleaned or cleaned in seen_email:
            return
        seen_email.add(cleaned)
        emails.append(cleaned)

    for match in AFFILIATION_ROR_URL_RE.findall(value):
        add_ror(match)
    for match in AFFILIATION_ROR_HOST_RE.findall(value):
        add_ror(match)
    for match in AFFILIATION_ROR_SPACED_URL_RE.findall(value):
        add_ror(match)
    for match in AFFILIATION_ROR_SPACED_HOST_RE.findall(value):
        add_ror(match)
    for match in AFFILIATION_ROR_PREFIX_RE.findall(value):
        add_ror(match)
    for match in AFFILIATION_GRID_RE.findall(value):
        add_grid(match)
    for match in AFFILIATION_EMAIL_RE.findall(value):
        add_email(match)

    return ror_ids, grid_ids, emails


def _normalize_ror_signal(value: str) -> str:
    collapsed = re.sub(r"\s+", "", value.strip().lower())
    if not collapsed:
        return ""
    if collapsed.startswith("ror.org/"):
        collapsed = f"https://{collapsed}"
    if re.fullmatch(r"0[0-9a-z]{8}", collapsed):
        collapsed = f"https://ror.org/{collapsed}"
    if not re.fullmatch(r"https?://(?:www\.)?ror\.org/0[0-9a-z]{8}", collapsed):
        return ""
    collapsed = re.sub(r"^https?://(?:www\.)?", "https://", collapsed)
    return collapsed


def _normalize_grid_signal(value: str) -> str:
    collapsed = re.sub(r"\s+", "", value.strip().lower())
    if not collapsed.startswith("grid."):
        return ""
    match = re.match(r"^grid\.[a-z0-9]+\.[a-z0-9]+", collapsed)
    if match is None:
        return ""
    return match.group(0)


def _grid_signal_candidates(value: str) -> list[str]:
    normalized = _normalize_grid_signal(value)
    if not normalized:
        return []
    candidates: list[str] = [normalized]
    match = re.fullmatch(r"(grid\.[a-z0-9]+\.)([a-z0-9]+)", normalized)
    if match is None:
        return candidates
    prefix = match.group(1)
    suffix = match.group(2)
    if len(suffix) <= 4:
        return candidates
    for width in (2, 1, 3, 4):
        if len(suffix) < width:
            continue
        candidate = f"{prefix}{suffix[:width]}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _contains_institution_marker(value: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", normalize_text(value)))
    return bool(tokens & set(INSTITUTION_MARKER_WORDS))


def _conjunction_split_candidates(value: str) -> list[str]:
    # Some AD lines encode multiple institutions in one clause:
    # "... Hospital and Harvard Medical School". Split these so each institution
    # can be matched independently, but only when both sides look institutional.
    normalized_value = _normalize_institution_label(value)
    parts = [_normalize_institution_label(part) for part in AFFILIATION_AND_DELIMITER_SPLIT.split(normalized_value) if part.strip()]
    if len(parts) < 2:
        return []
    marked_parts = [part for part in parts if _contains_institution_marker(part)]
    if len(marked_parts) < 2:
        return []
    return marked_parts


def _is_match_worthy_affiliation_candidate(value: str) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return False
    if normalized in CONJUNCTION_ONLY_FRAGMENTS:
        return False
    if _looks_like_geo_only_phrase(normalized):
        return False
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if not tokens:
        return False
    if any(token in AFFILIATION_ANCHOR_WORDS for token in tokens):
        return True
    if any(token in ORG_SUFFIX_WORDS for token in tokens):
        return True
    if any(token in UNIT_ONLY_WORDS for token in tokens):
        return False
    if len(tokens) == 1:
        return tokens[0] in SHORT_DISTINCTIVE_TOKENS
    return True


def _looks_like_geo_only_phrase(normalized: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if not tokens:
        return False
    geo_like = 0
    for token in tokens:
        if token in US_STATE_CODES or token in GEO_TOKENS or POSTAL_CODE_TOKEN.fullmatch(token):
            geo_like += 1
    if geo_like == len(tokens):
        return True
    if len(tokens) <= 4 and geo_like >= len(tokens) - 1:
        return True
    return False


def _uc_campus_from_text(value: str) -> str:
    clauses: list[str] = []
    for piece in AFFILIATION_PIECE_SPLIT.split(value):
        if not piece.strip():
            continue
        clauses.extend(clause.strip() for clause in piece.split(",") if clause.strip())
    return _uc_campus_from_clauses(clauses)


def _institution_candidates(piece: str, *, uc_campus_hint: str = "") -> list[tuple[int, str, str]]:
    clauses = [clause.strip() for clause in piece.split(",") if clause.strip()]
    candidates: list[tuple[int, str, str]] = []
    for clause in clauses:
        display = _expand_institution_clause(clause, clauses, uc_campus_hint=uc_campus_hint)
        normalized = normalize_text(display)
        if not normalized:
            continue
        key = _institution_key(display)
        if not key or UNIT_PREFIX_PATTERN.match(normalized):
            continue
        score = 0
        if any(hint in normalized for hint in INSTITUTION_HINTS):
            score += 3
        if any(hint in normalized for hint in SECONDARY_INSTITUTION_HINTS):
            score += 1
        if any(word in normalized for word in UNIT_WORDS):
            score -= 2
        if score <= 0:
            continue
        candidates.append((score, key, display))
    return candidates


def _expand_institution_clause(clause: str, clauses: list[str], *, uc_campus_hint: str = "") -> str:
    display = _normalize_institution_label(clause)
    normalized = normalize_text(display)
    if normalized in {"university of california", "university of california system"}:
        campus = _uc_campus_from_clauses(clauses) or uc_campus_hint
        if campus:
            return f"University of California, {campus}"
    return display


def _uc_campus_from_clauses(clauses: list[str]) -> str:
    for clause in clauses:
        normalized_clause = normalize_text(_normalize_institution_label(clause))
        if not normalized_clause:
            continue
        for alias, campus in UC_CAMPUS_ALIASES.items():
            if alias in normalized_clause:
                return campus
    return ""


def _institution_key(value: str) -> str:
    normalized = normalize_text(value)
    normalized = AMPERSAND.sub(" and ", normalized)
    normalized = APOSTROPHE_VARIANTS.sub("'", normalized)
    return normalize_token(normalized)


def _institution_distinctive_tokens(value: str) -> set[str]:
    normalized = normalize_text(value)
    normalized = AMPERSAND.sub(" and ", normalized)
    normalized = APOSTROPHE_VARIANTS.sub("'", normalized)
    raw_tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    distinctive: set[str] = set()
    for token in raw_tokens:
        if token in GENERIC_INSTITUTION_TOKENS:
            continue
        if token.isdigit():
            continue
        if len(token) < 3 and token not in SHORT_DISTINCTIVE_TOKENS:
            continue
        distinctive.add(token)
    return distinctive


def _has_distinctive_institution_overlap(left: str, right: str) -> bool:
    left_tokens = _institution_distinctive_tokens(left)
    right_tokens = _institution_distinctive_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return bool(left_tokens & right_tokens)


def _clean_clause_label(clause: str) -> str:
    return TRAILING_PUNCT.sub("", " ".join(clause.split()))


def _normalize_institution_label(clause: str) -> str:
    cleaned = APOSTROPHE_VARIANTS.sub("'", clause)
    cleaned = AMPERSAND.sub(" and ", cleaned)
    return _clean_clause_label(cleaned)


def build_clusters(
    citations: list[Citation],
    target: TargetName,
    target_orcid: str = "",
) -> tuple[list[Cluster], list[AuthorMatch]]:
    cluster_mentions: dict[str, list[AuthorMatch]] = defaultdict(list)
    matches: list[AuthorMatch] = []

    for citation in citations:
        citation_matches: list[AuthorMatch] = []
        for author in citation.authors:
            matched, method = match_author(author, target, target_orcid=target_orcid)
            if not matched or method is None:
                continue

            author_orcid = normalize_orcid(author.orcid)
            if method == "orcid" and author_orcid:
                cluster_id = f"orcid:{author_orcid}"
            else:
                aff_key = affiliation_blocks_fingerprint(
                    _author_record_affiliation_blocks(author),
                    fallback_affiliation=author.affiliation or "",
                )
                cluster_id = f"{normalize_token(author.last_name)}|{extract_initials(author.fore_name)}|{aff_key}"
                if aff_key == "no-affiliation":
                    # Keep no-affiliation candidates independent so users can include/exclude per citation.
                    cluster_id = f"{cluster_id}|pmid:{citation.pmid}"

            citation_matches.append(
                AuthorMatch(
                    pmid=citation.pmid,
                    position=author.position,
                    method=method,
                    author=author,
                    cluster_id=cluster_id,
                )
            )

        if target_orcid:
            orcid_matches = [match for match in citation_matches if match.method == "orcid"]
            if orcid_matches:
                citation_matches = orcid_matches

        for match in citation_matches:
            matches.append(match)
            cluster_mentions[match.cluster_id].append(match)

    cluster_mentions, merged_cluster_id_map = _merge_subset_affiliation_clusters(cluster_mentions)
    for match in matches:
        match.cluster_id = merged_cluster_id_map.get(match.cluster_id, match.cluster_id)

    clusters: list[Cluster] = []
    for cluster_id, cluster_matches in cluster_mentions.items():
        years = sorted(
            {
                citation.print_year
                for citation in citations
                for cm in cluster_matches
                if citation.pmid == cm.pmid and citation.print_year is not None
            }
        )
        affiliations = _cluster_affiliation_labels(cluster_matches)
        exemplar = cluster_matches[0].author
        label = f"{exemplar.last_name}, {exemplar.fore_name or exemplar.initials}"
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                label=label,
                mention_count=len(cluster_matches),
                years=years,
                affiliations=affiliations or ["(no affiliation listed)"],
            )
        )

    clusters.sort(key=lambda c: c.mention_count, reverse=True)
    return clusters, matches


def _merge_subset_affiliation_clusters(
    cluster_mentions: dict[str, list[AuthorMatch]],
) -> tuple[dict[str, list[AuthorMatch]], dict[str, str]]:
    mention_counts = {cluster_id: len(matches) for cluster_id, matches in cluster_mentions.items()}

    by_name_key: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for cluster_id in cluster_mentions:
        parsed = _parse_name_affiliation_cluster_id(cluster_id)
        if parsed is None:
            continue
        name_key, affiliation_keys = parsed
        if not affiliation_keys:
            continue
        by_name_key[name_key].append((cluster_id, affiliation_keys))

    merge_to: dict[str, str] = {}
    for candidates in by_name_key.values():
        for cluster_id, affiliation_keys in candidates:
            supersets = [
                (other_id, other_keys)
                for other_id, other_keys in candidates
                if cluster_id != other_id and affiliation_keys < other_keys and mention_counts.get(other_id, 0) >= mention_counts.get(cluster_id, 0)
            ]
            if not supersets:
                continue
            supersets.sort(
                key=lambda item: (mention_counts.get(item[0], 0), len(item[1]), item[0]),
                reverse=True,
            )
            merge_to[cluster_id] = supersets[0][0]

    if not merge_to:
        identity_map = {cluster_id: cluster_id for cluster_id in cluster_mentions}
        return cluster_mentions, identity_map

    def canonical_cluster_id(cluster_id: str) -> str:
        current = cluster_id
        seen: set[str] = set()
        while current in merge_to and current not in seen:
            seen.add(current)
            current = merge_to[current]
        return current

    merged: dict[str, list[AuthorMatch]] = defaultdict(list)
    canonical_map: dict[str, str] = {}
    for cluster_id, items in cluster_mentions.items():
        canonical = canonical_cluster_id(cluster_id)
        canonical_map[cluster_id] = canonical
        merged[canonical].extend(items)
    return merged, canonical_map


def _parse_name_affiliation_cluster_id(cluster_id: str) -> tuple[str, set[str]] | None:
    if cluster_id.startswith("orcid:"):
        return None
    parts = cluster_id.split("|")
    if len(parts) < 3:
        return None
    if parts[-1].startswith("pmid:"):
        # Keep no-affiliation per-PMID clusters independent by design.
        return None
    name_key = "|".join(parts[:2])
    affiliation_keys = {part for part in parts[2:] if part}
    return name_key, affiliation_keys


def _cluster_affiliation_labels(matches: list[AuthorMatch]) -> list[str]:
    seen: set[str] = set()
    labels: dict[str, str] = {}
    for match in matches:
        for raw_affiliation in _author_record_affiliation_blocks(match.author):
            names = _extract_institution_names(raw_affiliation)
            if not names:
                names = [
                    _normalize_institution_label(part)
                    for part in AFFILIATION_PIECE_SPLIT.split(raw_affiliation)
                    if _normalize_institution_label(part)
                    and normalize_text(_normalize_institution_label(part)) not in CONJUNCTION_ONLY_FRAGMENTS
                ]
            for name in names:
                key = _institution_key(name)
                if not key:
                    key = normalize_token(name)
                if not key or key in seen:
                    continue
                seen.add(key)
                labels[key] = _normalize_institution_label(name)
    return sorted(labels.values(), key=str.lower)


def citation_in_window(citation: Citation, start_year: int | None, end_year: int | None) -> bool:
    if start_year is None or end_year is None:
        return True
    if citation.print_year is None:
        return False
    return start_year <= citation.print_year <= end_year


@dataclass(slots=True)
class AnalysisResult:
    rows: list[ReportRow]
    uncertain_indices: list[int]


def _pmid_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _normalize_doi_for_match(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def _superseded_preprint_targets(citations: list[Citation]) -> dict[str, list[str]]:
    by_pmid = {citation.pmid: citation for citation in citations}
    non_preprint_by_doi: dict[str, set[str]] = defaultdict(set)
    reverse_update_links: dict[str, set[str]] = defaultdict(set)

    for citation in citations:
        if not is_preprint_article(citation):
            doi = _normalize_doi_for_match(citation.doi)
            if doi:
                non_preprint_by_doi[doi].add(citation.pmid)
        if is_preprint_article(citation):
            continue
        for prior_pmid in citation.update_of_pmids:
            reverse_update_links[prior_pmid].add(citation.pmid)

    superseded: dict[str, list[str]] = {}
    for citation in citations:
        if not is_preprint_article(citation):
            continue
        targets: set[str] = set()
        for updated_pmid in citation.update_in_pmids:
            linked = by_pmid.get(updated_pmid)
            if linked is not None and is_preprint_article(linked):
                continue
            targets.add(updated_pmid)
        targets.update(reverse_update_links.get(citation.pmid, set()))

        doi = _normalize_doi_for_match(citation.doi)
        if doi:
            targets.update(non_preprint_by_doi.get(doi, set()))

        targets.discard(citation.pmid)
        if not targets:
            continue
        superseded[citation.pmid] = sorted(targets, key=_pmid_sort_key)

    return superseded


def analyze_citations(
    citations: list[Citation],
    matches: list[AuthorMatch],
    selected_cluster_ids: set[str],
    start_year: int | None,
    end_year: int | None,
    forced_include_pmids: set[str] | None = None,
    excluded_pmids: set[str] | None = None,
) -> AnalysisResult:
    if forced_include_pmids is None:
        forced_include_pmids = set()
    if excluded_pmids is None:
        excluded_pmids = set()
    by_pmid: dict[str, list[AuthorMatch]] = defaultdict(list)
    for match in matches:
        by_pmid[match.pmid].append(match)

    rows: list[ReportRow] = []
    uncertain_indices: list[int] = []
    superseded_preprints = _superseded_preprint_targets(citations)

    for citation in citations:
        force_included = citation.pmid in forced_include_pmids
        if citation.pmid in excluded_pmids and not force_included:
            continue
        if not citation_in_window(citation, start_year, end_year) and not force_included:
            continue

        matched_mentions = by_pmid.get(citation.pmid, [])
        selected_mentions = [m for m in matched_mentions if m.cluster_id in selected_cluster_ids]
        if not selected_mentions and not force_included:
            continue
        # Forced-excluded citations can be explicitly restored at step 1.
        if force_included:
            selected_mentions = matched_mentions

        reasons: list[str] = []

        has_definitive_orcid = any(m.method == "orcid" for m in selected_mentions)

        if len(matched_mentions) > 1 and not has_definitive_orcid:
            reasons.append("multiple matching authors with same/similar name")

        if any(m.method == "initials" for m in selected_mentions) and not has_definitive_orcid:
            reasons.append("matched by initials fallback")

        if citation.source_for_roles == "pubmed":
            reasons.append("PMC missing; co-first/co-senior not detectable")

        first_positions = {1} | citation.co_first_positions
        last_position = len(citation.authors)
        senior_positions = {last_position} | citation.co_senior_positions
        selected_positions = {m.position for m in selected_mentions}

        counted_first = bool(first_positions & selected_positions)
        counted_senior = bool(senior_positions & selected_positions)
        preprint_article = is_preprint_article(citation)
        review_article = is_review_article(citation)
        counted_review_senior = review_article and counted_senior
        superseded_by = list(superseded_preprints.get(citation.pmid, []))
        superseded_preprint = preprint_article and bool(superseded_by)

        row = ReportRow(
            citation=citation,
            counted_overall=not (review_article or preprint_article),
            counted_first=counted_first,
            counted_senior=counted_senior,
            counted_review_senior=counted_review_senior,
            is_review=review_article,
            is_preprint=preprint_article,
            uncertainty_reasons=reasons,
            include=not reasons and not superseded_preprint,
            forced_include=force_included,
            matched_positions=set(selected_positions),
            preprint_superseded_by=superseded_by,
        )
        rows.append(row)
        if reasons:
            uncertain_indices.append(len(rows) - 1)

    rows.sort(
        key=lambda r: (
            r.citation.print_year is None,
            -(r.citation.print_year or 0),
            r.citation.journal.lower(),
        )
    )
    # Recompute after sorting so review indices map to the rendered row order.
    uncertain_indices = [idx for idx, row in enumerate(rows) if row.uncertainty_reasons]

    return AnalysisResult(rows=rows, uncertain_indices=uncertain_indices)


def format_summary(
    rows: list[ReportRow],
    start_year: int | None,
    end_year: int | None,
) -> str:
    window_label = f"{start_year}-{end_year}" if start_year is not None and end_year is not None else "all years"
    included = [r for r in rows if r.include]
    overall_included = [r for r in included if r.counted_overall]
    total = len(overall_included)
    first_senior = [r for r in overall_included if r.counted_first or r.counted_senior]
    review_senior_total = sum(1 for r in included if r.counted_review_senior)
    preprint_total = sum(1 for r in included if r.is_preprint)
    detail = _first_senior_detail(first_senior)
    return (
        f"Peer-reviewed Publications ({window_label}): {total} total, "
        f"{len(first_senior)} 1st/Sr author ({detail}). "
        f"In addition: {review_senior_total} review(s) as Sr author; "
        f"{preprint_total} preprint(s)."
    )


def _first_senior_detail(rows: list[ReportRow]) -> str:
    if not rows:
        return "none"

    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        year = row.citation.print_year
        if year is not None:
            grouped[row.citation.journal].append(year)
        else:
            grouped[row.citation.journal].append(-1)

    parts: list[str] = []
    # Sort by descending count then journal name for deterministic output.
    for journal, years in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0].lower())):
        count = len(years)
        year_counts: dict[int, int] = defaultdict(int)
        unknown_year_count = 0
        for year in years:
            if year >= 0:
                year_counts[year] += 1
            else:
                unknown_year_count += 1

        year_tokens: list[str] = []
        for year in sorted(year_counts):
            year_count = year_counts[year]
            if year_count > 1:
                year_tokens.append(f"{year_count}x{year}")
            else:
                year_tokens.append(str(year))
        if unknown_year_count > 0:
            if unknown_year_count > 1:
                year_tokens.append(f"{unknown_year_count}xn/a")
            else:
                year_tokens.append("n/a")

        if year_tokens:
            year_text = ", ".join(year_tokens)
            parts.append(f"{count} {journal} {year_text}")
        else:
            parts.append(f"{count} {journal}")
    return "; ".join(parts)
