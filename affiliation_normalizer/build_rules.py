#!/usr/bin/env python3
"""Compile affiliation rules from the curated seed/policy files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STOPWORDS = {"of", "the", "and", "for", "at", "in", "on", "to", "a", "an"}
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")
EMAIL_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$")
IN_WORD_APOSTROPHE_RE = re.compile(r"(?<=\w)'(?=\w)")

DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
APOSTROPHE_CHARS = "'\u2018\u2019\u2032\u02bc"
DASH_TRANSLATION = str.maketrans({ch: " " for ch in DASH_CHARS})
APOSTROPHE_TRANSLATION = str.maketrans({ch: "'" for ch in APOSTROPHE_CHARS})


@dataclass(frozen=True)
class AliasRule:
    alias: str
    alias_norm: str
    canonical_id: str
    alias_type: str
    policy: str


def normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.translate(DASH_TRANSLATION)
    folded = folded.translate(APOSTROPHE_TRANSLATION)
    folded = IN_WORD_APOSTROPHE_RE.sub("", folded)
    folded = unicodedata.normalize("NFKD", folded)
    folded = folded.encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().replace("&", " and ")
    folded = folded.replace("'", " ")
    folded = NON_ALNUM_RE.sub(" ", folded)
    return WHITESPACE_RE.sub(" ", folded).strip()


def normalize_email_domain(domain: str) -> str:
    raw = domain.strip().lower()
    if not raw:
        return ""

    if raw.startswith("mailto:"):
        raw = raw[len("mailto:") :]
    raw = raw.lstrip("@")
    if "@" in raw:
        raw = raw.rsplit("@", 1)[-1]
    raw = raw.strip(".")
    if not raw or ".." in raw:
        return ""

    if not EMAIL_DOMAIN_RE.fullmatch(raw):
        return ""
    return raw


def normalize_email_domains_field(value: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[|,;]", value or ""):
        normalized = normalize_email_domain(token)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def acronym(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name)
    sig = [w for w in words if w.lower() not in STOPWORDS]
    if len(sig) < 2:
        return ""
    return "".join(w[0] for w in sig).upper()


def generate_aliases(row: dict[str, str]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    canonical = (row.get("canonical_name") or "").strip()
    nih_name = (row.get("nih_reporter_name") or "").strip()
    city = (row.get("org_city") or "").strip()
    state = (row.get("org_state") or "").strip()

    if canonical:
        aliases.append((canonical, "canonical_name"))
        aliases.append((canonical.replace(",", " "), "canonical_name_nocomma"))
        aliases.append((canonical.replace("-", " "), "canonical_name_nohyphen"))

        ac = acronym(canonical)
        if 3 <= len(ac) <= 8:
            aliases.append((ac, "acronym"))

    if nih_name:
        aliases.append((nih_name, "nih_name"))
        aliases.append((nih_name.replace(",", " "), "nih_name_nocomma"))

    # UC campus convenience aliases.
    low = canonical.lower()
    prefix = "university of california,"
    if low.startswith(prefix):
        campus = canonical[len(prefix) :].strip()
        if campus:
            aliases.append((f"UC {campus}", "uc_campus"))
            aliases.append((f"U.C. {campus}", "uc_campus"))
            aliases.append((f"University of California {campus}", "uc_campus"))

    # Location-qualified variants often seen in AD text.
    if canonical and city and state:
        aliases.append((f"{canonical}, {city}, {state}", "canonical_with_geo"))

    out: list[tuple[str, str]] = []
    seen = set()
    for alias, alias_type in aliases:
        key = (alias.strip(), alias_type)
        if key[0] and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def load_alias_policy(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    policy: dict[str, str] = {}
    explicit_aliases: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            alias = (row.get("alias") or "").strip()
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            this_policy = (row.get("policy") or "").strip().lower()
            normalized_policy = this_policy or "review_only"
            policy[alias_norm] = normalized_policy

            if normalized_policy == "allow":
                candidates = [
                    token.strip()
                    for token in str(row.get("candidate_canonical_ids") or "").split("|")
                    if token.strip()
                ]
                if len(candidates) == 1:
                    explicit_aliases.append((alias, candidates[0]))
    return policy, explicit_aliases


def load_precedence_rules(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []

    out: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            active = (row.get("active") or "true").strip().lower()
            if active in {"0", "false", "no"}:
                continue
            preferred = (row.get("preferred_canonical_id") or "").strip()
            demoted = (row.get("demoted_canonical_id") or "").strip()
            if not preferred or not demoted:
                continue
            out.append(
                {
                    "preferred": preferred,
                    "demoted": demoted,
                    "reason": (row.get("reason") or "").strip(),
                }
            )
    return out


def build_rules(
    master_path: Path,
    alias_policy_path: Path,
    precedence_path: Path | None,
) -> dict:
    alias_policy, explicit_aliases = load_alias_policy(alias_policy_path)
    precedence_rules = load_precedence_rules(precedence_path)

    institutions: dict[str, dict[str, str]] = {}
    alias_rules: list[AliasRule] = []

    with master_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "canonical_id",
            "canonical_name",
            "org_city",
            "org_state",
            "org_country",
            "ror_id",
            "openalex_id",
            "nih_reporter_name",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in master file: {', '.join(sorted(missing))}")

        for row in reader:
            canonical_id = (row.get("canonical_id") or "").strip()
            if not canonical_id:
                continue
            email_domains = normalize_email_domains_field((row.get("email_domains") or "").strip())
            institutions[canonical_id] = {
                "canonical_id": canonical_id,
                "canonical_name": (row.get("canonical_name") or "").strip(),
                "city": (row.get("org_city") or "").strip().title(),
                "state": (row.get("org_state") or "").strip().upper(),
                "country": (row.get("org_country") or "").strip().upper(),
                "ror_id": (row.get("ror_id") or "").strip(),
                "grid_id": (row.get("grid_id") or "").strip(),
                "email_domains": "|".join(email_domains),
                "openalex_id": (row.get("openalex_id") or "").strip(),
                "nih_reporter_name": (row.get("nih_reporter_name") or "").strip(),
                "nih_reporter_ipf_code": (row.get("nih_reporter_ipf_code") or "").strip(),
            }

            for alias, alias_type in generate_aliases(row):
                alias_norm = normalize_text(alias)
                if not alias_norm:
                    continue
                policy = alias_policy.get(alias_norm, "allow")
                if policy == "deny":
                    continue
                alias_rules.append(
                    AliasRule(
                        alias=alias,
                        alias_norm=alias_norm,
                        canonical_id=canonical_id,
                        alias_type=alias_type,
                        policy=policy,
                    )
                )

    for alias, canonical_id in explicit_aliases:
        if canonical_id not in institutions:
            raise ValueError(f"Alias policy references unknown canonical_id: {canonical_id} (alias={alias!r})")
        alias_norm = normalize_text(alias)
        if not alias_norm:
            continue
        alias_rules.append(
            AliasRule(
                alias=alias,
                alias_norm=alias_norm,
                canonical_id=canonical_id,
                alias_type="manual_alias",
                policy="allow",
            )
        )

    # Dedupe alias rules.
    unique: dict[tuple[str, str, str], AliasRule] = {}
    for rule in alias_rules:
        key = (rule.alias_norm, rule.canonical_id, rule.policy)
        if key not in unique:
            unique[key] = rule

    deduped_rules = sorted(
        unique.values(),
        key=lambda r: (r.alias_norm, r.canonical_id, r.policy, r.alias_type, r.alias),
    )

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "master_path": str(master_path),
            "alias_policy_path": str(alias_policy_path),
            "precedence_path": str(precedence_path) if precedence_path else "",
            "institution_count": len(institutions),
            "alias_rule_count": len(deduped_rules),
        },
        "institutions": institutions,
        "alias_rules": [
            {
                "alias": r.alias,
                "alias_norm": r.alias_norm,
                "canonical_id": r.canonical_id,
                "alias_type": r.alias_type,
                "policy": r.policy,
            }
            for r in deduped_rules
        ],
        "precedence_rules": precedence_rules,
    }


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).with_name("data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("niaid_org_seed_master.csv"),
        help="Master institution CSV file.",
    )
    parser.add_argument(
        "--alias-policy",
        type=Path,
        default=Path("alias_policy_review.tsv"),
        help="Alias policy TSV file.",
    )
    parser.add_argument(
        "--precedence",
        type=Path,
        default=Path("canonical_precedence.tsv"),
        help="Canonical precedence TSV file (optional).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data_dir / "rules.json",
        help="Output JSON rules artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = build_rules(
        master_path=args.master,
        alias_policy_path=args.alias_policy,
        precedence_path=args.precedence if args.precedence.exists() else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rules, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Institution count: {rules['metadata']['institution_count']}")
    print(f"Alias rule count: {rules['metadata']['alias_rule_count']}")
    print(f"Precedence rules: {len(rules['precedence_rules'])}")


if __name__ == "__main__":
    main()
