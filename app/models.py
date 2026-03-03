from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class Author:
    position: int
    last_name: str
    fore_name: str
    initials: str
    affiliation: str
    orcid: str = ""


@dataclass(slots=True)
class Citation:
    pmid: str
    pmcid: str | None
    title: str
    journal: str
    print_year: int | None
    print_date: str
    authors: list[Author]
    source_for_roles: Literal["pmc", "pubmed"]
    publication_types: list[str] = field(default_factory=list)
    co_first_positions: set[int] = field(default_factory=set)
    co_senior_positions: set[int] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    doi: str | None = None


@dataclass(slots=True)
class TargetName:
    raw: str
    surname: str
    given: str
    initials: str
    style: Literal["full", "initials"] = "full"


@dataclass(slots=True)
class AuthorMatch:
    pmid: str
    position: int
    method: Literal["orcid", "given", "initials"]
    author: Author
    cluster_id: str


@dataclass(slots=True)
class Cluster:
    cluster_id: str
    label: str
    mention_count: int
    years: list[int]
    affiliations: list[str]


@dataclass(slots=True)
class ReportRow:
    citation: Citation
    counted_overall: bool
    counted_first: bool
    counted_senior: bool
    counted_review_senior: bool = False
    is_review: bool = False
    uncertainty_reasons: list[str] = field(default_factory=list)
    include: bool = True
    forced_include: bool = False
    matched_positions: set[int] = field(default_factory=set)
