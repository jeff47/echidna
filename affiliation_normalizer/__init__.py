"""Affiliation normalization package."""

from .matcher import (
    AffiliationNormalizer,
    match_affiliation,
    match_email_domain,
    match_grid,
    match_record,
    match_ror,
)
from .models import MatchResult

__all__ = [
    "AffiliationNormalizer",
    "MatchResult",
    "match_affiliation",
    "match_email_domain",
    "match_grid",
    "match_record",
    "match_ror",
]
