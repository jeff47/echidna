"""Affiliation normalization package."""

from .matcher import AffiliationNormalizer, match_affiliation
from .models import MatchResult

__all__ = [
    "AffiliationNormalizer",
    "MatchResult",
    "match_affiliation",
]

