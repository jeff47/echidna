# affiliation_normalizer

Normalize affiliation strings to a curated canonical institution record.

## Quick use

```python
from affiliation_normalizer import match_affiliation

result = match_affiliation(
    "Yale Cancer Center, Yale University, New Haven, CT, USA. david.calderwood@yale.edu."
)

if result.status == "matched":
    print(result.standardized_name)  # Yale University, New Haven, CT
else:
    print(result.status, result.reason, result.candidate_ids)
```

```python
from affiliation_normalizer import match_ror

result = match_ror("https://ror.org/03v76x132")
if result.status == "matched":
    print(result.canonical_name)  # Yale University
```

```python
from affiliation_normalizer import match_grid

result = match_grid("https://www.grid.ac/institutes/grid.47100.32")
if result.status == "matched":
    print(result.canonical_name)  # Yale University
```

```python
from affiliation_normalizer import match_email_domain

result = match_email_domain("med.yale.edu")
if result.status == "matched":
    print(result.canonical_name)
```

```python
from affiliation_normalizer import match_record

result = match_record(
    ror_id="03v76x132",  # highest priority
    email="person@yale.edu",  # second priority
    affiliation_text="Yale University, New Haven, CT",  # fallback
)
```

## Return contract

- `status`: `matched | ambiguous | not_found`
- `canonical_id`, `canonical_name`, `standardized_name`
- `city`, `state`, `country`
- `ror_id`, `grid_id`, `openalex_id`
- `reason`, `candidate_ids`, `matched_aliases`, `confidence`
- ROR lookup: `match_ror(ror_id: str)` (supports bare IDs and `https://ror.org/...` URLs)
- GRID lookup: `match_grid(grid_id: str)` (supports bare IDs and legacy GRID URLs)
- Email lookup: `match_email_domain(email_domain: str)` (supports domains or full email addresses)
- Priority lookup: `match_record(...)` using `ROR/GRID > email > text`

## Rule rebuild

When `niaid_org_seed_master.csv` or `alias_policy_review.tsv` changes:

```bash
python -m affiliation_normalizer.build_rules
```

This writes `affiliation_normalizer/data/rules.json`.

## Limitations

1. US-centric and seed-driven: out-of-seed institutions often return `not_found`.
2. `review_only` aliases do not auto-resolve.
3. Multi-institution strings can still return `ambiguous`.
4. Runtime matching is alias/normalization based; regex is not the primary mechanism.
