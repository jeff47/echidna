# ECHIDNA

Extract Citations, Handle Institutional Data, and Normalize Authors.

Precision-first webapp for publication counting from PMID lists with author disambiguation and optional manual review.

## Features

- Browser workflow with low-friction human review:
  - Step 1: confirm author identity by author+affiliation clusters
  - Step 2 (conditional): citation-level include/exclude (year, journal, title) only when ambiguity remains after Step 1
  - Step 3 (optional): review only uncertain citations
  - Final: one-line summary + citation table
- PMID input supports both pasted text and optional `.txt` file upload (one PMID per line).
- Uses PubMed metadata for citation details and print dates.
- Uses PMC XML as the primary source for author-level metadata (author list, affiliations, co-first/co-senior notes).
- Falls back to PubMed author/order data only when PMC is unavailable or parsing fails, and marks fallback in output.
- Optional ORCiD cross-checks: publication identifier matches (DOI/PMID/PMCID) and affiliation name matches are shown as badges in review/report tables.
- Publication-type filtering controls with default exclusions (preprints, editorials, comments, letters, news, etc.), each selectable on Step 0.
- Review articles are tracked separately: senior-author reviews are counted separately and excluded from the overall publication total.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Build metadata in footer

Set build-time values when creating the Docker image so the footer shows a fixed build timestamp and app commit:

```bash
ECHIDNA_BUILD_DATETIME="$(date -u '+%Y-%m-%d %H:%M:%S UTC')" \
ECHIDNA_APP_COMMIT="$(git rev-parse HEAD)" \
docker compose build
docker compose up -d
```

## Runtime Logging

- Set `ECHIDNA_LOG_LEVEL` in the service environment to `DEBUG` when troubleshooting metadata extraction.
- Logs include PMID-level notes for PMCID mapping, PMC parsing, and affiliation backfill behavior.
- Set `NCBI_API_KEY` (optional but recommended) to reduce `429 Too Many Requests` responses from NCBI E-utilities.
- Optional `NCBI_EMAIL` can also be set for API identification.
- Multi-step run state is persisted on disk for multi-worker deployments; default path is `/tmp/echidna-runs` and can be overridden with `ECHIDNA_RUNS_DIR`.

## Test

```bash
python -m pytest
```

## Counting rules

- Overall: selected investigator appears in citation and citation year is within configured range.
- First: investigator is first author, or is marked co-first in PMC XML.
- Senior: investigator is last author, or is marked co-senior/joint-supervising in PMC XML.
- If no PMC role data is available, first/senior uses strict list order and the table is flagged as `pubmed` source.

## Current limitations

- In-memory run state (not persistent across process restarts).
- No authentication/authorization.
- Co-first/co-senior extraction relies on text heuristics in PMC footnotes and may miss uncommon phrasing.
