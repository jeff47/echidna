# ECHIDNA

Extract Citations, Handle Institutional Data, and Normalize Authors.

Precision-first webapp for publication counting from PMID lists with author disambiguation and optional manual review.

## Features

- Browser workflow with low-friction human review:
  - Step 1: confirm author identity by author+affiliation clusters
  - Step 2 (conditional): citation-level include/exclude (year, journal, title) only when ambiguity remains after Step 1
  - Step 3 (optional): review only uncertain citations
  - Final: one-line summary + citation table
- Accepted `/disambiguate` submissions redirect to a live progress page and automatically advance to Step 1 when server-side clustering finishes.
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
- Set `ECHIDNA_PERF_LOG=1` in `.runtime/echidna.env` to emit opt-in one-line timing summaries for `/disambiguate`, NCBI fetches, ORCiD matching, and cluster analysis.
- Logs include PMID-level notes for PMCID mapping, PMC parsing, and affiliation backfill behavior.
- Set `NCBI_API_KEY` (optional but recommended) to reduce `429 Too Many Requests` responses from NCBI E-utilities.
- Optional `NCBI_EMAIL` can also be set for API identification.
- `ECHIDNA_NCBI_FETCH_WORKERS` controls parallel chunk fetches during `/disambiguate`; default is `4`.
- Multi-step run state is persisted on disk for multi-worker deployments; default path is `/tmp/echidna-runs` and can be overridden with `ECHIDNA_RUNS_DIR`.
- Usage audit records are stored in SQLite at `ECHIDNA_USAGE_DB_PATH`. Default path is the runtime volume at `usage.db` next to `ECHIDNA_RUNS_DIR`.
- Persisted run state expires after `ECHIDNA_RUN_TTL_SECONDS` seconds. Default is `86400` (24 hours). Expired runs are rejected and cleaned up opportunistically on save/load.
- `ECHIDNA_MAX_CONCURRENT_DISAMBIGUATIONS` limits concurrent expensive `/disambiguate` runs per app process. Default is `1`.
- When the app is saturated it returns `503 Server busy, retry shortly.` with a `Retry-After` header. `ECHIDNA_BUSY_RETRY_AFTER_SECONDS` controls that header value and defaults to `30`.
- Successful `POST /disambiguate` submissions return a `303` redirect to `/runs/{run_id}/progress`; the browser polls `/runs/{run_id}/status` until results are available at `/runs/{run_id}`.
- Set `ECHIDNA_ADMIN_USER` and `ECHIDNA_ADMIN_PASSWORD_HASH` to enable the password-protected `GET /admin/usage` endpoint.
- `ECHIDNA_ADMIN_PASSWORD` is still accepted as a compatibility fallback, but `ECHIDNA_ADMIN_PASSWORD_HASH` is preferred so the password is not stored in plaintext.

## Reverse Proxy

- `docker compose` now starts an internal Nginx sidecar at `echidna_proxy:8080`.
- Route external traffic to `echidna_proxy`, not directly to `echidna`.
- The proxy rate-limits:
  - `POST /disambiguate` to `5` requests/minute per client IP with burst `2`
  - `GET /orcid-search` to `20` requests/minute per client IP with burst `10`

## Usage Audit

- `GET /admin/usage` returns recent audited runs as JSON.
- Each usage item includes a `rerun_url`. Open `GET /admin/usage/{id}/rerun` to load the saved request back into the normal start form and rerun it through the standard `/disambiguate` flow.
- The endpoint is protected with HTTP Basic auth using `ECHIDNA_ADMIN_USER` and `ECHIDNA_ADMIN_PASSWORD_HASH`.
- Audit records include the submitted author name, PMID text or uploaded file contents, parsed PMID list, and run outcome.
- The app does not currently link usage-audit records to client IP addresses or user accounts.
- Query parameters:
  - `limit` limits the number of rows returned, default `50`
  - `author_name` filters by partial author name
  - `status` filters by exact run status

Generate a password hash for `ECHIDNA_ADMIN_PASSWORD_HASH` with:

```bash
python -m app.admin_password_hash
```

The command prompts for the password twice and prints the hash to stdout.

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
