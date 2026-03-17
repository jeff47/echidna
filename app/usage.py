from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(db_path.parent, 0o700)
    connection = sqlite3.connect(db_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            run_id TEXT NOT NULL,
            author_name TEXT NOT NULL,
            target_orcid TEXT NOT NULL,
            start_year INTEGER,
            end_year INTEGER,
            all_years INTEGER NOT NULL,
            include_preprints INTEGER NOT NULL,
            excluded_type_terms_json TEXT NOT NULL,
            submitted_pmids_text TEXT NOT NULL,
            uploaded_filename TEXT NOT NULL,
            uploaded_size_bytes INTEGER NOT NULL,
            parsed_pmids_json TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT NOT NULL,
            citation_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_runs_created_at ON usage_runs(created_at DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_runs_author_name ON usage_runs(author_name)"
    )
    if db_path.exists():
        os.chmod(db_path, 0o600)
    return connection


def record_usage_run(
    db_path: Path,
    *,
    run_id: str,
    author_name: str,
    target_orcid: str,
    start_year: int | None,
    end_year: int | None,
    all_years: bool,
    include_preprints: bool,
    excluded_type_terms: list[str],
    submitted_pmids_text: str,
    uploaded_filename: str,
    uploaded_size_bytes: int,
    parsed_pmids: list[str],
    status: str,
    error_message: str = "",
    citation_count: int = 0,
    error_count: int = 0,
) -> int:
    now = datetime.now(timezone.utc).timestamp()
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO usage_runs (
                created_at,
                updated_at,
                run_id,
                author_name,
                target_orcid,
                start_year,
                end_year,
                all_years,
                include_preprints,
                excluded_type_terms_json,
                submitted_pmids_text,
                uploaded_filename,
                uploaded_size_bytes,
                parsed_pmids_json,
                status,
                error_message,
                citation_count,
                error_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                now,
                run_id,
                author_name,
                target_orcid,
                start_year,
                end_year,
                int(all_years),
                int(include_preprints),
                json.dumps(excluded_type_terms, separators=(",", ":")),
                submitted_pmids_text,
                uploaded_filename,
                uploaded_size_bytes,
                json.dumps(parsed_pmids, separators=(",", ":")),
                status,
                error_message,
                citation_count,
                error_count,
            ),
        )
        return int(cursor.lastrowid)


def update_usage_run(
    db_path: Path,
    record_id: int,
    *,
    run_id: str | None = None,
    status: str | None = None,
    error_message: str | None = None,
    citation_count: int | None = None,
    error_count: int | None = None,
) -> None:
    assignments: list[str] = ["updated_at = ?"]
    values: list[Any] = [datetime.now(timezone.utc).timestamp()]
    if run_id is not None:
        assignments.append("run_id = ?")
        values.append(run_id)
    if status is not None:
        assignments.append("status = ?")
        values.append(status)
    if error_message is not None:
        assignments.append("error_message = ?")
        values.append(error_message)
    if citation_count is not None:
        assignments.append("citation_count = ?")
        values.append(citation_count)
    if error_count is not None:
        assignments.append("error_count = ?")
        values.append(error_count)
    values.append(record_id)
    with _connect(db_path) as connection:
        connection.execute(
            f"UPDATE usage_runs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )


def fetch_usage_runs(
    db_path: Path,
    *,
    limit: int,
    author_name: str = "",
    status: str = "",
) -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            created_at,
            updated_at,
            run_id,
            author_name,
            target_orcid,
            start_year,
            end_year,
            all_years,
            include_preprints,
            excluded_type_terms_json,
            submitted_pmids_text,
            uploaded_filename,
            uploaded_size_bytes,
            parsed_pmids_json,
            status,
            error_message,
            citation_count,
            error_count
        FROM usage_runs
    """
    clauses: list[str] = []
    params: list[Any] = []
    if author_name.strip():
        clauses.append("author_name LIKE ?")
        params.append(f"%{author_name.strip()}%")
    if status.strip():
        clauses.append("status = ?")
        params.append(status.strip())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        created_at = float(row["created_at"])
        updated_at = float(row["updated_at"])
        records.append(
            {
                "id": int(row["id"]),
                "created_at": datetime.fromtimestamp(created_at, timezone.utc).isoformat(),
                "updated_at": datetime.fromtimestamp(updated_at, timezone.utc).isoformat(),
                "run_id": str(row["run_id"]),
                "author_name": str(row["author_name"]),
                "target_orcid": str(row["target_orcid"]),
                "start_year": row["start_year"],
                "end_year": row["end_year"],
                "all_years": bool(row["all_years"]),
                "include_preprints": bool(row["include_preprints"]),
                "excluded_type_terms": json.loads(str(row["excluded_type_terms_json"])),
                "submitted_pmids_text": str(row["submitted_pmids_text"]),
                "uploaded_filename": str(row["uploaded_filename"]),
                "uploaded_size_bytes": int(row["uploaded_size_bytes"]),
                "parsed_pmids": json.loads(str(row["parsed_pmids_json"])),
                "status": str(row["status"]),
                "error_message": str(row["error_message"]),
                "citation_count": int(row["citation_count"]),
                "error_count": int(row["error_count"]),
            }
        )
    return records
