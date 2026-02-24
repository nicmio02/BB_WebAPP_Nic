# samples_store.py
"""
Database layer for the soil analysis pipeline.

Tables:
  - extracted_samples:  raw long-format extraction output (traceability)
  - scored_samples:     wide-format matched + unit-converted values (one row per param)
  - parameter_mappings: per-PDF manual overrides (kept for future manual corrections)
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

SAMPLES_DB_PATH = "samples.db"


def get_conn(db_path: str = SAMPLES_DB_PATH) -> sqlite3.Connection:
    """Open (and lazily initialize) the samples database."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist yet."""

    # Raw extraction output — kept for traceability / debugging
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS extracted_samples (
            user_id   TEXT NOT NULL,
            pdf_id    TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            parameter TEXT NOT NULL,
            unit      TEXT,
            value     TEXT,
            PRIMARY KEY (user_id, pdf_id, sample_id, parameter, unit)
        )
        """
    )

    # Scored (matched + converted) values — one row per canonical parameter
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scored_samples (
            user_id   TEXT NOT NULL,
            pdf_id    TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            parameter TEXT NOT NULL,
            value     REAL,
            PRIMARY KEY (user_id, pdf_id, sample_id, parameter)
        )
        """
    )

    # Per-PDF manual mapping overrides (retained for potential future use)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parameter_mappings (
            user_id           TEXT NOT NULL,
            pdf_id            TEXT NOT NULL,
            source_param      TEXT NOT NULL,
            target_eigenschap TEXT NOT NULL,
            PRIMARY KEY (user_id, pdf_id, source_param)
        )
        """
    )

    conn.commit()


# =====================================================================
# Raw extraction storage (traceability)
# =====================================================================

def save_extraction_results(
    conn: sqlite3.Connection,
    user_id: str,
    pdf_id: str,
    results: pd.DataFrame,
) -> None:
    """
    Persist raw extracted results (long format) into extracted_samples.
    Expected columns: sample_id, parameter, unit, value.
    """
    if results.empty:
        return

    cols = {c.lower(): c for c in results.columns}
    for key in ("sample_id", "parameter", "value"):
        if key not in cols:
            raise ValueError(f"results missing required column '{key}'")

    for _, row in results.iterrows():
        sample    = row[cols["sample_id"]]
        parameter = row[cols["parameter"]]
        unit      = row.get(cols.get("unit"), None)
        value     = row[cols["value"]]

        if pd.isna(unit):
            unit = None
        if pd.isna(value):
            value = None

        conn.execute(
            """
            INSERT OR REPLACE INTO extracted_samples
                (user_id, pdf_id, sample_id, parameter, unit, value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, pdf_id, sample, parameter, unit, value),
        )

    conn.commit()


# =====================================================================
# Scored (matched) sample storage
# =====================================================================

def save_scored_sample(
    conn: sqlite3.Connection,
    user_id: str,
    pdf_id: str,
    sample_id: str,
    scoring_df: pd.DataFrame,
) -> None:
    """
    Persist the wide-format scoring DataFrame produced by matcher.match_parameters
    into scored_samples (long format in DB for flexibility).

    Parameters
    ----------
    scoring_df : pd.DataFrame
        Single-row wide DF with canonical parameter names as columns and
        numeric values. Missing params have value -1 (stored as NULL).
    """
    if scoring_df.empty:
        return

    row = scoring_df.iloc[0]
    for param, value in row.items():
        # Treat -1 sentinel as NULL (missing)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            db_value = None if value == -1 else float(value)
        else:
            db_value = None

        conn.execute(
            """
            INSERT OR REPLACE INTO scored_samples
                (user_id, pdf_id, sample_id, parameter, value)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, pdf_id, sample_id, str(param), db_value),
        )

    conn.commit()


def load_scored_sample(
    conn: sqlite3.Connection,
    user_id: str,
    pdf_id: str,
    sample_id: str,
    required_cols: list[str],
) -> pd.DataFrame | None:
    """
    Reconstruct a wide 1-row DataFrame from scored_samples for a given sample.
    Returns None if no data is stored.
    """
    cur = conn.execute(
        """
        SELECT parameter, value
        FROM scored_samples
        WHERE user_id = ? AND pdf_id = ? AND sample_id = ?
        """,
        (user_id, pdf_id, sample_id),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    # Build wide dict
    data = {}
    for param, value in rows:
        data[param] = float(value) if value is not None else np.nan

    # Ensure all required columns exist
    for col in required_cols:
        if col not in data:
            data[col] = np.nan

    wide = pd.DataFrame([data])
    wide["SampleID"] = sample_id
    wide["DateProcessed"] = datetime.now().strftime("%Y-%m-%d")

    # Reorder columns
    main_cols = ["SampleID", "DateProcessed"] + [c for c in required_cols if c in wide.columns]
    other_cols = [c for c in wide.columns if c not in main_cols]
    wide = wide[main_cols + other_cols]

    return wide


# =====================================================================
# Listing helpers
# =====================================================================

def list_sample_pairs(conn: sqlite3.Connection, user_id: str) -> list[tuple[str, str]]:
    """Return distinct (pdf_id, sample_id) pairs for a user from scored_samples."""
    cur = conn.execute(
        """
        SELECT DISTINCT pdf_id, sample_id
        FROM scored_samples
        WHERE user_id = ?
        ORDER BY pdf_id, sample_id
        """,
        (user_id,),
    )
    return cur.fetchall()


def list_raw_sample_pairs(conn: sqlite3.Connection, user_id: str) -> list[tuple[str, str]]:
    """Return distinct (pdf_id, sample_id) pairs from raw extracted_samples."""
    cur = conn.execute(
        """
        SELECT DISTINCT pdf_id, sample_id
        FROM extracted_samples
        WHERE user_id = ?
        ORDER BY pdf_id, sample_id
        """,
        (user_id,),
    )
    return cur.fetchall()


# =====================================================================
# Manual mapping overrides (retained for future use)
# =====================================================================

def get_parameter_mappings(conn: sqlite3.Connection, user_id: str, pdf_id: str) -> dict[str, str]:
    """Retrieve manual mappings for a specific PDF."""
    cur = conn.execute(
        "SELECT source_param, target_eigenschap FROM parameter_mappings WHERE user_id = ? AND pdf_id = ?",
        (user_id, pdf_id),
    )
    return {row[0]: row[1] for row in cur.fetchall()}


def update_parameter_mapping(
    conn: sqlite3.Connection, user_id: str, pdf_id: str,
    source_param: str, target_eigenschap: str,
) -> None:
    """Insert or update a manual mapping for a specific PDF."""
    if not source_param or not target_eigenschap:
        return

    if target_eigenschap == "RESET":
        conn.execute(
            "DELETE FROM parameter_mappings WHERE user_id = ? AND pdf_id = ? AND source_param = ?",
            (user_id, pdf_id, source_param),
        )
    else:
        conn.execute(
            """
            INSERT OR REPLACE INTO parameter_mappings
                (user_id, pdf_id, source_param, target_eigenschap)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, pdf_id, source_param, target_eigenschap),
        )
    conn.commit()


def get_global_mappings(conn: sqlite3.Connection) -> dict[str, str]:
    """Return global mappings (if the legacy table exists, else empty)."""
    try:
        cur = conn.execute("SELECT source_param, target_eigenschap FROM global_parameter_mappings")
        return {row[0]: row[1] for row in cur.fetchall()}
    except Exception:
        return {}


def get_local_mappings(conn: sqlite3.Connection, user_id: str, pdf_id: str) -> dict[str, str]:
    """Return per-PDF manual mapping overrides."""
    return get_parameter_mappings(conn, user_id, pdf_id)


def get_combined_mappings(conn: sqlite3.Connection, user_id: str, pdf_id: str) -> dict[str, str]:
    """Returns global merged with local (local wins)."""
    global_map = get_global_mappings(conn)
    local_map = get_local_mappings(conn, user_id, pdf_id)
    return {**global_map, **local_map}
