#!/usr/bin/env python3

"""
BURST FINAL DATABASE WRITE — TRANSACTIONAL VALIDATION

Input:
    burst_final_precommit_v1.csv

Database:
    PostgreSQL
    risk_emergence.news_indicators

Default:
    DRY RUN — all changes rolled back.

Permanent update:
    python burst_final_database_write.py --commit

Safety:
    - Database credentials from environment variables
    - Frozen five-theme mapping
    - Exact CSV/database key matching
    - Existing burst columns must be NULL
    - Transactional UPDATE
    - Post-update validation
    - CSV/database value comparison
    - Automatic rollback on failure
"""

import argparse
import os
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from psycopg2 import sql
from psycopg2.extras import execute_values


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "burst_final_precommit_v1.csv"

SCHEMA_NAME = "risk_emergence"
INDICATORS_TABLE = "news_indicators"
THEMES_TABLE = "candidate_themes"

PERIOD_TYPE = "month"

EXPECTED_ROWS = 600
EXPECTED_THEMES = 5
EXPECTED_MONTHS_PER_THEME = 120

EXPECTED_WARMUP_PER_THEME = 12
EXPECTED_WARMUP_TOTAL = 60
EXPECTED_EVALUABLE_TOTAL = 540

EXPECTED_THEME_MAP = {
    "cyber_risk": 1,
    "climate_risk": 2,
    "pandemic_risk": 3,
    "supply_chain_risk": 4,
    "ai_risk": 5,
}

REQUIRED_COLUMNS = {
    "theme_name",
    "period_start",
    "burst_state_final",
    "burst_intensity_final",
}


# =============================================================================
# UTILITIES
# =============================================================================

def header(title):

    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def fail(message):
    raise RuntimeError(message)


def print_check(label, condition, detail=None):

    status = "PASS" if condition else "FAIL"

    print(f"{status} | {label}")

    if detail is not None:
        print(f"       {detail}")

    if not condition:
        fail(label)


def to_python_value(value):

    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def get_connection():

    required = [
        "PGDATABASE",
        "PGUSER",
        "PGPASSWORD",
    ]

    missing = [
        name
        for name in required
        if not os.environ.get(name)
    ]

    if missing:
        raise RuntimeError(
            "Missing PostgreSQL environment variables: "
            + ", ".join(missing)
        )

    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Validate and optionally commit frozen "
            "burst variables to PostgreSQL."
        )
    )

    parser.add_argument(
        "--commit",
        action="store_true",
        help=(
            "Permanently commit the update. "
            "Without this flag, the transaction "
            "is rolled back."
        ),
    )

    return parser.parse_args()


# =============================================================================
# LOAD AND VALIDATE CSV
# =============================================================================

def load_and_validate_csv():

    header("1. LOAD AND VALIDATE PRE-COMMIT DATASET")

    if not INPUT_FILE.exists():
        fail(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print_check(
        "Expected total number of rows",
        len(df) == EXPECTED_ROWS,
        f"observed={len(df)}, expected={EXPECTED_ROWS}",
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    print_check(
        "Required columns are present",
        len(missing_columns) == 0,
        f"missing={sorted(missing_columns)}",
    )

    df["period_start"] = pd.to_datetime(
        df["period_start"],
        errors="coerce",
    )

    print_check(
        "All period_start values are valid dates",
        df["period_start"].notna().all(),
    )

    df["period_start"] = (
        df["period_start"].dt.date
    )

    print_check(
        "No missing theme names",
        df["theme_name"].notna().all(),
    )

    print_check(
        "No duplicate theme-month keys in CSV",
        not df.duplicated(
            ["theme_name", "period_start"]
        ).any(),
    )

    print_check(
        "Expected number of themes in CSV",
        df["theme_name"].nunique() == EXPECTED_THEMES,
        (
            f"observed={df['theme_name'].nunique()}, "
            f"expected={EXPECTED_THEMES}"
        ),
    )

    counts = (
        df.groupby("theme_name")
        .size()
        .to_dict()
    )

    print_check(
        "Exactly 120 rows per theme in CSV",
        (
            len(counts) == EXPECTED_THEMES
            and all(
                value == EXPECTED_MONTHS_PER_THEME
                for value in counts.values()
            )
        ),
        str(counts),
    )

    observed_themes = set(df["theme_name"])

    print_check(
        "CSV themes exactly match frozen theme set",
        observed_themes == set(EXPECTED_THEME_MAP),
        f"observed={sorted(observed_themes)}",
    )

    # -------------------------------------------------------------------------
    # VALIDATE STATE AND INTENSITY
    # -------------------------------------------------------------------------

    df["burst_state_final"] = pd.to_numeric(
        df["burst_state_final"],
        errors="coerce",
    )

    df["burst_intensity_final"] = pd.to_numeric(
        df["burst_intensity_final"],
        errors="coerce",
    )

    state_nonnull = df["burst_state_final"].notna()

    intensity_nonnull = (
        df["burst_intensity_final"].notna()
    )

    print_check(
        "Exactly 540 evaluable burst_state values",
        int(state_nonnull.sum()) == EXPECTED_EVALUABLE_TOTAL,
        f"observed={int(state_nonnull.sum())}",
    )

    print_check(
        "Exactly 540 evaluable burst_intensity values",
        int(intensity_nonnull.sum()) == EXPECTED_EVALUABLE_TOTAL,
        f"observed={int(intensity_nonnull.sum())}",
    )

    print_check(
        "Exactly 60 warm-up burst_state NULL values",
        int((~state_nonnull).sum()) == EXPECTED_WARMUP_TOTAL,
    )

    print_check(
        "Exactly 60 warm-up burst_intensity NULL values",
        int((~intensity_nonnull).sum()) == EXPECTED_WARMUP_TOTAL,
    )

    print_check(
        "State and intensity NULL masks match exactly",
        state_nonnull.equals(intensity_nonnull),
    )

    print_check(
        "Evaluable burst_state values are only 0 or 1",
        df.loc[
            state_nonnull,
            "burst_state_final",
        ].isin([0, 1]).all(),
    )

    print_check(
        "Burst intensity is finite",
        np.isfinite(
            df.loc[
                intensity_nonnull,
                "burst_intensity_final",
            ].to_numpy(dtype=float)
        ).all(),
    )

    print_check(
        "Burst intensity is never negative",
        (
            df.loc[
                intensity_nonnull,
                "burst_intensity_final",
            ] >= 0
        ).all(),
    )

    inactive = (
        df["burst_state_final"] == 0
    )

    active = (
        df["burst_state_final"] == 1
    )

    print_check(
        "Inactive months have zero intensity",
        (
            df.loc[
                inactive,
                "burst_intensity_final",
            ] == 0
        ).all(),
    )

    print_check(
        "Active months have positive intensity",
        (
            df.loc[
                active,
                "burst_intensity_final",
            ] > 0
        ).all(),
    )

    # -------------------------------------------------------------------------
    # WARM-UP DISTRIBUTION
    # -------------------------------------------------------------------------

    warmup_counts = (
        df.assign(
            is_warmup=~state_nonnull
        )
        .groupby("theme_name")["is_warmup"]
        .sum()
    )

    print_check(
        "Exactly 12 warm-up months per theme",
        (
            warmup_counts == EXPECTED_WARMUP_PER_THEME
        ).all(),
        str(warmup_counts.to_dict()),
    )

    return df


# =============================================================================
# DATABASE THEME MAP
# =============================================================================

def validate_theme_map(cur, df):

    header("3. VALIDATE DATABASE THEME MAP")

    query = sql.SQL(
        """
        SELECT theme_id, theme_name
        FROM {}.{}
        ORDER BY theme_id;
        """
    ).format(
        sql.Identifier(SCHEMA_NAME),
        sql.Identifier(THEMES_TABLE),
    )

    cur.execute(query)

    db_theme_rows = cur.fetchall()

    db_theme_map = {
        theme_name: theme_id
        for theme_id, theme_name in db_theme_rows
    }

    print_check(
        "Database contains expected five-theme mapping",
        db_theme_map == EXPECTED_THEME_MAP,
        f"observed={db_theme_map}",
    )

    df["theme_id"] = (
        df["theme_name"].map(db_theme_map)
    )

    print_check(
        "All CSV themes mapped to theme_id",
        df["theme_id"].notna().all(),
    )

    return df


# =============================================================================
# LOCK TARGET ROWS
# =============================================================================

def lock_and_validate_target_rows(cur, df):

    header("4. LOCK AND VALIDATE TARGET DATABASE ROWS")

    query = sql.SQL(
        """
        SELECT
            theme_id,
            period_start,
            burst_state,
            burst_intensity
        FROM {}.{}
        WHERE period_type = %s
        ORDER BY theme_id, period_start
        FOR UPDATE;
        """
    ).format(
        sql.Identifier(SCHEMA_NAME),
        sql.Identifier(INDICATORS_TABLE),
    )

    cur.execute(
        query,
        (PERIOD_TYPE,),
    )

    db_rows = cur.fetchall()

    print_check(
        "Exactly 600 target rows exist in database",
        len(db_rows) == EXPECTED_ROWS,
        (
            f"observed={len(db_rows)}, "
            f"expected={EXPECTED_ROWS}"
        ),
    )

    db_keys = {
        (theme_id, period_start)
        for theme_id, period_start, _, _ in db_rows
    }

    csv_keys = {
        (int(row.theme_id), row.period_start)
        for row in df.itertuples(index=False)
    }

    print_check(
        "CSV and database key sets match exactly",
        csv_keys == db_keys,
        (
            f"csv_keys={len(csv_keys)}, "
            f"db_keys={len(db_keys)}, "
            f"csv_only={len(csv_keys - db_keys)}, "
            f"db_only={len(db_keys - csv_keys)}"
        ),
    )

    existing_state_nonnull = sum(
        state is not None
        for _, _, state, _ in db_rows
    )

    existing_intensity_nonnull = sum(
        intensity is not None
        for _, _, _, intensity in db_rows
    )

    print_check(
        "Target burst_state values are currently all NULL",
        existing_state_nonnull == 0,
        f"non_null={existing_state_nonnull}",
    )

    print_check(
        "Target burst_intensity values are currently all NULL",
        existing_intensity_nonnull == 0,
        f"non_null={existing_intensity_nonnull}",
    )


# =============================================================================
# BUILD UPDATE PAYLOAD
# =============================================================================

def build_update_payload(df):

    header("5. BUILD UPDATE PAYLOAD")

    update_rows = []

    for row in df.itertuples(index=False):

        state = to_python_value(
            row.burst_state_final
        )

        intensity = to_python_value(
            row.burst_intensity_final
        )

        if state is not None:
            state = int(state)

        if intensity is not None:
            intensity = float(intensity)

        update_rows.append(
            (
                int(row.theme_id),
                row.period_start,
                PERIOD_TYPE,
                state,
                intensity,
            )
        )

    print_check(
        "Update payload contains exactly 600 rows",
        len(update_rows) == EXPECTED_ROWS,
        f"observed={len(update_rows)}",
    )

    return update_rows


# =============================================================================
# EXECUTE TRANSACTIONAL UPDATE
# =============================================================================

def execute_update(cur, update_rows):

    header("6. EXECUTE TRANSACTIONAL UPDATE")

    query = sql.SQL(
        """
        UPDATE {}.{} AS ni
        SET
            burst_state = v.burst_state::smallint,
            burst_intensity = v.burst_intensity::real
        FROM (
            VALUES %s
        ) AS v(
            theme_id,
            period_start,
            period_type,
            burst_state,
            burst_intensity
        )
        WHERE ni.theme_id = v.theme_id::integer
          AND ni.period_start = v.period_start::date
          AND ni.period_type = v.period_type::varchar;
        """
    ).format(
        sql.Identifier(SCHEMA_NAME),
        sql.Identifier(INDICATORS_TABLE),
    )

    execute_values(
        cur,
        query.as_string(cur),
        update_rows,
        page_size=EXPECTED_ROWS,
    )

    affected_rows = cur.rowcount

    print_check(
        "Exactly 600 database rows updated",
        affected_rows == EXPECTED_ROWS,
        (
            f"observed={affected_rows}, "
            f"expected={EXPECTED_ROWS}"
        ),
    )


# =============================================================================
# POST-UPDATE VALIDATION
# =============================================================================

def validate_database_state(cur, df):

    header("7. POST-UPDATE VALIDATION INSIDE TRANSACTION")

    query = sql.SQL(
        """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(burst_state) AS state_nonnull,
            COUNT(burst_intensity) AS intensity_nonnull,

            COUNT(*) FILTER (
                WHERE burst_state = 1
            ) AS active_months,

            COUNT(*) FILTER (
                WHERE burst_state = 0
            ) AS inactive_months,

            COUNT(*) FILTER (
                WHERE burst_state IS NULL
            ) AS warmup_state_null,

            COUNT(*) FILTER (
                WHERE burst_intensity IS NULL
            ) AS warmup_intensity_null,

            COUNT(*) FILTER (
                WHERE burst_state = 0
                  AND burst_intensity <> 0
            ) AS inactive_nonzero_intensity,

            COUNT(*) FILTER (
                WHERE burst_state = 1
                  AND burst_intensity <= 0
            ) AS active_nonpositive_intensity

        FROM {}.{}
        WHERE period_type = %s;
        """
    ).format(
        sql.Identifier(SCHEMA_NAME),
        sql.Identifier(INDICATORS_TABLE),
    )

    cur.execute(
        query,
        (PERIOD_TYPE,),
    )

    (
        total_rows,
        state_nonnull_db,
        intensity_nonnull_db,
        active_months_db,
        inactive_months_db,
        warmup_state_null_db,
        warmup_intensity_null_db,
        inactive_nonzero_db,
        active_nonpositive_db,
    ) = cur.fetchone()

    expected_active = int(
        (df["burst_state_final"] == 1).sum()
    )

    expected_inactive = int(
        (df["burst_state_final"] == 0).sum()
    )

    checks = [
        (
            "Database still contains exactly 600 monthly rows",
            total_rows == EXPECTED_ROWS,
        ),
        (
            "Database has exactly 540 non-NULL burst_state values",
            state_nonnull_db == EXPECTED_EVALUABLE_TOTAL,
        ),
        (
            "Database has exactly 540 non-NULL burst_intensity values",
            intensity_nonnull_db == EXPECTED_EVALUABLE_TOTAL,
        ),
        (
            "Database has exactly 60 warm-up state NULLs",
            warmup_state_null_db == EXPECTED_WARMUP_TOTAL,
        ),
        (
            "Database has exactly 60 warm-up intensity NULLs",
            warmup_intensity_null_db == EXPECTED_WARMUP_TOTAL,
        ),
        (
            "Database active-month count matches CSV",
            active_months_db == expected_active,
        ),
        (
            "Database inactive-month count matches CSV",
            inactive_months_db == expected_inactive,
        ),
        (
            "No inactive month has non-zero intensity",
            inactive_nonzero_db == 0,
        ),
        (
            "No active month has non-positive intensity",
            active_nonpositive_db == 0,
        ),
    ]

    for label, condition in checks:
        print_check(label, condition)


# =============================================================================
# EXACT CSV ↔ DATABASE COMPARISON
# =============================================================================

def validate_exact_values(cur, df):

    header("8. EXACT CSV ↔ DATABASE VALUE CHECK")

    query = sql.SQL(
        """
        SELECT
            theme_id,
            period_start,
            burst_state,
            burst_intensity
        FROM {}.{}
        WHERE period_type = %s
        ORDER BY theme_id, period_start;
        """
    ).format(
        sql.Identifier(SCHEMA_NAME),
        sql.Identifier(INDICATORS_TABLE),
    )

    cur.execute(
        query,
        (PERIOD_TYPE,),
    )

    updated_rows = cur.fetchall()

    db_after = pd.DataFrame(
        updated_rows,
        columns=[
            "theme_id",
            "period_start",
            "db_state",
            "db_intensity",
        ],
    )

    expected = df[
        [
            "theme_id",
            "period_start",
            "burst_state_final",
            "burst_intensity_final",
        ]
    ].copy()

    merged = expected.merge(
        db_after,
        on=["theme_id", "period_start"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    print_check(
        "Exact one-to-one merge contains 600 rows",
        (
            len(merged) == EXPECTED_ROWS
            and (merged["_merge"] == "both").all()
        ),
    )

    expected_state = pd.to_numeric(
        merged["burst_state_final"],
        errors="coerce",
    )

    database_state = pd.to_numeric(
        merged["db_state"],
        errors="coerce",
    )

    state_equal = (
        (
            expected_state.isna()
            & database_state.isna()
        )
        |
        (
            expected_state.notna()
            & database_state.notna()
            & (
                expected_state
                == database_state
            )
        )
    )

    print_check(
        "All burst_state values match CSV exactly",
        bool(state_equal.all()),
        f"mismatches={int((~state_equal).sum())}",
    )

    expected_intensity = pd.to_numeric(
        merged["burst_intensity_final"],
        errors="coerce",
    )

    database_intensity = pd.to_numeric(
        merged["db_intensity"],
        errors="coerce",
    )

    both_null = (
        expected_intensity.isna()
        & database_intensity.isna()
    )

    both_values = (
        expected_intensity.notna()
        & database_intensity.notna()
    )

    intensity_equal = both_null.copy()

    # PostgreSQL REAL is float4.
    # Numerical tolerance accounts for float32 rounding.

    intensity_equal.loc[both_values] = np.isclose(
        expected_intensity.loc[both_values],
        database_intensity.loc[both_values],
        rtol=1e-6,
        atol=1e-9,
    )

    print_check(
        "All burst_intensity values match CSV within REAL precision",
        bool(intensity_equal.all()),
        f"mismatches={int((~intensity_equal).sum())}",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_arguments()

    commit_final_update = args.commit

    header("BURST FINAL DATABASE WRITE — TRANSACTIONAL VALIDATION")

    print(f"\nInput file : {INPUT_FILE}")
    print(f"Schema     : {SCHEMA_NAME}")
    print(f"Table      : {INDICATORS_TABLE}")
    print(f"Period type: {PERIOD_TYPE}")
    print(f"Commit     : {commit_final_update}")

    if commit_final_update:

        print("\nWARNING: COMMIT MODE ACTIVE")
        print(
            "This execution can permanently modify "
            "the PostgreSQL database."
        )

    else:

        print("\nSAFE MODE ACTIVE")
        print(
            "All database changes will be "
            "rolled back at the end."
        )

    # -------------------------------------------------------------------------
    # 1. VALIDATE CSV BEFORE CONNECTING
    # -------------------------------------------------------------------------

    df = load_and_validate_csv()

    # -------------------------------------------------------------------------
    # 2. CONNECT TO DATABASE
    # -------------------------------------------------------------------------

    header("2. DATABASE CONNECTION")

    conn = None

    try:

        conn = get_connection()

        conn.autocommit = False

        print(
            "PASS | PostgreSQL connection established"
        )

        print(
            "PASS | Transaction mode enabled"
        )

        with conn.cursor() as cur:

            # ---------------------------------------------------------------
            # 3. THEME MAP
            # ---------------------------------------------------------------

            df = validate_theme_map(
                cur,
                df,
            )

            # ---------------------------------------------------------------
            # 4. LOCK ROWS
            # ---------------------------------------------------------------

            lock_and_validate_target_rows(
                cur,
                df,
            )

            # ---------------------------------------------------------------
            # 5. UPDATE PAYLOAD
            # ---------------------------------------------------------------

            update_rows = build_update_payload(
                df
            )

            # ---------------------------------------------------------------
            # 6. UPDATE
            # ---------------------------------------------------------------

            execute_update(
                cur,
                update_rows,
            )

            # ---------------------------------------------------------------
            # 7. DATABASE VALIDATION
            # ---------------------------------------------------------------

            validate_database_state(
                cur,
                df,
            )

            # ---------------------------------------------------------------
            # 8. VALUE COMPARISON
            # ---------------------------------------------------------------

            validate_exact_values(
                cur,
                df,
            )

        # ---------------------------------------------------------------------
        # 9. TRANSACTION DECISION
        # ---------------------------------------------------------------------

        header("9. TRANSACTION DECISION")

        if commit_final_update:

            conn.commit()

            print("COMMIT EXECUTED")
            print(
                "Database changes are now permanent."
            )

        else:

            conn.rollback()

            print("ROLLBACK EXECUTED")
            print(
                "Dry run completed successfully."
            )
            print(
                "Database changes were not committed."
            )

        header("FINAL STATUS: PASS")

        if commit_final_update:

            print(
                "Burst state and intensity were "
                "successfully written to the database."
            )

        else:

            print(
                "The complete write path passed all checks, "
                "but changes were intentionally rolled back."
            )

    except Exception as exc:

        if conn is not None:
            conn.rollback()

        header("FINAL STATUS: FAIL")

        print(f"\nERROR: {exc}")

        print(
            "\nTransaction rolled back. "
            "No partial update was committed "
            "by this transaction."
        )

        raise

    finally:

        if conn is not None:

            conn.close()

            print(
                "\nDatabase connection closed."
            )


if __name__ == "__main__":

    try:
        main()

    except Exception:
        sys.exit(1)