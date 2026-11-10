import os
from collections import defaultdict

import numpy as np
import psycopg2
from psycopg2.extras import execute_values


# =============================================================================
# CONFIGURATION — FROZEN SEMANTIC DRIFT SPECIFICATION
# =============================================================================

SEMANTIC_DRIFT_VERSION = "semantic_drift_v1"

EMBEDDING_MODEL = "all-MiniLM-L6-v2_chunk254_mean_l2_v1"
THEMATIC_METHOD = "weakly_supervised"
THEMATIC_MODEL = "weak_v3_continuous_frozen"

START_DATE = "2014-01-01"
END_DATE = "2024-01-01"

N_THEMES = 5
EMBEDDING_DIM = 384
FETCH_SIZE = 5000

# Frozen after split-half reliability validation.
MIN_POSITIVE_ARTICLES = 50

# Frozen after temporal-reference validation.
HISTORY_WINDOW_MONTHS = 12
MIN_HISTORY_DEPTH = 3

# Safe default. Enable explicitly with SEMANTIC_DRIFT_WRITE=1.
WRITE_TO_DB = (
    os.getenv("SEMANTIC_DRIFT_WRITE", "0") == "1"
)

# Expected outputs from the validated pilots.
EXPECTED_RELIABLE_CENTROIDS = 461
EXPECTED_DRIFT_OBSERVATIONS = 443


# =============================================================================
# DATABASE
# =============================================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": (
        os.getenv("THESIS_DB_PASSWORD")
        or os.getenv("DB_PASSWORD")
    ),
}


def get_connection():

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. "
            "Set THESIS_DB_PASSWORD or DB_PASSWORD."
        )

    return psycopg2.connect(**DB_CONFIG)


# =============================================================================
# HELPERS
# =============================================================================

def parse_embedding(value):

    x = np.asarray(
        value,
        dtype=np.float32,
    )

    if x.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"expected {(EMBEDDING_DIM,)}, got {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Embedding contains non-finite values."
        )

    return x


def l2_normalize(x):

    norm = float(
        np.linalg.norm(x)
    )

    if not np.isfinite(norm) or norm <= 0:
        return None

    return x / norm


def cosine_distance(a, b):

    similarity = float(
        np.dot(a, b)
    )

    similarity = float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )

    return 1.0 - similarity


def month_difference(later, earlier):

    return (
        (later.year - earlier.year) * 12
        + later.month
        - earlier.month
    )


# =============================================================================
# STEP 1 — BUILD MONTHLY THEME CENTROIDS
# =============================================================================

def build_reliable_centroids(conn):
    """
    Build frozen score-weighted monthly theme centroids.

    Only observations with positive weak-supervision scores contribute.
    Theme-month groups are retained only when they contain at least
    MIN_POSITIVE_ARTICLES positive-score observations.

    For each retained group:

        raw_centroid =
            sum(score_i * embedding_i) / sum(score_i)

        centroid =
            raw_centroid / ||raw_centroid||_2
    """

    accumulators = defaultdict(
        lambda: {
            "weighted_sum": np.zeros(
                EMBEDDING_DIM,
                dtype=np.float64,
            ),
            "score_mass": 0.0,
            "n_positive": 0,
        }
    )

    sql = """
        SELECT
            s.theme_id,
            DATE_TRUNC(
                'month',
                a.publication_date
            )::date AS month,
            s.score,
            e.embedding
        FROM risk_emergence.v_valid_articles AS a
        JOIN risk_emergence.news_theme_scores_final AS s
            ON s.article_id = a.article_id
        JOIN risk_emergence.news_embeddings AS e
            ON e.article_id = a.article_id
        WHERE a.publication_date >= %s
          AND a.publication_date < %s
          AND s.method = %s
          AND s.score > 0
          AND e.model_name = %s;
    """

    print()
    print("STEP 1 — Building monthly theme centroids...")

    n_rows = 0

    with conn.cursor(
        name="semantic_drift_production_cursor"
    ) as cur:

        cur.itersize = FETCH_SIZE

        cur.execute(
            sql,
            (
                START_DATE,
                END_DATE,
                THEMATIC_METHOD,
                EMBEDDING_MODEL,
            ),
        )

        for (
            theme_id,
            month,
            score,
            embedding,
        ) in cur:

            theme_id = int(
                theme_id
            )

            if not 1 <= theme_id <= N_THEMES:
                raise ValueError(
                    f"Unexpected theme_id: {theme_id}"
                )

            w = float(score)

            if not np.isfinite(w) or w <= 0:
                raise ValueError(
                    f"Invalid thematic score: {w}"
                )

            x = parse_embedding(
                embedding
            ).astype(
                np.float64,
                copy=False,
            )

            key = (
                theme_id,
                month,
            )

            accumulators[key][
                "weighted_sum"
            ] += w * x

            accumulators[key][
                "score_mass"
            ] += w

            accumulators[key][
                "n_positive"
            ] += 1

            n_rows += 1

            if n_rows % 100_000 == 0:
                print(
                    f"Processed {n_rows:,} "
                    f"positive article-theme rows..."
                )

    if n_rows == 0:
        raise RuntimeError(
            "No positive weak-score observations found."
        )

    centroids = {}

    for key, acc in accumulators.items():

        if (
            acc["n_positive"]
            < MIN_POSITIVE_ARTICLES
        ):
            continue

        score_mass = float(
            acc["score_mass"]
        )

        if (
            not np.isfinite(score_mass)
            or score_mass <= 0
        ):
            raise ValueError(
                f"Invalid score mass for {key}: "
                f"{score_mass}"
            )

        raw_centroid = (
            acc["weighted_sum"]
            / score_mass
        )

        centroid = l2_normalize(
            raw_centroid
        )

        if centroid is None:
            raise ValueError(
                f"Invalid centroid for {key}"
            )

        centroid = centroid.astype(
            np.float32,
            copy=False,
        )

        centroid_norm = float(
            np.linalg.norm(centroid)
        )

        if not np.isclose(
            centroid_norm,
            1.0,
            atol=1e-5,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Centroid normalization failed "
                f"for {key}: norm={centroid_norm}"
            )

        centroids[key] = {
            "centroid": centroid,
            "n_positive": int(
                acc["n_positive"]
            ),
            "score_mass": score_mass,
        }

    print()
    print(
        f"Positive rows processed : "
        f"{n_rows:,}"
    )

    print(
        f"Positive-mass groups    : "
        f"{len(accumulators):,}"
    )

    print(
        f"Reliable centroids      : "
        f"{len(centroids):,}"
    )

    return centroids


# =============================================================================
# STEP 2 — ORGANIZE CENTROIDS BY THEME
# =============================================================================

def organize_by_theme(centroids):

    by_theme = defaultdict(list)

    for (
        theme_id,
        month,
    ), info in centroids.items():

        by_theme[theme_id].append(
            {
                "month": month,
                "centroid": info["centroid"],
                "n_positive": info["n_positive"],
                "score_mass": info["score_mass"],
            }
        )

    for theme_id in by_theme:

        by_theme[theme_id].sort(
            key=lambda x: x["month"]
        )

    return by_theme


# =============================================================================
# STEP 3 — COMPUTE FINAL SEMANTIC DRIFT
# =============================================================================

def compute_semantic_drift(by_theme):

    print()
    print(
        "STEP 2 — Computing final semantic drift..."
    )

    results = []

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        items = by_theme.get(
            theme_id,
            [],
        )

        for current_idx, current in enumerate(
            items
        ):

            current_month = current[
                "month"
            ]

            historical = []
            historical_months = []

            for past_idx in range(
                current_idx
            ):

                past = items[
                    past_idx
                ]

                gap = month_difference(
                    current_month,
                    past["month"],
                )

                if (
                    1
                    <= gap
                    <= HISTORY_WINDOW_MONTHS
                ):

                    historical.append(
                        past["centroid"].astype(
                            np.float64,
                            copy=False,
                        )
                    )

                    historical_months.append(
                        past["month"]
                    )

            history_depth = len(
                historical
            )

            if (
                history_depth
                < MIN_HISTORY_DEPTH
            ):
                continue

            reference_raw = np.mean(
                np.stack(
                    historical,
                    axis=0,
                ),
                axis=0,
            )

            reference = l2_normalize(
                reference_raw
            )

            if reference is None:
                raise ValueError(
                    "Invalid historical reference "
                    f"for theme={theme_id}, "
                    f"month={current_month}"
                )

            drift = cosine_distance(
                current["centroid"],
                reference,
            )

            if (
                not np.isfinite(drift)
                or drift < -1e-6
                or drift > 2.0 + 1e-6
            ):
                raise ValueError(
                    "Invalid semantic drift "
                    f"for theme={theme_id}, "
                    f"month={current_month}: "
                    f"{drift}"
                )

            results.append(
                {
                    "theme_id": theme_id,
                    "period_start": current_month,
                    "period_type": "month",
                    "semantic_drift": float(
                        drift
                    ),
                    "history_depth": int(
                        history_depth
                    ),
                    "n_positive": int(
                        current["n_positive"]
                    ),
                    "score_mass": float(
                        current["score_mass"]
                    ),
                    "history_first": min(
                        historical_months
                    ),
                    "history_last": max(
                        historical_months
                    ),
                }
            )

    results.sort(
        key=lambda x: (
            x["theme_id"],
            x["period_start"],
        )
    )

    print(
        f"Semantic drift observations: "
        f"{len(results):,}"
    )

    return results


# =============================================================================
# STEP 4 — VALIDATION
# =============================================================================

def validate_results(
    centroids,
    results,
):

    print()
    print("=" * 72)
    print("PRODUCTION VALIDATION")
    print("=" * 72)

    print(
        f"Reliable centroids expected : "
        f"{EXPECTED_RELIABLE_CENTROIDS}"
    )

    print(
        f"Reliable centroids actual   : "
        f"{len(centroids)}"
    )

    if (
        len(centroids)
        != EXPECTED_RELIABLE_CENTROIDS
    ):
        raise RuntimeError(
            "Reliable centroid count differs "
            "from validated pilot."
        )

    print()
    print(
        f"Drift observations expected : "
        f"{EXPECTED_DRIFT_OBSERVATIONS}"
    )

    print(
        f"Drift observations actual   : "
        f"{len(results)}"
    )

    if (
        len(results)
        != EXPECTED_DRIFT_OBSERVATIONS
    ):
        raise RuntimeError(
            "Semantic drift observation count "
            "differs from validated pilot."
        )

    keys = [
        (
            x["theme_id"],
            x["period_start"],
            x["period_type"],
        )
        for x in results
    ]

    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Duplicate "
            "(theme_id, period_start, period_type) "
            "keys detected."
        )

    if not results:
        raise RuntimeError(
            "No semantic drift results generated."
        )

    drift = np.asarray(
        [
            x["semantic_drift"]
            for x in results
        ],
        dtype=np.float64,
    )

    depths = np.asarray(
        [
            x["history_depth"]
            for x in results
        ],
        dtype=np.int64,
    )

    if not np.all(
        np.isfinite(drift)
    ):
        raise RuntimeError(
            "Non-finite semantic drift detected."
        )

    if np.any(drift < -1e-6):
        raise RuntimeError(
            "Negative cosine distance detected."
        )

    if np.any(
        drift > 2.0 + 1e-6
    ):
        raise RuntimeError(
            "Cosine distance above 2 detected."
        )

    if np.any(
        depths < MIN_HISTORY_DEPTH
    ):
        raise RuntimeError(
            "Observation below frozen "
            "minimum history depth detected."
        )

    if np.any(
        depths > HISTORY_WINDOW_MONTHS
    ):
        raise RuntimeError(
            "History depth exceeds frozen "
            "calendar window."
        )

    print()
    print(
        f"Drift min    : "
        f"{np.min(drift):.6f}"
    )

    print(
        f"Drift mean   : "
        f"{np.mean(drift):.6f}"
    )

    print(
        f"Drift median : "
        f"{np.median(drift):.6f}"
    )

    print(
        f"Drift p95    : "
        f"{np.quantile(drift, 0.95):.6f}"
    )

    print(
        f"Drift max    : "
        f"{np.max(drift):.6f}"
    )

    print()
    print("Observations by theme:")

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        subset = [
            x
            for x in results
            if x["theme_id"] == theme_id
        ]

        if not subset:
            print(
                f"  Theme {theme_id}: n=0"
            )
            continue

        values = np.asarray(
            [
                x["semantic_drift"]
                for x in subset
            ],
            dtype=np.float64,
        )

        print(
            f"  Theme {theme_id}: "
            f"n={len(subset):3d}, "
            f"mean={np.mean(values):.6f}, "
            f"median={np.median(values):.6f}, "
            f"min={np.min(values):.6f}, "
            f"max={np.max(values):.6f}"
        )

    print()
    print("History depth distribution:")

    for depth in sorted(
        np.unique(depths)
    ):

        print(
            f"  depth={depth:2d}: "
            f"{np.sum(depths == depth):3d}"
        )


# =============================================================================
# STEP 5 — PREVIEW
# =============================================================================

def preview_results(results):

    print()
    print("=" * 72)
    print("RESULT PREVIEW")
    print("=" * 72)

    print()
    print("First 15 observations:")

    for x in results[:15]:

        print(
            f"theme={x['theme_id']} | "
            f"month={x['period_start']} | "
            f"drift={x['semantic_drift']:.8f} | "
            f"depth={x['history_depth']} | "
            f"n_positive={x['n_positive']} | "
            f"history="
            f"{x['history_first']} -> "
            f"{x['history_last']}"
        )

    print()
    print("Last 15 observations:")

    for x in results[-15:]:

        print(
            f"theme={x['theme_id']} | "
            f"month={x['period_start']} | "
            f"drift={x['semantic_drift']:.8f} | "
            f"depth={x['history_depth']} | "
            f"n_positive={x['n_positive']} | "
            f"history="
            f"{x['history_first']} -> "
            f"{x['history_last']}"
        )


# =============================================================================
# STEP 6 — DATABASE WRITE
# =============================================================================

def write_results(
    conn,
    results,
):

    print()
    print("=" * 72)
    print("DATABASE WRITE")
    print("=" * 72)

    if not WRITE_TO_DB:

        print(
            "WRITE_TO_DB = False"
        )

        print(
            "No database changes performed."
        )

        return

    if (
        len(results)
        != EXPECTED_DRIFT_OBSERVATIONS
    ):
        raise RuntimeError(
            "Refusing database write: "
            f"expected {EXPECTED_DRIFT_OBSERVATIONS} "
            f"observations, got {len(results)}."
        )

    computation_notes = (
        f"{SEMANTIC_DRIFT_VERSION}; "
        f"embedding={EMBEDDING_MODEL}; "
        f"theme_method={THEMATIC_METHOD}; "
        f"theme_model={THEMATIC_MODEL}; "
        "centroid=positive_score_weighted_l2; "
        f"min_positive_articles={MIN_POSITIVE_ARTICLES}; "
        "reference=equal_weight_reliable_monthly_centroids; "
        f"history_window_months={HISTORY_WINDOW_MONTHS}; "
        f"min_history_depth={MIN_HISTORY_DEPTH}; "
        "distance=1-cosine"
    )

    insert_sql = """
        INSERT INTO risk_emergence.news_indicators (
            theme_id,
            period_start,
            period_type,
            semantic_drift,
            computation_notes
        )
        VALUES %s;
    """

    rows = [
        (
            x["theme_id"],
            x["period_start"],
            x["period_type"],
            x["semantic_drift"],
            computation_notes,
        )
        for x in results
    ]

    print(
        f"Preparing to insert "
        f"{len(rows):,} rows..."
    )

    try:

        with conn.cursor() as cur:

            # Refuse to overwrite an existing V1 materialization.
            cur.execute(
                """
                SELECT COUNT(*)
                FROM risk_emergence.news_indicators
                WHERE period_type = 'month'
                  AND semantic_drift IS NOT NULL
                  AND computation_notes LIKE %s;
                """,
                (
                    f"{SEMANTIC_DRIFT_VERSION};%",
                ),
            )

            existing_rows = int(
                cur.fetchone()[0]
            )

            print(
                "Existing semantic-drift V1 rows: "
                f"{existing_rows}"
            )

            if existing_rows != 0:
                raise RuntimeError(
                    "Refusing database write: "
                    f"{SEMANTIC_DRIFT_VERSION} "
                    "already exists."
                )

            execute_values(
                cur,
                insert_sql,
                rows,
                page_size=500,
            )

            # Validate the inserted materialization before commit.
            cur.execute(
                """
                SELECT
                    COUNT(*) AS n_rows,
                    COUNT(semantic_drift) AS n_drift,
                    MIN(semantic_drift) AS min_drift,
                    AVG(semantic_drift) AS mean_drift,
                    MAX(semantic_drift) AS max_drift
                FROM risk_emergence.news_indicators
                WHERE period_type = 'month'
                  AND semantic_drift IS NOT NULL
                  AND computation_notes LIKE %s;
                """,
                (
                    f"{SEMANTIC_DRIFT_VERSION};%",
                ),
            )

            (
                n_rows,
                n_drift,
                min_drift,
                mean_drift,
                max_drift,
            ) = cur.fetchone()

            n_rows = int(n_rows)
            n_drift = int(n_drift)

            print()
            print("Pre-commit validation:")

            print(
                f"  rows          : {n_rows}"
            )

            print(
                f"  drift non-null: {n_drift}"
            )

            print(
                f"  min           : "
                f"{float(min_drift):.6f}"
            )

            print(
                f"  mean          : "
                f"{float(mean_drift):.6f}"
            )

            print(
                f"  max           : "
                f"{float(max_drift):.6f}"
            )

            if (
                n_rows
                != EXPECTED_DRIFT_OBSERVATIONS
            ):
                raise RuntimeError(
                    "Pre-commit validation failed: "
                    f"expected "
                    f"{EXPECTED_DRIFT_OBSERVATIONS} "
                    f"rows, got {n_rows}."
                )

            if (
                n_drift
                != EXPECTED_DRIFT_OBSERVATIONS
            ):
                raise RuntimeError(
                    "Pre-commit validation failed: "
                    "NULL semantic drift detected."
                )

        conn.commit()

        print()
        print("COMMIT successful.")

        print(
            f"{len(results):,} semantic-drift "
            f"observations written."
        )

    except Exception:

        conn.rollback()

        print()
        print(
            "ROLLBACK executed. "
            "No partial write retained."
        )

        raise


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 72)
    print("SEMANTIC DRIFT — PRODUCTION V1")
    print("=" * 72)

    print(
        f"Embedding model      : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Thematic method      : "
        f"{THEMATIC_METHOD}"
    )

    print(
        f"Thematic model       : "
        f"{THEMATIC_MODEL}"
    )

    print(
        f"Period               : "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        f"Min positive articles: "
        f"{MIN_POSITIVE_ARTICLES}"
    )

    print(
        f"History window       : "
        f"{HISTORY_WINDOW_MONTHS} months"
    )

    print(
        f"Min history depth    : "
        f"{MIN_HISTORY_DEPTH}"
    )

    print(
        f"WRITE_TO_DB          : "
        f"{WRITE_TO_DB}"
    )

    conn = get_connection()

    try:

        centroids = (
            build_reliable_centroids(
                conn
            )
        )

        by_theme = organize_by_theme(
            centroids
        )

        results = compute_semantic_drift(
            by_theme
        )

        validate_results(
            centroids,
            results,
        )

        preview_results(
            results
        )

        write_results(
            conn,
            results,
        )

    finally:

        conn.close()

    print()
    print("=" * 72)

    if WRITE_TO_DB:
        print(
            "PRODUCTION RUN COMPLETE"
        )
    else:
        print(
            "PRODUCTION VALIDATION COMPLETE — "
            "NO DATABASE WRITE"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()