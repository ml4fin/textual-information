import hashlib
import os

import numpy as np
import psycopg2


# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "all-MiniLM-L6-v2_chunk254_mean_l2_v1"
METHOD = "weakly_supervised"

START_DATE = "2014-01-01"
END_DATE = "2024-01-01"

N_THEMES = 5
N_MONTHS = 120

EMBEDDING_DIM = 384
FETCH_SIZE = 5000

# Minimum support included in the split-half analysis.
MIN_N = 10

# Number of deterministic split-half replicates.
N_SPLITS = 10


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
    """Convert a PostgreSQL embedding to a float32 NumPy vector."""

    x = np.asarray(
        value,
        dtype=np.float32,
    )

    if x.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"expected {(EMBEDDING_DIM,)}, "
            f"got {x.shape}."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Non-finite embedding values detected."
        )

    return x


def l2_normalize(x):
    """Return the L2-normalized vector, or None for an invalid norm."""

    norm = float(
        np.linalg.norm(x)
    )

    if not np.isfinite(norm) or norm <= 0:
        return None

    return x / norm


def deterministic_half(article_id, split_id):
    """
    Deterministically assign an article to split half A or B.

    A cryptographic hash is used instead of Python's built-in hash()
    to ensure stable assignments across processes and executions.
    """

    value = (
        f"{int(article_id)}|{split_id}"
        .encode("utf-8")
    )

    digest = hashlib.blake2b(
        value,
        digest_size=8,
    ).digest()

    number = int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )

    return number % 2


# =============================================================================
# SPLIT-HALF ACCUMULATORS
# =============================================================================

def build_split_accumulators(conn):
    """
    Build score-weighted split-half accumulators for each
    theme × month combination.

    For each deterministic split replicate, positive-score articles
    are assigned to half A or B. Each half preserves the same
    continuous weak-score weighting used for the candidate semantic
    centroid.
    """

    accumulators = {}

    sql = """
        SELECT
            a.article_id,
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
          AND e.model_name = %s
        ORDER BY
            a.publication_date,
            s.theme_id,
            a.article_id;
    """

    print()
    print(
        "Streaming positive weak-score observations..."
    )

    with conn.cursor(
        name="split_half_validation_cursor"
    ) as cur:

        cur.itersize = FETCH_SIZE

        cur.execute(
            sql,
            (
                START_DATE,
                END_DATE,
                METHOD,
                MODEL_NAME,
            ),
        )

        n_rows = 0

        for (
            article_id,
            theme_id,
            month,
            score,
            embedding,
        ) in cur:

            x = parse_embedding(
                embedding
            ).astype(
                np.float64,
                copy=False,
            )

            w = float(score)

            if not np.isfinite(w) or w <= 0:
                raise ValueError(
                    f"Invalid positive score: {w}"
                )

            theme_id = int(
                theme_id
            )

            if not 1 <= theme_id <= N_THEMES:
                raise RuntimeError(
                    f"Unexpected theme_id: {theme_id}"
                )

            key = (
                theme_id,
                month,
            )

            if key not in accumulators:

                accumulators[key] = {
                    "n_positive": 0,
                    "score_mass": 0.0,
                    "splits": [
                        {
                            "sum_a": np.zeros(
                                EMBEDDING_DIM,
                                dtype=np.float64,
                            ),
                            "sum_b": np.zeros(
                                EMBEDDING_DIM,
                                dtype=np.float64,
                            ),
                            "mass_a": 0.0,
                            "mass_b": 0.0,
                            "n_a": 0,
                            "n_b": 0,
                        }
                        for _ in range(N_SPLITS)
                    ],
                }

            group = accumulators[key]

            group["n_positive"] += 1
            group["score_mass"] += w

            for split_id in range(
                N_SPLITS
            ):

                half = deterministic_half(
                    article_id,
                    split_id,
                )

                split = group[
                    "splits"
                ][split_id]

                if half == 0:

                    split["sum_a"] += (
                        w * x
                    )

                    split["mass_a"] += w
                    split["n_a"] += 1

                else:

                    split["sum_b"] += (
                        w * x
                    )

                    split["mass_b"] += w
                    split["n_b"] += 1

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

    print()
    print(
        f"Total positive rows processed: "
        f"{n_rows:,}"
    )

    print(
        f"Theme-month groups with positive mass: "
        f"{len(accumulators):,}"
    )

    return accumulators


# =============================================================================
# SPLIT-HALF RELIABILITY
# =============================================================================

def compute_reliability(accumulators):

    results = []

    skipped_low_n = 0
    skipped_invalid_split = 0

    for (
        theme_id,
        month,
    ), group in accumulators.items():

        n_positive = group[
            "n_positive"
        ]

        if n_positive < MIN_N:
            skipped_low_n += 1
            continue

        similarities = []

        for split in group["splits"]:

            if (
                split["n_a"] == 0
                or split["n_b"] == 0
                or split["mass_a"] <= 0
                or split["mass_b"] <= 0
            ):
                continue

            raw_a = (
                split["sum_a"]
                / split["mass_a"]
            )

            raw_b = (
                split["sum_b"]
                / split["mass_b"]
            )

            centroid_a = l2_normalize(
                raw_a
            )

            centroid_b = l2_normalize(
                raw_b
            )

            if (
                centroid_a is None
                or centroid_b is None
            ):
                continue

            similarity = float(
                np.dot(
                    centroid_a,
                    centroid_b,
                )
            )

            similarity = float(
                np.clip(
                    similarity,
                    -1.0,
                    1.0,
                )
            )

            similarities.append(
                similarity
            )

        n_valid_splits = len(
            similarities
        )

        if n_valid_splits == 0:
            skipped_invalid_split += 1
            continue

        sims = np.asarray(
            similarities,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(sims)
        ):
            raise RuntimeError(
                f"Non-finite split-half similarities "
                f"for theme={theme_id}, month={month}."
            )

        results.append(
            {
                "theme": theme_id,
                "month": month,
                "n_positive": n_positive,
                "score_mass": group[
                    "score_mass"
                ],
                "n_valid_splits": n_valid_splits,
                "reliability_mean": float(
                    np.mean(sims)
                ),
                "reliability_median": float(
                    np.median(sims)
                ),
                "reliability_min": float(
                    np.min(sims)
                ),
                "reliability_max": float(
                    np.max(sims)
                ),
            }
        )

    print()
    print("=" * 72)
    print("SPLIT-HALF COMPUTATION")
    print("=" * 72)

    print(
        f"Groups with n >= {MIN_N}: "
        f"{len(results):,}"
    )

    print(
        f"Groups skipped for n < {MIN_N}: "
        f"{skipped_low_n:,}"
    )

    print(
        f"Groups skipped for invalid splits: "
        f"{skipped_invalid_split:,}"
    )

    return results


# =============================================================================
# SUMMARY
# =============================================================================

def summarize(results):

    print()
    print("=" * 72)
    print("SPLIT-HALF RELIABILITY")
    print("=" * 72)

    if not results:
        print(
            "No valid results."
        )
        return

    overall = np.asarray(
        [
            result["reliability_mean"]
            for result in results
        ],
        dtype=np.float64,
    )

    print()
    print(
        "Overall theme-month reliability:"
    )

    print(
        f"  n      = {len(overall)}"
    )

    print(
        f"  mean   = {np.mean(overall):.6f}"
    )

    print(
        f"  median = {np.median(overall):.6f}"
    )

    print(
        f"  p05    = {np.quantile(overall, 0.05):.6f}"
    )

    print(
        f"  p25    = {np.quantile(overall, 0.25):.6f}"
    )

    print(
        f"  p75    = {np.quantile(overall, 0.75):.6f}"
    )

    print(
        f"  p95    = {np.quantile(overall, 0.95):.6f}"
    )

    print(
        f"  min    = {np.min(overall):.6f}"
    )

    print(
        f"  max    = {np.max(overall):.6f}"
    )

    print()
    print(
        "By n_positive support:"
    )

    support_bands = [
        (10, 19),
        (20, 29),
        (30, 49),
        (50, 99),
        (100, 299),
        (300, 999),
        (1000, None),
    ]

    for low, high in support_bands:

        if high is None:

            vals = [
                result["reliability_mean"]
                for result in results
                if result["n_positive"] >= low
            ]

            label = f">= {low}"

        else:

            vals = [
                result["reliability_mean"]
                for result in results
                if (
                    low
                    <= result["n_positive"]
                    <= high
                )
            ]

            label = f"{low}-{high}"

        if not vals:

            print(
                f"  support {label:9s}: "
                f"n=0"
            )

            continue

        arr = np.asarray(
            vals,
            dtype=np.float64,
        )

        print(
            f"  support {label:9s}: "
            f"n={len(arr):3d}, "
            f"mean={np.mean(arr):.6f}, "
            f"median={np.median(arr):.6f}, "
            f"p05={np.quantile(arr, 0.05):.6f}, "
            f"min={np.min(arr):.6f}"
        )

    print()
    print(
        "By theme:"
    )

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        vals = [
            result["reliability_mean"]
            for result in results
            if result["theme"] == theme_id
        ]

        if not vals:

            print(
                f"  Theme {theme_id}: "
                f"no valid groups"
            )

            continue

        arr = np.asarray(
            vals,
            dtype=np.float64,
        )

        print(
            f"  Theme {theme_id}: "
            f"n={len(arr):3d}, "
            f"mean={np.mean(arr):.6f}, "
            f"median={np.median(arr):.6f}, "
            f"p05={np.quantile(arr, 0.05):.6f}, "
            f"min={np.min(arr):.6f}"
        )

    print()
    print(
        "Candidate minimum-n thresholds:"
    )

    thresholds = [
        10,
        20,
        30,
        50,
        100,
    ]

    for threshold in thresholds:

        subset = [
            result
            for result in results
            if result["n_positive"] >= threshold
        ]

        if not subset:
            continue

        arr = np.asarray(
            [
                result["reliability_mean"]
                for result in subset
            ],
            dtype=np.float64,
        )

        print()
        print(
            f"  n_positive >= {threshold}"
        )

        print(
            f"    groups retained : "
            f"{len(subset)} / "
            f"{N_THEMES * N_MONTHS}"
        )

        print(
            f"    mean reliability: "
            f"{np.mean(arr):.6f}"
        )

        print(
            f"    median          : "
            f"{np.median(arr):.6f}"
        )

        print(
            f"    p05             : "
            f"{np.quantile(arr, 0.05):.6f}"
        )

        print(
            f"    minimum         : "
            f"{np.min(arr):.6f}"
        )

        for theme_id in range(
            1,
            N_THEMES + 1,
        ):

            n_theme = sum(
                1
                for result in subset
                if result["theme"] == theme_id
            )

            print(
                f"      Theme {theme_id}: "
                f"{n_theme:3d} / {N_MONTHS}"
            )


# =============================================================================
# LOWEST RELIABILITY GROUPS
# =============================================================================

def show_lowest(results, n=20):

    print()
    print("=" * 72)
    print("LOWEST RELIABILITY THEME-MONTHS")
    print("=" * 72)

    if not results:
        print(
            "No valid results."
        )
        return

    ordered = sorted(
        results,
        key=lambda result: result[
            "reliability_mean"
        ],
    )

    for result in ordered[:n]:

        print(
            f"Theme={result['theme']} | "
            f"month={result['month']} | "
            f"n={result['n_positive']:,} | "
            f"mass={result['score_mass']:.6f} | "
            f"reliability="
            f"{result['reliability_mean']:.6f} | "
            f"split_min="
            f"{result['reliability_min']:.6f} | "
            f"valid_splits="
            f"{result['n_valid_splits']}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if MIN_N < 2:
        raise ValueError(
            "MIN_N must be at least 2."
        )

    if N_SPLITS <= 0:
        raise ValueError(
            "N_SPLITS must be positive."
        )

    print("=" * 72)
    print(
        "SEMANTIC DRIFT - CENTROID "
        "SPLIT-HALF VALIDATION V1"
    )
    print("=" * 72)

    print(
        f"Embedding model : "
        f"{MODEL_NAME}"
    )

    print(
        f"Thematic method : "
        f"{METHOD}"
    )

    print(
        f"Period          : "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        f"Minimum n       : "
        f"{MIN_N}"
    )

    print(
        f"Split replicates: "
        f"{N_SPLITS}"
    )

    conn = get_connection()

    try:

        accumulators = (
            build_split_accumulators(
                conn
            )
        )

        results = compute_reliability(
            accumulators
        )

        summarize(
            results
        )

        show_lowest(
            results,
            n=20,
        )

    finally:

        conn.close()

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()