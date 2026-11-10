import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import psycopg2


# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "all-MiniLM-L6-v2_chunk254_mean_l2_v1"

METHODS = [
    "keyword",
    "supervised_v3",
    "weakly_supervised",
]

TOP_FRACTION = 0.10

START_DATE = "2014-01-01"
END_DATE = "2024-01-01"

N_THEMES = 5
N_MONTHS = 120

EMBEDDING_DIM = 384
FETCH_SIZE = 5000


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


def normalize_vector(x):
    """Return the L2-normalized vector."""

    norm = float(
        np.linalg.norm(x)
    )

    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            f"Invalid centroid norm: {norm}"
        )

    return x / norm


# =============================================================================
# CENTROID CONSTRUCTION
# =============================================================================

def build_monthly_centroids(conn):
    """
    Build monthly centroids for each method × theme × month combination.

    Within each group, only the top TOP_FRACTION of articles ranked by
    thematic score are selected. Ranking is performed independently for
    each method × theme × month combination.

    The centroid is the arithmetic mean of the selected embeddings,
    followed by L2 normalization:

        raw_centroid =
            sum(embedding_i) / n_selected

        centroid =
            raw_centroid / ||raw_centroid||_2

    No database writes are performed.
    """

    accumulators = defaultdict(
        lambda: {
            "vector_sum": np.zeros(
                EMBEDDING_DIM,
                dtype=np.float64,
            ),
            "n_articles": 0,
            "n_total": None,
        }
    )

    sql = """
        WITH ranked AS (
            SELECT
                s.article_id,
                s.method,
                s.theme_id,
                DATE_TRUNC(
                    'month',
                    a.publication_date
                )::date AS month,
                s.score,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        s.method,
                        s.theme_id,
                        DATE_TRUNC(
                            'month',
                            a.publication_date
                        )
                    ORDER BY
                        s.score DESC,
                        s.article_id
                ) AS rn,

                COUNT(*) OVER (
                    PARTITION BY
                        s.method,
                        s.theme_id,
                        DATE_TRUNC(
                            'month',
                            a.publication_date
                        )
                ) AS n_total

            FROM risk_emergence.v_valid_articles AS a
            JOIN risk_emergence.news_theme_scores_final AS s
                ON s.article_id = a.article_id

            WHERE a.publication_date >= %s
              AND a.publication_date < %s
              AND s.method = ANY(%s)
        )

        SELECT
            r.method,
            r.theme_id,
            r.month,
            r.score,
            e.embedding,
            r.n_total

        FROM ranked AS r

        JOIN risk_emergence.news_embeddings AS e
            ON e.article_id = r.article_id

        WHERE r.rn <= GREATEST(
            1,
            CEIL(r.n_total * %s)
        )
          AND e.model_name = %s

        ORDER BY
            r.month,
            r.method,
            r.theme_id,
            r.rn;
    """

    cursor_name = "semantic_drift_percentile_cursor"

    with conn.cursor(
        name=cursor_name
    ) as cur:

        cur.itersize = FETCH_SIZE

        cur.execute(
            sql,
            (
                START_DATE,
                END_DATE,
                METHODS,
                TOP_FRACTION,
                MODEL_NAME,
            ),
        )

        n_rows = 0

        for (
            method,
            theme_id,
            month,
            score,
            embedding,
            n_total,
        ) in cur:

            x = parse_embedding(
                embedding
            )

            score = float(score)

            if not np.isfinite(score):
                raise ValueError(
                    f"Non-finite score: {score}"
                )

            if n_total <= 0:
                raise ValueError(
                    f"Invalid group size: {n_total}"
                )

            key = (
                method,
                int(theme_id),
                month,
            )

            acc = accumulators[key]

            if acc["n_total"] is None:
                acc["n_total"] = int(n_total)

            elif acc["n_total"] != int(n_total):
                raise RuntimeError(
                    f"Inconsistent group size for {key}: "
                    f"{acc['n_total']} vs {n_total}."
                )

            acc["vector_sum"] += x.astype(
                np.float64,
                copy=False,
            )

            acc["n_articles"] += 1

            n_rows += 1

            if n_rows % 100_000 == 0:
                print(
                    f"Processed {n_rows:,} "
                    f"selected article-theme rows..."
                )

    if n_rows == 0:
        raise RuntimeError(
            "No article-theme observations "
            "matched the configured specification."
        )

    print()
    print(
        f"Total selected rows processed: "
        f"{n_rows:,}"
    )

    centroids = {}

    for key, acc in accumulators.items():

        if acc["n_articles"] <= 0:
            raise ValueError(
                f"Invalid selected article count "
                f"for {key}."
            )

        expected_selected = max(
            1,
            int(
                np.ceil(
                    acc["n_total"]
                    * TOP_FRACTION
                )
            ),
        )

        if acc["n_articles"] != expected_selected:
            raise RuntimeError(
                f"Top-fraction selection mismatch "
                f"for {key}: "
                f"selected={acc['n_articles']}, "
                f"expected={expected_selected}."
            )

        raw_centroid = (
            acc["vector_sum"]
            / acc["n_articles"]
        )

        raw_norm = float(
            np.linalg.norm(
                raw_centroid
            )
        )

        centroid = normalize_vector(
            raw_centroid
        ).astype(
            np.float32,
            copy=False,
        )

        centroids[key] = {
            "centroid": centroid,
            "n_articles": acc["n_articles"],
            "n_total": acc["n_total"],
            "raw_norm": raw_norm,
        }

    return centroids


# =============================================================================
# VALIDATION
# =============================================================================

def validate_centroids(centroids):

    if not centroids:
        raise RuntimeError(
            "No centroids were constructed."
        )

    print()
    print("=" * 70)
    print("CENTROID VALIDATION")
    print("=" * 70)

    theoretical_total = (
        len(METHODS)
        * N_THEMES
        * N_MONTHS
    )

    print(
        f"Centroids constructed: "
        f"{len(centroids):,}"
    )

    print(
        f"Theoretical combinations "
        f"({len(METHODS)} methods × "
        f"{N_THEMES} themes × "
        f"{N_MONTHS} months): "
        f"{theoretical_total:,}"
    )

    counts = defaultdict(int)

    normalized_norms = []
    raw_norms = []
    article_counts = []
    selection_fractions = []

    for (
        method,
        theme_id,
        month,
    ), info in centroids.items():

        if method not in METHODS:
            raise RuntimeError(
                f"Unexpected method: {method}"
            )

        if not 1 <= theme_id <= N_THEMES:
            raise RuntimeError(
                f"Unexpected theme_id: {theme_id}"
            )

        counts[method] += 1

        normalized_norms.append(
            float(
                np.linalg.norm(
                    info["centroid"]
                )
            )
        )

        raw_norms.append(
            info["raw_norm"]
        )

        article_counts.append(
            info["n_articles"]
        )

        selection_fractions.append(
            info["n_articles"]
            / info["n_total"]
        )

    normalized_norms = np.asarray(
        normalized_norms,
        dtype=np.float64,
    )

    raw_norms = np.asarray(
        raw_norms,
        dtype=np.float64,
    )

    article_counts = np.asarray(
        article_counts,
        dtype=np.int64,
    )

    selection_fractions = np.asarray(
        selection_fractions,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            normalized_norms
        )
    ):
        raise RuntimeError(
            "Non-finite normalized centroid norms detected."
        )

    if not np.allclose(
        normalized_norms,
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Centroid normalization validation failed."
        )

    print()

    for method in METHODS:
        print(
            f"{method:20s}: "
            f"{counts[method]:,} centroids"
        )

    print()

    print(
        "Normalized centroid norm: "
        f"min={normalized_norms.min():.8f}, "
        f"max={normalized_norms.max():.8f}"
    )

    print(
        "Raw centroid norm: "
        f"min={raw_norms.min():.6f}, "
        f"median={np.median(raw_norms):.6f}, "
        f"max={raw_norms.max():.6f}"
    )

    print(
        "Selected articles per centroid: "
        f"min={article_counts.min():,}, "
        f"median={int(np.median(article_counts)):,}, "
        f"max={article_counts.max():,}"
    )

    print(
        "Realized selection fraction: "
        f"min={selection_fractions.min():.4f}, "
        f"median={np.median(selection_fractions):.4f}, "
        f"max={selection_fractions.max():.4f}"
    )


# =============================================================================
# CROSS-METHOD COMPARISON
# =============================================================================

def compare_methods(centroids):

    print()
    print("=" * 70)
    print("CROSS-METHOD CENTROID SIMILARITY")
    print("=" * 70)

    for method_a, method_b in combinations(
        METHODS,
        2,
    ):

        similarities = []

        for theme_id in range(
            1,
            N_THEMES + 1,
        ):

            values_a = {
                month: info
                for (
                    method,
                    theme,
                    month,
                ), info in centroids.items()
                if method == method_a
                and theme == theme_id
            }

            values_b = {
                month: info
                for (
                    method,
                    theme,
                    month,
                ), info in centroids.items()
                if method == method_b
                and theme == theme_id
            }

            common_months = sorted(
                set(values_a)
                & set(values_b)
            )

            for month in common_months:

                a = values_a[
                    month
                ]["centroid"]

                b = values_b[
                    month
                ]["centroid"]

                similarity = float(
                    np.dot(
                        a,
                        b,
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
                    (
                        theme_id,
                        month,
                        similarity,
                    )
                )

        values = np.asarray(
            [
                similarity
                for _, _, similarity
                in similarities
            ],
            dtype=np.float64,
        )

        print()
        print(
            f"{method_a}  vs  {method_b}"
        )

        print(
            f"N comparisons: "
            f"{len(values):,}"
        )

        if len(values) == 0:
            print(
                "No common observations."
            )
            continue

        print(
            f"mean   = "
            f"{np.mean(values):.6f}"
        )

        print(
            f"median = "
            f"{np.median(values):.6f}"
        )

        print(
            f"p05    = "
            f"{np.quantile(values, 0.05):.6f}"
        )

        print(
            f"p25    = "
            f"{np.quantile(values, 0.25):.6f}"
        )

        print(
            f"p75    = "
            f"{np.quantile(values, 0.75):.6f}"
        )

        print(
            f"p95    = "
            f"{np.quantile(values, 0.95):.6f}"
        )

        print(
            f"min    = "
            f"{np.min(values):.6f}"
        )

        print(
            f"max    = "
            f"{np.max(values):.6f}"
        )

        print()
        print(
            "Mean similarity by theme:"
        )

        for theme_id in range(
            1,
            N_THEMES + 1,
        ):

            theme_values = np.asarray(
                [
                    similarity
                    for (
                        theme,
                        _month,
                        similarity,
                    ) in similarities
                    if theme == theme_id
                ],
                dtype=np.float64,
            )

            if len(theme_values) > 0:

                print(
                    f"  theme {theme_id}: "
                    f"{np.mean(theme_values):.6f} "
                    f"(n={len(theme_values)})"
                )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not 0.0 < TOP_FRACTION <= 1.0:
        raise ValueError(
            "TOP_FRACTION must be in (0, 1]."
        )

    print("=" * 70)
    print(
        "SEMANTIC DRIFT - TOP-PERCENTILE "
        "CENTROID PILOT V1"
    )
    print("=" * 70)

    print(
        f"Embedding model : "
        f"{MODEL_NAME}"
    )

    print(
        f"Top fraction    : "
        f"{TOP_FRACTION:.2%}"
    )

    print(
        f"Period          : "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        f"Methods         : "
        f"{METHODS}"
    )

    conn = get_connection()

    try:

        centroids = build_monthly_centroids(
            conn
        )

        validate_centroids(
            centroids
        )

        compare_methods(
            centroids
        )

    finally:
        conn.close()

    print()
    print("=" * 70)
    print("PILOT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()