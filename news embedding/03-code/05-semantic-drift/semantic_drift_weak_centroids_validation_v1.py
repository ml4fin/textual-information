import os
from collections import defaultdict

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

# Candidate reliability rule; not yet frozen.
MIN_POSITIVE_ARTICLES = 10


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
    """Return the L2-normalized vector."""

    norm = float(
        np.linalg.norm(x)
    )

    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            f"Invalid vector norm: {norm}"
        )

    return x / norm


# =============================================================================
# CENTROID CONSTRUCTION
# =============================================================================

def build_centroids(conn):
    """
    Build score-weighted monthly centroids for the selected method.

    Only observations with strictly positive thematic scores are used.

    For each theme × month combination:

        raw_centroid =
            sum(score_i * embedding_i) / sum(score_i)

        centroid =
            raw_centroid / ||raw_centroid||_2

    No database writes are performed.
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
        name="weak_centroid_validation_cursor"
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
            theme_id,
            month,
            score,
            embedding,
        ) in cur:

            x = parse_embedding(
                embedding
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

            acc = accumulators[key]

            acc["weighted_sum"] += (
                w
                * x.astype(
                    np.float64,
                    copy=False,
                )
            )

            acc["score_mass"] += w
            acc["n_positive"] += 1

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

    centroids = {}

    for key, acc in accumulators.items():

        if (
            not np.isfinite(
                acc["score_mass"]
            )
            or acc["score_mass"] <= 0
        ):
            raise ValueError(
                f"Invalid score mass for {key}: "
                f"{acc['score_mass']}"
            )

        if acc["n_positive"] <= 0:
            raise ValueError(
                f"Invalid positive article count "
                f"for {key}."
            )

        raw_centroid = (
            acc["weighted_sum"]
            / acc["score_mass"]
        )

        raw_norm = float(
            np.linalg.norm(
                raw_centroid
            )
        )

        centroid = l2_normalize(
            raw_centroid
        ).astype(
            np.float32,
            copy=False,
        )

        centroids[key] = {
            "centroid": centroid,
            "n_positive": acc["n_positive"],
            "score_mass": acc["score_mass"],
            "raw_norm": raw_norm,
        }

    return centroids


# =============================================================================
# SUPPORT VALIDATION
# =============================================================================

def validate_support(centroids):

    print()
    print("=" * 72)
    print("SUPPORT VALIDATION")
    print("=" * 72)

    theoretical_total = (
        N_THEMES
        * N_MONTHS
    )

    print(
        f"Theme-months with positive score mass: "
        f"{len(centroids):,} / {theoretical_total:,}"
    )

    thresholds = [
        1,
        5,
        10,
        20,
        30,
        50,
        100,
    ]

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        values = [
            info["n_positive"]
            for (
                theme,
                _month,
            ), info in centroids.items()
            if theme == theme_id
        ]

        print()
        print(
            f"Theme {theme_id}"
        )

        if not values:
            print(
                "  No positive-score months."
            )
            continue

        arr = np.asarray(
            values,
            dtype=np.int64,
        )

        print(
            f"  months with positive mass : "
            f"{len(arr)} / {N_MONTHS}"
        )

        print(
            f"  n_positive min/median/max : "
            f"{arr.min():,} / "
            f"{np.median(arr):.1f} / "
            f"{arr.max():,}"
        )

        for threshold in thresholds:

            n_valid = int(
                np.sum(
                    arr >= threshold
                )
            )

            print(
                f"  n >= {threshold:3d} : "
                f"{n_valid:3d} / {N_MONTHS}"
            )


# =============================================================================
# CANDIDATE RELIABILITY RULE
# =============================================================================

def filter_valid_centroids(centroids):

    valid = {
        key: info
        for key, info in centroids.items()
        if (
            info["n_positive"]
            >= MIN_POSITIVE_ARTICLES
            and info["score_mass"] > 0
        )
    }

    print()
    print("=" * 72)
    print("CANDIDATE RELIABILITY RULE")
    print("=" * 72)

    print(
        f"Minimum positive articles: "
        f"{MIN_POSITIVE_ARTICLES}"
    )

    print(
        f"Valid theme-month centroids: "
        f"{len(valid):,} / "
        f"{N_THEMES * N_MONTHS:,}"
    )

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        n_valid = sum(
            1
            for (
                theme,
                _month,
            ) in valid
            if theme == theme_id
        )

        print(
            f"  Theme {theme_id}: "
            f"{n_valid:3d} / {N_MONTHS}"
        )

    return valid


# =============================================================================
# CENTROID GEOMETRY
# =============================================================================

def validate_centroid_geometry(valid):

    if not valid:
        raise RuntimeError(
            "No centroids satisfy the candidate "
            "reliability rule."
        )

    print()
    print("=" * 72)
    print("CENTROID GEOMETRY")
    print("=" * 72)

    normalized_norms = np.asarray(
        [
            np.linalg.norm(
                info["centroid"]
            )
            for info in valid.values()
        ],
        dtype=np.float64,
    )

    raw_norms = np.asarray(
        [
            info["raw_norm"]
            for info in valid.values()
        ],
        dtype=np.float64,
    )

    score_masses = np.asarray(
        [
            info["score_mass"]
            for info in valid.values()
        ],
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

    if not np.all(
        np.isfinite(
            raw_norms
        )
    ):
        raise RuntimeError(
            "Non-finite raw centroid norms detected."
        )

    if not np.all(
        np.isfinite(
            score_masses
        )
    ):
        raise RuntimeError(
            "Non-finite score masses detected."
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

    print(
        "Normalized norm: "
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
        "Score mass: "
        f"min={score_masses.min():.6f}, "
        f"median={np.median(score_masses):.6f}, "
        f"max={score_masses.max():.6f}"
    )


# =============================================================================
# CONSECUTIVE-MONTH STABILITY
# =============================================================================

def analyze_consecutive_months(valid):
    """
    Diagnose consecutive-month centroid stability.

    This analysis is used only to assess whether low-support centroids
    exhibit abnormal instability. It does not define the final semantic
    drift measure.
    """

    print()
    print("=" * 72)
    print("CONSECUTIVE-MONTH STABILITY DIAGNOSTIC")
    print("=" * 72)

    observations = []

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        theme_items = sorted(
            [
                (
                    month,
                    info,
                )
                for (
                    theme,
                    month,
                ), info in valid.items()
                if theme == theme_id
            ],
            key=lambda item: item[0],
        )

        for i in range(
            1,
            len(theme_items),
        ):

            prev_month, prev_info = (
                theme_items[i - 1]
            )

            curr_month, curr_info = (
                theme_items[i]
            )

            month_difference = (
                (curr_month.year - prev_month.year)
                * 12
                + curr_month.month
                - prev_month.month
            )

            if month_difference != 1:
                continue

            similarity = float(
                np.dot(
                    prev_info["centroid"],
                    curr_info["centroid"],
                )
            )

            similarity = float(
                np.clip(
                    similarity,
                    -1.0,
                    1.0,
                )
            )

            min_support = min(
                prev_info["n_positive"],
                curr_info["n_positive"],
            )

            observations.append(
                {
                    "theme": theme_id,
                    "similarity": similarity,
                    "min_support": min_support,
                }
            )

    print(
        f"Consecutive valid pairs: "
        f"{len(observations):,}"
    )

    if not observations:
        return

    similarities = np.asarray(
        [
            observation["similarity"]
            for observation in observations
        ],
        dtype=np.float64,
    )

    print()
    print(
        "Overall cosine similarity:"
    )

    print(
        f"  mean   = "
        f"{np.mean(similarities):.6f}"
    )

    print(
        f"  median = "
        f"{np.median(similarities):.6f}"
    )

    print(
        f"  p05    = "
        f"{np.quantile(similarities, 0.05):.6f}"
    )

    print(
        f"  p95    = "
        f"{np.quantile(similarities, 0.95):.6f}"
    )

    print(
        f"  min    = "
        f"{np.min(similarities):.6f}"
    )

    print()
    print(
        "By minimum support of adjacent pair:"
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
                observation["similarity"]
                for observation in observations
                if observation["min_support"] >= low
            ]

            label = f">= {low}"

        else:

            vals = [
                observation["similarity"]
                for observation in observations
                if (
                    low
                    <= observation["min_support"]
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

        vals = np.asarray(
            vals,
            dtype=np.float64,
        )

        print(
            f"  support {label:9s}: "
            f"n={len(vals):3d}, "
            f"mean={np.mean(vals):.6f}, "
            f"median={np.median(vals):.6f}, "
            f"min={np.min(vals):.6f}"
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
            observation["similarity"]
            for observation in observations
            if observation["theme"] == theme_id
        ]

        if not vals:

            print(
                f"  Theme {theme_id}: "
                f"no consecutive pairs"
            )

            continue

        vals = np.asarray(
            vals,
            dtype=np.float64,
        )

        print(
            f"  Theme {theme_id}: "
            f"n={len(vals):3d}, "
            f"mean={np.mean(vals):.6f}, "
            f"median={np.median(vals):.6f}, "
            f"p05={np.quantile(vals, 0.05):.6f}, "
            f"min={np.min(vals):.6f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if MIN_POSITIVE_ARTICLES <= 0:
        raise ValueError(
            "MIN_POSITIVE_ARTICLES must be positive."
        )

    print("=" * 72)
    print(
        "SEMANTIC DRIFT - WEAK CENTROIDS "
        "VALIDATION V1"
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
        f"Candidate min n : "
        f"{MIN_POSITIVE_ARTICLES}"
    )

    conn = get_connection()

    try:

        centroids = build_centroids(
            conn
        )

        validate_support(
            centroids
        )

        valid = filter_valid_centroids(
            centroids
        )

        validate_centroid_geometry(
            valid
        )

        analyze_consecutive_months(
            valid
        )

    finally:
        conn.close()

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()