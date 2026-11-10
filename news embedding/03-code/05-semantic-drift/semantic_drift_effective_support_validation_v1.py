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

MIN_N_FOR_RELIABILITY = 10
N_SPLITS = 10

# Current candidate support rule under evaluation.
CANDIDATE_MIN_N = 50


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
            f"expected {(EMBEDDING_DIM,)}, got {x.shape}."
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

    BLAKE2b is used instead of Python's built-in hash() so that
    assignments remain stable across processes and executions.
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
# DATA LOADING AND ACCUMULATION
# =============================================================================

def build_accumulators(conn):
    """
    Build theme-month accumulators for effective support and
    split-half reliability.

    Effective support is defined as:

        N_eff = (sum_i w_i)^2 / sum_i(w_i^2)

    where w_i is the continuous weak-supervision score.

    The split-half accumulators preserve the same score weighting used
    in the candidate semantic centroid.
    """

    groups = {}

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
        name="effective_support_cursor"
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

            w = float(score)

            if not np.isfinite(w) or w <= 0:
                raise ValueError(
                    f"Invalid positive score: {w}"
                )

            x = parse_embedding(
                embedding
            ).astype(
                np.float64,
                copy=False,
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

            if key not in groups:

                groups[key] = {
                    "n_positive": 0,
                    "sum_w": 0.0,
                    "sum_w2": 0.0,
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

            group = groups[key]

            group["n_positive"] += 1
            group["sum_w"] += w
            group["sum_w2"] += w * w

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
        f"{len(groups):,}"
    )

    return groups


# =============================================================================
# EFFECTIVE SUPPORT AND RELIABILITY
# =============================================================================

def compute_results(groups):

    results = []

    invalid_reliability_groups = 0

    for (
        theme_id,
        month,
    ), group in groups.items():

        n_positive = group[
            "n_positive"
        ]

        sum_w = float(
            group["sum_w"]
        )

        sum_w2 = float(
            group["sum_w2"]
        )

        if (
            not np.isfinite(sum_w)
            or not np.isfinite(sum_w2)
            or sum_w <= 0
            or sum_w2 <= 0
        ):
            raise ValueError(
                f"Invalid weight statistics for "
                f"theme={theme_id}, month={month}."
            )

        n_eff = (
            (sum_w * sum_w)
            / sum_w2
        )

        if not np.isfinite(n_eff):
            raise ValueError(
                f"Non-finite N_eff for "
                f"theme={theme_id}, month={month}."
            )

        if n_eff < 1.0 - 1e-6:
            raise ValueError(
                f"N_eff < 1 for "
                f"theme={theme_id}, month={month}: "
                f"N_eff={n_eff:.8f}."
            )

        if n_eff > n_positive + 1e-6:
            raise ValueError(
                f"N_eff > n_positive for "
                f"theme={theme_id}, month={month}: "
                f"N_eff={n_eff:.8f}, "
                f"n={n_positive}."
            )

        similarities = []

        if (
            n_positive
            >= MIN_N_FOR_RELIABILITY
        ):

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

        reliability = None
        reliability_min = None
        n_valid_splits = len(
            similarities
        )

        if similarities:

            sims = np.asarray(
                similarities,
                dtype=np.float64,
            )

            if not np.all(
                np.isfinite(sims)
            ):
                raise RuntimeError(
                    f"Non-finite reliability values for "
                    f"theme={theme_id}, month={month}."
                )

            reliability = float(
                np.mean(sims)
            )

            reliability_min = float(
                np.min(sims)
            )

        elif (
            n_positive
            >= MIN_N_FOR_RELIABILITY
        ):

            invalid_reliability_groups += 1

        results.append(
            {
                "theme": theme_id,
                "month": month,
                "n_positive": n_positive,
                "score_mass": sum_w,
                "n_eff": float(n_eff),
                "eff_ratio": float(
                    n_eff / n_positive
                ),
                "n_valid_splits": n_valid_splits,
                "reliability": reliability,
                "reliability_min": reliability_min,
            }
        )

    print()
    print(
        f"Groups without valid reliability "
        f"despite n >= {MIN_N_FOR_RELIABILITY}: "
        f"{invalid_reliability_groups:,}"
    )

    return results


# =============================================================================
# EFFECTIVE SUPPORT SUMMARY
# =============================================================================

def summarize_effective_support(results):

    print()
    print("=" * 72)
    print("EFFECTIVE SUPPORT")
    print("=" * 72)

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        subset = [
            result
            for result in results
            if result["theme"] == theme_id
        ]

        if not subset:
            continue

        n_eff = np.asarray(
            [
                result["n_eff"]
                for result in subset
            ],
            dtype=np.float64,
        )

        ratio = np.asarray(
            [
                result["eff_ratio"]
                for result in subset
            ],
            dtype=np.float64,
        )

        print()
        print(
            f"Theme {theme_id}"
        )

        print(
            f"  months               : "
            f"{len(subset)}"
        )

        print(
            f"  N_eff min/median/max : "
            f"{np.min(n_eff):.2f} / "
            f"{np.median(n_eff):.2f} / "
            f"{np.max(n_eff):.2f}"
        )

        print(
            f"  N_eff/n p05/med/p95  : "
            f"{np.quantile(ratio, 0.05):.3f} / "
            f"{np.median(ratio):.3f} / "
            f"{np.quantile(ratio, 0.95):.3f}"
        )


# =============================================================================
# RELIABILITY BY EFFECTIVE SUPPORT
# =============================================================================

def reliability_by_effective_support(results):

    valid = [
        result
        for result in results
        if result["reliability"] is not None
    ]

    print()
    print("=" * 72)
    print("RELIABILITY BY EFFECTIVE SUPPORT")
    print("=" * 72)

    if not valid:
        print(
            "No groups with valid reliability."
        )
        return

    bands = [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 50),
        (50, 100),
        (100, 300),
        (300, None),
    ]

    for low, high in bands:

        if high is None:

            subset = [
                result
                for result in valid
                if result["n_eff"] >= low
            ]

            label = f">= {low}"

        else:

            subset = [
                result
                for result in valid
                if (
                    low
                    <= result["n_eff"]
                    < high
                )
            ]

            label = f"{low}-{high}"

        if not subset:

            print(
                f"  N_eff {label:9s}: n=0"
            )

            continue

        arr = np.asarray(
            [
                result["reliability"]
                for result in subset
            ],
            dtype=np.float64,
        )

        print(
            f"  N_eff {label:9s}: "
            f"n={len(arr):3d}, "
            f"mean={np.mean(arr):.6f}, "
            f"median={np.median(arr):.6f}, "
            f"p05={np.quantile(arr, 0.05):.6f}, "
            f"min={np.min(arr):.6f}"
        )


# =============================================================================
# CURRENT CANDIDATE SUPPORT RULE
# =============================================================================

def inspect_candidate(results):

    subset = [
        result
        for result in results
        if (
            result["n_positive"]
            >= CANDIDATE_MIN_N
            and result["reliability"]
            is not None
        )
    ]

    print()
    print("=" * 72)
    print(
        f"CURRENT CANDIDATE: "
        f"n_positive >= {CANDIDATE_MIN_N}"
    )
    print("=" * 72)

    print(
        f"Groups retained: "
        f"{len(subset)} / "
        f"{N_THEMES * N_MONTHS}"
    )

    if not subset:
        print(
            "No groups satisfy the candidate rule."
        )
        return

    n_eff = np.asarray(
        [
            result["n_eff"]
            for result in subset
        ],
        dtype=np.float64,
    )

    ratio = np.asarray(
        [
            result["eff_ratio"]
            for result in subset
        ],
        dtype=np.float64,
    )

    reliability = np.asarray(
        [
            result["reliability"]
            for result in subset
        ],
        dtype=np.float64,
    )

    print(
        f"N_eff min/median/p05: "
        f"{np.min(n_eff):.2f} / "
        f"{np.median(n_eff):.2f} / "
        f"{np.quantile(n_eff, 0.05):.2f}"
    )

    print(
        f"N_eff/n min/median/p05: "
        f"{np.min(ratio):.3f} / "
        f"{np.median(ratio):.3f} / "
        f"{np.quantile(ratio, 0.05):.3f}"
    )

    print(
        f"Reliability mean/median/p05/min: "
        f"{np.mean(reliability):.6f} / "
        f"{np.median(reliability):.6f} / "
        f"{np.quantile(reliability, 0.05):.6f} / "
        f"{np.min(reliability):.6f}"
    )

    print()
    print(
        f"Coverage by theme:"
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
            f"  Theme {theme_id}: "
            f"{n_theme:3d} / {N_MONTHS}"
        )

    print()
    print(
        "Lowest N_eff among retained groups:"
    )

    ordered = sorted(
        subset,
        key=lambda result: result[
            "n_eff"
        ],
    )

    for result in ordered[:20]:

        print(
            f"Theme={result['theme']} | "
            f"month={result['month']} | "
            f"n={result['n_positive']:,} | "
            f"N_eff={result['n_eff']:.2f} | "
            f"ratio={result['eff_ratio']:.3f} | "
            f"mass={result['score_mass']:.3f} | "
            f"reliability="
            f"{result['reliability']:.6f} | "
            f"valid_splits="
            f"{result['n_valid_splits']}"
        )


# =============================================================================
# CORRELATIONS
# =============================================================================

def correlations(results):

    valid = [
        result
        for result in results
        if result["reliability"] is not None
    ]

    print()
    print("=" * 72)
    print("CORRELATIONS WITH SPLIT-HALF RELIABILITY")
    print("=" * 72)

    if len(valid) < 2:
        print(
            "Insufficient observations for correlations."
        )
        return

    n_positive = np.asarray(
        [
            result["n_positive"]
            for result in valid
        ],
        dtype=np.float64,
    )

    n_eff = np.asarray(
        [
            result["n_eff"]
            for result in valid
        ],
        dtype=np.float64,
    )

    score_mass = np.asarray(
        [
            result["score_mass"]
            for result in valid
        ],
        dtype=np.float64,
    )

    reliability = np.asarray(
        [
            result["reliability"]
            for result in valid
        ],
        dtype=np.float64,
    )

    corr_n = np.corrcoef(
        np.log1p(n_positive),
        reliability,
    )[0, 1]

    corr_neff = np.corrcoef(
        np.log1p(n_eff),
        reliability,
    )[0, 1]

    corr_mass = np.corrcoef(
        np.log1p(score_mass),
        reliability,
    )[0, 1]

    print(
        "corr(log1p(n_positive), reliability) = "
        f"{corr_n:.6f}"
    )

    print(
        "corr(log1p(N_eff), reliability)      = "
        f"{corr_neff:.6f}"
    )

    print(
        "corr(log1p(score_mass), reliability) = "
        f"{corr_mass:.6f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if MIN_N_FOR_RELIABILITY < 2:
        raise ValueError(
            "MIN_N_FOR_RELIABILITY must be at least 2."
        )

    if N_SPLITS <= 0:
        raise ValueError(
            "N_SPLITS must be positive."
        )

    if CANDIDATE_MIN_N < MIN_N_FOR_RELIABILITY:
        raise ValueError(
            "CANDIDATE_MIN_N cannot be lower than "
            "MIN_N_FOR_RELIABILITY."
        )

    print("=" * 72)
    print(
        "SEMANTIC DRIFT - EFFECTIVE SUPPORT "
        "VALIDATION V1"
    )
    print("=" * 72)

    print(
        f"Embedding model : {MODEL_NAME}"
    )

    print(
        f"Thematic method : {METHOD}"
    )

    print(
        f"Period          : "
        f"{START_DATE} -> {END_DATE}"
    )

    print(
        f"Reliability min n: "
        f"{MIN_N_FOR_RELIABILITY}"
    )

    print(
        f"Candidate min n : "
        f"{CANDIDATE_MIN_N}"
    )

    print(
        f"Split replicates: "
        f"{N_SPLITS}"
    )

    conn = get_connection()

    try:

        groups = build_accumulators(
            conn
        )

        results = compute_results(
            groups
        )

        summarize_effective_support(
            results
        )

        reliability_by_effective_support(
            results
        )

        inspect_candidate(
            results
        )

        correlations(
            results
        )

    finally:

        conn.close()

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()