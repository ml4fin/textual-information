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

# Frozen after centroid reliability validation.
MIN_POSITIVE_ARTICLES = 50

# Diagnostic candidate rolling-history windows; not yet frozen.
ROLLING_WINDOWS = [3, 6, 12, 24]

# Diagnostic candidate minimum depth for the 12-month reference.
CANDIDATE_MIN_REFERENCE_DEPTH = 3


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


def cosine_distance(a, b):
    """Return cosine distance between two L2-normalized vectors."""

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
    """Return the number of calendar months between two dates."""

    return (
        (later.year - earlier.year) * 12
        + later.month
        - earlier.month
    )


# =============================================================================
# BUILD FROZEN CENTROIDS
# =============================================================================

def build_centroids(conn):
    """
    Build score-weighted monthly centroids using the frozen
    minimum-support rule.

    Only positive weak-supervision scores are included. A centroid is
    retained only when at least MIN_POSITIVE_ARTICLES contribute to the
    corresponding theme-month group.

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
          AND e.model_name = %s
        ORDER BY
            a.publication_date,
            s.theme_id,
            a.article_id;
    """

    print()
    print(
        "Building frozen reliable centroids..."
    )

    with conn.cursor(
        name="temporal_reference_cursor"
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
                    f"Processed {n_rows:,} rows..."
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

        raw_centroid = (
            acc["weighted_sum"]
            / acc["score_mass"]
        )

        centroid = l2_normalize(
            raw_centroid
        )

        if centroid is None:
            raise ValueError(
                f"Invalid centroid for {key}."
            )

        centroid = centroid.astype(
            np.float32,
            copy=False,
        )

        norm = float(
            np.linalg.norm(centroid)
        )

        if not np.isclose(
            norm,
            1.0,
            atol=1e-5,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Centroid normalization failed "
                f"for {key}: norm={norm}."
            )

        centroids[key] = {
            "centroid": centroid,
            "n_positive": acc[
                "n_positive"
            ],
            "score_mass": acc[
                "score_mass"
            ],
        }

    print()
    print(
        f"Positive rows processed: "
        f"{n_rows:,}"
    )

    print(
        f"Reliable centroids: "
        f"{len(centroids):,} / "
        f"{N_THEMES * N_MONTHS:,}"
    )

    return centroids


# =============================================================================
# ORGANIZE BY THEME
# =============================================================================

def organize_by_theme(centroids):
    """Organize reliable centroids chronologically within each theme."""

    by_theme = defaultdict(list)

    for (
        theme_id,
        month,
    ), info in centroids.items():

        by_theme[theme_id].append(
            (
                month,
                info["centroid"],
            )
        )

    for theme_id in by_theme:

        by_theme[theme_id].sort(
            key=lambda item: item[0]
        )

    return by_theme


# =============================================================================
# GAP STRUCTURE
# =============================================================================

def analyze_gaps(by_theme):

    print()
    print("=" * 72)
    print("GAP STRUCTURE BETWEEN RELIABLE CENTROIDS")
    print("=" * 72)

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        items = by_theme.get(
            theme_id,
            [],
        )

        gaps = []

        for idx in range(
            1,
            len(items),
        ):

            gap = month_difference(
                items[idx][0],
                items[idx - 1][0],
            )

            gaps.append(
                gap
            )

        print()
        print(
            f"Theme {theme_id}: "
            f"reliable months={len(items)}"
        )

        if not gaps:
            print(
                "  no gaps available"
            )
            continue

        gaps = np.asarray(
            gaps,
            dtype=np.int64,
        )

        print(
            f"  consecutive (gap=1): "
            f"{np.sum(gaps == 1)}"
        )

        print(
            f"  gap > 1            : "
            f"{np.sum(gaps > 1)}"
        )

        print(
            f"  max gap            : "
            f"{np.max(gaps)} months"
        )

        print(
            f"  median gap         : "
            f"{np.median(gaps):.1f} months"
        )


# =============================================================================
# REFERENCE COVERAGE
# =============================================================================

def analyze_reference_coverage(by_theme):

    print()
    print("=" * 72)
    print("REFERENCE COVERAGE")
    print("=" * 72)

    print()
    print(
        "PREVIOUS CALENDAR MONTH "
        "(requires t-1 to be reliable)"
    )

    total_previous = 0

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        items = by_theme.get(
            theme_id,
            [],
        )

        available_months = {
            month
            for month, _ in items
        }

        n_valid = 0

        for current_month, _ in items:

            previous_exists = any(
                month_difference(
                    current_month,
                    previous_month,
                ) == 1
                for previous_month
                in available_months
            )

            if previous_exists:
                n_valid += 1

        total_previous += n_valid

        print(
            f"  Theme {theme_id}: "
            f"{n_valid:3d} drift observations"
        )

    print(
        f"  TOTAL: {total_previous}"
    )

    print()
    print(
        "PREVIOUS AVAILABLE RELIABLE MONTH "
        "(skips gaps)"
    )

    total_previous_available = 0

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        n_reliable = len(
            by_theme.get(
                theme_id,
                [],
            )
        )

        n_valid = max(
            0,
            n_reliable - 1,
        )

        total_previous_available += (
            n_valid
        )

        print(
            f"  Theme {theme_id}: "
            f"{n_valid:3d} drift observations"
        )

    print(
        f"  TOTAL: "
        f"{total_previous_available}"
    )

    print()
    print(
        "FIXED FIRST RELIABLE MONTH"
    )

    total_fixed = 0

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        n_reliable = len(
            by_theme.get(
                theme_id,
                [],
            )
        )

        n_valid = max(
            0,
            n_reliable - 1,
        )

        total_fixed += n_valid

        print(
            f"  Theme {theme_id}: "
            f"{n_valid:3d} drift observations"
        )

    print(
        f"  TOTAL: {total_fixed}"
    )

    print()
    print(
        "ROLLING CALENDAR HISTORY"
    )

    for window in ROLLING_WINDOWS:

        print()
        print(
            f"  Window = previous "
            f"{window} calendar months"
        )

        total = 0

        for theme_id in range(
            1,
            N_THEMES + 1,
        ):

            items = by_theme.get(
                theme_id,
                [],
            )

            n_valid = 0

            for current_idx, (
                current_month,
                _current_centroid,
            ) in enumerate(items):

                has_history = False

                for past_idx in range(
                    current_idx
                ):

                    past_month, _ = (
                        items[past_idx]
                    )

                    gap = month_difference(
                        current_month,
                        past_month,
                    )

                    if 1 <= gap <= window:
                        has_history = True
                        break

                if has_history:
                    n_valid += 1

            total += n_valid

            print(
                f"    Theme {theme_id}: "
                f"{n_valid:3d}"
            )

        print(
            f"    TOTAL: {total}"
        )


# =============================================================================
# ROLLING REFERENCE DEPTH
# =============================================================================

def analyze_rolling_reference_depth(by_theme):

    print()
    print("=" * 72)
    print("ROLLING REFERENCE DEPTH")
    print("=" * 72)

    for window in ROLLING_WINDOWS:

        print()
        print(
            f"Window = previous "
            f"{window} calendar months"
        )

        pooled_depths = []

        for theme_id in range(
            1,
            N_THEMES + 1,
        ):

            items = by_theme.get(
                theme_id,
                [],
            )

            depths = []

            for current_idx, (
                current_month,
                _current_centroid,
            ) in enumerate(items):

                n_history = 0

                for past_idx in range(
                    current_idx
                ):

                    past_month, _ = (
                        items[past_idx]
                    )

                    gap = month_difference(
                        current_month,
                        past_month,
                    )

                    if 1 <= gap <= window:
                        n_history += 1

                if n_history > 0:
                    depths.append(
                        n_history
                    )

            if not depths:

                print(
                    f"  Theme {theme_id}: n=0"
                )
                continue

            arr = np.asarray(
                depths,
                dtype=np.int64,
            )

            pooled_depths.extend(
                arr.tolist()
            )

            print(
                f"  Theme {theme_id}: "
                f"n={len(arr):3d}, "
                f"min={np.min(arr):2d}, "
                f"p05={np.quantile(arr, 0.05):.1f}, "
                f"median={np.median(arr):.1f}, "
                f"max={np.max(arr):2d}, "
                f"depth>=3={np.sum(arr >= 3):3d}, "
                f"depth>=6={np.sum(arr >= 6):3d}"
            )

        if pooled_depths:

            arr = np.asarray(
                pooled_depths,
                dtype=np.int64,
            )

            print(
                f"  ALL: "
                f"n={len(arr):3d}, "
                f"min={np.min(arr):2d}, "
                f"median={np.median(arr):.1f}, "
                f"depth>=3={np.sum(arr >= 3):3d}, "
                f"depth>=6={np.sum(arr >= 6):3d}"
            )


# =============================================================================
# 12-MONTH REFERENCE DEPTH VALIDATION
# =============================================================================

def validate_reference_depth(by_theme):

    window = 12

    print()
    print("=" * 72)
    print("12-MONTH REFERENCE DEPTH VALIDATION")
    print("=" * 72)

    observations = []

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        items = by_theme.get(
            theme_id,
            [],
        )

        for current_idx, (
            current_month,
            current_centroid,
        ) in enumerate(items):

            historical = []

            for past_idx in range(
                current_idx
            ):

                past_month, past_centroid = (
                    items[past_idx]
                )

                gap = month_difference(
                    current_month,
                    past_month,
                )

                if 1 <= gap <= window:

                    historical.append(
                        past_centroid.astype(
                            np.float64,
                            copy=False,
                        )
                    )

            depth = len(
                historical
            )

            if (
                depth
                < CANDIDATE_MIN_REFERENCE_DEPTH
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
                continue

            drift = cosine_distance(
                current_centroid,
                reference,
            )

            observations.append(
                {
                    "theme": theme_id,
                    "month": current_month,
                    "depth": depth,
                    "drift": drift,
                }
            )

    print()
    print(
        f"Observations with depth >= "
        f"{CANDIDATE_MIN_REFERENCE_DEPTH}: "
        f"{len(observations)}"
    )

    if not observations:
        print(
            "No observations available for "
            "reference-depth validation."
        )
        return

    print()
    print(
        "Drift by actual reference depth:"
    )

    depth_bands = [
        (3, 5),
        (6, 8),
        (9, 11),
        (12, 12),
    ]

    for low, high in depth_bands:

        vals = [
            observation["drift"]
            for observation in observations
            if (
                low
                <= observation["depth"]
                <= high
            )
        ]

        if not vals:

            print(
                f"  depth {low}-{high}: n=0"
            )
            continue

        arr = np.asarray(
            vals,
            dtype=np.float64,
        )

        print(
            f"  depth {low}-{high}: "
            f"n={len(arr):3d}, "
            f"mean={np.mean(arr):.6f}, "
            f"median={np.median(arr):.6f}, "
            f"p05={np.quantile(arr, 0.05):.6f}, "
            f"p95={np.quantile(arr, 0.95):.6f}, "
            f"max={np.max(arr):.6f}"
        )

    print()
    print(
        "Depth 3-5 observations by theme:"
    )

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        subset = [
            observation
            for observation in observations
            if (
                observation["theme"]
                == theme_id
                and 3
                <= observation["depth"]
                <= 5
            )
        ]

        if not subset:

            print(
                f"  Theme {theme_id}: n=0"
            )
            continue

        arr = np.asarray(
            [
                observation["drift"]
                for observation in subset
            ],
            dtype=np.float64,
        )

        print(
            f"  Theme {theme_id}: "
            f"n={len(arr):3d}, "
            f"mean={np.mean(arr):.6f}, "
            f"median={np.median(arr):.6f}, "
            f"max={np.max(arr):.6f}"
        )

    print()
    print(
        "Candidate minimum reference depths:"
    )

    for min_depth in [
        3,
        6,
        12,
    ]:

        subset = [
            observation
            for observation in observations
            if (
                observation["depth"]
                >= min_depth
            )
        ]

        print()
        print(
            f"  minimum depth >= "
            f"{min_depth}"
        )

        print(
            f"    observations : "
            f"{len(subset)}"
        )

        if not subset:
            continue

        arr = np.asarray(
            [
                observation["drift"]
                for observation in subset
            ],
            dtype=np.float64,
        )

        print(
            f"    mean          : "
            f"{np.mean(arr):.6f}"
        )

        print(
            f"    median        : "
            f"{np.median(arr):.6f}"
        )

        print(
            f"    p95           : "
            f"{np.quantile(arr, 0.95):.6f}"
        )

        for theme_id in range(
            1,
            N_THEMES + 1,
        ):

            n_theme = sum(
                1
                for observation in subset
                if (
                    observation["theme"]
                    == theme_id
                )
            )

            print(
                f"      Theme {theme_id}: "
                f"{n_theme:3d}"
            )

    print()
    print(
        "All observations with depth 3-5:"
    )

    low_depth = sorted(
        [
            observation
            for observation in observations
            if (
                3
                <= observation["depth"]
                <= 5
            )
        ],
        key=lambda observation: (
            observation["theme"],
            observation["month"],
        ),
    )

    for observation in low_depth:

        print(
            f"Theme={observation['theme']} | "
            f"month={observation['month']} | "
            f"depth={observation['depth']} | "
            f"drift={observation['drift']:.6f}"
        )


# =============================================================================
# CANDIDATE DRIFT SERIES
# =============================================================================

def compute_candidate_series(by_theme):

    series = defaultdict(list)

    for theme_id in range(
        1,
        N_THEMES + 1,
    ):

        items = by_theme.get(
            theme_id,
            [],
        )

        if not items:
            continue

        # Previous calendar month.
        for current_idx, (
            current_month,
            current_centroid,
        ) in enumerate(items):

            for past_idx in range(
                current_idx - 1,
                -1,
                -1,
            ):

                past_month, past_centroid = (
                    items[past_idx]
                )

                gap = month_difference(
                    current_month,
                    past_month,
                )

                if gap == 1:

                    series[
                        (
                            "previous_calendar",
                            theme_id,
                        )
                    ].append(
                        cosine_distance(
                            current_centroid,
                            past_centroid,
                        )
                    )

                    break

                if gap > 1:
                    break

        # Previous available reliable month.
        for idx in range(
            1,
            len(items),
        ):

            _, previous_centroid = (
                items[idx - 1]
            )

            _, current_centroid = (
                items[idx]
            )

            series[
                (
                    "previous_available",
                    theme_id,
                )
            ].append(
                cosine_distance(
                    current_centroid,
                    previous_centroid,
                )
            )

        # Fixed first reliable month.
        first_reference = (
            items[0][1]
        )

        for idx in range(
            1,
            len(items),
        ):

            _, current_centroid = (
                items[idx]
            )

            series[
                (
                    "fixed_first",
                    theme_id,
                )
            ].append(
                cosine_distance(
                    current_centroid,
                    first_reference,
                )
            )

        # Rolling calendar history.
        for window in ROLLING_WINDOWS:

            for current_idx, (
                current_month,
                current_centroid,
            ) in enumerate(items):

                historical = []

                for past_idx in range(
                    current_idx
                ):

                    past_month, past_centroid = (
                        items[past_idx]
                    )

                    gap = month_difference(
                        current_month,
                        past_month,
                    )

                    if 1 <= gap <= window:

                        historical.append(
                            past_centroid.astype(
                                np.float64,
                                copy=False,
                            )
                        )

                if not historical:
                    continue

                # Reliable historical months receive equal weight.
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
                    continue

                series[
                    (
                        f"rolling_{window}",
                        theme_id,
                    )
                ].append(
                    cosine_distance(
                        current_centroid,
                        reference,
                    )
                )

    return series


# =============================================================================
# CANDIDATE SERIES SUMMARY
# =============================================================================

def summarize_series(series):

    print()
    print("=" * 72)
    print("CANDIDATE DRIFT DISTRIBUTIONS")
    print("=" * 72)

    methods = [
        "previous_calendar",
        "previous_available",
        "fixed_first",
    ] + [
        f"rolling_{window}"
        for window in ROLLING_WINDOWS
    ]

    for method in methods:

        print()
        print(
            method.upper()
        )

        pooled = []

        for theme_id in range(
            1,
            N_THEMES + 1,
        ):

            values = series.get(
                (
                    method,
                    theme_id,
                ),
                [],
            )

            if not values:

                print(
                    f"  Theme {theme_id}: "
                    f"n=0"
                )

                continue

            arr = np.asarray(
                values,
                dtype=np.float64,
            )

            pooled.extend(
                arr.tolist()
            )

            print(
                f"  Theme {theme_id}: "
                f"n={len(arr):3d}, "
                f"mean={np.mean(arr):.6f}, "
                f"median={np.median(arr):.6f}, "
                f"p95={np.quantile(arr, 0.95):.6f}, "
                f"max={np.max(arr):.6f}"
            )

        if pooled:

            pooled = np.asarray(
                pooled,
                dtype=np.float64,
            )

            print(
                f"  ALL: "
                f"n={len(pooled):3d}, "
                f"mean={np.mean(pooled):.6f}, "
                f"median={np.median(pooled):.6f}, "
                f"p95={np.quantile(pooled, 0.95):.6f}, "
                f"max={np.max(pooled):.6f}"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if MIN_POSITIVE_ARTICLES <= 0:
        raise ValueError(
            "MIN_POSITIVE_ARTICLES must be positive."
        )

    if not ROLLING_WINDOWS:
        raise ValueError(
            "ROLLING_WINDOWS cannot be empty."
        )

    if any(
        window <= 0
        for window in ROLLING_WINDOWS
    ):
        raise ValueError(
            "All rolling windows must be positive."
        )

    print("=" * 72)
    print(
        "SEMANTIC DRIFT - TEMPORAL "
        "REFERENCE PILOT V1"
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
        f"Frozen min n    : "
        f"{MIN_POSITIVE_ARTICLES}"
    )

    print(
        f"Rolling windows : "
        f"{ROLLING_WINDOWS}"
    )

    conn = get_connection()

    try:

        centroids = build_centroids(
            conn
        )

    finally:

        conn.close()

    if not centroids:
        raise RuntimeError(
            "No reliable centroids were constructed."
        )

    by_theme = organize_by_theme(
        centroids
    )

    analyze_gaps(
        by_theme
    )

    analyze_reference_coverage(
        by_theme
    )

    analyze_rolling_reference_depth(
        by_theme
    )

    validate_reference_depth(
        by_theme
    )

    series = compute_candidate_series(
        by_theme
    )

    summarize_series(
        series
    )

    print()
    print("=" * 72)
    print("PILOT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()