import os
import psycopg2
import numpy as np
import faiss

from datetime import date, timedelta
from time import perf_counter


# =============================================================================
# PARAMETERS
# =============================================================================

MODEL_NAME = "all-MiniLM-L6-v2_chunk254_mean_l2_v1"

START_DATE = date(2015, 1, 1)
END_DATE = date(2024, 1, 1)  # exclusive

WINDOW_DAYS = 365
K = 10

K_PRIME = 200
HNSW_M = 48
HNSW_EF_CONSTRUCTION = 400
HNSW_EF_SEARCH = 400

FETCH_SIZE = 5000
EMBEDDING_DIM = 384
NORM_TOLERANCE = 1e-4

SEARCH_METHOD = "faiss_hnsw_monthly_rebuild"

# Keep this at 1 for the first controlled production execution.
# After validating the first month, it can be increased.
MAX_NEW_MONTHS_PER_RUN = 1


# =============================================================================
# DATABASE
# =============================================================================

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": (
        os.getenv("THESIS_DB_PASSWORD")
        or os.getenv("DB_PASSWORD")
    ),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
}


def get_connection():

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. "
            "Set THESIS_DB_PASSWORD or DB_PASSWORD."
        )

    return psycopg2.connect(**DB_CONFIG)


# =============================================================================
# QUERIES
# =============================================================================

ARTICLE_QUERY = """
SELECT
    a.article_id,
    a.publication_date,
    a.source_code,
    a.title,
    e.embedding
FROM risk_emergence.v_valid_articles AS a
JOIN risk_emergence.news_embeddings AS e
    ON e.article_id = a.article_id
WHERE e.model_name = %s
  AND a.publication_date >= %s
  AND a.publication_date < %s
ORDER BY
    a.publication_date,
    a.article_id;
"""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_articles(
    conn,
    start_date,
    end_date,
    cursor_name,
):
    """
    Load articles and embeddings for a half-open date interval.
    """

    ids_parts = []
    dates_parts = []
    sources = []
    titles = []
    vectors_parts = []

    stream = conn.cursor(name=cursor_name)
    stream.itersize = FETCH_SIZE

    try:

        stream.execute(
            ARTICLE_QUERY,
            (
                MODEL_NAME,
                start_date,
                end_date,
            ),
        )

        total_rows = 0

        while True:

            rows = stream.fetchmany(FETCH_SIZE)

            if not rows:
                break

            ids_parts.append(
                np.asarray(
                    [row[0] for row in rows],
                    dtype=np.int64,
                )
            )

            dates_parts.append(
                np.asarray(
                    [row[1] for row in rows],
                    dtype=object,
                )
            )

            sources.extend(
                [row[2] for row in rows]
            )

            titles.extend(
                [row[3] for row in rows]
            )

            vectors_parts.append(
                np.asarray(
                    [row[4] for row in rows],
                    dtype=np.float32,
                )
            )

            total_rows += len(rows)

    finally:
        stream.close()

    if total_rows == 0:
        raise ValueError(
            f"No rows found for "
            f"{start_date} <= date < {end_date}."
        )

    ids = np.concatenate(ids_parts)
    dates = np.concatenate(dates_parts)

    vectors = np.vstack(
        vectors_parts
    ).astype(
        np.float32,
        copy=False,
    )

    if vectors.shape != (
        len(ids),
        EMBEDDING_DIM,
    ):
        raise ValueError(
            f"Unexpected embedding matrix: "
            f"{vectors.shape}"
        )

    if len(ids) != len(dates):
        raise ValueError(
            "IDs and dates have different lengths."
        )

    if len(ids) != len(sources):
        raise ValueError(
            "IDs and sources have different lengths."
        )

    if len(ids) != len(titles):
        raise ValueError(
            "IDs and titles have different lengths."
        )

    if len(np.unique(ids)) != len(ids):
        raise ValueError(
            "Duplicate article IDs detected."
        )

    if not np.all(np.isfinite(vectors)):
        raise ValueError(
            "Non-finite embedding values detected."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=NORM_TOLERANCE,
        rtol=0.0,
    ):
        max_error = float(
            np.max(
                np.abs(norms - 1.0)
            )
        )

        raise ValueError(
            "Embeddings are not sufficiently "
            f"L2-normalized. Max error: {max_error:.8f}"
        )

    return (
        ids,
        dates,
        sources,
        titles,
        vectors,
    )


# =============================================================================
# FAISS
# =============================================================================

def build_hnsw(
    vectors,
    ids,
):
    """
    Build the validated HNSW index.

    Embeddings are L2-normalized, so inner product corresponds
    to cosine similarity.
    """

    if K_PRIME < K:
        raise ValueError(
            "K_PRIME must be greater than or equal to K."
        )

    if HNSW_EF_SEARCH < K_PRIME:
        raise ValueError(
            "HNSW_EF_SEARCH must be greater than "
            "or equal to K_PRIME."
        )

    base = faiss.IndexHNSWFlat(
        EMBEDDING_DIM,
        HNSW_M,
        faiss.METRIC_INNER_PRODUCT,
    )

    base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    base.hnsw.efSearch = HNSW_EF_SEARCH

    index = faiss.IndexIDMap2(base)

    index.add_with_ids(
        vectors,
        ids,
    )

    return index, base


# =============================================================================
# MONTH STATUS
# =============================================================================

def get_month_status(
    conn,
    month_start,
    month_end,
):

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT COUNT(*)
            FROM risk_emergence.v_valid_articles AS a
            JOIN risk_emergence.news_embeddings AS e
              ON e.article_id = a.article_id
            WHERE e.model_name = %s
              AND a.publication_date >= %s
              AND a.publication_date < %s;
            """,
            (
                MODEL_NAME,
                month_start,
                month_end,
            ),
        )

        expected = cur.fetchone()[0]

        cur.execute(
            """
            SELECT COUNT(*)
            FROM risk_emergence.news_novelty AS n
            JOIN risk_emergence.v_valid_articles AS a
              ON a.article_id = n.article_id
            WHERE n.model_name = %s
              AND n.k_neighbors = %s
              AND n.window_days = %s
              AND a.publication_date >= %s
              AND a.publication_date < %s;
            """,
            (
                MODEL_NAME,
                K,
                WINDOW_DAYS,
                month_start,
                month_end,
            ),
        )

        stored = cur.fetchone()[0]

    if stored > expected:
        raise RuntimeError(
            f"Invalid month state for "
            f"{month_start:%Y-%m}: "
            f"stored={stored:,} > expected={expected:,}."
        )

    return expected, stored


# =============================================================================
# DATABASE WRITER
# =============================================================================

def write_month_results(
    conn,
    article_ids,
    novelty_scores,
):

    if len(article_ids) != len(novelty_scores):
        raise ValueError(
            "article_ids and novelty_scores "
            "have different lengths."
        )

    if len(article_ids) == 0:
        raise ValueError(
            "Cannot write an empty month."
        )

    rows = [
        (
            int(article_id),
            MODEL_NAME,
            float(novelty_score),
            K,
            WINDOW_DAYS,
            SEARCH_METHOD,
            K_PRIME,
            HNSW_M,
            HNSW_EF_CONSTRUCTION,
            HNSW_EF_SEARCH,
        )
        for article_id, novelty_score
        in zip(
            article_ids,
            novelty_scores,
            strict=True,
        )
    ]

    sql = """
        INSERT INTO risk_emergence.news_novelty (
            article_id,
            model_name,
            novelty_score,
            k_neighbors,
            window_days,
            search_method,
            k_prime,
            hnsw_m,
            ef_construction,
            ef_search
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (
            article_id,
            model_name,
            k_neighbors,
            window_days
        )
        DO NOTHING;
    """

    with conn.cursor() as cur:
        cur.executemany(
            sql,
            rows,
        )

    return len(rows)


# =============================================================================
# GLOBAL STATUS
# =============================================================================

def get_global_status(conn):

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT COUNT(*)
            FROM risk_emergence.v_valid_articles AS a
            JOIN risk_emergence.news_embeddings AS e
              ON e.article_id = a.article_id
            WHERE e.model_name = %s
              AND a.publication_date >= %s
              AND a.publication_date < %s;
            """,
            (
                MODEL_NAME,
                START_DATE,
                END_DATE,
            ),
        )

        total_targets = cur.fetchone()[0]

        cur.execute(
            """
            SELECT COUNT(*)
            FROM risk_emergence.news_novelty AS n
            JOIN risk_emergence.v_valid_articles AS a
              ON a.article_id = n.article_id
            WHERE n.model_name = %s
              AND n.k_neighbors = %s
              AND n.window_days = %s
              AND a.publication_date >= %s
              AND a.publication_date < %s;
            """,
            (
                MODEL_NAME,
                K,
                WINDOW_DAYS,
                START_DATE,
                END_DATE,
            ),
        )

        completed = cur.fetchone()[0]

    if completed > total_targets:
        raise RuntimeError(
            f"Invalid global state: "
            f"completed={completed:,} > "
            f"targets={total_targets:,}."
        )

    return total_targets, completed


# =============================================================================
# DATE UTILITY
# =============================================================================

def get_next_month(month_start):

    if month_start.month == 12:
        return date(
            month_start.year + 1,
            1,
            1,
        )

    return date(
        month_start.year,
        month_start.month + 1,
        1,
    )


# =============================================================================
# PROCESS ONE MONTH
# =============================================================================

def process_month(
    conn,
    month_start,
    month_end,
    expected,
):

    month_timer = perf_counter()

    history_start = (
        month_start
        - timedelta(days=WINDOW_DAYS)
    )

    print()
    print("=" * 72)
    print(f"PROCESSING MONTH: {month_start:%Y-%m}")
    print("=" * 72)

    print(
        f"Target interval      : "
        f"{month_start} <= date < {month_end}"
    )

    print(
        f"History interval     : "
        f"{history_start} <= date < {month_start}"
    )

    print(
        f"Expected articles    : "
        f"{expected:,}"
    )

    print()
    print("LOADING HISTORY")

    t0 = perf_counter()

    (
        history_ids,
        history_dates,
        _history_sources,
        _history_titles,
        history_vectors,
    ) = load_articles(
        conn,
        history_start,
        month_start,
        f"history_{month_start:%Y%m}",
    )

    history_load_seconds = perf_counter() - t0

    print(
        f"History rows         : "
        f"{len(history_ids):,}"
    )

    print(
        f"History matrix       : "
        f"{history_vectors.shape}"
    )

    print(
        f"History load         : "
        f"{history_load_seconds:.3f} s"
    )

    if len(history_ids) < K:
        raise RuntimeError(
            f"Insufficient history for "
            f"{month_start:%Y-%m}."
        )

    print()
    print("LOADING TARGET MONTH")

    t0 = perf_counter()

    (
        target_ids,
        target_dates,
        _target_sources,
        _target_titles,
        target_vectors,
    ) = load_articles(
        conn,
        month_start,
        month_end,
        f"targets_{month_start:%Y%m}",
    )

    target_load_seconds = perf_counter() - t0

    print(
        f"Target rows          : "
        f"{len(target_ids):,}"
    )

    print(
        f"Target matrix        : "
        f"{target_vectors.shape}"
    )

    print(
        f"Target load          : "
        f"{target_load_seconds:.3f} s"
    )

    if len(target_ids) != expected:
        raise RuntimeError(
            f"Target count mismatch for "
            f"{month_start:%Y-%m}: "
            f"loaded={len(target_ids):,}, "
            f"expected={expected:,}."
        )

    print()
    print("BUILDING HNSW")

    t0 = perf_counter()

    index, hnsw_base = build_hnsw(
        history_vectors,
        history_ids,
    )

    build_seconds = perf_counter() - t0

    print(
        f"Initial index        : "
        f"{index.ntotal:,}"
    )

    print(
        f"HNSW build           : "
        f"{build_seconds:.3f} s"
    )

    if index.ntotal != len(history_ids):
        raise RuntimeError(
            "Initial FAISS index size mismatch."
        )

    id_to_date = {
        int(article_id): publication_date
        for article_id, publication_date
        in zip(
            history_ids,
            history_dates,
            strict=True,
        )
    }

    month_result_ids = []
    month_result_scores = []

    incomplete = 0
    causality_violations = 0

    expired_counts = []

    search_seconds = 0.0
    add_seconds = 0.0

    unique_dates = sorted(
        set(
            target_dates.tolist()
        )
    )

    print()
    print("DAILY CAUSAL PROCESSING")

    print(
        "date         targets   complete   incomplete   "
        "index_before   expired_mean   expired_max"
    )

    for current_date in unique_dates:

        day_mask = (
            target_dates == current_date
        )

        day_ids = target_ids[
            day_mask
        ]

        day_vectors = target_vectors[
            day_mask
        ]

        index_before = int(
            index.ntotal
        )

        # Search before adding current-day articles to prevent look-ahead.
        t0 = perf_counter()

        similarities, neighbor_ids = index.search(
            day_vectors,
            K_PRIME,
        )

        search_seconds += perf_counter() - t0

        valid_lower_bound = (
            current_date
            - timedelta(days=WINDOW_DAYS)
        )

        day_complete = 0
        day_incomplete = 0
        day_expired_counts = []

        for row_idx, article_id in enumerate(
            day_ids
        ):

            valid_sims = []
            expired_raw = 0

            for similarity, candidate_id in zip(
                similarities[row_idx],
                neighbor_ids[row_idx],
                strict=True,
            ):

                candidate_id = int(
                    candidate_id
                )

                if candidate_id == -1:
                    continue

                candidate_date = id_to_date.get(
                    candidate_id
                )

                if candidate_date is None:
                    raise RuntimeError(
                        "FAISS returned unknown "
                        f"article_id={candidate_id}."
                    )

                if candidate_date >= current_date:

                    causality_violations += 1

                    raise RuntimeError(
                        "Temporal causality violation: "
                        f"target={int(article_id)}, "
                        f"target_date={current_date}, "
                        f"candidate={candidate_id}, "
                        f"candidate_date={candidate_date}."
                    )

                if candidate_date < valid_lower_bound:

                    expired_raw += 1
                    continue

                if len(valid_sims) < K:

                    valid_sims.append(
                        float(
                            np.clip(
                                similarity,
                                -1.0,
                                1.0,
                            )
                        )
                    )

            expired_counts.append(
                expired_raw
            )

            day_expired_counts.append(
                expired_raw
            )

            if len(valid_sims) < K:

                incomplete += 1
                day_incomplete += 1
                continue

            novelty = (
                1.0
                - float(
                    np.mean(
                        np.asarray(
                            valid_sims,
                            dtype=np.float32,
                        )
                    )
                )
            )

            novelty = float(
                np.clip(
                    novelty,
                    0.0,
                    2.0,
                )
            )

            month_result_ids.append(
                int(article_id)
            )

            month_result_scores.append(
                novelty
            )

            day_complete += 1

        # Current-day articles become historical only after all same-day
        # targets have been evaluated.
        t0 = perf_counter()

        index.add_with_ids(
            day_vectors,
            day_ids,
        )

        add_seconds += perf_counter() - t0

        for article_id in day_ids:

            id_to_date[
                int(article_id)
            ] = current_date

        if day_expired_counts:

            expired_mean = float(
                np.mean(
                    day_expired_counts
                )
            )

            expired_max = int(
                np.max(
                    day_expired_counts
                )
            )

        else:

            expired_mean = 0.0
            expired_max = 0

        print(
            f"{current_date}   "
            f"{len(day_ids):7,d}   "
            f"{day_complete:8,d}   "
            f"{day_incomplete:10,d}   "
            f"{index_before:12,d}   "
            f"{expired_mean:12.4f}   "
            f"{expired_max:11,d}"
        )

    print()
    print("MONTH VALIDATION")

    final_index_expected = (
        len(history_ids)
        + len(target_ids)
    )

    print(
        f"Expected articles    : "
        f"{expected:,}"
    )

    print(
        f"Computed novelty     : "
        f"{len(month_result_ids):,}"
    )

    print(
        f"Incomplete           : "
        f"{incomplete:,}"
    )

    print(
        f"Causality violations : "
        f"{causality_violations:,}"
    )

    print(
        f"Initial index        : "
        f"{len(history_ids):,}"
    )

    print(
        f"Final index          : "
        f"{index.ntotal:,}"
    )

    print(
        f"Expected final index : "
        f"{final_index_expected:,}"
    )

    if incomplete != 0:
        raise RuntimeError(
            f"Found {incomplete:,} incomplete "
            f"articles in {month_start:%Y-%m}. "
            "Month will NOT be persisted."
        )

    if causality_violations != 0:
        raise RuntimeError(
            f"Found {causality_violations:,} "
            "causality violations."
        )

    if len(month_result_ids) != expected:
        raise RuntimeError(
            "Monthly result count mismatch."
        )

    if len(set(month_result_ids)) != expected:
        raise RuntimeError(
            "Duplicate article IDs in "
            "monthly results."
        )

    if index.ntotal != final_index_expected:
        raise RuntimeError(
            "Final FAISS index size mismatch."
        )

    novelty_array = np.asarray(
        month_result_scores,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(novelty_array)
    ):
        raise RuntimeError(
            "Non-finite novelty values detected."
        )

    if (
        np.any(novelty_array < 0.0)
        or np.any(novelty_array > 2.0)
    ):
        raise RuntimeError(
            "Novelty outside [0, 2]."
        )

    print(
        f"Novelty min          : "
        f"{novelty_array.min():.8f}"
    )

    print(
        f"Novelty mean         : "
        f"{novelty_array.mean():.8f}"
    )

    print(
        f"Novelty median       : "
        f"{np.median(novelty_array):.8f}"
    )

    print(
        f"Novelty max          : "
        f"{novelty_array.max():.8f}"
    )

    if expired_counts:

        print(
            f"Mean expired raw     : "
            f"{np.mean(expired_counts):.4f}"
        )

        print(
            f"Max expired raw      : "
            f"{np.max(expired_counts):,}"
        )

    print(
        "MONTH VALIDATION     : PASS"
    )

    print()
    print("DATABASE WRITE")

    t0 = perf_counter()

    rows_attempted = write_month_results(
        conn,
        month_result_ids,
        month_result_scores,
    )

    write_seconds = perf_counter() - t0

    expected_tx, stored_tx = get_month_status(
        conn,
        month_start,
        month_end,
    )

    print(
        f"Rows attempted       : "
        f"{rows_attempted:,}"
    )

    print(
        f"Rows visible in TX   : "
        f"{stored_tx:,}"
    )

    if expected_tx != expected:
        raise RuntimeError(
            "Expected target count changed "
            "during transaction."
        )

    if stored_tx != expected:
        raise RuntimeError(
            f"Transactional DB verification failed: "
            f"stored={stored_tx:,}, "
            f"expected={expected:,}."
        )

    print(
        "IN-TRANSACTION "
        "VALIDATION      : PASS"
    )

    # Commit the complete month atomically.
    conn.commit()

    expected_after, stored_after = get_month_status(
        conn,
        month_start,
        month_end,
    )

    if expected_after != expected:
        conn.rollback()

        raise RuntimeError(
            "Expected target count changed "
            "after commit."
        )

    if stored_after != expected:
        conn.rollback()

        raise RuntimeError(
            f"Post-commit verification failed: "
            f"stored={stored_after:,}, "
            f"expected={expected:,}."
        )

    # get_month_status() opened a new read transaction after the commit.
    # No writes occurred in it, so close it before processing the next month.
    conn.rollback()

    month_seconds = perf_counter() - month_timer

    print(
        f"Rows after commit    : "
        f"{stored_after:,}"
    )

    print(
        f"DB write             : "
        f"{write_seconds:.3f} s"
    )

    print(
        f"Search time          : "
        f"{search_seconds:.3f} s"
    )

    print(
        f"Incremental add      : "
        f"{add_seconds:.3f} s"
    )

    print(
        f"Month runtime        : "
        f"{month_seconds:.3f} s"
    )

    print(
        f"MONTH {month_start:%Y-%m}       : "
        "COMMITTED"
    )

    return stored_after


# =============================================================================
# RESUMABLE PRODUCTION RUNNER
# =============================================================================

def main():

    total_start = perf_counter()

    print("=" * 72)
    print("NOVELTY PRODUCTION - RESUMABLE MONTHLY RUNNER")
    print("=" * 72)

    print(f"Model                : {MODEL_NAME}")
    print(f"Production period    : {START_DATE} <= date < {END_DATE}")
    print(f"K                    : {K}")
    print(f"Window               : {WINDOW_DAYS} days")
    print(f"K'                   : {K_PRIME}")
    print(f"HNSW M               : {HNSW_M}")
    print(f"efConstruction       : {HNSW_EF_CONSTRUCTION}")
    print(f"efSearch             : {HNSW_EF_SEARCH}")
    print(
        f"Max new months/run   : "
        f"{MAX_NEW_MONTHS_PER_RUN}"
    )

    conn = get_connection()

    processed_new_months = 0
    skipped_complete_months = 0

    try:

        total_targets, completed_before = (
            get_global_status(conn)
        )

        print()
        print("INITIAL GLOBAL STATUS")

        print(
            f"Global targets       : "
            f"{total_targets:,}"
        )

        print(
            f"Global completed     : "
            f"{completed_before:,}"
        )

        print(
            f"Global remaining     : "
            f"{total_targets - completed_before:,}"
        )

        month_start = START_DATE

        while month_start < END_DATE:

            month_end = get_next_month(
                month_start
            )

            expected, stored = get_month_status(
                conn,
                month_start,
                month_end,
            )

            if expected <= 0:
                raise RuntimeError(
                    f"EMPTY month detected: "
                    f"{month_start:%Y-%m}."
                )

            if stored == expected:

                print()
                print(
                    f"{month_start:%Y-%m} | "
                    f"expected={expected:,} | "
                    f"stored={stored:,} | "
                    "COMPLETE | SKIP"
                )

                skipped_complete_months += 1
                month_start = month_end
                continue

            if 0 < stored < expected:

                raise RuntimeError(
                    f"PARTIAL month detected for "
                    f"{month_start:%Y-%m}: "
                    f"stored={stored:,}, "
                    f"expected={expected:,}. "
                    "Production aborted."
                )

            if stored != 0:
                raise RuntimeError(
                    f"Unexpected month state for "
                    f"{month_start:%Y-%m}: "
                    f"stored={stored:,}, "
                    f"expected={expected:,}."
                )

            if (
                processed_new_months
                >= MAX_NEW_MONTHS_PER_RUN
            ):

                print()
                print(
                    "Safety limit reached. "
                    "No additional month will be computed."
                )

                break

            print()
            print(
                f"{month_start:%Y-%m} | "
                f"expected={expected:,} | "
                f"stored=0 | "
                "NOT STARTED | COMPUTE"
            )

            process_month(
                conn,
                month_start,
                month_end,
                expected,
            )

            processed_new_months += 1
            month_start = month_end

        total_targets_after, completed_after = (
            get_global_status(conn)
        )

        if total_targets_after != total_targets:
            raise RuntimeError(
                "Global target count changed "
                "during production."
            )

        rows_added = (
            completed_after
            - completed_before
        )

        total_seconds = (
            perf_counter()
            - total_start
        )

        print()
        print("=" * 72)
        print("RUN SUMMARY")
        print("=" * 72)

        print(
            f"Complete months skipped : "
            f"{skipped_complete_months:,}"
        )

        print(
            f"New months committed    : "
            f"{processed_new_months:,}"
        )

        print(
            f"Rows before run         : "
            f"{completed_before:,}"
        )

        print(
            f"Rows after run          : "
            f"{completed_after:,}"
        )

        print(
            f"Rows added this run     : "
            f"{rows_added:,}"
        )

        print(
            f"Global remaining        : "
            f"{total_targets_after - completed_after:,}"
        )

        print(
            f"Total runtime           : "
            f"{total_seconds:.3f} s"
        )

        print()
        print(
            "RESUMABLE MONTHLY RUNNER: PASS"
        )

    except Exception:

        # Previously committed monthly checkpoints remain untouched.
        conn.rollback()

        print()
        print(
            "PRODUCTION RUN FAILED - "
            "CURRENT TRANSACTION ROLLED BACK"
        )

        raise

    finally:

        conn.close()


if __name__ == "__main__":
    main()