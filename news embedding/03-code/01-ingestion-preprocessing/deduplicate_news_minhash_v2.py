import os
import re
import time
from collections import OrderedDict

import psycopg2
from datasketch import MinHash, MinHashLSH
from psycopg2.extras import execute_batch

try:
    import psutil
except ImportError:
    psutil = None


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

SCHEMA = "risk_emergence"
TABLE = "news_articles"

NUM_PERM = 128
LSH_THRESHOLD = 0.90
VERIFY_JACCARD_THRESHOLD = 0.90
SHINGLE_SIZE = 5
MIN_WORDS = 30

FETCH_SIZE = 2000
UPDATE_BATCH_SIZE = 2000
SHINGLE_CACHE_SIZE = 50000
PROGRESS_EVERY = 10000
EXAMPLE_LIMIT = 20

# Set to an integer to run the procedure on a limited benchmark sample.
MAX_ARTICLES = None

# Set to False to run the procedure without persisting database changes.
COMMIT_CHANGES = True


_whitespace_pattern = re.compile(r"\s+")
_token_pattern = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?",
    re.I,
)

_process = (
    psutil.Process(os.getpid())
    if psutil
    else None
)


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def ram_gb():
    """Return the current Python process memory usage in GB."""

    if _process is None:
        return None

    return _process.memory_info().rss / (1024 ** 3)


def normalize_text(text):
    """Normalize article text before shingling."""

    if not text:
        return ""

    text = _whitespace_pattern.sub(
        " ",
        text.lower(),
    ).strip()

    return " ".join(
        _token_pattern.findall(text)
    )


def make_shingles(text):
    """
    Generate word shingles for an article.

    Articles shorter than MIN_WORDS are excluded from MinHash comparison.
    """

    words = normalize_text(text).split()

    if len(words) < MIN_WORDS:
        return None, len(words)

    shingles = {
        " ".join(
            words[i:i + SHINGLE_SIZE]
        ).encode("utf-8")
        for i in range(
            len(words) - SHINGLE_SIZE + 1
        )
    }

    return shingles, len(words)


def make_minhash(shingles):
    """Generate a MinHash signature from a set of shingles."""

    minhash = MinHash(
        num_perm=NUM_PERM
    )

    try:
        minhash.update_batch(shingles)

    except AttributeError:
        # Compatibility fallback for datasketch versions
        # without MinHash.update_batch().
        for shingle in shingles:
            minhash.update(shingle)

    return minhash


def jaccard(a, b):
    """Compute exact Jaccard similarity between two shingle sets."""

    if not a or not b:
        return 0.0

    union_size = len(a | b)

    return (
        len(a & b) / union_size
        if union_size
        else 0.0
    )


# ============================================================================
# SHINGLE CACHE
# ============================================================================

class LRUCache:
    """Bounded LRU cache used to limit in-memory shingle storage."""

    def __init__(self, size):
        self.size = size
        self.data = OrderedDict()

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)

        while len(self.data) > self.size:
            self.data.popitem(last=False)

    def get(self, key):
        value = self.data.get(key)

        if value is not None:
            self.data.move_to_end(key)

        return value

    def __len__(self):
        return len(self.data)


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def get_connection():
    """Open a PostgreSQL connection."""

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. "
            "Set the DB_PASSWORD environment variable."
        )

    return psycopg2.connect(**DB_CONFIG)


def fetch_shingles(conn, article_id):
    """
    Reconstruct shingles for an LSH candidate that is no longer cached.
    """

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT body
            FROM {SCHEMA}.{TABLE}
            WHERE article_id = %s
            """,
            (int(article_id),),
        )

        row = cur.fetchone()

    return (
        make_shingles(row[0])[0]
        if row
        else None
    )


def flush_true_updates(conn, article_ids):
    """Persist verified near-duplicate flags in batches."""

    if not article_ids:
        return

    with conn.cursor() as cur:
        execute_batch(
            cur,
            f"""
            UPDATE {SCHEMA}.{TABLE}
            SET is_duplicate_verified = TRUE
            WHERE article_id = %s
            """,
            [
                (article_id,)
                for article_id in article_ids
            ],
            page_size=UPDATE_BATCH_SIZE,
        )

    article_ids.clear()


# ============================================================================
# PROGRESS REPORTING
# ============================================================================

def print_progress(
    processed,
    total,
    near_duplicates,
    candidates,
    start_time,
    cache,
    peak_ram,
):
    """Print processing speed, progress, memory usage, and ETA."""

    elapsed = time.time() - start_time

    speed = (
        processed / elapsed
        if elapsed
        else 0
    )

    eta = (
        (total - processed) / speed / 60
        if speed
        else 0
    )

    current_ram = ram_gb()

    if current_ram is not None:
        peak_ram[0] = max(
            peak_ram[0],
            current_ram,
        )

        memory_info = (
            f" | RAM={current_ram:.2f} GB "
            f"(peak={peak_ram[0]:.2f})"
        )
    else:
        memory_info = ""

    percentage = (
        processed / total * 100
        if total
        else 0.0
    )

    print(
        f"[PROGRESS] {processed:,}/{total:,} "
        f"({percentage:6.2f}%) | "
        f"near_dup={near_duplicates:,} | "
        f"LSH_candidates={candidates:,} | "
        f"cache={len(cache):,} | "
        f"speed={speed:,.0f} art/s | "
        f"ETA={eta:.1f} min"
        f"{memory_info}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run MinHash-LSH near-duplicate detection on eligible news articles."""

    start_time = time.time()
    peak_ram = [ram_gb() or 0.0]

    conn = get_connection()
    conn.autocommit = False

    print("Database connection established.")

    required_columns = {
        "article_id",
        "publication_date",
        "body",
        "is_english",
        "is_duplicate_flag",
        "is_duplicate_verified",
        "is_excluded",
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            """,
            (SCHEMA, TABLE),
        )

        columns = {
            row[0]
            for row in cur.fetchall()
        }

    missing_columns = (
        required_columns - columns
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print("Schema validation completed successfully.")

    where_clause = """
        is_english = TRUE
        AND COALESCE(is_duplicate_flag, FALSE) = FALSE
        AND COALESCE(is_excluded, FALSE) = FALSE
        AND body IS NOT NULL
        AND BTRIM(body) <> ''
    """

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.{TABLE}
            WHERE {where_clause}
            """
        )

        available = cur.fetchone()[0]

    total = (
        min(available, MAX_ARTICLES)
        if MAX_ARTICLES
        else available
    )

    print(
        f"Articles eligible for MinHash: "
        f"{available:,}"
    )
    print(
        f"Articles to process: {total:,}"
    )
    print(
        f"Parameters: "
        f"shingles={SHINGLE_SIZE}-word | "
        f"num_perm={NUM_PERM} | "
        f"LSH threshold={LSH_THRESHOLD:.2f} | "
        f"Jaccard verification="
        f"{VERIFY_JACCARD_THRESHOLD:.2f}"
    )
    print(
        f"COMMIT_CHANGES={COMMIT_CHANGES}"
    )

    if psutil:
        print(
            "Initial Python process RAM: "
            f"{ram_gb():.3f} GB"
        )
    else:
        print(
            "psutil not installed: "
            "RAM monitoring disabled."
        )

    # Reset existing verified flags before recomputing the complete
    # near-duplicate classification. Dry runs revert this update.
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {SCHEMA}.{TABLE}
            SET is_duplicate_verified = FALSE
            WHERE is_duplicate_verified
                  IS DISTINCT FROM FALSE
            """
        )

    lsh = MinHashLSH(
        threshold=LSH_THRESHOLD,
        num_perm=NUM_PERM,
    )

    cache = LRUCache(
        SHINGLE_CACHE_SIZE
    )

    read_cur = conn.cursor(
        name="minhash_cursor_v2"
    )

    read_cur.itersize = FETCH_SIZE

    query = f"""
        SELECT
            article_id,
            publication_date,
            body
        FROM {SCHEMA}.{TABLE}
        WHERE {where_clause}
        ORDER BY
            publication_date,
            article_id
    """

    if MAX_ARTICLES:
        query += (
            f" LIMIT {int(MAX_ARTICLES)}"
        )

    read_cur.execute(query)

    processed = 0
    too_short = 0
    near_duplicates = 0
    candidates_total = 0
    jaccard_checks = 0
    db_fallbacks = 0

    true_updates = []
    examples = []

    for (
        article_id,
        publication_date,
        body,
    ) in read_cur:

        processed += 1

        shingles, word_count = (
            make_shingles(body)
        )

        if not shingles:
            too_short += 1

        else:
            minhash = make_minhash(
                shingles
            )

            candidate_keys = lsh.query(
                minhash
            )

            candidates_total += len(
                candidate_keys
            )

            best_key = None
            best_jaccard = 0.0

            # LSH generates candidates; exact Jaccard similarity
            # determines whether a candidate is a verified duplicate.
            for key in candidate_keys:
                other_shingles = cache.get(
                    key
                )

                if other_shingles is None:
                    db_fallbacks += 1

                    other_shingles = (
                        fetch_shingles(
                            conn,
                            key,
                        )
                    )

                    if other_shingles:
                        cache.put(
                            key,
                            other_shingles,
                        )

                if not other_shingles:
                    continue

                jaccard_checks += 1

                score = jaccard(
                    shingles,
                    other_shingles,
                )

                if score > best_jaccard:
                    best_jaccard = score
                    best_key = key

                if (
                    score
                    >= VERIFY_JACCARD_THRESHOLD
                ):
                    break

            duplicate = (
                best_key is not None
                and best_jaccard
                >= VERIFY_JACCARD_THRESHOLD
            )

            if duplicate:
                near_duplicates += 1

                if COMMIT_CHANGES:
                    true_updates.append(
                        article_id
                    )

                    if (
                        len(true_updates)
                        >= UPDATE_BATCH_SIZE
                    ):
                        flush_true_updates(
                            conn,
                            true_updates,
                        )

                if (
                    len(examples)
                    < EXAMPLE_LIMIT
                ):
                    examples.append(
                        (
                            article_id,
                            int(best_key),
                            publication_date,
                            best_jaccard,
                            word_count,
                        )
                    )

            else:
                # Only canonical/non-duplicate articles enter the LSH index.
                # Subsequent articles are therefore compared against retained
                # representatives rather than against already detected copies.
                key = str(article_id)

                lsh.insert(
                    key,
                    minhash,
                )

                cache.put(
                    key,
                    shingles,
                )

        if (
            processed % PROGRESS_EVERY == 0
            or processed == total
        ):
            print_progress(
                processed,
                total,
                near_duplicates,
                candidates_total,
                start_time,
                cache,
                peak_ram,
            )

    if COMMIT_CHANGES:
        flush_true_updates(
            conn,
            true_updates,
        )

    read_cur.close()

    if COMMIT_CHANGES:
        conn.commit()
        action = "COMMIT"
    else:
        conn.rollback()
        action = "ROLLBACK (dry-run)"

    elapsed = time.time() - start_time
    final_ram = ram_gb()

    near_duplicate_share = (
        near_duplicates / processed * 100
        if processed
        else 0.0
    )

    average_speed = (
        processed / elapsed
        if elapsed
        else 0.0
    )

    print()
    print("=" * 80)
    print("MINHASH NEAR-DUPLICATION REPORT")
    print("=" * 80)
    print(
        f"Articles processed                 : "
        f"{processed:,}"
    )
    print(
        f"Articles below minimum length      : "
        f"{too_short:,}"
    )
    print(
        f"Candidates returned by LSH         : "
        f"{candidates_total:,}"
    )
    print(
        f"Exact Jaccard comparisons          : "
        f"{jaccard_checks:,}"
    )
    print(
        f"PostgreSQL shingle fallbacks       : "
        f"{db_fallbacks:,}"
    )
    print(
        f"Verified near-duplicates           : "
        f"{near_duplicates:,}"
    )
    print(
        f"Near-duplicate share               : "
        f"{near_duplicate_share:.4f}%"
    )
    print(
        f"Final shingle cache size           : "
        f"{len(cache):,}"
    )

    if final_ram is not None:
        peak_ram[0] = max(
            peak_ram[0],
            final_ram,
        )

        print(
            f"Final Python process RAM           : "
            f"{final_ram:.3f} GB"
        )
        print(
            f"Observed peak RAM                  : "
            f"{peak_ram[0]:.3f} GB"
        )

    print(
        f"Database action                    : "
        f"{action}"
    )
    print(
        f"Total execution time               : "
        f"{elapsed / 60:.2f} min"
    )
    print(
        f"Average processing speed           : "
        f"{average_speed:,.1f} art/s"
    )

    if examples:
        print()
        print("NEAR-DUPLICATE EXAMPLES")

        for (
            duplicate_id,
            original_id,
            publication_date,
            score,
            word_count,
        ) in examples:

            print(
                f"  duplicate={duplicate_id} "
                f"original={original_id} "
                f"date={publication_date} "
                f"jaccard={score:.4f} "
                f"words={word_count}"
            )

    conn.close()


if __name__ == "__main__":
    main()