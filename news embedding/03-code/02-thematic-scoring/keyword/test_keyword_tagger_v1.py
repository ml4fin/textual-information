#!/usr/bin/env python3

"""
Diagnostic test for seed-word matching.

The script loads a deterministic sample of valid articles and the active
candidate themes, then checks which seed words occur in each article.

For each positive article-theme match, it reports:
- the fraction of the theme's seed words found in the article;
- the matched seed words;
- a short text context around each match.

This script is intended for qualitative inspection of the seed-word
matching logic. It does not write results to the database.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

import psycopg2
from psycopg2.extensions import connection as PgConnection


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

SCHEMA_NAME = "risk_emergence"
THEMES_TABLE = "candidate_themes"
VALID_ARTICLES_VIEW = "v_valid_articles"

SAMPLE_SIZE = 1000
CONTEXT_WINDOW = 80
MAX_RESULTS_TO_PRINT = 50


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Theme:
    theme_id: int
    theme_name: str
    seed_words: list[str]


@dataclass
class Article:
    article_id: int
    title: str
    body: str


# =============================================================================
# DATABASE
# =============================================================================

def connect_to_database() -> PgConnection:
    """Open a PostgreSQL connection."""

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. "
            "Set the DB_PASSWORD environment variable."
        )

    return psycopg2.connect(**DB_CONFIG)


def load_active_themes(
    conn: PgConnection,
) -> list[Theme]:
    """Load active themes and their seed words."""

    query = f"""
        SELECT
            theme_id,
            theme_name,
            seed_words
        FROM {SCHEMA_NAME}.{THEMES_TABLE}
        WHERE is_active = TRUE
        ORDER BY theme_id;
    """

    with conn.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()

    return [
        Theme(
            theme_id=theme_id,
            theme_name=theme_name,
            seed_words=seed_words or [],
        )
        for theme_id, theme_name, seed_words in rows
    ]


def load_article_sample(
    conn: PgConnection,
    sample_size: int,
) -> list[Article]:
    """
    Load a deterministic sample of valid articles.

    Ordering by article_id avoids the cost of ORDER BY random() on the
    full corpus. The resulting sample is intended for diagnostic testing,
    not as a statistically random sample.
    """

    query = f"""
        SELECT
            article_id,
            title,
            body
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
        ORDER BY article_id
        LIMIT %s;
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (sample_size,),
        )
        rows = cursor.fetchall()

    return [
        Article(
            article_id=article_id,
            title=title or "",
            body=body or "",
        )
        for article_id, title, body in rows
    ]


# =============================================================================
# SEED-WORD MATCHING
# =============================================================================

def normalize_text(
    text: Optional[str],
) -> str:
    """Lowercase text and normalize whitespace."""

    if not text:
        return ""

    text = text.lower()
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_seed_word(
    text: str,
    seed_word: str,
) -> bool:
    """
    Check whether a seed word or phrase occurs as a bounded expression.

    Word boundaries prevent short seeds such as "AI" from matching
    substrings inside unrelated words such as "said".
    """

    normalized_seed = normalize_text(
        seed_word
    )

    if not normalized_seed:
        return False

    pattern = (
        rf"(?<!\w)"
        rf"{re.escape(normalized_seed)}"
        rf"(?!\w)"
    )

    return (
        re.search(pattern, text)
        is not None
    )


def extract_context(
    text: str,
    seed_word: str,
    window: int = CONTEXT_WINDOW,
) -> str:
    """Return a short context around the first occurrence of a seed."""

    normalized_seed = normalize_text(
        seed_word
    )

    if not normalized_seed:
        return ""

    pattern = (
        rf"(?<!\w)"
        rf"{re.escape(normalized_seed)}"
        rf"(?!\w)"
    )

    match = re.search(
        pattern,
        text,
    )

    if match is None:
        return ""

    start = max(
        0,
        match.start() - window,
    )

    end = min(
        len(text),
        match.end() + window,
    )

    return text[start:end].replace(
        "\n",
        " ",
    )


def calculate_keyword_score(
    article: Article,
    theme: Theme,
) -> tuple[float, list[str]]:
    """
    Compute the fraction of a theme's seed words found in an article.

    The score is used only for this diagnostic inspection and is distinct
    from the PostgreSQL ts_rank-based keyword baseline.
    """

    complete_text = normalize_text(
        f"{article.title} {article.body}"
    )

    total_seed_words = len(
        theme.seed_words
    )

    if total_seed_words == 0:
        return 0.0, []

    matched_seed_words = [
        seed_word
        for seed_word in theme.seed_words
        if contains_seed_word(
            complete_text,
            seed_word,
        )
    ]

    score = (
        len(matched_seed_words)
        / total_seed_words
    )

    return score, matched_seed_words


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the qualitative seed-word matching test."""

    conn = None

    try:
        conn = connect_to_database()

        themes = load_active_themes(
            conn
        )

        articles = load_article_sample(
            conn,
            SAMPLE_SIZE,
        )

        print(
            f"Themes loaded: {len(themes)}"
        )
        print(
            f"Articles loaded: {len(articles)}"
        )

        print()
        print("THEMES AND SEED WORDS")

        for theme in themes:
            print(
                f"- {theme.theme_id} | "
                f"{theme.theme_name} | "
                f"{theme.seed_words}"
            )

        results_with_matches = []

        for article in articles:
            complete_text = normalize_text(
                f"{article.title} {article.body}"
            )

            for theme in themes:
                score, matched_words = (
                    calculate_keyword_score(
                        article,
                        theme,
                    )
                )

                if score <= 0:
                    continue

                contexts = [
                    {
                        "word": word,
                        "context": extract_context(
                            complete_text,
                            word,
                        ),
                    }
                    for word in matched_words
                ]

                results_with_matches.append(
                    {
                        "article_id":
                            article.article_id,
                        "title":
                            article.title,
                        "theme_id":
                            theme.theme_id,
                        "theme_name":
                            theme.theme_name,
                        "score":
                            score,
                        "matched_words":
                            matched_words,
                        "contexts":
                            contexts,
                    }
                )

        results_with_matches.sort(
            key=lambda row: row["score"],
            reverse=True,
        )

        print()
        print("RESULTS WITH SCORE ABOVE ZERO")
        print(
            f"Total results: "
            f"{len(results_with_matches)}"
        )

        for result in (
            results_with_matches[
                :MAX_RESULTS_TO_PRINT
            ]
        ):
            print("-" * 100)
            print(
                f"article_id: "
                f"{result['article_id']}"
            )
            print(
                f"title: "
                f"{result['title']}"
            )
            print(
                f"theme: "
                f"{result['theme_name']}"
            )
            print(
                f"diagnostic score: "
                f"{result['score']:.4f}"
            )
            print("Matched seeds:")

            for context in result["contexts"]:
                print(
                    f"  - {context['word']}"
                )
                print(
                    f"    {context['context']}"
                )

    except psycopg2.Error as exc:
        print("PostgreSQL error:")
        print(exc)
        raise

    except Exception as exc:
        print("Execution error:")
        print(exc)
        raise

    finally:
        if conn is not None:
            conn.close()
            print()
            print(
                "Database connection closed."
            )


if __name__ == "__main__":
    main()