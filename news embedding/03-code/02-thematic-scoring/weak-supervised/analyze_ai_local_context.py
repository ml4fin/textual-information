#!/usr/bin/env python3

"""
Exploratory analysis of local lexical context for the AI-related risk theme.

The script uses human-reviewed AI-related examples and extracts a local
token window around the annotated theme evidence. Positive and negative
contexts are then compared using document prevalence and TF-IDF.

Candidate terms are ranked using:

    (positive_share - negative_share)
    *
    (positive_mean_tfidf - negative_mean_tfidf)

A separate title-level comparison is also produced.

This is a diagnostic analysis of the reviewed sample and does not modify
the database or produce article-level thematic scores.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = os.getenv(
    "GOLD_REVIEW_FILE",
    "gold_review_high.csv",
)

OUTPUT_FILE = os.getenv(
    "AI_LOCAL_CONTEXT_OUTPUT_FILE",
    "ai_local_context_candidates.csv",
)

CONTEXT_FILE = os.getenv(
    "AI_LOCAL_CONTEXT_FILE",
    "ai_local_contexts.csv",
)

TRUE_COL = "human_review_decision"
THEME_COL = "theme_name"

AI_THEME = "AI-Related Risk"

CONTEXT_WINDOW = 50
MIN_DOCUMENT_FREQUENCY = 2
MAX_DOCUMENT_FREQUENCY = 0.95

TOP_CONTEXT_RESULTS = 50
TOP_TITLE_RESULTS = 40


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def tokenize(text: str) -> list[str]:
    """Tokenize text using lowercase alphanumeric and hyphenated terms."""

    if pd.isna(text):
        return []

    return re.findall(
        r"[a-z0-9\-]+",
        str(text).lower(),
    )


def normalize_phrase(value: str) -> str:
    """Normalize an annotated theme-evidence phrase."""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def find_phrase_positions(
    tokens: list[str],
    phrase: str,
) -> list[int]:
    """Return token positions where the complete phrase begins."""

    phrase_tokens = tokenize(
        phrase
    )

    if not phrase_tokens:
        return []

    phrase_length = len(
        phrase_tokens
    )

    positions = []

    for i in range(
        len(tokens) - phrase_length + 1
    ):
        if (
            tokens[i:i + phrase_length]
            == phrase_tokens
        ):
            positions.append(i)

    return positions


def extract_context(
    text: str,
    phrase: str,
    window: int = CONTEXT_WINDOW,
) -> dict:
    """
    Extract token windows around all exact occurrences of a theme phrase.

    If the annotated phrase cannot be located exactly after tokenization,
    the full tokenized article is retained as a fallback and the record is
    marked with found_exact=False.
    """

    tokens = tokenize(
        text
    )

    positions = find_phrase_positions(
        tokens,
        phrase,
    )

    if not positions:
        return {
            "context": " ".join(tokens),
            "found_exact": False,
            "n_mentions": 0,
        }

    contexts = []

    for position in positions:
        start = max(
            0,
            position - window,
        )

        end = min(
            len(tokens),
            position + window + 1,
        )

        contexts.extend(
            tokens[start:end]
        )

    return {
        "context": " ".join(contexts),
        "found_exact": True,
        "n_mentions": len(positions),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

df = pd.read_csv(
    INPUT_FILE
)

df[TRUE_COL] = pd.to_numeric(
    df[TRUE_COL],
    errors="coerce",
)

df = df.dropna(
    subset=[TRUE_COL]
).copy()

df[TRUE_COL] = (
    df[TRUE_COL]
    .astype(int)
)

ai = df[
    df[THEME_COL] == AI_THEME
].copy()

ai["label_name"] = np.where(
    ai[TRUE_COL] == 1,
    "POSITIVE",
    "NEGATIVE",
)

ai["full_text"] = (
    ai["title"]
    .fillna("")
    .astype(str)
    + " "
    + ai["body"]
    .fillna("")
    .astype(str)
)

ai["theme_phrase"] = (
    ai["theme_evidence"]
    .apply(normalize_phrase)
)


# =============================================================================
# LOCAL CONTEXT EXTRACTION
# =============================================================================

contexts = []

for _, row in ai.iterrows():

    result = extract_context(
        row["full_text"],
        row["theme_phrase"],
        CONTEXT_WINDOW,
    )

    contexts.append(
        {
            "annotation_id":
                row["annotation_id"],

            "label":
                int(row[TRUE_COL]),

            "label_name":
                row["label_name"],

            "title":
                row["title"],

            "theme_evidence":
                row["theme_phrase"],

            "found_exact":
                result["found_exact"],

            "n_mentions":
                result["n_mentions"],

            "context":
                result["context"],
        }
    )

context_df = pd.DataFrame(
    contexts
)


# =============================================================================
# SAMPLE SUMMARY
# =============================================================================

print("=" * 115)
print("AI LOCAL CONTEXT ANALYSIS")
print("=" * 115)

print(
    f"\nAI records: {len(context_df):,}"
)

print(
    f"Context window: +/- "
    f"{CONTEXT_WINDOW} tokens"
)

print(
    "\nExact theme phrase found:"
)

print(
    context_df["found_exact"]
    .value_counts()
    .to_string()
)


# =============================================================================
# POSITIVE CONTEXT INSPECTION
# =============================================================================

print("\n" + "=" * 115)
print("POSITIVE LOCAL CONTEXTS")
print("=" * 115)

for _, row in context_df[
    context_df["label"] == 1
].iterrows():

    print(
        f"\nID {row['annotation_id']}"
    )

    print(
        f"Title: {row['title']}"
    )

    print(
        f"Theme: {row['theme_evidence']}"
    )

    print(
        f"Exact match: {row['found_exact']}"
    )

    print("Context:")

    print(
        row["context"]
    )


# =============================================================================
# LOCAL-CONTEXT TF-IDF
# =============================================================================

positive_mask = (
    context_df["label"].values
    == 1
)

negative_mask = (
    context_df["label"].values
    == 0
)

n_positive = int(
    positive_mask.sum()
)

n_negative = int(
    negative_mask.sum()
)

if n_positive == 0 or n_negative == 0:
    raise ValueError(
        "Both positive and negative AI examples are required "
        "for the local-context comparison."
    )

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=MIN_DOCUMENT_FREQUENCY,
    max_df=MAX_DOCUMENT_FREQUENCY,
    token_pattern=(
        r"(?u)\b[a-zA-Z]"
        r"[a-zA-Z\-]+\b"
    ),
)

X = vectorizer.fit_transform(
    context_df["context"]
)

terms = np.array(
    vectorizer.get_feature_names_out()
)


# =============================================================================
# DOCUMENT PREVALENCE
# =============================================================================

binary_X = (
    X > 0
).astype(int)

positive_docs = np.asarray(
    binary_X[
        positive_mask
    ].sum(axis=0)
).ravel()

negative_docs = np.asarray(
    binary_X[
        negative_mask
    ].sum(axis=0)
).ravel()

positive_share = (
    positive_docs
    / n_positive
)

negative_share = (
    negative_docs
    / n_negative
)


# =============================================================================
# MEAN TF-IDF
# =============================================================================

positive_mean = np.asarray(
    X[
        positive_mask
    ].mean(axis=0)
).ravel()

negative_mean = np.asarray(
    X[
        negative_mask
    ].mean(axis=0)
).ravel()

share_difference = (
    positive_share
    - negative_share
)

tfidf_difference = (
    positive_mean
    - negative_mean
)

# Candidate terms receive high values when they are both more prevalent
# and more strongly weighted in positive local contexts.
discrimination_score = (
    share_difference
    * tfidf_difference
)


# =============================================================================
# LOCAL-CONTEXT CANDIDATES
# =============================================================================

rows = []

for i, term in enumerate(terms):

    # Require support from at least two positive contexts.
    if (
        positive_docs[i]
        < MIN_DOCUMENT_FREQUENCY
    ):
        continue

    if (
        positive_share[i]
        <= negative_share[i]
    ):
        continue

    rows.append(
        {
            "candidate":
                term,

            "positive_docs":
                int(positive_docs[i]),

            "negative_docs":
                int(negative_docs[i]),

            "positive_share":
                positive_share[i],

            "negative_share":
                negative_share[i],

            "share_difference":
                share_difference[i],

            "positive_mean_tfidf":
                positive_mean[i],

            "negative_mean_tfidf":
                negative_mean[i],

            "discrimination_score":
                discrimination_score[i],
        }
    )

results = pd.DataFrame(
    rows
)

if not results.empty:
    results = results.sort_values(
        [
            "discrimination_score",
            "positive_docs",
        ],
        ascending=[
            False,
            False,
        ],
    )


# =============================================================================
# LOCAL-CONTEXT REPORT
# =============================================================================

print("\n" + "=" * 115)
print("TOP LOCAL AI CONTEXT CANDIDATES")
print("=" * 115)

if results.empty:
    print(
        "No candidate terms satisfied the selection criteria."
    )
else:
    print(
        results
        .head(TOP_CONTEXT_RESULTS)
        .to_string(
            index=False
        )
    )


# =============================================================================
# TITLE-LEVEL COMPARISON
# =============================================================================

print("\n" + "=" * 115)
print("TITLE WORDS: POSITIVE VS NEGATIVE")
print("=" * 115)

title_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=MIN_DOCUMENT_FREQUENCY,
)

title_X = (
    title_vectorizer
    .fit_transform(
        context_df["title"]
        .fillna("")
    )
)

title_terms = np.array(
    title_vectorizer
    .get_feature_names_out()
)

title_binary = (
    title_X > 0
).astype(int)

title_positive_docs = np.asarray(
    title_binary[
        positive_mask
    ].sum(axis=0)
).ravel()

title_negative_docs = np.asarray(
    title_binary[
        negative_mask
    ].sum(axis=0)
).ravel()

title_rows = []

for i, term in enumerate(
    title_terms
):

    if (
        title_positive_docs[i]
        < MIN_DOCUMENT_FREQUENCY
    ):
        continue

    positive_title_share = (
        title_positive_docs[i]
        / n_positive
    )

    negative_title_share = (
        title_negative_docs[i]
        / n_negative
    )

    if (
        positive_title_share
        <= negative_title_share
    ):
        continue

    title_rows.append(
        {
            "candidate":
                term,

            "positive_docs":
                int(
                    title_positive_docs[i]
                ),

            "negative_docs":
                int(
                    title_negative_docs[i]
                ),

            "positive_share":
                positive_title_share,

            "negative_share":
                negative_title_share,

            "share_difference":
                (
                    positive_title_share
                    - negative_title_share
                ),
        }
    )

title_results = pd.DataFrame(
    title_rows
)

if not title_results.empty:
    title_results = (
        title_results.sort_values(
            [
                "share_difference",
                "positive_docs",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    print(
        title_results
        .head(TOP_TITLE_RESULTS)
        .to_string(
            index=False
        )
    )

else:
    print(
        "No title terms satisfied the selection criteria."
    )


# =============================================================================
# SAVE
# =============================================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

context_df.to_csv(
    CONTEXT_FILE,
    index=False,
)

print("\n" + "=" * 115)
print("FILES SAVED")
print("=" * 115)

print(OUTPUT_FILE)
print(CONTEXT_FILE)

print(
    "\nAnalysis completed."
)