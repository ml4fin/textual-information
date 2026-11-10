#!/usr/bin/env python3

"""
Exploratory analysis of lexical context for the AI-related risk theme.

The script compares human-reviewed positive and negative AI-related examples
and identifies unigram and bigram candidates that are more characteristic
of positive cases.

Candidate terms are evaluated using:
1. document prevalence in positive and negative examples;
2. mean TF-IDF in positive and negative examples;
3. a discrimination score defined as:

   (positive_share - negative_share)
   *
   (positive_mean_tfidf - negative_mean_tfidf)

The output is intended for qualitative analysis and potential refinement
of theme-specific lexical signals.
"""

import os

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
    "AI_CONTEXT_OUTPUT_FILE",
    "ai_context_candidates.csv",
)

TRUE_COL = "human_review_decision"
THEME_COL = "theme_name"

AI_THEME = "AI-Related Risk"

MIN_DOCUMENT_FREQUENCY = 2
MAX_DOCUMENT_FREQUENCY = 0.90
TOP_RESULTS_TO_PRINT = 50


# =============================================================================
# DATA LOADING
# =============================================================================

df = pd.read_csv(INPUT_FILE)

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

ai["class_type"] = np.where(
    ai[TRUE_COL] == 1,
    "POSITIVE",
    "NEGATIVE",
)

ai["text"] = (
    ai["title"]
    .fillna("")
    .astype(str)
    + " "
    + ai["body"]
    .fillna("")
    .astype(str)
)


# =============================================================================
# SAMPLE SUMMARY
# =============================================================================

print("=" * 110)
print("AI CONTEXT ANALYSIS")
print("=" * 110)

print(
    f"\nAI records: {len(ai):,}"
)

print("\nLabels:")

print(
    ai["class_type"]
    .value_counts()
    .to_string()
)


# =============================================================================
# TITLE INSPECTION
# =============================================================================

print("\n" + "=" * 110)
print("AI POSITIVE TITLES")
print("=" * 110)

for _, row in ai[
    ai["class_type"] == "POSITIVE"
].iterrows():

    print(
        f"\nID {row['annotation_id']}"
    )

    print(
        f"Theme evidence: "
        f"{row['theme_evidence']}"
    )

    print(row["title"])


print("\n" + "=" * 110)
print("AI NEGATIVE TITLES")
print("=" * 110)

for _, row in ai[
    ai["class_type"] == "NEGATIVE"
].iterrows():

    print(
        f"\nID {row['annotation_id']}"
    )

    print(
        f"Theme evidence: "
        f"{row['theme_evidence']}"
    )

    print(row["title"])


# =============================================================================
# TF-IDF
# =============================================================================

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
    ai["text"]
)

terms = np.array(
    vectorizer.get_feature_names_out()
)

positive_mask = (
    ai["class_type"].values
    == "POSITIVE"
)

negative_mask = (
    ai["class_type"].values
    == "NEGATIVE"
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

n_positive = int(
    positive_mask.sum()
)

n_negative = int(
    negative_mask.sum()
)

if n_positive == 0 or n_negative == 0:
    raise ValueError(
        "Both positive and negative AI examples are required "
        "for the discrimination analysis."
    )

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


# =============================================================================
# DISCRIMINATION SCORE
# =============================================================================

share_difference = (
    positive_share
    - negative_share
)

tfidf_difference = (
    positive_mean
    - negative_mean
)

# The score rewards terms that are both more prevalent and more strongly
# weighted in positive examples than in negative examples.
discrimination_score = (
    share_difference
    * tfidf_difference
)


# =============================================================================
# CANDIDATE TERMS
# =============================================================================

rows = []

for i, term in enumerate(terms):

    # Require support from at least two positive documents to reduce
    # candidates driven by isolated examples.
    if positive_docs[i] < 2:
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

results = pd.DataFrame(rows)

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
# CANDIDATE REPORT
# =============================================================================

print("\n" + "=" * 110)
print("TOP AI POSITIVE CONTEXT CANDIDATES")
print("=" * 110)

if results.empty:
    print(
        "No candidate terms satisfied the selection criteria."
    )
else:
    print(
        results
        .head(TOP_RESULTS_TO_PRINT)
        .to_string(
            index=False
        )
    )


# =============================================================================
# THEME-EVIDENCE DISTRIBUTION
# =============================================================================

print("\n" + "=" * 110)
print("THEME EVIDENCE: POSITIVE VS NEGATIVE")
print("=" * 110)

theme_rows = []

for label, group in ai.groupby(
    "class_type"
):

    counts = (
        group["theme_evidence"]
        .fillna("")
        .astype(str)
        .str.lower()
        .value_counts()
    )

    for signal, count in counts.items():
        theme_rows.append(
            {
                "class_type":
                    label,

                "theme_signal":
                    signal,

                "n":
                    count,
            }
        )

theme_df = pd.DataFrame(
    theme_rows
)

if theme_df.empty:
    print(
        "No theme-evidence values available."
    )
else:
    print(
        theme_df
        .sort_values(
            [
                "theme_signal",
                "class_type",
            ]
        )
        .to_string(
            index=False
        )
    )


# =============================================================================
# SAVE
# =============================================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 110)
print("FILE SAVED")
print("=" * 110)

print(OUTPUT_FILE)

print(
    "\nAnalysis completed."
)