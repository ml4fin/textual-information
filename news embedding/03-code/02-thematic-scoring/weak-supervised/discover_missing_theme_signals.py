#!/usr/bin/env python3

"""
Discovery of potentially missing lexical theme signals.

The analysis focuses on records classified by the rule-based procedure as
"no_theme_signal" and compares:

- false negatives (FN): human-positive cases missed by the rule;
- true negatives (TN): human-negative cases correctly rejected.

For each theme, unigram and bigram candidates are evaluated using their
document prevalence and mean TF-IDF in FN and TN examples.

The discrimination score is defined as:

    (FN_share - TN_share)
    *
    (FN_mean_tfidf - TN_mean_tfidf)

Terms that are more prevalent among false negatives may indicate lexical
signals that are missing from the current theme-detection rules.

This is an exploratory diagnostic analysis and does not modify the database.
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
    "MISSING_THEME_SIGNAL_OUTPUT",
    "missing_theme_signal_candidates.csv",
)

TRUE_COL = "human_review_decision"
PRED_COL = "suggested_label"
THEME_COL = "theme_name"

MIN_DOCUMENT_FREQUENCY = 2
MAX_DOCUMENT_FREQUENCY = 0.90
TOP_RESULTS_PER_THEME = 30


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

df[PRED_COL] = pd.to_numeric(
    df[PRED_COL],
    errors="coerce",
)

df = df.dropna(
    subset=[
        TRUE_COL,
        PRED_COL,
    ]
).copy()

df[TRUE_COL] = (
    df[TRUE_COL]
    .astype(int)
)

df[PRED_COL] = (
    df[PRED_COL]
    .astype(int)
)


# =============================================================================
# TEXT
# =============================================================================

df["text"] = (
    df["title"]
    .fillna("")
    .astype(str)
    + " "
    + df["body"]
    .fillna("")
    .astype(str)
)


# =============================================================================
# NO_THEME_SIGNAL SUBSET
# =============================================================================

subset = df[
    df["rule_reason"]
    == "no_theme_signal"
].copy()

subset["class_type"] = np.where(
    subset[TRUE_COL] == 1,
    "FN",
    "TN",
)


print("=" * 110)
print("DISCOVERY OF MISSING THEME SIGNALS")
print("=" * 110)

print(
    f"\nno_theme_signal records: "
    f"{len(subset):,}"
)

print("\nClass distribution:")

print(
    subset["class_type"]
    .value_counts()
    .to_string()
)


# =============================================================================
# THEME ANALYSIS
# =============================================================================

def analyse_theme(
    theme_df: pd.DataFrame,
    theme_name: str,
) -> list[dict]:
    """
    Identify terms that distinguish false negatives from true negatives
    within a single theme.
    """

    false_negatives = theme_df[
        theme_df["class_type"] == "FN"
    ]

    true_negatives = theme_df[
        theme_df["class_type"] == "TN"
    ]

    print("\n" + "#" * 110)
    print(theme_name)
    print("#" * 110)

    print(
        f"FN: {len(false_negatives):,} | "
        f"TN: {len(true_negatives):,}"
    )

    if false_negatives.empty:
        print(
            "No false negatives available "
            "for this theme."
        )
        return []

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

    try:
        X = vectorizer.fit_transform(
            theme_df["text"]
        )

    except ValueError:
        print(
            "Insufficient vocabulary for TF-IDF analysis."
        )
        return []

    terms = np.array(
        vectorizer.get_feature_names_out()
    )

    fn_mask = (
        theme_df["class_type"].values
        == "FN"
    )

    tn_mask = (
        theme_df["class_type"].values
        == "TN"
    )


    # =============================================================================
    # MEAN TF-IDF
    # =============================================================================

    fn_mean = np.asarray(
        X[
            fn_mask
        ].mean(axis=0)
    ).ravel()

    if true_negatives.empty:
        tn_mean = np.zeros_like(
            fn_mean
        )
    else:
        tn_mean = np.asarray(
            X[
                tn_mask
            ].mean(axis=0)
        ).ravel()


    # =============================================================================
    # DOCUMENT PREVALENCE
    # =============================================================================

    binary_X = (
        X > 0
    ).astype(int)

    fn_docs = np.asarray(
        binary_X[
            fn_mask
        ].sum(axis=0)
    ).ravel()

    tn_docs = np.asarray(
        binary_X[
            tn_mask
        ].sum(axis=0)
    ).ravel()

    fn_share = (
        fn_docs
        / len(false_negatives)
    )

    if true_negatives.empty:
        tn_share = np.zeros_like(
            tn_docs,
            dtype=float,
        )
    else:
        tn_share = (
            tn_docs
            / len(true_negatives)
        )


    # =============================================================================
    # DISCRIMINATION SCORE
    # =============================================================================

    tfidf_difference = (
        fn_mean
        - tn_mean
    )

    prevalence_difference = (
        fn_share
        - tn_share
    )

    # High values identify terms that are both more prevalent and more
    # strongly weighted in false negatives than in true negatives.
    discrimination_score = (
        tfidf_difference
        * prevalence_difference
    )


    # =============================================================================
    # CANDIDATE TERMS
    # =============================================================================

    rows = []

    for i, term in enumerate(
        terms
    ):

        # Require support from at least two false-negative documents.
        if (
            fn_docs[i]
            < MIN_DOCUMENT_FREQUENCY
        ):
            continue

        if (
            fn_share[i]
            <= tn_share[i]
        ):
            continue

        rows.append(
            {
                "theme_name":
                    theme_name,

                "candidate":
                    term,

                "FN_docs":
                    int(fn_docs[i]),

                "TN_docs":
                    int(tn_docs[i]),

                "FN_share":
                    fn_share[i],

                "TN_share":
                    tn_share[i],

                "share_difference":
                    prevalence_difference[i],

                "FN_mean_tfidf":
                    fn_mean[i],

                "TN_mean_tfidf":
                    tn_mean[i],

                "discrimination_score":
                    discrimination_score[i],
            }
        )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        print(
            "No candidate terms satisfied "
            "the selection criteria."
        )
        return []

    results = results.sort_values(
        [
            "discrimination_score",
            "FN_docs",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print(
        f"\nTOP {TOP_RESULTS_PER_THEME} CANDIDATES:"
    )

    print(
        results
        .head(TOP_RESULTS_PER_THEME)
        .to_string(
            index=False
        )
    )

    return results.to_dict(
        orient="records"
    )


# =============================================================================
# RUN BY THEME
# =============================================================================

all_results = []

for theme_name, group in subset.groupby(
    THEME_COL
):

    theme_results = analyse_theme(
        group,
        theme_name,
    )

    all_results.extend(
        theme_results
    )


# =============================================================================
# SAVE
# =============================================================================

result_df = pd.DataFrame(
    all_results
)

if not result_df.empty:
    result_df = result_df.sort_values(
        [
            "theme_name",
            "discrimination_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

result_df.to_csv(
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