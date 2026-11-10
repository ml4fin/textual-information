#!/usr/bin/env python3

"""
Supervised Model V3 — Final Development Fit / Production Candidate

Fit the selected V3 hybrid architecture on all 485 development observations.

Architecture:
    shared word TF-IDF
    + shared character TF-IDF
    + explicit theme × text TF-IDF
    -> Logistic Regression

Frozen 5-fold out-of-fold development performance:
    ROC-AUC = 0.815162
    Average Precision = 0.659727

Selected global development threshold:
    0.34

This is not yet the final frozen supervised model. It is fitted only to
support a production sanity check on the same 1,000 production articles
previously used for the V1 diagnostic.

No production observations are used for training.
No final-test observations are used for training.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "SUPERVISED_CV_DATASET",
        "supervised_cv_dataset.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "SUPERVISED_V3_CANDIDATE_DIR",
        "supervised_v3_production_candidate",
    )
)

MODEL_FILE = OUTPUT_DIR / "supervised_v3_candidate.joblib"

METADATA_FILE = (
    OUTPUT_DIR
    / "supervised_v3_candidate_metadata.json"
)


# =============================================================================
# FROZEN DEVELOPMENT SPECIFICATION
# =============================================================================

EXPECTED_N = 485
EXPECTED_NEGATIVES = 337
EXPECTED_POSITIVES = 148

EXPECTED_THEMES = [
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
]

RANDOM_STATE = 42
C_VALUE = 4.0
SELECTED_THRESHOLD = 0.34
CALIBRATION = "none"


# =============================================================================
# FEATURE SPECIFICATION
# =============================================================================

WORD_MAX_FEATURES = 25_000
CHAR_MAX_FEATURES = 25_000
INTERACTION_MAX_FEATURES = 50_000

SEPARATOR = "|||"

TOKEN_PATTERN = re.compile(
    r"(?u)\b\w\w+\b"
)


# =============================================================================
# DATA VALIDATION
# =============================================================================

REQUIRED_COLUMNS = {
    "article_id",
    "theme_name",
    "title",
    "body",
    "human_review_decision",
    "cv_fold",
}


# =============================================================================
# TEXT REPRESENTATION
# =============================================================================

def clean_text(value) -> str:

    if pd.isna(value):
        return ""

    return str(value)


def normalize_theme(theme: str) -> str:

    theme = clean_text(theme).lower()

    theme = re.sub(
        r"[^a-z0-9]+",
        "_",
        theme,
    )

    return theme.strip("_")


def build_v3_document(
    theme_name: str,
    title: str,
    body: str,
) -> str:

    theme_key = normalize_theme(
        theme_name
    )

    return (
        f"{theme_key} {SEPARATOR} "
        f"{clean_text(title)} {SEPARATOR} "
        f"{clean_text(body)}"
    )


def build_v3_document_from_row(
    row,
) -> str:

    return build_v3_document(
        theme_name=row["theme_name"],
        title=row["title"],
        body=row["body"],
    )


def parse_v3_document(
    document: str,
):

    parts = document.split(
        SEPARATOR,
        maxsplit=2,
    )

    if len(parts) != 3:
        raise ValueError(
            "Invalid V3 document. Expected format: "
            "'theme ||| title ||| body'."
        )

    theme_key = parts[0].strip()
    title = parts[1].strip()
    body = parts[2].strip()

    return (
        theme_key,
        title,
        body,
    )


# =============================================================================
# SHARED TEXT EXTRACTOR
# =============================================================================

class SharedTextExtractor(
    BaseEstimator,
    TransformerMixin,
):

    def fit(
        self,
        X,
        y=None,
    ):
        return self

    def transform(
        self,
        X,
    ):

        output = []

        for document in X:

            _, title, body = parse_v3_document(
                document
            )

            output.append(
                f"TITLE: {title} BODY: {body}"
            )

        return output


# =============================================================================
# THEME × TEXT ANALYZER
# =============================================================================

def theme_interaction_analyzer(
    document: str,
):

    theme_key, title, body = parse_v3_document(
        document
    )

    text = (
        clean_text(title)
        + " "
        + clean_text(body)
    ).lower()

    tokens = TOKEN_PATTERN.findall(
        text
    )

    for token in tokens:
        yield f"{theme_key}__{token}"

    for i in range(len(tokens) - 1):
        yield (
            f"{theme_key}__"
            f"{tokens[i]}_{tokens[i + 1]}"
        )


# =============================================================================
# MODEL
# =============================================================================

def build_model() -> Pipeline:

    shared_word = Pipeline(
        [
            (
                "extract_shared_text",
                SharedTextExtractor(),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=WORD_MAX_FEATURES,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    shared_char = Pipeline(
        [
            (
                "extract_shared_text",
                SharedTextExtractor(),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=CHAR_MAX_FEATURES,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    interaction = TfidfVectorizer(
        analyzer=theme_interaction_analyzer,
        lowercase=False,
        min_df=2,
        max_df=0.95,
        max_features=INTERACTION_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
    )

    features = FeatureUnion(
        [
            (
                "shared_word",
                shared_word,
            ),
            (
                "shared_char",
                shared_char,
            ),
            (
                "theme_interaction",
                interaction,
            ),
        ]
    )

    classifier = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# =============================================================================
# FEATURE COUNTS
# =============================================================================

def get_feature_counts(
    model: Pipeline,
):

    features = model.named_steps[
        "features"
    ]

    transformers = dict(
        features.transformer_list
    )

    shared_word = transformers[
        "shared_word"
    ]

    shared_char = transformers[
        "shared_char"
    ]

    interaction = transformers[
        "theme_interaction"
    ]

    n_word = len(
        shared_word.named_steps[
            "tfidf"
        ].vocabulary_
    )

    n_char = len(
        shared_char.named_steps[
            "tfidf"
        ].vocabulary_
    )

    n_interaction = len(
        interaction.vocabulary_
    )

    return {
        "shared_word": int(n_word),
        "shared_char": int(n_char),
        "theme_interaction": int(
            n_interaction
        ),
        "total": int(
            n_word
            + n_char
            + n_interaction
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "SUPERVISED MODEL V3 — "
        "FULL DEVELOPMENT FIT / PRODUCTION CANDIDATE"
    )
    print("=" * 110)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nObservations loaded: "
        f"{len(df):,}"
    )

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df["label"] = (
        pd.to_numeric(
            df[
                "human_review_decision"
            ],
            errors="raise",
        )
        .astype(int)
    )

    invalid_labels = (
        set(df["label"].unique())
        - {0, 1}
    )

    if invalid_labels:
        raise RuntimeError(
            "Invalid labels: "
            f"{sorted(invalid_labels)}"
        )

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} observations, "
            f"found {len(df)}."
        )

    negatives = int(
        (df["label"] == 0).sum()
    )

    positives = int(
        (df["label"] == 1).sum()
    )

    print(
        f"Negatives: {negatives:,}"
    )
    print(
        f"Positives: {positives:,}"
    )

    if (
        negatives != EXPECTED_NEGATIVES
        or positives != EXPECTED_POSITIVES
    ):
        raise RuntimeError(
            "Label distribution does not match "
            "the frozen development set."
        )

    if df["article_id"].isna().any():
        raise RuntimeError(
            "Missing article IDs in development set."
        )

    unique_articles = df[
        "article_id"
    ].nunique()

    print(
        f"Unique article IDs: "
        f"{unique_articles:,}"
    )

    if unique_articles != EXPECTED_N:
        raise RuntimeError(
            "The development set does not contain "
            "485 unique article IDs."
        )

    actual_themes = sorted(
        df[
            "theme_name"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if actual_themes != sorted(
        EXPECTED_THEMES
    ):
        raise RuntimeError(
            "Theme list does not match "
            "the expected development specification."
        )

    print(
        "\nTheme distribution:"
    )

    theme_table = (
        df.groupby(
            "theme_name"
        )["label"]
        .agg(
            n="size",
            positives="sum",
        )
    )

    theme_table[
        "negatives"
    ] = (
        theme_table["n"]
        - theme_table["positives"]
    )

    print(
        theme_table.to_string()
    )

    print(
        "\nBuilding V3 documents..."
    )

    df["v3_document"] = df.apply(
        build_v3_document_from_row,
        axis=1,
    )

    print("[OK]")

    print(
        "\nBuilding model pipeline..."
    )

    model = build_model()

    print("[OK]")

    print(
        "\nFitting V3 on all "
        "485 development observations..."
    )

    model.fit(
        df["v3_document"],
        df["label"],
    )

    print(
        "[OK] Training complete"
    )

    feature_counts = get_feature_counts(
        model
    )

    print(
        "\nFeature counts:"
    )
    print(
        f"  shared word       = "
        f"{feature_counts['shared_word']:,}"
    )
    print(
        f"  shared char       = "
        f"{feature_counts['shared_char']:,}"
    )
    print(
        f"  theme interaction = "
        f"{feature_counts['theme_interaction']:,}"
    )
    print(
        f"  TOTAL             = "
        f"{feature_counts['total']:,}"
    )

    # These predictions are used only to verify the fitted model output.
    # They are not valid out-of-sample performance estimates.
    print()
    print("=" * 110)
    print(
        "IN-SAMPLE TECHNICAL SANITY CHECK"
    )
    print("=" * 110)

    probabilities = model.predict_proba(
        df["v3_document"]
    )[:, 1]

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Model produced non-finite probabilities."
        )

    if (
        probabilities.min() < 0.0
        or probabilities.max() > 1.0
    ):
        raise RuntimeError(
            "Model produced probabilities "
            "outside [0, 1]."
        )

    print(
        f"\nProbability min:    "
        f"{probabilities.min():.6f}"
    )
    print(
        f"Probability mean:   "
        f"{probabilities.mean():.6f}"
    )
    print(
        f"Probability median: "
        f"{np.median(probabilities):.6f}"
    )
    print(
        f"Probability max:    "
        f"{probabilities.max():.6f}"
    )

    predicted_positive = int(
        (
            probabilities
            >= SELECTED_THRESHOLD
        ).sum()
    )

    print(
        f"\nPredicted positive "
        f"@ {SELECTED_THRESHOLD:.2f}: "
        f"{predicted_positive:,} / "
        f"{len(df):,}"
    )

    print(
        "\nNOTE: These are not performance metrics. "
        "The model was fitted on the same "
        "485 observations."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving model..."
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"[OK]\n{MODEL_FILE}"
    )

    metadata = {
        "model_name":
            "Supervised Model V3",

        "model_version":
            "supervised_v3_candidate",

        "architecture":
            (
                "Hybrid Shared + "
                "Theme Text Interactions"
            ),

        "status":
            "FITTED_PRODUCTION_SANITY_CANDIDATE",

        "trained_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "development_data": {
            "n":
                int(len(df)),

            "unique_articles":
                int(unique_articles),

            "positives":
                positives,

            "negatives":
                negatives,

            "role":
                (
                    "development / "
                    "model selection / "
                    "threshold selection"
                ),

            "not_final_test":
                True,
        },

        "features": {
            "shared_word_tfidf": {
                "ngram_range":
                    [1, 2],

                "min_df":
                    2,

                "max_df":
                    0.95,

                "max_features":
                    WORD_MAX_FEATURES,

                "stop_words":
                    "english",

                "sublinear_tf":
                    True,
            },

            "shared_char_tfidf": {
                "analyzer":
                    "char_wb",

                "ngram_range":
                    [3, 5],

                "min_df":
                    2,

                "max_features":
                    CHAR_MAX_FEATURES,

                "sublinear_tf":
                    True,
            },

            "theme_interaction_tfidf": {
                "representation":
                    (
                        "theme-conditioned "
                        "word unigrams and bigrams"
                    ),

                "min_df":
                    2,

                "max_df":
                    0.95,

                "max_features":
                    INTERACTION_MAX_FEATURES,

                "sublinear_tf":
                    True,
            },

            "fitted_feature_counts":
                feature_counts,
        },

        "classifier": {
            "type":
                "LogisticRegression",

            "C":
                C_VALUE,

            "penalty":
                "l2",

            "solver":
                "liblinear",

            "class_weight":
                "balanced",

            "max_iter":
                2000,

            "random_state":
                RANDOM_STATE,
        },

        "threshold": {
            "type":
                "single_global_threshold",

            "value":
                SELECTED_THRESHOLD,

            "selection_data":
                "development OOF",

            "selection_objective":
                "maximum global F1",

            "tie_break_1":
                "maximum balanced accuracy",

            "tie_break_2":
                "closest to 0.50",
        },

        "calibration":
            CALIBRATION,

        "development_oof": {
            "roc_auc":
                0.815162,

            "average_precision":
                0.659727,

            "threshold":
                SELECTED_THRESHOLD,

            "f1":
                0.629213,

            "balanced_accuracy":
                0.735945,

            "precision":
                0.538462,

            "recall":
                0.756757,

            "specificity":
                0.715134,

            "tn":
                241,

            "fp":
                96,

            "fn":
                36,

            "tp":
                112,
        },

        "production_status": {
            "full_corpus_scoring":
                False,

            "sanity_check_required":
                True,

            "sanity_check_n_articles":
                1000,

            "note":
                (
                    "Candidate must pass cross-theme "
                    "production sanity diagnostics "
                    "before final freeze."
                ),
        },

        "methodological_note":
            (
                "Model fitted on all 485 development observations "
                "only after architecture and global threshold were "
                "selected using frozen out-of-fold development "
                "predictions. In-sample predictions generated during "
                "this script are technical sanity checks only and "
                "must not be reported as performance estimates."
            ),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write("\n")

    print(
        "\nMetadata saved:"
    )
    print(
        METADATA_FILE
    )

    print()
    print("=" * 110)
    print("COMPLETE")
    print("=" * 110)

    print(
        f"""
Model:
  {MODEL_FILE}

Metadata:
  {METADATA_FILE}

Status:
  FITTED_PRODUCTION_SANITY_CANDIDATE

Selected threshold:
  {SELECTED_THRESHOLD:.2f}

Do not run full-corpus scoring yet.

Next:
  Score only the same 1,000 production articles used for the V1 diagnostic.
"""
    )


if __name__ == "__main__":
    main()