#!/usr/bin/env python3

"""
Supervised Model Tuning — Frozen Cross-Validation Search

Compare a predefined set of TF-IDF feature representations and logistic
regression regularization strengths using the fixed development-set folds.

For each candidate model:
    1. fit the feature representation on the training folds only;
    2. fit logistic regression on the resulting training features;
    3. generate out-of-fold probabilities for every development observation;
    4. compute global, fold-level, and theme-level ranking metrics.

Model comparison is based on a predefined composite selection score combining:
    - global ROC-AUC;
    - global Average Precision;
    - macro theme ROC-AUC;
    - macro theme Average Precision.

No threshold-dependent metric is used for this model-ranking stage.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "SUPERVISED_CV_DATASET",
        "supervised_cv_dataset.csv",
    )
)

OUTPUT_RESULTS = Path(
    os.getenv(
        "SUPERVISED_MODEL_TUNING_RESULTS",
        "supervised_model_tuning.csv",
    )
)

OUTPUT_OOF = Path(
    os.getenv(
        "SUPERVISED_MODEL_TUNING_OOF",
        "supervised_model_tuning_oof.csv",
    )
)


# =============================================================================
# DATA SPECIFICATION
# =============================================================================

TRUE_COL = "human_review_decision"
TEXT_COL = "model_text"
FOLD_COL = "cv_fold"
THEME_COL = "theme_name"
ID_COL = "annotation_id"

EXPECTED_N = 485
EXPECTED_FOLDS = {1, 2, 3, 4, 5}

REQUIRED_COLUMNS = {
    ID_COL,
    TRUE_COL,
    TEXT_COL,
    FOLD_COL,
    THEME_COL,
}


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

RANDOM_STATE = 42

C_VALUES = [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]


# =============================================================================
# FEATURE BUILDERS
# =============================================================================

def word_unigram():

    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
    )


def word_bigram():

    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
    )


def word_bigram_min3():

    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
    )


def word_plus_char():

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=25_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=25_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


FEATURE_CONFIGS = {
    "word_1_1": word_unigram,
    "word_1_2": word_bigram,
    "word_1_2_min3": word_bigram_min3,
    "word_char": word_plus_char,
}


# =============================================================================
# HELPERS
# =============================================================================

def safe_metric(
    metric_function,
    y_true,
    y_score,
):

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        metric_function(
            y_true,
            y_score,
        )
    )


def safe_theme_name(
    theme,
):

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(theme).lower(),
    )

    return value.strip("_")


# =============================================================================
# LOAD AND VALIDATE DATA
# =============================================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
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

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} development "
            f"observations, found {len(df)}."
        )

    df[TRUE_COL] = pd.to_numeric(
        df[TRUE_COL],
        errors="raise",
    ).astype(int)

    df[FOLD_COL] = pd.to_numeric(
        df[FOLD_COL],
        errors="raise",
    ).astype(int)

    invalid_labels = (
        set(df[TRUE_COL].unique())
        - {0, 1}
    )

    if invalid_labels:
        raise RuntimeError(
            "Invalid labels: "
            f"{sorted(invalid_labels)}"
        )

    actual_folds = set(
        df[FOLD_COL].unique()
    )

    if actual_folds != EXPECTED_FOLDS:
        raise RuntimeError(
            "Unexpected CV folds. "
            f"Expected {sorted(EXPECTED_FOLDS)}, "
            f"found {sorted(actual_folds)}."
        )

    if df[ID_COL].isna().any():
        raise RuntimeError(
            "Missing annotation IDs."
        )

    if df[ID_COL].duplicated().any():
        raise RuntimeError(
            "Duplicate annotation IDs."
        )

    if df[THEME_COL].isna().any():
        raise RuntimeError(
            "Missing theme names."
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
    )

    if (
        df[TEXT_COL]
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Empty model_text observations detected."
        )

    # Use a positional index so OOF predictions can be mapped
    # deterministically back to the original observations.
    df = df.reset_index(
        drop=True
    )

    return df


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    df,
    feature_name,
    feature_builder,
    c_value,
):

    model_name = (
        f"{feature_name}__C_{c_value}"
    )

    print()
    print("=" * 120)
    print(model_name)
    print("=" * 120)

    oof_probability = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    fold_auc = []
    fold_ap = []

    for fold in sorted(
        EXPECTED_FOLDS
    ):

        train_mask = (
            df[FOLD_COL] != fold
        )

        validation_mask = (
            df[FOLD_COL] == fold
        )

        X_train = df.loc[
            train_mask,
            TEXT_COL,
        ]

        y_train = df.loc[
            train_mask,
            TRUE_COL,
        ]

        X_validation = df.loc[
            validation_mask,
            TEXT_COL,
        ]

        y_validation = df.loc[
            validation_mask,
            TRUE_COL,
        ]

        if y_train.nunique() < 2:
            raise RuntimeError(
                f"{model_name}, fold {fold}: "
                "training set contains only one class."
            )

        features = feature_builder()

        # Feature extraction is fitted exclusively on the training fold.
        X_train_matrix = (
            features.fit_transform(
                X_train
            )
        )

        X_validation_matrix = (
            features.transform(
                X_validation
            )
        )

        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

        model.fit(
            X_train_matrix,
            y_train,
        )

        probability = (
            model.predict_proba(
                X_validation_matrix
            )[:, 1]
        )

        if not np.isfinite(
            probability
        ).all():
            raise RuntimeError(
                f"{model_name}, fold {fold}: "
                "non-finite probabilities."
            )

        validation_positions = np.flatnonzero(
            validation_mask.to_numpy()
        )

        oof_probability[
            validation_positions
        ] = probability

        fold_auc.append(
            safe_metric(
                roc_auc_score,
                y_validation,
                probability,
            )
        )

        fold_ap.append(
            safe_metric(
                average_precision_score,
                y_validation,
                probability,
            )
        )

    if np.isnan(
        oof_probability
    ).any():
        raise RuntimeError(
            f"Missing OOF predictions "
            f"for {model_name}."
        )

    global_auc = safe_metric(
        roc_auc_score,
        df[TRUE_COL],
        oof_probability,
    )

    global_ap = safe_metric(
        average_precision_score,
        df[TRUE_COL],
        oof_probability,
    )

    theme_auc_values = []
    theme_ap_values = []
    theme_metrics = {}

    for theme, group in df.groupby(
        THEME_COL,
        sort=True,
    ):

        positions = group.index.to_numpy()

        probabilities_theme = (
            oof_probability[
                positions
            ]
        )

        auc_theme = safe_metric(
            roc_auc_score,
            group[TRUE_COL],
            probabilities_theme,
        )

        ap_theme = safe_metric(
            average_precision_score,
            group[TRUE_COL],
            probabilities_theme,
        )

        theme_auc_values.append(
            auc_theme
        )

        theme_ap_values.append(
            ap_theme
        )

        theme_key = safe_theme_name(
            theme
        )

        theme_metrics[
            f"auc_{theme_key}"
        ] = auc_theme

        theme_metrics[
            f"ap_{theme_key}"
        ] = ap_theme

    macro_theme_auc = float(
        np.nanmean(
            theme_auc_values
        )
    )

    macro_theme_ap = float(
        np.nanmean(
            theme_ap_values
        )
    )

    result = {
        "model":
            model_name,

        "feature_config":
            feature_name,

        "C":
            c_value,

        "global_roc_auc":
            global_auc,

        "global_average_precision":
            global_ap,

        "mean_fold_roc_auc":
            float(
                np.nanmean(
                    fold_auc
                )
            ),

        "std_fold_roc_auc":
            float(
                np.nanstd(
                    fold_auc
                )
            ),

        "mean_fold_average_precision":
            float(
                np.nanmean(
                    fold_ap
                )
            ),

        "std_fold_average_precision":
            float(
                np.nanstd(
                    fold_ap
                )
            ),

        "macro_theme_roc_auc":
            macro_theme_auc,

        "macro_theme_average_precision":
            macro_theme_ap,
    }

    result.update(
        theme_metrics
    )

    oof = pd.DataFrame(
        {
            "annotation_id":
                df[ID_COL].to_numpy(),

            "theme_name":
                df[THEME_COL].to_numpy(),

            "human_review_decision":
                df[TRUE_COL].to_numpy(),

            "cv_fold":
                df[FOLD_COL].to_numpy(),

            "model":
                model_name,

            "oof_probability":
                oof_probability,
        }
    )

    print(
        f"Global ROC-AUC:       "
        f"{global_auc:.4f}"
    )

    print(
        f"Global AP:            "
        f"{global_ap:.4f}"
    )

    print(
        f"Macro-theme ROC-AUC: "
        f"{macro_theme_auc:.4f}"
    )

    print(
        f"Macro-theme AP:      "
        f"{macro_theme_ap:.4f}"
    )

    return result, oof


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "SUPERVISED MODEL TUNING"
    )
    print("=" * 120)

    df = load_dataset()

    print(
        f"\nDevelopment observations: "
        f"{len(df):,}"
    )

    print(
        f"Positive observations:    "
        f"{int(df[TRUE_COL].sum()):,}"
    )

    print(
        f"Negative observations:    "
        f"{int((df[TRUE_COL] == 0).sum()):,}"
    )

    print(
        f"CV folds:                 "
        f"{sorted(EXPECTED_FOLDS)}"
    )

    result_rows = []
    oof_frames = []

    for (
        feature_name,
        feature_builder,
    ) in FEATURE_CONFIGS.items():

        for c_value in C_VALUES:

            result, oof = evaluate_model(
                df=df,
                feature_name=feature_name,
                feature_builder=feature_builder,
                c_value=c_value,
            )

            result_rows.append(
                result
            )

            oof_frames.append(
                oof
            )

    results = pd.DataFrame(
        result_rows
    )

    # The selection score was predefined for model comparison and does
    # not use threshold-dependent classification metrics.
    results["selection_score"] = (
        0.35
        * results[
            "global_roc_auc"
        ]
        + 0.35
        * results[
            "global_average_precision"
        ]
        + 0.15
        * results[
            "macro_theme_roc_auc"
        ]
        + 0.15
        * results[
            "macro_theme_average_precision"
        ]
    )

    results = (
        results
        .sort_values(
            [
                "selection_score",
                "global_average_precision",
                "global_roc_auc",
                "model",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 120)
    print(
        "MODEL RANKING"
    )
    print("=" * 120)

    display_columns = [
        "model",
        "global_roc_auc",
        "global_average_precision",
        "macro_theme_roc_auc",
        "macro_theme_average_precision",
        "mean_fold_roc_auc",
        "std_fold_roc_auc",
        "selection_score",
    ]

    print(
        results[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    best = results.iloc[0]

    print()
    print("=" * 120)
    print(
        "TOP MODEL"
    )
    print("=" * 120)

    for column in display_columns:
        print(
            f"{column}: "
            f"{best[column]}"
        )

    oof_results = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    expected_oof_rows = (
        len(df)
        * len(FEATURE_CONFIGS)
        * len(C_VALUES)
    )

    if len(oof_results) != expected_oof_rows:
        raise RuntimeError(
            "Unexpected number of stored "
            "OOF predictions."
        )

    OUTPUT_RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_OOF.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    oof_results.to_csv(
        OUTPUT_OOF,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "FILES SAVED"
    )
    print("=" * 120)

    print(
        OUTPUT_RESULTS
    )

    print(
        OUTPUT_OOF
    )

    print(
        "\nModel tuning complete."
    )


if __name__ == "__main__":
    main()