#!/usr/bin/env python3

"""
Supervised Model — Development Threshold Selection

Select the global classification threshold for the previously selected
supervised model using its frozen out-of-fold development predictions.

Selected model:
    word_char__C_4.0

Threshold selection:
    1. maximize global F1;
    2. break ties using global balanced accuracy;
    3. break remaining ties using the threshold closest to 0.50.

The threshold grid ranges from 0.20 to 0.80 in increments of 0.01.

Theme-level metrics are reported as diagnostics but are not used as the
primary threshold-selection criterion.

All metrics in this script are based on out-of-fold development predictions.
No final-test observations are used.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "SUPERVISED_MODEL_TUNING_OOF",
        "supervised_model_tuning_oof.csv",
    )
)

OUTPUT_THRESHOLD_RESULTS = Path(
    os.getenv(
        "SUPERVISED_THRESHOLD_RESULTS",
        "supervised_threshold_tuning.csv",
    )
)

OUTPUT_FINAL_OOF = Path(
    os.getenv(
        "SUPERVISED_SELECTED_MODEL_OOF",
        "supervised_selected_model_oof.csv",
    )
)


# =============================================================================
# DEVELOPMENT SPECIFICATION
# =============================================================================

SELECTED_MODEL = "word_char__C_4.0"

TRUE_COL = "human_review_decision"
THEME_COL = "theme_name"
PROB_COL = "oof_probability"

EXPECTED_N = 485
EXPECTED_NEGATIVES = 337
EXPECTED_POSITIVES = 148

THRESHOLDS = np.arange(
    0.20,
    0.801,
    0.01,
)

REQUIRED_COLUMNS = {
    "annotation_id",
    "model",
    TRUE_COL,
    THEME_COL,
    PROB_COL,
    "cv_fold",
}


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    y_true,
    y_pred,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }


# =============================================================================
# LOAD AND VALIDATE
# =============================================================================

def load_selected_model_oof():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"OOF predictions not found:\n"
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

    df = df[
        df["model"] == SELECTED_MODEL
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No OOF predictions found for "
            f"model '{SELECTED_MODEL}'."
        )

    df[TRUE_COL] = pd.to_numeric(
        df[TRUE_COL],
        errors="raise",
    ).astype(int)

    df[PROB_COL] = pd.to_numeric(
        df[PROB_COL],
        errors="raise",
    )

    invalid_labels = (
        set(df[TRUE_COL].unique())
        - {0, 1}
    )

    if invalid_labels:
        raise RuntimeError(
            "Invalid labels: "
            f"{sorted(invalid_labels)}"
        )

    if df[PROB_COL].isna().any():
        raise RuntimeError(
            "Missing OOF probabilities."
        )

    if not np.isfinite(
        df[PROB_COL].to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite OOF probabilities."
        )

    if (
        (df[PROB_COL] < 0.0)
        | (df[PROB_COL] > 1.0)
    ).any():
        raise RuntimeError(
            "OOF probabilities outside [0,1]."
        )

    if df[THEME_COL].isna().any():
        raise RuntimeError(
            "Missing theme names."
        )

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} OOF observations "
            f"for the selected model, found {len(df)}."
        )

    if df["annotation_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate annotation IDs in selected-model "
            "OOF predictions."
        )

    positives = int(
        (df[TRUE_COL] == 1).sum()
    )

    negatives = int(
        (df[TRUE_COL] == 0).sum()
    )

    if (
        positives != EXPECTED_POSITIVES
        or negatives != EXPECTED_NEGATIVES
    ):
        raise RuntimeError(
            "Label distribution does not match the "
            "frozen development set."
        )

    return df.reset_index(
        drop=True
    )


# =============================================================================
# THRESHOLD EVALUATION
# =============================================================================

def evaluate_thresholds(
    df,
):

    rows = []

    y_true = df[
        TRUE_COL
    ].to_numpy()

    probabilities = df[
        PROB_COL
    ].to_numpy()

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        row = {
            "threshold": round(
                float(threshold),
                2,
            ),
        }

        row.update(
            compute_metrics(
                y_true,
                predictions,
            )
        )

        theme_f1 = []
        theme_balanced_accuracy = []

        for _, group in df.groupby(
            THEME_COL,
            sort=True,
        ):

            theme_true = group[
                TRUE_COL
            ].to_numpy()

            theme_prediction = (
                group[PROB_COL].to_numpy()
                >= threshold
            ).astype(int)

            theme_f1.append(
                f1_score(
                    theme_true,
                    theme_prediction,
                    zero_division=0,
                )
            )

            theme_balanced_accuracy.append(
                balanced_accuracy_score(
                    theme_true,
                    theme_prediction,
                )
            )

        row["macro_theme_f1"] = float(
            np.mean(
                theme_f1
            )
        )

        row[
            "macro_theme_balanced_accuracy"
        ] = float(
            np.mean(
                theme_balanced_accuracy
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "SUPERVISED MODEL — DEVELOPMENT THRESHOLD SELECTION"
    )
    print("=" * 120)

    df = load_selected_model_oof()

    print(
        f"\nSelected model: "
        f"{SELECTED_MODEL}"
    )

    print(
        f"OOF observations: "
        f"{len(df):,}"
    )

    print(
        f"Positive: "
        f"{int((df[TRUE_COL] == 1).sum()):,}"
    )

    print(
        f"Negative: "
        f"{int((df[TRUE_COL] == 0).sum()):,}"
    )

    results = evaluate_thresholds(
        df
    )

    # Threshold selection is based on global development OOF performance.
    # Theme-level metrics are retained only as robustness diagnostics.
    results["distance_from_050"] = (
        results["threshold"]
        - 0.50
    ).abs()

    ranked = (
        results
        .sort_values(
            by=[
                "f1",
                "balanced_accuracy",
                "distance_from_050",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    best = ranked.iloc[0]

    best_threshold = float(
        best["threshold"]
    )

    display_columns = [
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
        "macro_theme_f1",
        "macro_theme_balanced_accuracy",
    ]

    print()
    print("=" * 120)
    print(
        "TOP 15 THRESHOLDS BY GLOBAL F1"
    )
    print("=" * 120)

    print(
        ranked[
            display_columns
        ]
        .head(15)
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print(
        "SELECTED THRESHOLD"
    )
    print("=" * 120)

    for column in display_columns:

        value = best[column]

        if column in {
            "TN",
            "FP",
            "FN",
            "TP",
        }:
            print(
                f"{column:32s}: "
                f"{int(value)}"
            )

        else:
            print(
                f"{column:32s}: "
                f"{value:.6f}"
            )

    default_rows = results[
        np.isclose(
            results["threshold"],
            0.50,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    if len(default_rows) != 1:
        raise RuntimeError(
            "Could not identify exactly one "
            "0.50 threshold row."
        )

    default_row = (
        default_rows.iloc[0]
    )

    print()
    print("=" * 120)
    print(
        "COMPARISON: DEFAULT 0.50 VS SELECTED THRESHOLD"
    )
    print("=" * 120)

    comparison = pd.DataFrame(
        [
            {
                "scenario": "default_0.50",
                **{
                    column: default_row[column]
                    for column in display_columns
                },
            },
            {
                "scenario": "selected",
                **{
                    column: best[column]
                    for column in display_columns
                },
            },
        ]
    )

    print(
        comparison
        .round(4)
        .to_string(
            index=False
        )
    )

    # Inspect the local performance surface around the selected threshold.
    lower = best_threshold - 0.05
    upper = best_threshold + 0.05

    local = results[
        (results["threshold"] >= lower)
        & (results["threshold"] <= upper)
    ].copy()

    print()
    print("=" * 120)
    print(
        "LOCAL THRESHOLD STABILITY ±0.05"
    )
    print("=" * 120)

    print(
        local[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print(
        "PER-THEME METRICS AT SELECTED THRESHOLD"
    )
    print("=" * 120)

    theme_rows = []

    for theme, group in df.groupby(
        THEME_COL,
        sort=True,
    ):

        theme_true = group[
            TRUE_COL
        ].to_numpy()

        theme_prediction = (
            group[PROB_COL].to_numpy()
            >= best_threshold
        ).astype(int)

        metrics = compute_metrics(
            theme_true,
            theme_prediction,
        )

        row = {
            "theme_name": theme,
            "n": len(group),
            "positives": int(
                theme_true.sum()
            ),
        }

        row.update(
            metrics
        )

        theme_rows.append(
            row
        )

    theme_results = pd.DataFrame(
        theme_rows
    )

    print(
        theme_results
        .round(4)
        .to_string(
            index=False
        )
    )

    df[
        "selected_threshold"
    ] = best_threshold

    df[
        "oof_prediction_selected"
    ] = (
        df[PROB_COL]
        >= best_threshold
    ).astype(int)

    OUTPUT_THRESHOLD_RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FINAL_OOF.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_THRESHOLD_RESULTS,
        index=False,
    )

    df.to_csv(
        OUTPUT_FINAL_OOF,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "FILES SAVED"
    )
    print("=" * 120)

    print(
        OUTPUT_THRESHOLD_RESULTS
    )

    print(
        OUTPUT_FINAL_OOF
    )

    print(
        "\nThreshold selection complete."
    )


if __name__ == "__main__":
    main()