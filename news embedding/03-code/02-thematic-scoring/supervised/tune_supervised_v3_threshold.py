#!/usr/bin/env python3

"""
SUPERVISED V3 — GLOBAL THRESHOLD SELECTION
==========================================

Select one global classification threshold using only the out-of-fold
predictions from the 485-observation development set.

Input:
    supervised_v3_oof_predictions.csv

Selection rule:
    1. maximize global F1;
    2. break ties by maximizing balanced accuracy;
    3. break remaining ties using the threshold closest to 0.50;
    4. use the lower threshold as a deterministic final tie-break.

Threshold grid:
    0.20 to 0.80 inclusive, step 0.01.

No model fitting is performed in this script. No final-test or production
observations are used. Theme-level metrics are diagnostic only and do not
affect threshold selection.

All reported performance metrics remain development estimates.
"""

from __future__ import annotations

import json
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
        "SUPERVISED_V3_OOF_FILE",
        "supervised_v3_oof_predictions.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "SUPERVISED_V3_THRESHOLD_DIR",
        "supervised_v3_threshold",
    )
)

GRID_FILE = (
    OUTPUT_DIR
    / "supervised_v3_threshold_grid.csv"
)

SELECTED_FILE = (
    OUTPUT_DIR
    / "supervised_v3_selected_threshold.json"
)

THEME_METRICS_FILE = (
    OUTPUT_DIR
    / "supervised_v3_selected_threshold_by_theme.csv"
)

ERRORS_FILE = (
    OUTPUT_DIR
    / "supervised_v3_oof_errors_at_selected_threshold.csv"
)


# =============================================================================
# DEVELOPMENT SPECIFICATION
# =============================================================================

EXPECTED_N = 485
EXPECTED_NEGATIVES = 337
EXPECTED_POSITIVES = 148

PROBABILITY_COLUMN = "supervised_v3_oof_probability"
LABEL_COLUMN = "label"
THEME_COLUMN = "theme_name"

THRESHOLDS = np.round(
    np.arange(
        0.20,
        0.801,
        0.01,
    ),
    2,
)

REQUIRED_COLUMNS = {
    LABEL_COLUMN,
    PROBABILITY_COLUMN,
    THEME_COLUMN,
    "article_id",
    "cv_fold",
}


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    probability,
    threshold,
):

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(
            prediction.sum()
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),
    }


# =============================================================================
# LOAD AND VALIDATE
# =============================================================================

def load_oof_predictions():

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

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} observations, "
            f"found {len(df)}."
        )

    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="raise",
    ).astype(int)

    df[PROBABILITY_COLUMN] = pd.to_numeric(
        df[PROBABILITY_COLUMN],
        errors="raise",
    )

    invalid_labels = (
        set(df[LABEL_COLUMN].unique())
        - {0, 1}
    )

    if invalid_labels:
        raise RuntimeError(
            "Invalid labels: "
            f"{sorted(invalid_labels)}"
        )

    probability = df[
        PROBABILITY_COLUMN
    ].to_numpy()

    if not np.isfinite(
        probability
    ).all():
        raise RuntimeError(
            "Missing or non-finite OOF probabilities."
        )

    if (
        (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise RuntimeError(
            "OOF probabilities outside [0,1]."
        )

    if df[THEME_COLUMN].isna().any():
        raise RuntimeError(
            "Missing theme names."
        )

    if df["article_id"].isna().any():
        raise RuntimeError(
            "Missing article IDs."
        )

    if df["cv_fold"].isna().any():
        raise RuntimeError(
            "Missing CV fold assignments."
        )

    negatives = int(
        (df[LABEL_COLUMN] == 0).sum()
    )

    positives = int(
        (df[LABEL_COLUMN] == 1).sum()
    )

    if (
        negatives != EXPECTED_NEGATIVES
        or positives != EXPECTED_POSITIVES
    ):
        raise RuntimeError(
            "Label distribution does not match "
            "the frozen development set."
        )

    return (
        df,
        df[LABEL_COLUMN].to_numpy(),
        probability,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "SUPERVISED V3 — "
        "GLOBAL OOF THRESHOLD SELECTION"
    )
    print("=" * 100)

    df, y_true, probability = (
        load_oof_predictions()
    )

    negatives = int(
        (y_true == 0).sum()
    )

    positives = int(
        (y_true == 1).sum()
    )

    print(
        f"\nObservations: "
        f"{len(df):,}"
    )

    print(
        f"Negative:     "
        f"{negatives:,}"
    )

    print(
        f"Positive:     "
        f"{positives:,}"
    )

    print(
        f"\nProbability min:    "
        f"{probability.min():.6f}"
    )

    print(
        f"Probability mean:   "
        f"{probability.mean():.6f}"
    )

    print(
        f"Probability median: "
        f"{np.median(probability):.6f}"
    )

    print(
        f"Probability max:    "
        f"{probability.max():.6f}"
    )

    print()
    print("=" * 100)
    print("THRESHOLD GRID")
    print("=" * 100)

    results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_true,
            probability,
            threshold,
        )

        metrics["distance_from_050"] = abs(
            float(threshold) - 0.50
        )

        results.append(
            metrics
        )

    grid_df = pd.DataFrame(
        results
    )

    # Threshold selection is based exclusively on global OOF development
    # performance. The final threshold tie-break ensures deterministic output.
    ranked = (
        grid_df
        .sort_values(
            by=[
                "f1",
                "balanced_accuracy",
                "distance_from_050",
                "threshold",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    selected = ranked.iloc[0]

    selected_threshold = float(
        selected["threshold"]
    )

    print(
        f"\nSELECTED THRESHOLD: "
        f"{selected_threshold:.2f}"
    )

    print(
        "\nSelection rule:"
    )

    print(
        "  1. maximum global F1"
    )

    print(
        "  2. maximum balanced accuracy"
    )

    print(
        "  3. closest to 0.50"
    )

    print(
        "  4. lower threshold"
    )

    print()
    print("=" * 100)
    print("TOP 10 THRESHOLDS")
    print("=" * 100)

    columns_to_show = [
        "threshold",
        "f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    print(
        ranked[
            columns_to_show
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    selected_metrics = calculate_metrics(
        y_true,
        probability,
        selected_threshold,
    )

    print()
    print("=" * 100)
    print(
        f"GLOBAL METRICS @ "
        f"{selected_threshold:.2f}"
    )
    print("=" * 100)

    for key in [
        "tn",
        "fp",
        "fn",
        "tp",
        "predicted_positive",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
    ]:

        value = selected_metrics[
            key
        ]

        if isinstance(
            value,
            (float, np.floating),
        ):
            print(
                f"{key:22s}: "
                f"{value:.6f}"
            )

        else:
            print(
                f"{key:22s}: "
                f"{value}"
            )

    reference_metrics = calculate_metrics(
        y_true,
        probability,
        0.50,
    )

    print()
    print("=" * 100)
    print("COMPARISON WITH 0.50")
    print("=" * 100)

    print(
        f"\nF1:                 "
        f"{reference_metrics['f1']:.6f}"
        f" -> "
        f"{selected_metrics['f1']:.6f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{reference_metrics['balanced_accuracy']:.6f}"
        f" -> "
        f"{selected_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Precision:           "
        f"{reference_metrics['precision']:.6f}"
        f" -> "
        f"{selected_metrics['precision']:.6f}"
    )

    print(
        f"Recall:              "
        f"{reference_metrics['recall']:.6f}"
        f" -> "
        f"{selected_metrics['recall']:.6f}"
    )

    # Theme-level metrics diagnose cross-theme behavior but do not influence
    # the global threshold selected above.
    print()
    print("=" * 100)
    print(
        "METRICS BY THEME @ "
        f"{selected_threshold:.2f}"
    )
    print("=" * 100)

    theme_results = []

    for theme_name, group in df.groupby(
        THEME_COLUMN,
        sort=True,
    ):

        y_theme = group[
            LABEL_COLUMN
        ].to_numpy()

        p_theme = group[
            PROBABILITY_COLUMN
        ].to_numpy()

        metrics = calculate_metrics(
            y_theme,
            p_theme,
            selected_threshold,
        )

        metrics["theme_name"] = (
            theme_name
        )

        metrics["n"] = int(
            len(group)
        )

        metrics["positives"] = int(
            y_theme.sum()
        )

        theme_results.append(
            metrics
        )

        print(
            f"\n{theme_name}"
        )

        print(
            f"  n / pos     = "
            f"{len(group)} / "
            f"{int(y_theme.sum())}"
        )

        print(
            f"  TN FP FN TP = "
            f"{metrics['tn']} "
            f"{metrics['fp']} "
            f"{metrics['fn']} "
            f"{metrics['tp']}"
        )

        print(
            f"  Precision   = "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall      = "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1          = "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"  BA          = "
            f"{metrics['balanced_accuracy']:.4f}"
        )

    df[
        "supervised_v3_selected_prediction"
    ] = (
        probability >= selected_threshold
    ).astype(int)

    df[
        "supervised_v3_error_type"
    ] = "CORRECT"

    fp_mask = (
        (df[LABEL_COLUMN] == 0)
        & (
            df[
                "supervised_v3_selected_prediction"
            ] == 1
        )
    )

    fn_mask = (
        (df[LABEL_COLUMN] == 1)
        & (
            df[
                "supervised_v3_selected_prediction"
            ] == 0
        )
    )

    df.loc[
        fp_mask,
        "supervised_v3_error_type",
    ] = "FP"

    df.loc[
        fn_mask,
        "supervised_v3_error_type",
    ] = "FN"

    errors_df = df[
        df[
            "supervised_v3_error_type"
        ].isin(
            ["FP", "FN"]
        )
    ].copy()

    # Distance from the selected decision boundary helps distinguish
    # borderline errors from high-confidence errors.
    errors_df[
        "distance_from_threshold"
    ] = (
        errors_df[
            PROBABILITY_COLUMN
        ]
        - selected_threshold
    ).abs()

    errors_df = (
        errors_df
        .sort_values(
            by=[
                "supervised_v3_error_type",
                "distance_from_threshold",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_df.sort_values(
        "threshold"
    ).to_csv(
        GRID_FILE,
        index=False,
    )

    pd.DataFrame(
        theme_results
    ).to_csv(
        THEME_METRICS_FILE,
        index=False,
    )

    errors_df.to_csv(
        ERRORS_FILE,
        index=False,
    )

    selected_json = {
        "model": (
            "Supervised Model V3 - "
            "Hybrid Shared + Theme Interactions"
        ),
        "status": (
            "THRESHOLD_SELECTED_DEVELOPMENT"
        ),
        "development_n": int(
            len(df)
        ),
        "threshold_selection": {
            "scope": "global",
            "grid_start": 0.20,
            "grid_end": 0.80,
            "grid_step": 0.01,
            "primary_objective": (
                "maximize_global_f1"
            ),
            "tie_break_1": (
                "maximize_balanced_accuracy"
            ),
            "tie_break_2": (
                "closest_to_0.50"
            ),
            "tie_break_3": (
                "lower_threshold"
            ),
            "selected_threshold": (
                selected_threshold
            ),
        },
        "selected_threshold_metrics": {
            key: (
                int(value)
                if isinstance(
                    value,
                    (int, np.integer),
                )
                else float(value)
            )
            for key, value
            in selected_metrics.items()
        },
        "reference_threshold_050": {
            key: (
                int(value)
                if isinstance(
                    value,
                    (int, np.integer),
                )
                else float(value)
            )
            for key, value
            in reference_metrics.items()
        },
        "ranking_metrics_from_v3_oof": {
            "roc_auc": 0.815162,
            "average_precision": 0.659727,
        },
        "calibration": (
            "none_at_this_stage"
        ),
        "methodological_note": (
            "The threshold was selected exclusively from "
            "out-of-fold predictions on the 485-observation "
            "development dataset. These metrics are development "
            "estimates and must not be interpreted as unbiased "
            "final-test performance. A single global threshold "
            "is used; no theme-specific thresholds are selected."
        ),
    }

    with open(
        SELECTED_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selected_json,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print(
        f"\nThreshold grid:\n"
        f"{GRID_FILE}"
    )

    print(
        f"\nSelected threshold:\n"
        f"{SELECTED_FILE}"
    )

    print(
        f"\nMetrics by theme:\n"
        f"{THEME_METRICS_FILE}"
    )

    print(
        f"\nOOF errors:\n"
        f"{ERRORS_FILE}"
    )

    print()
    print("=" * 100)
    print("NEXT STEP")
    print("=" * 100)

    print(
        """
Do not run full-corpus scoring yet.

Next:
1. review the selected threshold and theme-level metrics;
2. fit V3 on all 485 development observations;
3. score only the same 1,000-article production diagnostic sample;
4. inspect cross-theme correlations and within-article score ranges;
5. only then decide whether V3 can be frozen for final evaluation.
"""
    )


if __name__ == "__main__":
    main()