#!/usr/bin/env python3

"""
FINAL TEST — KEYWORD + WEAK V2.3.2 EVALUATION
=============================================

Evaluate two already-existing scoring methods on the frozen final test:

1. Keyword baseline
   model_version = seedwords_tsrank_v1
   decision rule = thematic score > 0

2. Weak Supervision V2.3.2
   model_version = weak_labels_v2_3_2_score_v1
   decision rule = thematic score >= 0.50

Methodological constraints
--------------------------
- No model fitting.
- No rule modification.
- No threshold selection.
- No calibration.
- Final-test human labels are used only for evaluation.
- Ranking metrics are computed from the continuous stored scores.
- Classification metrics are computed from the frozen binary rules.
"""

from __future__ import annotations

import hashlib
import json
import os

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "FINAL_TEST_KEYWORD_WEAK_SCORES",
        "final_test_existing_keyword_weak_scores.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "FINAL_TEST_KEYWORD_WEAK_EVAL_DIR",
        "keyword_weak_v232_evaluation",
    )
)

EXPECTED_ROWS = 500
EXPECTED_ARTICLES = 300
EXPECTED_THEMES = 5

KEYWORD_MODEL_VERSION = "seedwords_tsrank_v1"
WEAK_MODEL_VERSION = "weak_labels_v2_3_2_score_v1"

KEYWORD_METHOD_NAME = "keyword_seedwords_tsrank_v1"
WEAK_METHOD_NAME = "weak_supervision_v2_3_2"

KEYWORD_RULE = "thematic score > 0"
WEAK_RULE = "thematic score >= 0.50"

WEAK_THRESHOLD = 0.50


# =============================================================================
# OUTPUT FILES
# =============================================================================

OVERALL_FILE = (
    OUTPUT_DIR
    / "keyword_weak_v232_final_overall.csv"
)

THEME_FILE = (
    OUTPUT_DIR
    / "keyword_weak_v232_final_by_theme.csv"
)

BLOCK_FILE = (
    OUTPUT_DIR
    / "keyword_weak_v232_final_by_block.csv"
)

THEME_BLOCK_FILE = (
    OUTPUT_DIR
    / "keyword_weak_v232_final_by_theme_block.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "keyword_weak_v232_final_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "keyword_weak_v232_final_summary.json"
)


# =============================================================================
# HASHING
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# METRICS
# =============================================================================

def safe_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:

    if np.unique(y_true).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            scores,
        )
    )


def safe_ap(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:

    if np.unique(y_true).size < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            scores,
        )
    )


def calculate_metrics(
    df: pd.DataFrame,
    prediction_col: str,
    score_col: str,
) -> dict:

    y_true = (
        df["human_label"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df[prediction_col]
        .astype(int)
        .to_numpy()
    )

    scores = (
        df[score_col]
        .astype(float)
        .to_numpy()
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    balanced_accuracy = (
        (recall + specificity) / 2
        if np.isfinite(specificity)
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    return {
        "n":
            int(len(df)),

        "gold_negative":
            int((y_true == 0).sum()),

        "gold_positive":
            int((y_true == 1).sum()),

        "predicted_negative":
            int((y_pred == 0).sum()),

        "predicted_positive":
            int((y_pred == 1).sum()),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(recall),

        "specificity":
            float(specificity)
            if np.isfinite(specificity)
            else np.nan,

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "balanced_accuracy":
            float(balanced_accuracy)
            if np.isfinite(balanced_accuracy)
            else np.nan,

        "npv":
            float(npv)
            if np.isfinite(npv)
            else np.nan,

        "roc_auc":
            safe_auc(
                y_true,
                scores,
            ),

        "average_precision":
            safe_ap(
                y_true,
                scores,
            ),
    }


def grouped_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
    prediction_col: str,
    score_col: str,
) -> pd.DataFrame:

    rows = []

    grouper = (
        group_cols[0]
        if len(group_cols) == 1
        else group_cols
    )

    for keys, subset in df.groupby(
        grouper,
        dropna=False,
        sort=True,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        result = dict(
            zip(
                group_cols,
                keys,
            )
        )

        result.update(
            calculate_metrics(
                subset,
                prediction_col,
                score_col,
            )
        )

        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# METHOD EVALUATION
# =============================================================================

def evaluate_method(
    df: pd.DataFrame,
    method_name: str,
    prediction_col: str,
    score_col: str,
):

    print()
    print("=" * 110)
    print(method_name)
    print("=" * 110)

    overall = calculate_metrics(
        df,
        prediction_col,
        score_col,
    )

    print(
        f"""
N:                    {overall['n']}
Gold positives:       {overall['gold_positive']}
Predicted positives:  {overall['predicted_positive']}

Confusion matrix:
  TN = {overall['tn']}
  FP = {overall['fp']}
  FN = {overall['fn']}
  TP = {overall['tp']}

Accuracy:             {overall['accuracy']:.6f}
Precision:            {overall['precision']:.6f}
Recall:               {overall['recall']:.6f}
Specificity:          {overall['specificity']:.6f}
F1:                   {overall['f1']:.6f}
Balanced Accuracy:    {overall['balanced_accuracy']:.6f}
NPV:                  {overall['npv']:.6f}
ROC-AUC:              {overall['roc_auc']:.6f}
Average Precision:    {overall['average_precision']:.6f}
"""
    )

    by_theme = grouped_metrics(
        df,
        [
            "theme_id",
            "theme_name",
        ],
        prediction_col,
        score_col,
    )

    by_block = grouped_metrics(
        df,
        [
            "sample_block",
        ],
        prediction_col,
        score_col,
    )

    by_theme_block = grouped_metrics(
        df,
        [
            "sample_block",
            "theme_id",
            "theme_name",
        ],
        prediction_col,
        score_col,
    )

    print("-" * 110)
    print("BY THEME")
    print("-" * 110)

    print(
        by_theme.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("-" * 110)
    print("BY SAMPLE BLOCK")
    print("-" * 110)

    print(
        by_block.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    overall_df = pd.DataFrame(
        [
            {
                "method":
                    method_name,
                **overall,
            }
        ]
    )

    return (
        overall_df,
        by_theme,
        by_block,
        by_theme_block,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "FINAL TEST — "
        "KEYWORD + WEAK V2.3.2 EVALUATION"
    )
    print("=" * 110)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found:\n"
            f"{INPUT_FILE}"
        )

    input_hash = sha256_file(
        INPUT_FILE
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nRows loaded: "
        f"{len(df):,}"
    )

    print(
        f"Input SHA-256: "
        f"{input_hash}"
    )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    required = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "sample_block",
        "human_label",
        "keyword_score",
        "weak_score",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # Structural integrity
    # -------------------------------------------------------------------------

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(df)}."
        )

    if (
        df["article_id"].nunique()
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ARTICLES} "
            "unique articles."
        )

    if (
        df["theme_id"].nunique()
        != EXPECTED_THEMES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_THEMES} themes."
        )

    if (
        df["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id detected."
        )

    if df[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pairs detected."
        )

    identity_columns = [
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "sample_block",
    ]

    if df[
        identity_columns
    ].isna().any().any():

        raise RuntimeError(
            "Missing final-test identifiers "
            "or grouping information."
        )

    # -------------------------------------------------------------------------
    # Labels and scores
    # -------------------------------------------------------------------------

    df["human_label"] = pd.to_numeric(
        df["human_label"],
        errors="raise",
    )

    if (
        df["human_label"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing human labels."
        )

    if not set(
        df["human_label"].unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "human_label must contain "
            "only 0 and 1."
        )

    df["human_label"] = (
        df["human_label"]
        .astype(int)
    )

    for score_col in [
        "keyword_score",
        "weak_score",
    ]:

        df[score_col] = pd.to_numeric(
            df[score_col],
            errors="raise",
        )

        values = (
            df[score_col]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                f"{score_col} contains "
                "missing or non-finite values."
            )

        if (
            (values < 0.0).any()
            or
            (values > 1.0).any()
        ):
            raise RuntimeError(
                f"{score_col} contains values "
                "outside [0,1]."
            )

    print(
        "[OK] final-test structure, labels, "
        "and stored scores verified"
    )

    # -------------------------------------------------------------------------
    # Frozen decision rules
    # -------------------------------------------------------------------------

    df[
        "keyword_prediction"
    ] = (
        df["keyword_score"]
        > 0.0
    ).astype(int)

    df[
        "weak_v232_prediction"
    ] = (
        df["weak_score"]
        >= WEAK_THRESHOLD
    ).astype(int)

    print(
        f"\nLocked keyword rule: "
        f"{KEYWORD_RULE}"
    )

    print(
        f"Locked Weak V2.3.2 rule: "
        f"{WEAK_RULE}"
    )

    print(
        f"\nKeyword positives: "
        f"{int(df['keyword_prediction'].sum()):,} "
        f"/ {EXPECTED_ROWS:,}"
    )

    print(
        f"Weak V2.3.2 positives: "
        f"{int(df['weak_v232_prediction'].sum()):,} "
        f"/ {EXPECTED_ROWS:,}"
    )

    # -------------------------------------------------------------------------
    # Keyword evaluation
    # -------------------------------------------------------------------------

    (
        keyword_overall,
        keyword_theme,
        keyword_block,
        keyword_theme_block,
    ) = evaluate_method(
        df,
        KEYWORD_METHOD_NAME,
        "keyword_prediction",
        "keyword_score",
    )

    # -------------------------------------------------------------------------
    # Weak V2.3.2 evaluation
    # -------------------------------------------------------------------------

    (
        weak_overall,
        weak_theme,
        weak_block,
        weak_theme_block,
    ) = evaluate_method(
        df,
        WEAK_METHOD_NAME,
        "weak_v232_prediction",
        "weak_score",
    )

    # -------------------------------------------------------------------------
    # Combine evaluation artifacts
    # -------------------------------------------------------------------------

    overall = pd.concat(
        [
            keyword_overall,
            weak_overall,
        ],
        ignore_index=True,
    )

    keyword_theme[
        "method"
    ] = KEYWORD_METHOD_NAME

    weak_theme[
        "method"
    ] = WEAK_METHOD_NAME

    theme_metrics = pd.concat(
        [
            keyword_theme,
            weak_theme,
        ],
        ignore_index=True,
    )

    keyword_block[
        "method"
    ] = KEYWORD_METHOD_NAME

    weak_block[
        "method"
    ] = WEAK_METHOD_NAME

    block_metrics = pd.concat(
        [
            keyword_block,
            weak_block,
        ],
        ignore_index=True,
    )

    keyword_theme_block[
        "method"
    ] = KEYWORD_METHOD_NAME

    weak_theme_block[
        "method"
    ] = WEAK_METHOD_NAME

    theme_block_metrics = pd.concat(
        [
            keyword_theme_block,
            weak_theme_block,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Refuse silent overwrite
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        OVERALL_FILE,
        THEME_FILE,
        BLOCK_FILE,
        THEME_BLOCK_FILE,
        PREDICTIONS_FILE,
        SUMMARY_FILE,
    ]

    existing = [
        path
        for path in output_files
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Evaluation artifacts already exist; "
            "refusing silent overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    # -------------------------------------------------------------------------
    # Save CSV artifacts
    # -------------------------------------------------------------------------

    overall.to_csv(
        OVERALL_FILE,
        index=False,
    )

    theme_metrics.to_csv(
        THEME_FILE,
        index=False,
    )

    block_metrics.to_csv(
        BLOCK_FILE,
        index=False,
    )

    theme_block_metrics.to_csv(
        THEME_BLOCK_FILE,
        index=False,
    )

    df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Evaluation manifest
    # -------------------------------------------------------------------------

    summary = {
        "status":
            "FINAL_TEST_EVALUATION",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "input": {
            "path":
                str(INPUT_FILE),

            "sha256":
                input_hash,

            "rows":
                int(len(df)),

            "unique_articles":
                int(
                    df[
                        "article_id"
                    ].nunique()
                ),

            "themes":
                int(
                    df[
                        "theme_id"
                    ].nunique()
                ),

            "gold_positive":
                int(
                    df[
                        "human_label"
                    ].sum()
                ),
        },

        "keyword": {
            "method":
                KEYWORD_METHOD_NAME,

            "model_version":
                KEYWORD_MODEL_VERSION,

            "decision_rule":
                KEYWORD_RULE,

            "threshold_tuning_performed":
                False,

            "overall_metrics":
                keyword_overall
                .iloc[0]
                .drop(
                    labels=["method"]
                )
                .to_dict(),
        },

        "weak_v2_3_2": {
            "method":
                WEAK_METHOD_NAME,

            "model_version":
                WEAK_MODEL_VERSION,

            "decision_rule":
                WEAK_RULE,

            "threshold_tuning_performed":
                False,

            "overall_metrics":
                weak_overall
                .iloc[0]
                .drop(
                    labels=["method"]
                )
                .to_dict(),
        },

        "methodological_lock": {
            "model_training_performed":
                False,

            "rule_modification_performed":
                False,

            "threshold_selection_performed":
                False,

            "final_test_labels_used_only_for_evaluation":
                True,
        },

        "artifacts": {
            "overall":
                str(OVERALL_FILE),

            "by_theme":
                str(THEME_FILE),

            "by_block":
                str(BLOCK_FILE),

            "by_theme_block":
                str(THEME_BLOCK_FILE),

            "predictions":
                str(PREDICTIONS_FILE),
        },
    }

    # Convert NumPy scalar values before JSON serialization.
    summary = json.loads(
        json.dumps(
            summary,
            default=lambda value: (
                value.item()
                if isinstance(
                    value,
                    np.generic,
                )
                else None
            ),
        )
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

        file.write("\n")

    # -------------------------------------------------------------------------
    # Output fingerprints
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    for path in output_files:

        print(
            f"\n{path}"
        )

        print(
            f"SHA-256: "
            f"{sha256_file(path)}"
        )

    print()
    print("=" * 110)
    print(
        "KEYWORD + WEAK V2.3.2 "
        "FINAL EVALUATION COMPLETE"
    )
    print("=" * 110)

    print(
        "\nNo thresholds, rules, or model "
        "parameters were selected from "
        "final-test performance."
    )


if __name__ == "__main__":
    main()