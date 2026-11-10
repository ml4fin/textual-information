#!/usr/bin/env python3

"""
FREEZE SUPERVISED MODEL V3
==========================

Create the formal frozen specification of Supervised Model V3 after:

1. development on 485 manually labelled observations;
2. frozen 5-fold out-of-fold evaluation;
3. V1/V2/V3 architecture comparison;
4. global threshold selection from V3 OOF predictions;
5. a 1,000-article production structural sanity check.

This script does not:
    - retrain the model;
    - change the architecture or hyperparameters;
    - change the selected threshold;
    - introduce probability calibration;
    - use final-test data;
    - perform full-corpus production scoring.

After this freeze, the architecture, features, classifier hyperparameters,
global threshold, and calibration specification are locked.

The next predictive performance evaluation must use a new, untouched,
manually labelled final test set.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(
    os.getenv(
        "SUPERVISED_MODELS_DIR",
        "07_supervised_models",
    )
)

CANDIDATE_DIR = (
    BASE_DIR
    / "05_v3_hybrid"
    / "production_candidate"
)

CANDIDATE_MODEL = (
    CANDIDATE_DIR
    / "supervised_v3_candidate.joblib"
)

CANDIDATE_METADATA = (
    CANDIDATE_DIR
    / "supervised_v3_candidate_metadata.json"
)

THRESHOLD_FILE = (
    BASE_DIR
    / "05_v3_hybrid"
    / "threshold"
    / "supervised_v3_selected_threshold.json"
)

DIAGNOSTIC_DIR = (
    CANDIDATE_DIR
    / "diagnostic_1000"
)

DIAGNOSTIC_CORRELATIONS = (
    DIAGNOSTIC_DIR
    / "supervised_v3_diagnostic_1000_correlations.csv"
)

DIAGNOSTIC_ARTICLES = (
    DIAGNOSTIC_DIR
    / "supervised_v3_diagnostic_1000_article_diagnostics.csv"
)

FROZEN_DIR = (
    BASE_DIR
    / "06_frozen"
)

FROZEN_SPEC = (
    FROZEN_DIR
    / "supervised_model_v3_frozen.json"
)

FROZEN_MODEL = (
    FROZEN_DIR
    / "supervised_model_v3.joblib"
)


# =============================================================================
# EXPECTED FROZEN VALUES
# =============================================================================

EXPECTED_DEVELOPMENT_N = 485
EXPECTED_NEGATIVES = 337
EXPECTED_POSITIVES = 148

EXPECTED_C = 4.0
EXPECTED_THRESHOLD = 0.34

EXPECTED_AUC = 0.815162
EXPECTED_AP = 0.659727
EXPECTED_F1 = 0.629213
EXPECTED_BA = 0.735945

EXPECTED_PRODUCTION_N = 1_000
EXPECTED_PRODUCTION_SCORES = 5_000

EXPECTED_DIAGNOSTIC = {
    "mean_pairwise_correlation": 0.915674,
    "min_pairwise_correlation": 0.879207,
    "max_pairwise_correlation": 0.960006,
    "mean_score_range": 0.273085,
    "median_score_range": 0.276899,
    "mean_score_std": 0.107116,
}

EXPECTED_THEMES = [
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
]

FLOAT_TOLERANCE = 1e-6


# =============================================================================
# HELPERS
# =============================================================================

def assert_close(
    actual,
    expected,
    name,
    tolerance=FLOAT_TOLERANCE,
):

    if not np.isclose(
        float(actual),
        float(expected),
        atol=tolerance,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{name} does not match the frozen specification.\n"
            f"Actual:   {actual}\n"
            f"Expected: {expected}"
        )


def load_json(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def sha256_file(
    path: Path,
    chunk_size=1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_candidate_metadata(
    candidate,
):

    development = candidate[
        "development_data"
    ]

    if (
        int(development["n"])
        != EXPECTED_DEVELOPMENT_N
    ):
        raise RuntimeError(
            "Unexpected development-set size."
        )

    if (
        int(development["negatives"])
        != EXPECTED_NEGATIVES
    ):
        raise RuntimeError(
            "Unexpected number of development negatives."
        )

    if (
        int(development["positives"])
        != EXPECTED_POSITIVES
    ):
        raise RuntimeError(
            "Unexpected number of development positives."
        )

    if not development[
        "not_final_test"
    ]:
        raise RuntimeError(
            "Development data must be explicitly marked "
            "as non-final-test data."
        )

    classifier = candidate[
        "classifier"
    ]

    assert_close(
        classifier["C"],
        EXPECTED_C,
        "LogisticRegression C",
    )

    if classifier["penalty"] != "l2":
        raise RuntimeError(
            "Unexpected classifier penalty."
        )

    if classifier["solver"] != "liblinear":
        raise RuntimeError(
            "Unexpected classifier solver."
        )

    if classifier["class_weight"] != "balanced":
        raise RuntimeError(
            "Unexpected classifier class weight."
        )

    if int(
        classifier["max_iter"]
    ) != 2000:
        raise RuntimeError(
            "Unexpected classifier max_iter."
        )

    if int(
        classifier["random_state"]
    ) != 42:
        raise RuntimeError(
            "Unexpected classifier random_state."
        )

    assert_close(
        candidate["threshold"]["value"],
        EXPECTED_THRESHOLD,
        "Candidate threshold",
    )

    oof = candidate[
        "development_oof"
    ]

    assert_close(
        oof["roc_auc"],
        EXPECTED_AUC,
        "OOF ROC-AUC",
    )

    assert_close(
        oof["average_precision"],
        EXPECTED_AP,
        "OOF average precision",
    )

    assert_close(
        oof["f1"],
        EXPECTED_F1,
        "OOF F1",
    )

    assert_close(
        oof["balanced_accuracy"],
        EXPECTED_BA,
        "OOF balanced accuracy",
    )

    if candidate[
        "calibration"
    ] != "none":
        raise RuntimeError(
            "Calibration must remain frozen as 'none'."
        )


def validate_threshold_specification(
    threshold_data,
):

    selection = threshold_data[
        "threshold_selection"
    ]

    if selection[
        "scope"
    ] != "global":
        raise RuntimeError(
            "Threshold scope must be global."
        )

    assert_close(
        selection["selected_threshold"],
        EXPECTED_THRESHOLD,
        "OOF selected threshold",
    )

    assert_close(
        selection["grid_start"],
        0.20,
        "Threshold grid start",
    )

    assert_close(
        selection["grid_end"],
        0.80,
        "Threshold grid end",
    )

    assert_close(
        selection["grid_step"],
        0.01,
        "Threshold grid step",
    )


def validate_production_diagnostic():

    correlations = pd.read_csv(
        DIAGNOSTIC_CORRELATIONS
    )

    articles = pd.read_csv(
        DIAGNOSTIC_ARTICLES
    )

    if len(articles) != EXPECTED_PRODUCTION_N:
        raise RuntimeError(
            "Production diagnostic article count does not "
            "match the frozen specification."
        )

    required_article_columns = {
        "article_id",
        "score_range",
        "score_std",
    }

    missing = (
        required_article_columns
        - set(articles.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing production diagnostic columns: "
            f"{sorted(missing)}"
        )

    if (
        articles["article_id"].nunique()
        != EXPECTED_PRODUCTION_N
    ):
        raise RuntimeError(
            "Production diagnostic does not contain exactly "
            "1,000 unique article IDs."
        )

    score_range = pd.to_numeric(
        articles["score_range"],
        errors="raise",
    )

    score_std = pd.to_numeric(
        articles["score_std"],
        errors="raise",
    )

    assert_close(
        score_range.mean(),
        EXPECTED_DIAGNOSTIC[
            "mean_score_range"
        ],
        "Mean within-article score range",
    )

    assert_close(
        score_range.median(),
        EXPECTED_DIAGNOSTIC[
            "median_score_range"
        ],
        "Median within-article score range",
    )

    assert_close(
        score_std.mean(),
        EXPECTED_DIAGNOSTIC[
            "mean_score_std"
        ],
        "Mean within-article score standard deviation",
    )

    numeric_correlations = (
        correlations
        .select_dtypes(
            include=[np.number]
        )
        .to_numpy()
        .ravel()
    )

    numeric_correlations = (
        numeric_correlations[
            np.isfinite(
                numeric_correlations
            )
        ]
    )

    if numeric_correlations.size == 0:
        raise RuntimeError(
            "No numeric correlation values found in "
            "the production diagnostic."
        )

    # The diagnostic CSV must contain the pairwise correlation values
    # themselves. If its structure differs, validate the named summary
    # columns explicitly instead of inferring them here.
    assert_close(
        numeric_correlations.mean(),
        EXPECTED_DIAGNOSTIC[
            "mean_pairwise_correlation"
        ],
        "Mean pairwise correlation",
    )

    assert_close(
        numeric_correlations.min(),
        EXPECTED_DIAGNOSTIC[
            "min_pairwise_correlation"
        ],
        "Minimum pairwise correlation",
    )

    assert_close(
        numeric_correlations.max(),
        EXPECTED_DIAGNOSTIC[
            "max_pairwise_correlation"
        ],
        "Maximum pairwise correlation",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "SUPERVISED MODEL V3 — "
        "FORMAL DEVELOPMENT FREEZE"
    )
    print("=" * 110)

    required_files = [
        CANDIDATE_MODEL,
        CANDIDATE_METADATA,
        THRESHOLD_FILE,
        DIAGNOSTIC_CORRELATIONS,
        DIAGNOSTIC_ARTICLES,
    ]

    print(
        "\nChecking required artifacts..."
    )

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

        print(
            f"[OK] {path.name}"
        )

    candidate = load_json(
        CANDIDATE_METADATA
    )

    threshold_data = load_json(
        THRESHOLD_FILE
    )

    print()
    print("=" * 110)
    print(
        "VALIDATE DEVELOPMENT SPECIFICATION"
    )
    print("=" * 110)

    validate_candidate_metadata(
        candidate
    )

    print(
        f"[OK] development n = "
        f"{EXPECTED_DEVELOPMENT_N}"
    )

    print(
        f"[OK] negatives = "
        f"{EXPECTED_NEGATIVES}"
    )

    print(
        f"[OK] positives = "
        f"{EXPECTED_POSITIVES}"
    )

    print(
        f"[OK] LogisticRegression C = "
        f"{EXPECTED_C}"
    )

    print(
        "[OK] L2 / liblinear / balanced"
    )

    print(
        f"[OK] OOF ROC-AUC = "
        f"{EXPECTED_AUC:.6f}"
    )

    print(
        f"[OK] OOF AP      = "
        f"{EXPECTED_AP:.6f}"
    )

    print(
        f"[OK] OOF F1      = "
        f"{EXPECTED_F1:.6f}"
    )

    print(
        f"[OK] OOF BA      = "
        f"{EXPECTED_BA:.6f}"
    )

    print(
        "[OK] calibration = none"
    )

    print()
    print("=" * 110)
    print(
        "VALIDATE THRESHOLD SPECIFICATION"
    )
    print("=" * 110)

    validate_threshold_specification(
        threshold_data
    )

    print(
        f"[OK] global threshold = "
        f"{EXPECTED_THRESHOLD:.2f}"
    )

    print()
    print("=" * 110)
    print(
        "VALIDATE PRODUCTION STRUCTURAL DIAGNOSTIC"
    )
    print("=" * 110)

    validate_production_diagnostic()

    print(
        f"[OK] diagnostic articles = "
        f"{EXPECTED_PRODUCTION_N:,}"
    )

    print(
        "[OK] cross-theme correlation summary"
    )

    print(
        "[OK] within-article score variation"
    )

    FROZEN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 110)
    print(
        "FREEZE FITTED MODEL"
    )
    print("=" * 110)

    # The already-fitted candidate is copied byte-for-byte. No refitting occurs.
    shutil.copy2(
        CANDIDATE_MODEL,
        FROZEN_MODEL,
    )

    candidate_hash = sha256_file(
        CANDIDATE_MODEL
    )

    frozen_hash = sha256_file(
        FROZEN_MODEL
    )

    if candidate_hash != frozen_hash:
        raise RuntimeError(
            "Frozen model hash differs from candidate model hash."
        )

    print(
        f"\n[OK] copied without retraining:\n"
        f"{FROZEN_MODEL}"
    )

    print(
        f"\n[OK] SHA-256:\n"
        f"{frozen_hash}"
    )

    frozen_spec = {
        "model_name": (
            "Supervised Model V3 - "
            "Frozen Development Version"
        ),
        "model_version": "supervised_v3",
        "status": "FROZEN",
        "frozen_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model_artifact": {
            "filename": (
                FROZEN_MODEL.name
            ),
            "sha256": frozen_hash,
            "source_artifact": (
                CANDIDATE_MODEL.name
            ),
            "retrained_during_freeze": False,
        },
        "selection_history": {
            "v1": {
                "status": (
                    "rejected_after_production_structural_diagnostic"
                ),
                "development_roc_auc": 0.7943,
                "development_average_precision": 0.6135,
                "reason": (
                    "Near-perfect cross-theme production score "
                    "correlations of approximately 0.98-1.00 "
                    "indicated insufficient theme sensitivity."
                ),
            },
            "v2": {
                "status": "not_selected",
                "architecture": (
                    "theme_interactions_only"
                ),
                "development_roc_auc": 0.765458,
                "development_average_precision": 0.606942,
                "reason": (
                    "Explicit interactions improved structural "
                    "theme conditioning but reduced pooled "
                    "ranking performance."
                ),
            },
            "v3": {
                "status": "selected",
                "architecture": (
                    "hybrid shared word + shared char + "
                    "theme-text interactions"
                ),
                "reason": (
                    "Selected on development after architecture "
                    "comparison, followed by a production "
                    "structural sanity check confirming materially "
                    "greater theme sensitivity than V1."
                ),
            },
        },
        "development_data": {
            "n": EXPECTED_DEVELOPMENT_N,
            "unique_articles": EXPECTED_DEVELOPMENT_N,
            "negatives": EXPECTED_NEGATIVES,
            "positives": EXPECTED_POSITIVES,
            "role": (
                "development / model selection / "
                "threshold selection"
            ),
            "final_test": False,
            "note": (
                "These observations were repeatedly used during "
                "supervised model development and must not be "
                "interpreted as an untouched final test set."
            ),
        },
        "candidate_themes": EXPECTED_THEMES,
        "input_representation": {
            "production_document": (
                "<normalized_theme_key> ||| "
                "<title> ||| <full_body>"
            ),
            "body_truncation": False,
            "important": (
                "Production scoring must construct one document "
                "for every article-theme pair using the exact "
                "frozen V3 representation."
            ),
        },
        "features": {
            "shared_word_tfidf": {
                "text": "TITLE + BODY only",
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.95,
                "max_features": 25_000,
                "stop_words": "english",
                "sublinear_tf": True,
            },
            "shared_char_tfidf": {
                "text": "TITLE + BODY only",
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "min_df": 2,
                "max_features": 25_000,
                "sublinear_tf": True,
            },
            "theme_interaction_tfidf": {
                "text": "TITLE + BODY",
                "conditioning": "theme_key",
                "features": (
                    "theme-conditioned word unigrams and bigrams"
                ),
                "min_df": 2,
                "max_df": 0.95,
                "max_features": 50_000,
                "sublinear_tf": True,
                "norm": "l2",
            },
            "fitted_full_development_counts": (
                candidate[
                    "features"
                ][
                    "fitted_feature_counts"
                ]
            ),
        },
        "classifier": {
            "type": "LogisticRegression",
            "C": EXPECTED_C,
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42,
        },
        "threshold": {
            "scope": "global",
            "value": EXPECTED_THRESHOLD,
            "theme_specific_thresholds": False,
            "selection_data": (
                "5-fold OOF development predictions"
            ),
            "selection_grid": {
                "start": 0.20,
                "end": 0.80,
                "step": 0.01,
            },
            "selection_rule": [
                "maximize global F1",
                "tie-break maximize balanced accuracy",
                "tie-break closest to 0.50",
                "final deterministic tie-break: lower threshold",
            ],
        },
        "calibration": "none",
        "development_oof_performance": {
            "roc_auc": EXPECTED_AUC,
            "average_precision": EXPECTED_AP,
            "threshold": EXPECTED_THRESHOLD,
            "tn": 241,
            "fp": 96,
            "fn": 36,
            "tp": 112,
            "accuracy": 0.727835,
            "precision": 0.538462,
            "recall": 0.756757,
            "specificity": 0.715134,
            "f1": EXPECTED_F1,
            "balanced_accuracy": EXPECTED_BA,
            "interpretation": (
                "Development/model-selection estimates only; "
                "not unbiased final-test performance."
            ),
        },
        "development_oof_by_theme": {
            "AI-Related Risk": {
                "n": 62,
                "positives": 18,
                "roc_auc": 0.6667,
                "average_precision": 0.4425,
            },
            "Climate / Environmental Risk": {
                "n": 110,
                "positives": 40,
                "roc_auc": 0.7979,
                "average_precision": 0.7155,
            },
            "Cybersecurity Risk": {
                "n": 89,
                "positives": 19,
                "roc_auc": 0.7842,
                "average_precision": 0.4000,
            },
            "Pandemic Risk": {
                "n": 88,
                "positives": 43,
                "roc_auc": 0.8295,
                "average_precision": 0.8182,
            },
            "Supply Chain Risk": {
                "n": 136,
                "positives": 28,
                "roc_auc": 0.8750,
                "average_precision": 0.6181,
            },
        },
        "production_structural_diagnostic": {
            "status": "PASSED",
            "n_articles": EXPECTED_PRODUCTION_N,
            "n_article_theme_scores": EXPECTED_PRODUCTION_SCORES,
            "article_id_range": [
                859026,
                860068,
            ],
            "purpose": (
                "Structural sanity check only; "
                "not predictive performance evaluation."
            ),
            "threshold_positive_rate": 0.3960,
            **EXPECTED_DIAGNOSTIC,
            "reference_v1": (
                "V1 showed approximately 0.98-1.00 pairwise "
                "cross-theme correlations."
            ),
            "interpretation": (
                "V3 materially reduced the near-perfect "
                "cross-theme coupling observed for V1 and "
                "produced substantial within-article score "
                "variation when the target theme was changed."
            ),
            "limitation": (
                "The diagnostic production articles do not have "
                "gold labels for all five themes. This diagnostic "
                "therefore demonstrates structural theme "
                "sensitivity, not cross-theme predictive validity "
                "or corpus prevalence."
            ),
        },
        "known_limitation": {
            "training_structure": (
                "Each of the 485 development articles has a gold "
                "label for one candidate theme only."
            ),
            "production_structure": (
                "Each production article is scored under all five "
                "candidate themes."
            ),
            "implication": (
                "Counterfactual article-theme predictions for the "
                "other four themes are not directly supervised by "
                "gold labels in the current development dataset."
            ),
        },
        "freeze_policy": {
            "architecture_locked": True,
            "features_locked": True,
            "classifier_hyperparameters_locked": True,
            "threshold_locked": True,
            "calibration_locked": True,
            "development_set_closed_for_tuning": True,
            "production_diagnostic_closed_for_tuning": True,
            "allowed_next_evaluation": (
                "new unseen manually-labelled final test set"
            ),
            "no_further_tuning_before_final_test": True,
        },
        "final_test_status": {
            "evaluated": False,
            "performance_known": False,
            "required": True,
        },
        "full_corpus_status": {
            "scored": False,
            "allowed_before_final_evaluation": False,
        },
        "methodological_statement": (
            "Supervised Model V3 is frozen after model development "
            "on 485 manually labelled development observations and "
            "a separate unlabelled production structural sanity "
            "check. No further architecture, feature, "
            "hyperparameter, threshold, or calibration changes "
            "are permitted before evaluation on a new untouched "
            "manually labelled test set. All reported OOF metrics "
            "remain development estimates rather than final "
            "out-of-sample performance estimates."
        ),
    }

    with FROZEN_SPEC.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            frozen_spec,
            f,
            indent=2,
        )

    print()
    print("=" * 110)
    print(
        "FROZEN SPECIFICATION CREATED"
    )
    print("=" * 110)

    print(
        f"\nModel:\n"
        f"{FROZEN_MODEL}"
    )

    print(
        f"\nSpecification:\n"
        f"{FROZEN_SPEC}"
    )

    print()
    print("=" * 110)
    print(
        "FINAL STATUS"
    )
    print("=" * 110)

    print(
        f"""
SUPERVISED MODEL V3
STATUS: FROZEN

Architecture:
  hybrid shared + theme × text

Classifier:
  Logistic Regression
  C = {EXPECTED_C}
  class_weight = balanced

Global threshold:
  {EXPECTED_THRESHOLD:.2f}

Calibration:
  none

Development OOF:
  ROC-AUC = {EXPECTED_AUC:.6f}
  AP      = {EXPECTED_AP:.6f}
  F1      = {EXPECTED_F1:.6f}
  BA      = {EXPECTED_BA:.6f}

Production structural diagnostic:
  PASSED

Development performance is not final-test performance.

No further supervised tuning may use the 485 development observations
or the 1,000 production diagnostic articles.

NEXT:
  construct a new untouched manually-labelled final test set.
"""
    )


if __name__ == "__main__":
    main()