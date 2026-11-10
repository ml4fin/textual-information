#!/usr/bin/env python3

"""
FINAL TEST — WEAK V3 CONTINUOUS FROZEN EVALUATION
=================================================

Apply the frozen Weak V3 Continuous specification to the untouched
500-observation final test.

Methodological constraints
--------------------------
- No training.
- No refitting.
- No tuning.
- No calibration.
- No threshold selection.
- Candidate C was selected and frozen on development data.
- Final-test labels are used only for out-of-sample evaluation.
- The continuous score is a relevance/evidence index, not a
  calibrated probability.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from build_weak_v3_continuous_evidence import (
    extract_evidence,
)

from weak_supervision_v3_continuous_config import (
    VERSION,
    MODEL_VERSION,
    METHOD,
    SELECTED_CANDIDATE,
    PROXIMITY_WINDOW,
    DENSITY_REINFORCEMENT,
    SCORE_MIN,
    SCORE_MAX,
    DEVELOPMENT_N,
    DEVELOPMENT_POSITIVES,
    DEVELOPMENT_NEGATIVES,
    DEVELOPMENT_ROC_AUC,
    DEVELOPMENT_AVERAGE_PRECISION,
    FINAL_TEST_USED_FOR_SELECTION,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    os.getenv(
        "FINAL_EVALUATION_DIR",
        "08_final_evaluation",
    )
)

GOLD_FILE = Path(
    os.getenv(
        "FINAL_TEST_GOLD_FILE",
        str(
            BASE_DIR
            / "01_gold"
            / "final_test_gold_frozen.csv"
        ),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "FINAL_WEAK_V3_CONTINUOUS_DIR",
        str(
            BASE_DIR
            / "03_weak_supervision"
        ),
    )
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "final_weak_v3_continuous_predictions.csv"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "final_weak_v3_continuous_manifest.json"
)

EXPECTED_GOLD_SHA256 = (
    "d4e65929a1831f7823bd92546b100647"
    "96379a55e3924fd642e5366bade19e21"
)

EXPECTED_ROWS = 500
EXPECTED_ARTICLES = 300
EXPECTED_THEMES = 5

EXPECTED_POSITIVES = 26
EXPECTED_NEGATIVES = 474

EXPECTED_CANDIDATE = "score_C_density"
EXPECTED_WINDOW = 50
EXPECTED_DENSITY_REINFORCEMENT = 0.35

EXPECTED_SCORE_MIN = 0.0
EXPECTED_SCORE_MAX = 1.0

EXPECTED_THEME_LABELS = {
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
}

REQUIRED_EVIDENCE_COLUMNS = {
    "article_id",
    "theme_id",
    "theme_name",
    "human_label",
    "strong_frequency",
    "strong_density_1000",
    "proximity_strength",
}


# =============================================================================
# HASH
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# FROZEN SCORE TRANSFORMATIONS
# =============================================================================

def sat_count(
    x: float,
) -> float:

    x = max(
        float(x),
        0.0,
    )

    return float(
        1.0 - np.exp(-x)
    )


def density_component(
    x: float,
) -> float:

    x = max(
        float(x),
        0.0,
    )

    return float(
        x / (1.0 + x)
    )


def probabilistic_union(
    *components: float,
) -> float:

    product = 1.0

    for component in components:

        component = float(
            np.clip(
                component,
                0.0,
                1.0,
            )
        )

        product *= (
            1.0 - component
        )

    return float(
        np.clip(
            1.0 - product,
            0.0,
            1.0,
        )
    )


def frozen_score_c(
    row,
) -> float:
    """
    Frozen Candidate C relevance score.

    AI:
        proximity evidence only

    Pandemic:
        strong frequency
        + density reinforcement

    Climate / Cyber / Supply:
        strong frequency
        + proximity
        + density reinforcement
    """

    theme = str(
        row["theme_name"]
    )

    strong_frequency = float(
        row["strong_frequency"]
    )

    strong_density = float(
        row["strong_density_1000"]
    )

    proximity = float(
        row["proximity_strength"]
    )

    c_strong_frequency = sat_count(
        strong_frequency
    )

    c_strong_density = density_component(
        strong_density
    )

    density_reinforcement = (
        DENSITY_REINFORCEMENT
        * c_strong_density
    )

    if theme == "AI-Related Risk":

        score = proximity

    elif theme == "Pandemic Risk":

        score = probabilistic_union(
            c_strong_frequency,
            density_reinforcement,
        )

    elif theme in {
        "Climate / Environmental Risk",
        "Cybersecurity Risk",
        "Supply Chain Risk",
    }:

        score = probabilistic_union(
            c_strong_frequency,
            proximity,
            density_reinforcement,
        )

    else:

        raise RuntimeError(
            f"Unexpected theme: {theme!r}"
        )

    return float(
        np.clip(
            score,
            SCORE_MIN,
            SCORE_MAX,
        )
    )


# =============================================================================
# METRICS
# =============================================================================

def evaluate_metrics(
    y,
    score,
) -> dict:

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(y) != len(score):

        raise RuntimeError(
            "Label/score length mismatch."
        )

    if not np.isfinite(
        score
    ).all():

        raise RuntimeError(
            "Non-finite scores passed "
            "to evaluation."
        )

    if len(
        np.unique(y)
    ) < 2:

        return {
            "roc_auc": np.nan,
            "average_precision": np.nan,
        }

    return {
        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    score,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    y,
                    score,
                )
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "FINAL TEST — "
        "WEAK V3 CONTINUOUS FROZEN EVALUATION"
    )
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Frozen configuration
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("1. VERIFY FROZEN CONFIG")
    print("=" * 100)

    if (
        SELECTED_CANDIDATE
        != EXPECTED_CANDIDATE
    ):
        raise RuntimeError(
            "Selected candidate mismatch."
        )

    if (
        PROXIMITY_WINDOW
        != EXPECTED_WINDOW
    ):
        raise RuntimeError(
            "Proximity window mismatch."
        )

    if not np.isclose(
        DENSITY_REINFORCEMENT,
        EXPECTED_DENSITY_REINFORCEMENT,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Density reinforcement mismatch."
        )

    if not np.isclose(
        SCORE_MIN,
        EXPECTED_SCORE_MIN,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "SCORE_MIN mismatch."
        )

    if not np.isclose(
        SCORE_MAX,
        EXPECTED_SCORE_MAX,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "SCORE_MAX mismatch."
        )

    if FINAL_TEST_USED_FOR_SELECTION:

        raise RuntimeError(
            "Frozen configuration indicates "
            "that final-test data were used "
            "for specification selection."
        )

    print(
        f"\nVersion: {VERSION}"
    )

    print(
        f"Method: {METHOD}"
    )

    print(
        f"Model version: {MODEL_VERSION}"
    )

    print(
        f"Candidate: {SELECTED_CANDIDATE}"
    )

    print(
        f"Proximity window: "
        f"{PROXIMITY_WINDOW}"
    )

    print(
        f"Density reinforcement: "
        f"{DENSITY_REINFORCEMENT}"
    )

    print(
        f"Score range: "
        f"[{SCORE_MIN}, {SCORE_MAX}]"
    )

    print(
        "\n[OK] frozen configuration verified"
    )

    # -------------------------------------------------------------------------
    # Frozen gold
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("2. VERIFY FROZEN GOLD")
    print("=" * 100)

    if not GOLD_FILE.exists():

        raise FileNotFoundError(
            "Frozen gold not found:\n"
            f"{GOLD_FILE}"
        )

    actual_gold_hash = sha256_file(
        GOLD_FILE
    )

    if (
        actual_gold_hash
        != EXPECTED_GOLD_SHA256
    ):
        raise RuntimeError(
            "Frozen gold SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_GOLD_SHA256}\n"
            f"Found:    {actual_gold_hash}"
        )

    print(
        "\n[OK] frozen gold SHA-256 verified"
    )

    # -------------------------------------------------------------------------
    # Structural validation
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("3. STRUCTURAL CHECKS")
    print("=" * 100)

    gold = pd.read_csv(
        GOLD_FILE
    )

    required_columns = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "publication_date",
        "title",
        "body",
        "sample_block",
        "human_label",
    }

    missing = (
        required_columns
        - set(gold.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if len(gold) != EXPECTED_ROWS:

        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(gold)}."
        )

    n_articles = (
        gold[
            "article_id"
        ]
        .nunique()
    )

    if (
        n_articles
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ARTICLES} "
            f"unique articles, "
            f"found {n_articles}."
        )

    n_themes = (
        gold[
            "theme_id"
        ]
        .nunique()
    )

    if (
        n_themes
        != EXPECTED_THEMES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_THEMES} themes, "
            f"found {n_themes}."
        )

    if (
        gold[
            "final_test_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id."
        )

    if gold[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pairs."
        )

    actual_themes = set(
        gold[
            "theme_name"
        ]
        .dropna()
        .astype(str)
    )

    if (
        actual_themes
        != EXPECTED_THEME_LABELS
    ):
        raise RuntimeError(
            "Theme universe mismatch.\n"
            f"Found: {sorted(actual_themes)}"
        )

    theme_mapping = (
        gold[
            [
                "theme_id",
                "theme_name",
            ]
        ]
        .drop_duplicates()
    )

    if (
        theme_mapping[
            "theme_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "A theme_id maps to multiple "
            "theme names."
        )

    if (
        theme_mapping[
            "theme_name"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "A theme name maps to multiple "
            "theme IDs."
        )

    labels = pd.to_numeric(
        gold[
            "human_label"
        ],
        errors="raise",
    )

    if labels.isna().any():

        raise RuntimeError(
            "Missing human labels."
        )

    if not np.isfinite(
        labels.to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "Non-finite human labels."
        )

    if not set(
        labels.unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "human_label contains "
            "values outside {0,1}."
        )

    gold[
        "human_label"
    ] = labels.astype(int)

    positives = int(
        gold[
            "human_label"
        ]
        .sum()
    )

    negatives = int(
        len(gold)
        - positives
    )

    if (
        positives
        != EXPECTED_POSITIVES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_POSITIVES} "
            f"positive labels, "
            f"found {positives}."
        )

    if (
        negatives
        != EXPECTED_NEGATIVES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_NEGATIVES} "
            f"negative labels, "
            f"found {negatives}."
        )

    print(
        f"\nRows: {len(gold):,}"
    )

    print(
        f"Unique articles: "
        f"{n_articles:,}"
    )

    print(
        f"Themes: {n_themes}"
    )

    print(
        f"Gold negatives: "
        f"{negatives}"
    )

    print(
        f"Gold positives: "
        f"{positives}"
    )

    print(
        "\n[OK] structural checks passed"
    )

    # -------------------------------------------------------------------------
    # Frozen evidence extraction
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "4. EXTRACT FROZEN "
        "WEAK V3 EVIDENCE"
    )
    print("=" * 100)

    # Compatibility alias required by the frozen
    # development evidence-extraction interface.
    gold[
        "human_review_decision"
    ] = (
        gold[
            "human_label"
        ]
        .astype(int)
    )

    if not np.array_equal(
        gold[
            "human_review_decision"
        ].to_numpy(),
        gold[
            "human_label"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Label compatibility "
            "alias mismatch."
        )

    records = []

    for index, row in enumerate(
        gold.itertuples(
            index=False
        ),
        start=1,
    ):

        records.append(
            extract_evidence(
                row
            )
        )

        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/{len(gold)}"
            )

    evidence = pd.DataFrame(
        records
    )

    if len(evidence) != EXPECTED_ROWS:

        raise RuntimeError(
            "Evidence row count mismatch."
        )

    missing_evidence = (
        REQUIRED_EVIDENCE_COLUMNS
        - set(evidence.columns)
    )

    if missing_evidence:

        raise RuntimeError(
            "Missing evidence columns: "
            f"{sorted(missing_evidence)}"
        )

    if not np.array_equal(
        evidence[
            "article_id"
        ]
        .to_numpy(),
        gold[
            "article_id"
        ]
        .to_numpy(),
    ):
        raise RuntimeError(
            "Article alignment mismatch "
            "after evidence extraction."
        )

    if not np.array_equal(
        evidence[
            "theme_id"
        ]
        .to_numpy(),
        gold[
            "theme_id"
        ]
        .to_numpy(),
    ):
        raise RuntimeError(
            "Theme alignment mismatch "
            "after evidence extraction."
        )

    if not np.array_equal(
        evidence[
            "human_label"
        ]
        .to_numpy(),
        gold[
            "human_label"
        ]
        .to_numpy(),
    ):
        raise RuntimeError(
            "Human-label alignment mismatch "
            "after evidence extraction."
        )

    evidence_numeric_columns = [
        "strong_frequency",
        "strong_density_1000",
        "proximity_strength",
    ]

    for column in evidence_numeric_columns:

        evidence[
            column
        ] = pd.to_numeric(
            evidence[
                column
            ],
            errors="raise",
        )

        values = (
            evidence[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                f"Non-finite evidence "
                f"in {column}."
            )

        if (
            values < 0
        ).any():

            raise RuntimeError(
                f"Negative evidence "
                f"in {column}."
            )

    proximity_values = (
        evidence[
            "proximity_strength"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        proximity_values > 1
    ).any():

        raise RuntimeError(
            "proximity_strength "
            "outside [0,1]."
        )

    print(
        "\n[OK] frozen evidence extracted "
        "and validated"
    )

    # -------------------------------------------------------------------------
    # Frozen Candidate C
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "5. APPLY FROZEN CANDIDATE C"
    )
    print("=" * 100)

    scores = (
        evidence
        .apply(
            frozen_score_c,
            axis=1,
        )
        .to_numpy(
            dtype=float
        )
    )

    if (
        len(scores)
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            "Score row count mismatch."
        )

    if not np.isfinite(
        scores
    ).all():

        raise RuntimeError(
            "Non-finite scores detected."
        )

    if (
        (scores < SCORE_MIN).any()
        or
        (scores > SCORE_MAX).any()
    ):
        raise RuntimeError(
            "Scores outside frozen range."
        )

    n_zero = int(
        (scores == 0).sum()
    )

    zero_rate = float(
        np.mean(
            scores == 0
        )
    )

    print(
        f"\nScores generated: "
        f"{len(scores):,}"
    )

    print(
        f"Score min:    "
        f"{scores.min():.6f}"
    )

    print(
        f"Score mean:   "
        f"{scores.mean():.6f}"
    )

    print(
        f"Score median: "
        f"{np.median(scores):.6f}"
    )

    print(
        f"Score max:    "
        f"{scores.max():.6f}"
    )

    print(
        f"Zero scores:  "
        f"{n_zero:,} / "
        f"{len(scores):,} "
        f"({zero_rate:.2%})"
    )

    # -------------------------------------------------------------------------
    # Final-test ranking metrics
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "6. FINAL TEST — "
        "CONTINUOUS METRICS"
    )
    print("=" * 100)

    y = (
        gold[
            "human_label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    overall = evaluate_metrics(
        y,
        scores,
    )

    print(
        f"\nROC-AUC:           "
        f"{overall['roc_auc']:.6f}"
    )

    print(
        f"Average Precision: "
        f"{overall['average_precision']:.6f}"
    )

    print(
        "\nDevelopment reference "
        "(not refitted):"
    )

    print(
        f"Development ROC-AUC: "
        f"{DEVELOPMENT_ROC_AUC:.6f}"
    )

    print(
        f"Development AP:      "
        f"{DEVELOPMENT_AVERAGE_PRECISION:.6f}"
    )

    # -------------------------------------------------------------------------
    # Evaluation frame
    # -------------------------------------------------------------------------

    evaluation = gold[
        [
            "final_test_id",
            "article_id",
            "theme_id",
            "theme_name",
            "publication_date",
            "sample_block",
            "human_label",
        ]
    ].copy()

    evaluation[
        "weak_continuous_score"
    ] = scores

    evaluation[
        "method"
    ] = METHOD

    evaluation[
        "model_version"
    ] = MODEL_VERSION

    evaluation[
        "score_interpretation"
    ] = (
        "continuous weak evidence relevance; "
        "not calibrated probability"
    )

    # -------------------------------------------------------------------------
    # By theme
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("7. PERFORMANCE BY THEME")
    print("=" * 100)

    theme_records = []

    for (
        theme_id,
        theme_name
    ), part in evaluation.groupby(
        [
            "theme_id",
            "theme_name",
        ],
        sort=True,
    ):

        metrics = evaluate_metrics(
            part[
                "human_label"
            ],
            part[
                "weak_continuous_score"
            ],
        )

        theme_records.append(
            {
                "theme_id":
                    theme_id,

                "theme_name":
                    theme_name,

                "n":
                    int(
                        len(part)
                    ),

                "positives":
                    int(
                        part[
                            "human_label"
                        ]
                        .sum()
                    ),

                "prevalence":
                    float(
                        part[
                            "human_label"
                        ]
                        .mean()
                    ),

                "roc_auc":
                    metrics[
                        "roc_auc"
                    ],

                "average_precision":
                    metrics[
                        "average_precision"
                    ],

                "mean_score":
                    float(
                        part[
                            "weak_continuous_score"
                        ]
                        .mean()
                    ),

                "zero_score_rate":
                    float(
                        (
                            part[
                                "weak_continuous_score"
                            ]
                            == 0
                        )
                        .mean()
                    ),
            }
        )

    theme_table = pd.DataFrame(
        theme_records
    )

    print(
        "\n"
        + theme_table
        .round(6)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # By sample block
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "8. PERFORMANCE BY SAMPLE BLOCK"
    )
    print("=" * 100)

    block_records = []

    for block, part in evaluation.groupby(
        "sample_block",
        sort=True,
    ):

        metrics = evaluate_metrics(
            part[
                "human_label"
            ],
            part[
                "weak_continuous_score"
            ],
        )

        block_records.append(
            {
                "sample_block":
                    block,

                "n":
                    int(
                        len(part)
                    ),

                "positives":
                    int(
                        part[
                            "human_label"
                        ]
                        .sum()
                    ),

                "prevalence":
                    float(
                        part[
                            "human_label"
                        ]
                        .mean()
                    ),

                "roc_auc":
                    metrics[
                        "roc_auc"
                    ],

                "average_precision":
                    metrics[
                        "average_precision"
                    ],
            }
        )

    block_table = pd.DataFrame(
        block_records
    )

    print(
        "\n"
        + block_table
        .round(6)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Score distribution by gold label
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "9. SCORE DISTRIBUTION BY GOLD"
    )
    print("=" * 100)

    distribution_records = []

    for label, part in evaluation.groupby(
        "human_label",
        sort=True,
    ):

        values = (
            part[
                "weak_continuous_score"
            ]
        )

        distribution_records.append(
            {
                "gold":
                    int(label),

                "n":
                    int(
                        len(values)
                    ),

                "mean":
                    float(
                        values.mean()
                    ),

                "median":
                    float(
                        values.median()
                    ),

                "p75":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),

                "p90":
                    float(
                        values.quantile(
                            0.90
                        )
                    ),

                "max":
                    float(
                        values.max()
                    ),

                "zero_rate":
                    float(
                        (
                            values == 0
                        )
                        .mean()
                    ),
            }
        )

    distribution_table = (
        pd.DataFrame(
            distribution_records
        )
    )

    print(
        "\n"
        + distribution_table
        .round(6)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Save frozen final-test scores
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "10. SAVE AND LOCK FINAL SCORES"
    )
    print("=" * 100)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = [
        path
        for path in [
            PREDICTIONS_FILE,
            MANIFEST_FILE,
        ]
        if path.exists()
    ]

    if existing:

        raise FileExistsError(
            "Final Weak V3 continuous "
            "artifacts already exist; "
            "refusing silent overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    evaluation.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    predictions_hash = sha256_file(
        PREDICTIONS_FILE
    )

    manifest = {
        "status":
            "FINAL_TEST_EVALUATION_FROZEN",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "purpose": (
            "Untouched final-test evaluation "
            "of the frozen Weak V3 continuous "
            "relevance score."
        ),

        "method":
            METHOD,

        "model_version":
            MODEL_VERSION,

        "version":
            VERSION,

        "selected_candidate":
            SELECTED_CANDIDATE,

        "score_interpretation": (
            "Continuous weak-evidence relevance "
            "score in [0,1]; not a calibrated "
            "probability."
        ),

        "frozen_parameters": {
            "proximity_window":
                PROXIMITY_WINDOW,

            "density_reinforcement":
                DENSITY_REINFORCEMENT,

            "score_min":
                SCORE_MIN,

            "score_max":
                SCORE_MAX,
        },

        "development_reference": {
            "rows":
                DEVELOPMENT_N,

            "positives":
                DEVELOPMENT_POSITIVES,

            "negatives":
                DEVELOPMENT_NEGATIVES,

            "roc_auc":
                DEVELOPMENT_ROC_AUC,

            "average_precision":
                DEVELOPMENT_AVERAGE_PRECISION,
        },

        "gold": {
            "path":
                str(
                    GOLD_FILE
                ),

            "sha256":
                actual_gold_hash,

            "rows":
                int(
                    len(gold)
                ),

            "unique_articles":
                int(
                    gold[
                        "article_id"
                    ]
                    .nunique()
                ),

            "positives":
                positives,

            "negatives":
                negatives,
        },

        "final_test_metrics": {
            "roc_auc":
                overall[
                    "roc_auc"
                ],

            "average_precision":
                overall[
                    "average_precision"
                ],

            "score_min":
                float(
                    scores.min()
                ),

            "score_mean":
                float(
                    scores.mean()
                ),

            "score_median":
                float(
                    np.median(
                        scores
                    )
                ),

            "score_max":
                float(
                    scores.max()
                ),

            "zero_score_rate":
                zero_rate,
        },

        "methodological_lock": {
            "training_performed":
                False,

            "refit_performed":
                False,

            "calibration_performed":
                False,

            "threshold_selection_performed":
                False,

            "final_test_tuning_performed":
                False,

            "final_test_used_for_selection":
                False,

            "score_specification_changed":
                False,
        },
    }

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    manifest_hash = sha256_file(
        MANIFEST_FILE
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print(
        f"\nScores written to:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"\nPrediction SHA-256:\n"
        f"{predictions_hash}"
    )

    print(
        f"\nManifest written to:\n"
        f"{MANIFEST_FILE}"
    )

    print(
        f"\nManifest SHA-256:\n"
        f"{manifest_hash}"
    )

    print()
    print("=" * 100)
    print(
        "WEAK V3 CONTINUOUS FINAL-TEST "
        "EVALUATION SUCCESSFULLY FROZEN"
    )
    print("=" * 100)

    print(
        f"""
Rows evaluated:       {len(evaluation):,}
Unique articles:      {evaluation['article_id'].nunique():,}
Gold positives:       {positives:,}
Gold negatives:       {negatives:,}

ROC-AUC:              {overall['roc_auc']:.6f}
Average Precision:    {overall['average_precision']:.6f}

No training, refitting, calibration, threshold selection,
or final-test tuning was performed.
"""
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            "\nFATAL ERROR:"
        )

        print(
            repr(exc)
        )

        sys.exit(1)