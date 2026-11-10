#!/usr/bin/env python3

"""
WEAK SUPERVISION V3 — DEVELOPMENT EVALUATION
============================================

Purpose
-------
Evaluate the successive Weak Supervision V3 development stages on the
frozen manually reviewed development set:

    baseline
        -> A strong/weak refinement
        -> proximity rescue (window = 50)
        -> final Weak V3 with AI strict rule

IMPORTANT
---------
Human review labels are used here for development-stage evaluation.

This script is NOT a final-test evaluation and does NOT produce
final-test performance estimates.

The final Weak V3 rules must be frozen after this development stage
before application to the untouched final test.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "WEAK_V3_DEVELOPMENT_DATASET",
        "gold_review_high.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "WEAK_V3_DEVELOPMENT_EVAL_DIR",
        "weak_v3_development_evaluation",
    )
)

OUTPUT_PREDICTIONS = (
    OUTPUT_DIR
    / "weak_v3_development_predictions.csv"
)

OUTPUT_METRICS = (
    OUTPUT_DIR
    / "weak_v3_development_metrics.csv"
)

OUTPUT_THEME_METRICS = (
    OUTPUT_DIR
    / "weak_v3_development_metrics_by_theme.csv"
)

TRUE_COL = "human_review_decision"
PRED_COL = "suggested_label"
THEME_COL = "theme_name"

EXPECTED_N = 485
EXPECTED_NEGATIVES = 337
EXPECTED_POSITIVES = 148

WINDOW = 50


# =============================================================================
# A_STRONG_WEAK SIGNALS
# =============================================================================

STRONG_SIGNALS = {
    "Climate / Environmental Risk": {
        "carbon emissions",
        "wildfire",
        "wildfires",
        "extreme weather",
        "global warming",
        "climate change",
        "flooding",
        "floods",
    },

    "Cybersecurity Risk": {
        "cyberattacks",
        "hackers",
        "cyberattack",
    },

    "Pandemic Risk": {
        "coronavirus",
        "covid-19",
        "covid",
    },

    "Supply Chain Risk": {
        "supply chain",
        "supply chains",
    },
}


AMBIGUOUS_SIGNALS = {
    "Climate / Environmental Risk": {
        "drought",
        "flood",
        "heatwave",
    },

    "Cybersecurity Risk": {
        "hacked",
        "hacking",
        "hacker",
        "cybersecurity",
    },

    "Pandemic Risk": {
        "pandemic",
        "epidemic",
    },

    "Supply Chain Risk": set(),
}


# =============================================================================
# PROXIMITY SIGNALS
# =============================================================================

CLIMATE_THEME_SIGNALS = [
    "emission",
    "emissions",
    "carbon",
    "pollution",
    "hurricane",
    "hurricanes",
    "storm",
    "storms",
    "weather",
    "climate",
]


CYBER_OBJECT_SIGNALS = [
    "computer",
    "computers",
    "device",
    "devices",
    "software",
    "internet",
    "phone",
    "phones",
    "iphone",
]


CYBER_CONTEXT_SIGNALS = [
    "security",
    "privacy",
    "vulnerability",
    "vulnerabilities",
    "fraud",
    "unauthorized access",
    "unauthorised access",
    "security incident",
    "security incidents",
]


SUPPLY_OBJECT_SIGNALS = [
    "port",
    "ports",
    "oil",
    "food",
    "refinery",
    "refineries",
    "energy",
    "freight",
    "shipping",
    "delivery",
    "deliveries",
]


SUPPLY_DISRUPTION_SIGNALS = [
    "disruption",
    "disruptions",
    "disrupted",
    "shutdown",
    "shut",
    "shortage",
    "shortages",
    "recall",
    "blackout",
    "blackouts",
]


# =============================================================================
# AI STRICT SIGNALS
# =============================================================================

AI_STRICT_SIGNALS = [
    "regulator",
    "regulators",
    "regulation",
    "regulated",
    "rules",
    "law",
    "laws",
    "legal",

    "copyright",
    "copyrighted",
    "lawsuit",
    "lawsuits",
    "sue",
    "sues",
    "sued",
    "trade secret",
    "trade secrets",
    "stealing",
    "stolen",

    "crash",
    "crashes",
    "collision",
    "collisions",
    "accident",
    "accidents",
    "safety",
    "unsafe",

    "extremist",
    "extremism",
    "hateful",
    "offensive",
    "abuse",
    "misuse",

    "concern",
    "concerns",
    "fear",
    "fears",
    "panic",
    "danger",
    "dangerous",
    "risk",
    "risks",
    "threat",
    "threats",

    "weapon",
    "weapons",
    "killer",
    "lethal",

    "shut down",
    "shutdown",
    "suspended",
    "suspension",
    "investigate",
    "investigation",

    "antitrust",
]


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalize(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def has_risk_evidence(value) -> bool:

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value not in {
        "",
        "nan",
        "none",
    }


def tokenize(text) -> list[str]:

    if pd.isna(text):
        return []

    return re.findall(
        r"[a-z0-9\-]+",
        str(text).lower(),
    )


def phrase_positions(
    tokens: list[str],
    phrase: str,
) -> list[int]:

    target = tokenize(phrase)

    if not target:
        return []

    n = len(target)

    return [
        i
        for i in range(
            len(tokens) - n + 1
        )
        if tokens[i:i + n] == target
    ]


def minimum_distance(
    tokens: list[str],
    phrase1: str,
    phrase2: str,
):

    positions_1 = phrase_positions(
        tokens,
        phrase1,
    )

    positions_2 = phrase_positions(
        tokens,
        phrase2,
    )

    if (
        not positions_1
        or not positions_2
    ):
        return None

    return min(
        abs(a - b)
        for a in positions_1
        for b in positions_2
    )


def within_window(
    tokens: list[str],
    signals1,
    signals2,
    window: int,
) -> bool:

    for signal_1 in signals1:

        for signal_2 in signals2:

            distance = minimum_distance(
                tokens,
                signal_1,
                signal_2,
            )

            if (
                distance is not None
                and distance <= window
            ):
                return True

    return False


def ai_strict_match(
    tokens: list[str],
    theme_phrase: str,
    window: int,
) -> bool:

    if not theme_phrase:
        return False

    for signal in AI_STRICT_SIGNALS:

        distance = minimum_distance(
            tokens,
            theme_phrase,
            signal,
        )

        if (
            distance is not None
            and distance <= window
        ):
            return True

    return False


def parse_risk_evidence(
    value,
) -> list[str]:

    if pd.isna(value):
        return []

    value = (
        str(value)
        .lower()
        .strip()
    )

    if value in {
        "",
        "nan",
        "none",
    }:
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
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

    return {
        "n":
            int(len(y_true)),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
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
    }


# =============================================================================
# LOAD AND VALIDATE DEVELOPMENT DATA
# =============================================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Development dataset not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        TRUE_COL,
        PRED_COL,
        THEME_COL,
        "title",
        "body",
        "rule_reason",
        "theme_evidence",
        "risk_evidence",
    }

    missing = (
        required
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
    )

    df[PRED_COL] = pd.to_numeric(
        df[PRED_COL],
        errors="raise",
    )

    if (
        df[TRUE_COL].isna().any()
        or df[PRED_COL].isna().any()
    ):

        raise RuntimeError(
            "Missing development labels detected."
        )

    if not set(
        df[TRUE_COL].unique()
    ).issubset(
        {0, 1}
    ):

        raise RuntimeError(
            f"{TRUE_COL} must contain only 0/1."
        )

    if not set(
        df[PRED_COL].unique()
    ).issubset(
        {0, 1}
    ):

        raise RuntimeError(
            f"{PRED_COL} must contain only 0/1."
        )

    df[TRUE_COL] = (
        df[TRUE_COL]
        .astype(int)
    )

    df[PRED_COL] = (
        df[PRED_COL]
        .astype(int)
    )

    n_positive = int(
        df[TRUE_COL].sum()
    )

    n_negative = int(
        len(df) - n_positive
    )

    if (
        n_positive != EXPECTED_POSITIVES
        or n_negative != EXPECTED_NEGATIVES
    ):

        raise RuntimeError(
            "Unexpected development label distribution.\n"
            f"Expected negatives/positives: "
            f"{EXPECTED_NEGATIVES}/"
            f"{EXPECTED_POSITIVES}\n"
            f"Found negatives/positives: "
            f"{n_negative}/{n_positive}"
        )

    if df[THEME_COL].isna().any():

        raise RuntimeError(
            "Missing theme names detected."
        )

    return df


# =============================================================================
# WEAK V3 DEVELOPMENT STAGES
# =============================================================================

def apply_weak_v3(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["_text"] = (
        df["title"]
        .fillna("")
        .astype(str)
        + " "
        + df["body"]
        .fillna("")
        .astype(str)
    )

    df["_tokens"] = (
        df["_text"]
        .apply(tokenize)
    )

    df["_theme_signal"] = (
        df["theme_evidence"]
        .apply(normalize)
    )

    # Baseline generated by the original gold-review rules.
    df["pred_baseline"] = (
        df[PRED_COL]
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # A: strong / ambiguous signal refinement
    # -------------------------------------------------------------------------

    df["pred_A"] = (
        df["pred_baseline"]
    )

    for idx, row in df.iterrows():

        if (
            row["rule_reason"]
            != "single_theme_signal"
        ):
            continue

        theme = row[THEME_COL]
        signal = row["_theme_signal"]

        strong = STRONG_SIGNALS.get(
            theme,
            set(),
        )

        ambiguous = AMBIGUOUS_SIGNALS.get(
            theme,
            set(),
        )

        if signal in strong:

            df.at[
                idx,
                "pred_A",
            ] = 1

        elif signal in ambiguous:

            df.at[
                idx,
                "pred_A",
            ] = int(
                has_risk_evidence(
                    row["risk_evidence"]
                )
            )

    # -------------------------------------------------------------------------
    # Proximity rescue
    # -------------------------------------------------------------------------

    df["pred_proximity50"] = (
        df["pred_A"]
    )

    for idx, row in df.iterrows():

        if (
            row["rule_reason"]
            != "no_theme_signal"
        ):
            continue

        theme = row[THEME_COL]
        tokens = row["_tokens"]

        activated = False

        if (
            theme
            == "Climate / Environmental Risk"
        ):

            risk_terms = parse_risk_evidence(
                row["risk_evidence"]
            )

            if risk_terms:

                activated = within_window(
                    tokens,
                    CLIMATE_THEME_SIGNALS,
                    risk_terms,
                    WINDOW,
                )

        elif (
            theme
            == "Cybersecurity Risk"
        ):

            activated = within_window(
                tokens,
                CYBER_OBJECT_SIGNALS,
                CYBER_CONTEXT_SIGNALS,
                WINDOW,
            )

        elif (
            theme
            == "Supply Chain Risk"
        ):

            activated = within_window(
                tokens,
                SUPPLY_OBJECT_SIGNALS,
                SUPPLY_DISRUPTION_SIGNALS,
                WINDOW,
            )

        if activated:

            df.at[
                idx,
                "pred_proximity50",
            ] = 1

    # -------------------------------------------------------------------------
    # Final Weak V3: proximity result + AI strict rule
    # -------------------------------------------------------------------------

    df["pred_final_weak"] = (
        df["pred_proximity50"]
    )

    for idx, row in df.iterrows():

        if (
            row[THEME_COL]
            != "AI-Related Risk"
        ):
            continue

        activated = ai_strict_match(
            row["_tokens"],
            row["_theme_signal"],
            WINDOW,
        )

        if activated:

            df.at[
                idx,
                "pred_final_weak",
            ] = 1

    prediction_columns = [
        "pred_baseline",
        "pred_A",
        "pred_proximity50",
        "pred_final_weak",
    ]

    for column in prediction_columns:

        values = set(
            df[column]
            .astype(int)
            .unique()
        )

        if not values.issubset(
            {0, 1}
        ):

            raise RuntimeError(
                f"{column} contains "
                f"non-binary values: {values}"
            )

        df[column] = (
            df[column]
            .astype(int)
        )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "WEAK SUPERVISION V3 — "
        "DEVELOPMENT EVALUATION"
    )
    print("=" * 120)

    df = load_data()

    print(
        f"\nDevelopment observations: "
        f"{len(df):,}"
    )

    print(
        f"Gold negatives: "
        f"{int((df[TRUE_COL] == 0).sum()):,}"
    )

    print(
        f"Gold positives: "
        f"{int((df[TRUE_COL] == 1).sum()):,}"
    )

    df = apply_weak_v3(
        df
    )

    scenarios = {
        "baseline":
            "pred_baseline",

        "A_strong_weak":
            "pred_A",

        "A_proximity_50":
            "pred_proximity50",

        "FINAL_WEAK_V3":
            "pred_final_weak",
    }

    # -------------------------------------------------------------------------
    # Global metrics
    # -------------------------------------------------------------------------

    global_rows = []

    for scenario, column in scenarios.items():

        result = calculate_metrics(
            df[TRUE_COL],
            df[column],
        )

        result["scenario"] = scenario

        global_rows.append(
            result
        )

    results = pd.DataFrame(
        global_rows
    )[
        [
            "scenario",
            "n",
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
        ]
    ]

    print()
    print("GLOBAL DEVELOPMENT METRICS")
    print("-" * 120)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # -------------------------------------------------------------------------
    # Theme-level diagnostics
    # -------------------------------------------------------------------------

    theme_rows = []

    for scenario, column in scenarios.items():

        for theme, group in df.groupby(
            THEME_COL,
            sort=True,
        ):

            result = calculate_metrics(
                group[TRUE_COL],
                group[column],
            )

            result["scenario"] = scenario
            result["theme_name"] = theme

            theme_rows.append(
                result
            )

    theme_metrics = pd.DataFrame(
        theme_rows
    )

    for metric_name in [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]:

        print()
        print("=" * 120)
        print(
            f"{metric_name.upper()} BY THEME"
        )
        print("=" * 120)

        print(
            theme_metrics
            .pivot(
                index="theme_name",
                columns="scenario",
                values=metric_name,
            )
            .round(4)
            .to_string()
        )

    # -------------------------------------------------------------------------
    # Final-stage error diagnostics
    # -------------------------------------------------------------------------

    df["final_error_type"] = np.select(
        [
            (
                (df[TRUE_COL] == 1)
                & (df["pred_final_weak"] == 0)
            ),
            (
                (df[TRUE_COL] == 0)
                & (df["pred_final_weak"] == 1)
            ),
        ],
        [
            "FN",
            "FP",
        ],
        default="CORRECT",
    )

    print()
    print("=" * 120)
    print(
        "FINAL WEAK V3 DEVELOPMENT ERRORS BY THEME"
    )
    print("=" * 120)

    error_table = (
        df[
            df["final_error_type"]
            != "CORRECT"
        ]
        .groupby(
            [
                THEME_COL,
                "final_error_type",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        error_table.to_string()
    )

    # -------------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    theme_metrics.to_csv(
        OUTPUT_THEME_METRICS,
        index=False,
    )

    output_columns = [
        "annotation_id",
        "article_id",
        "theme_name",
        "title",
        TRUE_COL,
        "pred_baseline",
        "pred_A",
        "pred_proximity50",
        "pred_final_weak",
        "final_error_type",
        "rule_reason",
        "theme_evidence",
        "risk_evidence",
    ]

    existing_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    df[
        existing_columns
    ].to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print()
    print("=" * 120)
    print("FILES SAVED")
    print("=" * 120)

    print(
        OUTPUT_METRICS
    )

    print(
        OUTPUT_THEME_METRICS
    )

    print(
        OUTPUT_PREDICTIONS
    )

    print()
    print(
        "Development evaluation completed."
    )


if __name__ == "__main__":
    main()