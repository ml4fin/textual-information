#!/usr/bin/env python3

"""
Diagnostic analysis of rule-level classification errors.

The script uses human-reviewed records to investigate two rule outcomes:

1. single_theme_signal
   Compares true positives (TP) and false positives (FP) and estimates the
   empirical precision associated with each observed theme-evidence signal.

2. no_theme_signal
   Compares false negatives (FN) and true negatives (TN) and examines which
   risk-evidence signals are disproportionately associated with missed
   positive cases.

The analysis is descriptive and is intended to support qualitative inspection
and refinement of the rule-based labeling procedure.
"""

import os
from collections import Counter

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = os.getenv(
    "GOLD_REVIEW_FILE",
    "gold_review_high.csv",
)

OUT_SINGLE = os.getenv(
    "SINGLE_THEME_DISCRIMINATION_OUTPUT",
    "single_theme_signal_discrimination.csv",
)

OUT_NO_THEME = os.getenv(
    "NO_THEME_DISCRIMINATION_OUTPUT",
    "no_theme_signal_discrimination.csv",
)

TRUE_COL = "human_review_decision"
PRED_COL = "suggested_label"
THEME_COL = "theme_name"


# =============================================================================
# HELPERS
# =============================================================================

def split_evidence(value) -> list[str]:
    """Split pipe-delimited evidence into normalized individual signals."""

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return []

    return [
        signal.strip().lower()
        for signal in value.split("|")
        if signal.strip()
    ]


def classification_type(row) -> str:
    """Return TP, FP, FN, or TN from the human and predicted labels."""

    true_label = int(
        row[TRUE_COL]
    )

    predicted_label = int(
        row[PRED_COL]
    )

    if (
        true_label == 1
        and predicted_label == 1
    ):
        return "TP"

    if (
        true_label == 0
        and predicted_label == 1
    ):
        return "FP"

    if (
        true_label == 1
        and predicted_label == 0
    ):
        return "FN"

    if (
        true_label == 0
        and predicted_label == 0
    ):
        return "TN"

    return "UNKNOWN"


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

df["class_type"] = df.apply(
    classification_type,
    axis=1,
)


# =============================================================================
# SAMPLE SUMMARY
# =============================================================================

print("=" * 110)
print("RULE DISCRIMINATION ANALYSIS")
print("=" * 110)

print(
    f"\nTotal records: {len(df):,}"
)


# =============================================================================
# SINGLE_THEME_SIGNAL: TP VS FP
# =============================================================================

single = df[
    df["rule_reason"]
    == "single_theme_signal"
].copy()

print("\n" + "=" * 110)
print("SINGLE_THEME_SIGNAL: DISTRIBUTION")
print("=" * 110)

print(
    single["class_type"]
    .value_counts()
    .to_string()
)

print("\nBy theme:")

print(
    pd.crosstab(
        single[THEME_COL],
        single["class_type"],
    ).to_string()
)


# =============================================================================
# THEME-EVIDENCE DISCRIMINATION
# =============================================================================

single_rows = []

for theme, group in single.groupby(
    THEME_COL
):

    true_positives = group[
        group["class_type"] == "TP"
    ]

    false_positives = group[
        group["class_type"] == "FP"
    ]

    tp_counter = Counter()
    fp_counter = Counter()

    for value in true_positives[
        "theme_evidence"
    ]:
        tp_counter.update(
            split_evidence(value)
        )

    for value in false_positives[
        "theme_evidence"
    ]:
        fp_counter.update(
            split_evidence(value)
        )

    signals = sorted(
        set(tp_counter)
        | set(fp_counter)
    )

    for signal in signals:

        tp_count = tp_counter.get(
            signal,
            0,
        )

        fp_count = fp_counter.get(
            signal,
            0,
        )

        total = (
            tp_count
            + fp_count
        )

        empirical_precision = (
            tp_count / total
            if total > 0
            else 0.0
        )

        single_rows.append(
            {
                "theme_name":
                    theme,

                "signal":
                    signal,

                "TP_count":
                    tp_count,

                "FP_count":
                    fp_count,

                "total_positive_predictions":
                    total,

                "empirical_precision":
                    empirical_precision,
            }
        )

single_signal_df = pd.DataFrame(
    single_rows
)

if not single_signal_df.empty:
    single_signal_df = (
        single_signal_df.sort_values(
            [
                "theme_name",
                "empirical_precision",
                "total_positive_predictions",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )


# =============================================================================
# SINGLE_THEME_SIGNAL REPORT
# =============================================================================

print("\n" + "=" * 110)
print("SINGLE_THEME_SIGNAL: TP VS FP BY SIGNAL")
print("=" * 110)

if single_signal_df.empty:
    print(
        "No theme-evidence signals available."
    )

else:
    for theme, group in (
        single_signal_df.groupby(
            "theme_name"
        )
    ):

        print()
        print("#" * 90)
        print(theme)
        print("#" * 90)

        print(
            group.to_string(
                index=False
            )
        )


print("\n" + "=" * 110)
print("SINGLE_THEME_SIGNAL: TP AND FP EXAMPLES")
print("=" * 110)

for theme, group in single.groupby(
    THEME_COL
):

    print()
    print("#" * 100)
    print(theme)
    print("#" * 100)

    for class_type in [
        "TP",
        "FP",
    ]:

        subset = group[
            group["class_type"]
            == class_type
        ]

        print(
            f"\n--- {class_type} "
            f"({len(subset):,}) ---"
        )

        for _, row in subset.iterrows():

            print(
                f"\nID {row['annotation_id']} | "
                f"{row['theme_evidence']}"
            )

            print(
                row["title"]
            )


# =============================================================================
# NO_THEME_SIGNAL: FN VS TN
# =============================================================================

no_theme = df[
    df["rule_reason"]
    == "no_theme_signal"
].copy()

print("\n" + "=" * 110)
print("NO_THEME_SIGNAL: DISTRIBUTION")
print("=" * 110)

print(
    no_theme["class_type"]
    .value_counts()
    .to_string()
)

print("\nBy theme:")

print(
    pd.crosstab(
        no_theme[THEME_COL],
        no_theme["class_type"],
    ).to_string()
)


# =============================================================================
# RISK-EVIDENCE DISCRIMINATION
# =============================================================================

no_theme_rows = []

for theme, group in no_theme.groupby(
    THEME_COL
):

    false_negatives = group[
        group["class_type"] == "FN"
    ]

    true_negatives = group[
        group["class_type"] == "TN"
    ]

    fn_counter = Counter()
    tn_counter = Counter()

    for value in false_negatives[
        "risk_evidence"
    ]:
        fn_counter.update(
            split_evidence(value)
        )

    for value in true_negatives[
        "risk_evidence"
    ]:
        tn_counter.update(
            split_evidence(value)
        )

    signals = sorted(
        set(fn_counter)
        | set(tn_counter)
    )

    for signal in signals:

        fn_count = fn_counter.get(
            signal,
            0,
        )

        tn_count = tn_counter.get(
            signal,
            0,
        )

        total = (
            fn_count
            + tn_count
        )

        fn_share = (
            fn_count / total
            if total > 0
            else 0.0
        )

        no_theme_rows.append(
            {
                "theme_name":
                    theme,

                "risk_signal":
                    signal,

                "FN_count":
                    fn_count,

                "TN_count":
                    tn_count,

                "total_occurrences":
                    total,

                "FN_share":
                    fn_share,
            }
        )

no_theme_signal_df = pd.DataFrame(
    no_theme_rows
)

if not no_theme_signal_df.empty:
    no_theme_signal_df = (
        no_theme_signal_df.sort_values(
            [
                "theme_name",
                "FN_share",
                "total_occurrences",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )


# =============================================================================
# NO_THEME_SIGNAL REPORT
# =============================================================================

print("\n" + "=" * 110)
print("NO_THEME_SIGNAL: FN VS TN BY RISK EVIDENCE")
print("=" * 110)

if no_theme_signal_df.empty:
    print(
        "No risk-evidence signals available."
    )

else:
    for theme, group in (
        no_theme_signal_df.groupby(
            "theme_name"
        )
    ):

        print()
        print("#" * 90)
        print(theme)
        print("#" * 90)

        print(
            group.to_string(
                index=False
            )
        )


print("\n" + "=" * 110)
print("NO_THEME_SIGNAL: FALSE-NEGATIVE EXAMPLES")
print("=" * 110)

for theme, group in no_theme.groupby(
    THEME_COL
):

    false_negatives = group[
        group["class_type"] == "FN"
    ]

    if false_negatives.empty:
        continue

    print()
    print("#" * 100)
    print(theme)
    print("#" * 100)

    for _, row in (
        false_negatives.iterrows()
    ):

        print(
            f"\nID {row['annotation_id']}"
        )

        print(
            row["title"]
        )

        print(
            "Risk:",
            row["risk_evidence"],
        )


# =============================================================================
# SAVE
# =============================================================================

single_signal_df.to_csv(
    OUT_SINGLE,
    index=False,
)

no_theme_signal_df.to_csv(
    OUT_NO_THEME,
    index=False,
)

print("\n" + "=" * 110)
print("FILES SAVED")
print("=" * 110)

print(OUT_SINGLE)
print(OUT_NO_THEME)

print(
    "\nAnalysis completed."
)