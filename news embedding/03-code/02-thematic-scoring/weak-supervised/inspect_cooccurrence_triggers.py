#!/usr/bin/env python3

"""
Audit of lexical triggers used by the co-occurrence rules.

The script examines records changed by the combined A_plus_all scenario and
reconstructs which lexical signals activated the Climate, Cybersecurity, and
Supply Chain co-occurrence rules.

For each trigger, it reports how often the signal is associated with:
- FIXED_ERROR: an error corrected by the scenario;
- NEW_ERROR: an error introduced by the scenario.

It also computes:

    net_gain = fixed_errors - new_errors

and an empirical precision within the set of changed records:

    fixed_errors / (fixed_errors + new_errors)

This is a diagnostic analysis of rule behavior and does not modify the
underlying classification data.
"""

import os
import re
from collections import Counter

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

GOLD_FILE = os.getenv(
    "GOLD_REVIEW_FILE",
    "gold_review_high.csv",
)

CHANGES_FILE = os.getenv(
    "COOCCURRENCE_CHANGES_FILE",
    "cooccurrence_rule_changes.csv",
)

OUTPUT_FILE = os.getenv(
    "COOCCURRENCE_TRIGGER_AUDIT_OUTPUT",
    "cooccurrence_trigger_audit.csv",
)

SUMMARY_FILE = os.getenv(
    "COOCCURRENCE_TRIGGER_SUMMARY_OUTPUT",
    "cooccurrence_trigger_summary.csv",
)

SCENARIO_NAME = "A_plus_all"


# =============================================================================
# CO-OCCURRENCE SIGNALS
# =============================================================================

# These patterns must remain aligned with the signals used in the
# corresponding co-occurrence-rule simulation.

CLIMATE_THEME_SIGNALS = [
    r"\bemissions?\b",
    r"\bcarbon\b",
    r"\bpollution\b",
    r"\bhurricane\b",
    r"\bhurricanes\b",
    r"\bstorm\b",
    r"\bstorms\b",
    r"\bweather\b",
    r"\bclimate\b",
]

CYBER_OBJECT_SIGNALS = [
    r"\bcomputer\b",
    r"\bcomputers\b",
    r"\bdevice\b",
    r"\bdevices\b",
    r"\bsoftware\b",
    r"\binternet\b",
    r"\bphone\b",
    r"\bphones\b",
    r"\biphone\b",
]

CYBER_CONTEXT_SIGNALS = [
    r"\bsecurity\b",
    r"\bprivacy\b",
    r"\bvulnerability\b",
    r"\bvulnerabilities\b",
    r"\bfraud\b",
    r"\bunauthori[sz]ed access\b",
    r"\bsecurity incident\b",
    r"\bsecurity incidents\b",
]

SUPPLY_OBJECT_SIGNALS = [
    r"\bport\b",
    r"\bports\b",
    r"\boil\b",
    r"\bfood\b",
    r"\brefinery\b",
    r"\brefineries\b",
    r"\benergy\b",
    r"\bfreight\b",
    r"\bshipping\b",
    r"\bdelivery\b",
    r"\bdeliveries\b",
]

SUPPLY_DISRUPTION_SIGNALS = [
    r"\bdisruption\b",
    r"\bdisruptions\b",
    r"\bdisrupted\b",
    r"\bshutdown\b",
    r"\bshut\b",
    r"\bshortage\b",
    r"\bshortages\b",
    r"\brecall\b",
    r"\bblackout\b",
    r"\bblackouts\b",
]


# =============================================================================
# HELPERS
# =============================================================================

def clean_pattern(pattern: str) -> str:
    """Convert a regex pattern into a more readable signal label."""

    label = pattern.replace(
        r"\b",
        "",
    )

    return label


def matched_patterns(
    text: str,
    patterns: list[str],
) -> list[str]:
    """Return the configured patterns that match the supplied text."""

    if pd.isna(text):
        return []

    found = []

    for pattern in patterns:
        if re.search(
            pattern,
            str(text),
            flags=re.IGNORECASE,
        ):
            found.append(
                clean_pattern(pattern)
            )

    return found


def update_counter(
    counter: Counter,
    values: pd.Series,
) -> None:
    """Count individual pipe-delimited trigger labels."""

    for value in values:

        if pd.isna(value):
            continue

        for signal in str(value).split(
            " | "
        ):
            signal = signal.strip()

            if signal:
                counter[signal] += 1


def append_signal_summary(
    summary_rows: list[dict],
    group: pd.DataFrame,
    theme: str,
    signal_group: str,
    column: str,
) -> None:
    """Append fixed/new-error statistics for one trigger group."""

    fixed = group[
        group["effect"] == "FIXED_ERROR"
    ]

    errors = group[
        group["effect"] == "NEW_ERROR"
    ]

    fixed_counter = Counter()
    error_counter = Counter()

    update_counter(
        fixed_counter,
        fixed[column],
    )

    update_counter(
        error_counter,
        errors[column],
    )

    all_signals = (
        set(fixed_counter)
        | set(error_counter)
    )

    for signal in all_signals:

        fixed_n = fixed_counter[
            signal
        ]

        error_n = error_counter[
            signal
        ]

        total = (
            fixed_n
            + error_n
        )

        empirical_precision = (
            fixed_n / total
            if total > 0
            else 0.0
        )

        summary_rows.append(
            {
                "theme_name":
                    theme,

                "signal_group":
                    signal_group,

                "signal":
                    signal,

                "fixed_errors":
                    fixed_n,

                "new_errors":
                    error_n,

                "net_gain":
                    fixed_n - error_n,

                "empirical_precision":
                    empirical_precision,
            }
        )


# =============================================================================
# DATA LOADING
# =============================================================================

gold = pd.read_csv(
    GOLD_FILE
)

changes = pd.read_csv(
    CHANGES_FILE
)

changes = changes[
    changes["scenario"]
    == SCENARIO_NAME
].copy()

gold["text_full"] = (
    gold["title"]
    .fillna("")
    .astype(str)
    + " "
    + gold["body"]
    .fillna("")
    .astype(str)
)


# =============================================================================
# MERGE FULL TEXT
# =============================================================================

# Only text_full is added from the gold file. Keeping title out of this merge
# avoids ambiguous title_x/title_y columns when changes already contains it.
audit = changes.merge(
    gold[
        [
            "annotation_id",
            "text_full",
        ]
    ],
    on="annotation_id",
    how="left",
    validate="many_to_one",
)


# =============================================================================
# TRIGGER RECONSTRUCTION
# =============================================================================

rows = []

for _, row in audit.iterrows():

    theme = row["theme_name"]
    text = row["text_full"]

    first_group = []
    second_group = []

    if (
        theme
        == "Climate / Environmental Risk"
    ):
        first_group = matched_patterns(
            text,
            CLIMATE_THEME_SIGNALS,
        )

        # The Climate rule combines a lexical theme signal with the
        # pre-existing risk evidence rather than a second lexical group.
        if pd.notna(
            row["risk_evidence"]
        ):
            second_group = [
                str(
                    row["risk_evidence"]
                )
            ]

    elif (
        theme
        == "Cybersecurity Risk"
    ):
        first_group = matched_patterns(
            text,
            CYBER_OBJECT_SIGNALS,
        )

        second_group = matched_patterns(
            text,
            CYBER_CONTEXT_SIGNALS,
        )

    elif (
        theme
        == "Supply Chain Risk"
    ):
        first_group = matched_patterns(
            text,
            SUPPLY_OBJECT_SIGNALS,
        )

        second_group = matched_patterns(
            text,
            SUPPLY_DISRUPTION_SIGNALS,
        )

    rows.append(
        {
            "annotation_id":
                row["annotation_id"],

            "theme_name":
                theme,

            "effect":
                row["effect"],

            "title":
                row["title"],

            "trigger_group_1":
                " | ".join(
                    first_group
                ),

            "trigger_group_2":
                " | ".join(
                    second_group
                ),

            "risk_evidence":
                row["risk_evidence"],
        }
    )

result = pd.DataFrame(
    rows
)


# =============================================================================
# INDIVIDUAL CHANGES
# =============================================================================

print("=" * 120)
print("CO-OCCURRENCE TRIGGER AUDIT")
print("=" * 120)

for theme, group in result.groupby(
    "theme_name"
):

    print()
    print("#" * 120)
    print(theme)
    print("#" * 120)

    for effect in [
        "FIXED_ERROR",
        "NEW_ERROR",
    ]:

        subset = group[
            group["effect"] == effect
        ]

        print(
            f"\n--- {effect} "
            f"({len(subset):,}) ---"
        )

        for _, row in subset.iterrows():

            print(
                f"\nID {row['annotation_id']}"
            )

            print(
                f"Title: {row['title']}"
            )

            print(
                f"Trigger group 1: "
                f"{row['trigger_group_1']}"
            )

            print(
                f"Trigger group 2: "
                f"{row['trigger_group_2']}"
            )


# =============================================================================
# TRIGGER PERFORMANCE
# =============================================================================

summary_rows = []

for theme, group in result.groupby(
    "theme_name"
):

    append_signal_summary(
        summary_rows,
        group,
        theme,
        "group_1",
        "trigger_group_1",
    )

    # Climate group 2 contains pre-existing risk evidence rather than a
    # comparable lexical trigger group.
    if (
        theme
        != "Climate / Environmental Risk"
    ):
        append_signal_summary(
            summary_rows,
            group,
            theme,
            "group_2",
            "trigger_group_2",
        )

summary = pd.DataFrame(
    summary_rows
)

if not summary.empty:
    summary = summary.sort_values(
        [
            "theme_name",
            "signal_group",
            "net_gain",
            "fixed_errors",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )


# =============================================================================
# SUMMARY REPORT
# =============================================================================

print()
print("=" * 120)
print("TRIGGER PERFORMANCE")
print("=" * 120)

if summary.empty:
    print(
        "No trigger statistics available."
    )

else:
    print(
        summary.to_string(
            index=False
        )
    )


# =============================================================================
# SAVE
# =============================================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

print()
print("=" * 120)
print("FILES SAVED")
print("=" * 120)

print(OUTPUT_FILE)
print(SUMMARY_FILE)

print(
    "\nAnalysis completed."
)