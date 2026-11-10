#!/usr/bin/env python3

"""
BURST FINAL PRE-COMMIT — FROZEN SPECIFICATION

Input:
    data-1791021700092.csv

Output:
    burst_final_precommit_v1.csv

Frozen methodology:
    Historical window: 12 months
    Entry rolling Z: > 3.0
    Entry historical quantile: > Q95
    Persistence: coverage > historical rolling mean
    Exit: coverage <= historical rolling mean

Expected dataset:
    5 themes
    120 months per theme
    600 observations

Safety:
    - No database writes
    - Output saved only if all validation checks pass
    - Frozen parameters and expected results preserved
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION — FROZEN BURST SPECIFICATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data-1791021700092.csv"

OUTPUT_PATH = BASE_DIR / "burst_final_precommit_v1.csv"

WINDOW = 12
ENTRY_Z = 3.0
ENTRY_Q = 0.95

EXPECTED_THEMES = 5
EXPECTED_MONTHS_PER_THEME = 120

EXPECTED_ROWS = (
    EXPECTED_THEMES * EXPECTED_MONTHS_PER_THEME
)

EPS = 1e-12

REQUIRED_COLUMNS = {
    "theme_name",
    "period_start",
    "article_count",
    "article_count_norm",
}

EXPORT_COLUMNS = [
    "theme_name",
    "period_start",
    "article_count",
    "article_count_norm",
    "rolling_mean_12",
    "rolling_std_12",
    "rolling_z_12",
    "all_prior_q95",
    "evaluable",
    "entry_candidate",
    "burst_onset",
    "episode_id",
    "burst_state_final",
    "burst_intensity_final",
]

EXPECTED_RESULTS = {
    "ai_risk": {
        "onsets": 6,
        "burst_months": 26,
    },
    "climate_risk": {
        "onsets": 4,
        "burst_months": 22,
    },
    "cyber_risk": {
        "onsets": 3,
        "burst_months": 15,
    },
    "pandemic_risk": {
        "onsets": 1,
        "burst_months": 12,
    },
    "supply_chain_risk": {
        "onsets": 5,
        "burst_months": 35,
    },
}

KEY_ONSETS = [
    ("pandemic_risk", "2020-01-01"),
    ("ai_risk", "2022-12-01"),
    ("climate_risk", "2021-07-01"),
    ("cyber_risk", "2020-01-01"),
    ("cyber_risk", "2022-02-01"),
    ("supply_chain_risk", "2020-02-01"),
    ("supply_chain_risk", "2021-10-01"),
]

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)
pd.set_option(
    "display.float_format",
    lambda x: f"{x:.10f}",
)


# =============================================================================
# HELPERS
# =============================================================================

def header(title):

    print("\n" + "=" * 140)
    print(title)
    print("=" * 140)


def check(name, condition, details=""):

    status = "PASS" if condition else "FAIL"

    print(f"{status:4s} | {name}")

    if details:
        print(f"       {details}")

    return bool(condition)


# =============================================================================
# LOAD
# =============================================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["period_start"] = pd.to_datetime(
        df["period_start"],
        errors="coerce",
    )

    if df["period_start"].isna().any():
        raise ValueError(
            "Invalid or missing period_start values."
        )

    for column in [
        "article_count",
        "article_count_norm",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.sort_values(
            ["theme_name", "period_start"]
        )
        .reset_index(drop=True)
    )

    return df


# =============================================================================
# FROZEN BURST CONSTRUCTION
# =============================================================================

def build_burst_variables(df):

    parts = []

    for theme, g in df.groupby(
        "theme_name",
        sort=False,
    ):

        g = (
            g.copy()
            .sort_values("period_start")
            .reset_index(drop=True)
        )

        x = g["article_count_norm"].astype(float)

        # ---------------------------------------------------------------------
        # STRICTLY HISTORICAL ROLLING STATISTICS
        # ---------------------------------------------------------------------

        historical = x.shift(1)

        g["rolling_mean_12"] = (
            historical
            .rolling(
                WINDOW,
                min_periods=WINDOW,
            )
            .mean()
        )

        g["rolling_std_12"] = (
            historical
            .rolling(
                WINDOW,
                min_periods=WINDOW,
            )
            .std(ddof=1)
        )

        g["rolling_z_12"] = np.where(
            g["rolling_std_12"] > 0,
            (
                x - g["rolling_mean_12"]
            ) / g["rolling_std_12"],
            np.nan,
        )

        # ---------------------------------------------------------------------
        # STRICTLY HISTORICAL EXPANDING Q95
        # ---------------------------------------------------------------------

        g["all_prior_q95"] = (
            historical
            .expanding(
                min_periods=WINDOW
            )
            .quantile(ENTRY_Q)
        )

        # ---------------------------------------------------------------------
        # WARM-UP
        # ---------------------------------------------------------------------

        g["evaluable"] = (
            g["rolling_mean_12"].notna()
            &
            g["rolling_std_12"].notna()
            &
            g["all_prior_q95"].notna()
        )

        # ---------------------------------------------------------------------
        # ENTRY CANDIDATE
        # ---------------------------------------------------------------------

        g["entry_candidate"] = (
            g["evaluable"]
            &
            (g["rolling_z_12"] > ENTRY_Z)
            &
            (x > g["all_prior_q95"])
        )

        # ---------------------------------------------------------------------
        # FINAL STATE MACHINE
        # ---------------------------------------------------------------------

        burst_state = np.full(
            len(g),
            np.nan,
        )

        burst_intensity = np.full(
            len(g),
            np.nan,
        )

        burst_onset = np.zeros(
            len(g),
            dtype=int,
        )

        episode_id = np.full(
            len(g),
            np.nan,
        )

        active = False
        current_episode = 0

        for i in range(len(g)):

            if not bool(
                g.loc[i, "evaluable"]
            ):
                continue

            x_i = float(
                x.iloc[i]
            )

            mean_i = float(
                g.loc[i, "rolling_mean_12"]
            )

            candidate_i = bool(
                g.loc[i, "entry_candidate"]
            )

            # -----------------------------------------------------------------
            # INACTIVE
            # -----------------------------------------------------------------

            if not active:

                if candidate_i:

                    active = True
                    current_episode += 1

                    burst_state[i] = 1
                    burst_onset[i] = 1

                    episode_id[i] = (
                        current_episode
                    )

                    burst_intensity[i] = max(
                        0.0,
                        x_i - mean_i,
                    )

                else:

                    burst_state[i] = 0
                    burst_intensity[i] = 0.0

            # -----------------------------------------------------------------
            # ACTIVE
            # -----------------------------------------------------------------

            else:

                if x_i > mean_i:

                    burst_state[i] = 1

                    episode_id[i] = (
                        current_episode
                    )

                    burst_intensity[i] = max(
                        0.0,
                        x_i - mean_i,
                    )

                else:

                    active = False

                    # Same-month re-entry is allowed only if
                    # the full frozen entry condition holds.

                    if candidate_i:

                        active = True
                        current_episode += 1

                        burst_state[i] = 1
                        burst_onset[i] = 1

                        episode_id[i] = (
                            current_episode
                        )

                        burst_intensity[i] = max(
                            0.0,
                            x_i - mean_i,
                        )

                    else:

                        burst_state[i] = 0
                        burst_intensity[i] = 0.0

        g["burst_state_final"] = burst_state

        g["burst_intensity_final"] = (
            burst_intensity
        )

        g["burst_onset"] = burst_onset

        g["episode_id"] = episode_id

        parts.append(g)

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    out = (
        out.sort_values(
            ["theme_name", "period_start"]
        )
        .reset_index(drop=True)
    )

    return out


# =============================================================================
# VALIDATION
# =============================================================================

def validate_results(df, out):

    checks = []

    # -------------------------------------------------------------------------
    # 1. INPUT VALIDATION
    # -------------------------------------------------------------------------

    header("1. INPUT VALIDATION")

    checks.append(
        check(
            "Expected total number of rows",
            len(df) == EXPECTED_ROWS,
            (
                f"observed={len(df)}, "
                f"expected={EXPECTED_ROWS}"
            ),
        )
    )

    checks.append(
        check(
            "Expected number of themes",
            (
                df["theme_name"].nunique()
                == EXPECTED_THEMES
            ),
            (
                f"observed={df['theme_name'].nunique()}, "
                f"expected={EXPECTED_THEMES}"
            ),
        )
    )

    counts = df.groupby("theme_name").size()

    checks.append(
        check(
            "Exactly 120 monthly observations per theme",
            (
                counts == EXPECTED_MONTHS_PER_THEME
            ).all(),
            str(counts.to_dict()),
        )
    )

    checks.append(
        check(
            "No duplicate theme-month keys",
            not df.duplicated(
                ["theme_name", "period_start"]
            ).any(),
        )
    )

    checks.append(
        check(
            "No missing article_count_norm",
            df["article_count_norm"].notna().all(),
        )
    )

    checks.append(
        check(
            "Coverage is non-negative",
            (
                df["article_count_norm"] >= 0
            ).all(),
        )
    )

    # -------------------------------------------------------------------------
    # 2. STRUCTURAL VALIDATION
    # -------------------------------------------------------------------------

    header("2. STRUCTURAL VALIDATION")

    warmup = ~out["evaluable"]
    eval_mask = out["evaluable"]

    warmup_by_theme = (
        out.groupby("theme_name")["evaluable"]
        .apply(
            lambda values: (~values).sum()
        )
    )

    checks.append(
        check(
            "Exactly 12 warm-up months per theme",
            (
                warmup_by_theme == WINDOW
            ).all(),
            str(warmup_by_theme.to_dict()),
        )
    )

    checks.append(
        check(
            "Exactly 60 warm-up rows overall",
            warmup.sum()
            == EXPECTED_THEMES * WINDOW,
            (
                f"observed={warmup.sum()}, "
                f"expected={EXPECTED_THEMES * WINDOW}"
            ),
        )
    )

    checks.append(
        check(
            "Exactly 540 evaluable rows overall",
            eval_mask.sum()
            == (
                EXPECTED_ROWS
                - EXPECTED_THEMES * WINDOW
            ),
            (
                f"observed={eval_mask.sum()}, "
                f"expected="
                f"{EXPECTED_ROWS - EXPECTED_THEMES * WINDOW}"
            ),
        )
    )

    checks.append(
        check(
            "Warm-up burst_state is NULL",
            out.loc[
                warmup,
                "burst_state_final",
            ].isna().all(),
        )
    )

    checks.append(
        check(
            "Warm-up burst_intensity is NULL",
            out.loc[
                warmup,
                "burst_intensity_final",
            ].isna().all(),
        )
    )

    checks.append(
        check(
            "All evaluable rows have non-NULL burst_state",
            out.loc[
                eval_mask,
                "burst_state_final",
            ].notna().all(),
        )
    )

    checks.append(
        check(
            "All evaluable rows have non-NULL burst_intensity",
            out.loc[
                eval_mask,
                "burst_intensity_final",
            ].notna().all(),
        )
    )

    # -------------------------------------------------------------------------
    # 3. STATE / INTENSITY CONSISTENCY
    # -------------------------------------------------------------------------

    header("3. STATE / INTENSITY CONSISTENCY")

    checks.append(
        check(
            "Evaluable burst_state values are only 0 or 1",
            out.loc[
                eval_mask,
                "burst_state_final",
            ].isin([0, 1]).all(),
        )
    )

    checks.append(
        check(
            "Burst intensity is never negative",
            (
                out.loc[
                    eval_mask,
                    "burst_intensity_final",
                ] >= -EPS
            ).all(),
        )
    )

    inactive = (
        eval_mask
        &
        (out["burst_state_final"] == 0)
    )

    checks.append(
        check(
            "Inactive evaluable months have zero intensity",
            (
                out.loc[
                    inactive,
                    "burst_intensity_final",
                ].abs() <= EPS
            ).all(),
        )
    )

    active = (
        eval_mask
        &
        (out["burst_state_final"] == 1)
    )

    checks.append(
        check(
            "Active months have strictly positive intensity",
            (
                out.loc[
                    active,
                    "burst_intensity_final",
                ] > 0
            ).all(),
        )
    )

    expected_intensity = (
        out.loc[
            active,
            "article_count_norm",
        ]
        -
        out.loc[
            active,
            "rolling_mean_12",
        ]
    )

    checks.append(
        check(
            "Active intensity equals coverage minus rolling mean",
            np.allclose(
                out.loc[
                    active,
                    "burst_intensity_final",
                ],
                expected_intensity,
                rtol=0,
                atol=EPS,
            ),
        )
    )

    checks.append(
        check(
            "Every onset is an active burst month",
            (
                out.loc[
                    out["burst_onset"] == 1,
                    "burst_state_final",
                ] == 1
            ).all(),
        )
    )

    checks.append(
        check(
            "Every onset satisfies frozen entry rule",
            out.loc[
                out["burst_onset"] == 1,
                "entry_candidate",
            ].eq(True).all(),
        )
    )

    # -------------------------------------------------------------------------
    # 4. EPISODE SUMMARY
    # -------------------------------------------------------------------------

    header("4. FINAL EPISODE SUMMARY")

    episode_rows = []

    for theme, group in out.groupby("theme_name"):

        active_group = group[
            group["burst_state_final"] == 1
        ].copy()

        if active_group.empty:
            continue

        for eid, episode in active_group.groupby(
            "episode_id"
        ):

            episode_rows.append({
                "theme_name": theme,
                "episode_id": int(eid),
                "onset": episode[
                    "period_start"
                ].min(),
                "last_active": episode[
                    "period_start"
                ].max(),
                "duration_months": len(episode),
                "peak_coverage": episode[
                    "article_count_norm"
                ].max(),
                "peak_intensity": episode[
                    "burst_intensity_final"
                ].max(),
                "cumulative_intensity": episode[
                    "burst_intensity_final"
                ].sum(),
            })

    episode_columns = [
        "theme_name",
        "episode_id",
        "onset",
        "last_active",
        "duration_months",
        "peak_coverage",
        "peak_intensity",
        "cumulative_intensity",
    ]

    episodes = pd.DataFrame(
        episode_rows,
        columns=episode_columns,
    )

    print(
        episodes.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    print("\nCounts by theme:")

    theme_summary = (
        out.groupby("theme_name")
        .agg(
            total_months=(
                "period_start",
                "size",
            ),
            evaluable_months=(
                "evaluable",
                "sum",
            ),
            burst_months=(
                "burst_state_final",
                lambda s: int((s == 1).sum()),
            ),
            onset_count=(
                "burst_onset",
                "sum",
            ),
            max_intensity=(
                "burst_intensity_final",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        theme_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    # -------------------------------------------------------------------------
    # 5. FROZEN REGRESSION CHECKS
    # -------------------------------------------------------------------------

    header("5. REGRESSION CHECK AGAINST FROZEN DIAGNOSTICS")

    for theme, expected in EXPECTED_RESULTS.items():

        group = out[
            out["theme_name"] == theme
        ]

        actual_onsets = int(
            group["burst_onset"].sum()
        )

        actual_burst_months = int(
            (
                group["burst_state_final"] == 1
            ).sum()
        )

        checks.append(
            check(
                f"{theme}: expected onset count",
                actual_onsets == expected["onsets"],
                (
                    f"observed={actual_onsets}, "
                    f"expected={expected['onsets']}"
                ),
            )
        )

        checks.append(
            check(
                f"{theme}: expected active burst months",
                (
                    actual_burst_months
                    == expected["burst_months"]
                ),
                (
                    f"observed={actual_burst_months}, "
                    f"expected={expected['burst_months']}"
                ),
            )
        )

    # -------------------------------------------------------------------------
    # 6. KEY EPISODE CHECKS
    # -------------------------------------------------------------------------

    header("6. KEY EPISODE CHECKS")

    for theme, date_string in KEY_ONSETS:

        date = pd.Timestamp(date_string)

        row = out[
            (out["theme_name"] == theme)
            &
            (out["period_start"] == date)
        ]

        valid = (
            len(row) == 1
            and row.iloc[0]["burst_onset"] == 1
            and row.iloc[0]["burst_state_final"] == 1
        )

        checks.append(
            check(
                f"{theme} onset at {date.date()}",
                valid,
            )
        )

    # -------------------------------------------------------------------------
    # 7. WARM-UP BOUNDARY AUDIT
    # -------------------------------------------------------------------------

    header("7. WARM-UP BOUNDARY AUDIT")

    audit_columns = [
        "period_start",
        "article_count_norm",
        "rolling_mean_12",
        "rolling_std_12",
        "rolling_z_12",
        "all_prior_q95",
        "evaluable",
        "entry_candidate",
        "burst_state_final",
        "burst_intensity_final",
    ]

    for theme, group in out.groupby("theme_name"):

        group = group.sort_values("period_start")

        print(f"\n--- {theme} ---")

        print(
            group.iloc[10:15][audit_columns]
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # 8. FINAL STATUS
    # -------------------------------------------------------------------------

    header("8. FINAL VALIDATION STATUS")

    n_pass = sum(checks)
    n_total = len(checks)
    n_fail = n_total - n_pass

    print(f"\nChecks passed : {n_pass}")
    print(f"Checks failed : {n_fail}")
    print(f"Total checks  : {n_total}")

    if n_fail == 0:

        print("\nFINAL STATUS: PASS")

        print(
            "The dataset is internally consistent with "
            "the frozen burst specification."
        )

    else:

        print("\nFINAL STATUS: FAIL")

        print(
            "DO NOT write burst variables to the database. "
            "Investigate failed checks first."
        )

    return n_fail == 0


# =============================================================================
# MAIN
# =============================================================================

def main():

    header("BURST FINAL PRE-COMMIT BUILD")

    print(f"\nInput  : {INPUT_PATH}")
    print(f"Output : {OUTPUT_PATH}")

    print(
        f"""
Frozen specification

Input:
    x_k,t = article_count_norm

Historical baseline:
    previous {WINDOW} months only

Entry:
    rolling_z > {ENTRY_Z}
    AND
    x_k,t > all-prior historical Q{int(ENTRY_Q * 100)}

Persistence:
    while active, remain active iff
    x_k,t > rolling historical mean

Exit:
    first month with
    x_k,t <= rolling historical mean

Intensity:
    if burst_state = 1:
        x_k,t - rolling historical mean
    otherwise:
        0

Warm-up:
    first {WINDOW} months per theme:
        burst_state     = NULL
        burst_intensity = NULL
"""
    )

    df = load_data()

    out = build_burst_variables(df)

    validation_passed = validate_results(
        df,
        out,
    )

    # -------------------------------------------------------------------------
    # SAVE ONLY AFTER SUCCESSFUL VALIDATION
    # -------------------------------------------------------------------------

    if not validation_passed:

        raise RuntimeError(
            "\nBURST PRE-COMMIT VALIDATION FAILED.\n"
            "The output CSV was not written.\n"
            "Do not proceed to the database write stage."
        )

    out[EXPORT_COLUMNS].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    header("PRE-COMMIT DATASET SAVED")

    print(f"\nOutput:\n{OUTPUT_PATH}")

    print(f"\nRows: {len(out)}")

    print(
        "\nAll frozen validation checks passed."
    )

    print(
        "The dataset is eligible for the "
        "subsequent database write stage."
    )


if __name__ == "__main__":
    main()