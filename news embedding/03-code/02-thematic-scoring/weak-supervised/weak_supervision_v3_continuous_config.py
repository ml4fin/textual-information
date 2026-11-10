"""
Weak Supervision V3 Continuous — Frozen Development Specification.

Target:
    Continuous article-theme relevance score in [0, 1].

Interpretation:
    Higher scores indicate stronger Weak V3 evidence that an article
    is relevant to the specified financial risk theme.

Important:
    The score is an interpretable relevance index, not a calibrated
    probability.

Development set:
    n = 485
    positives = 148
    negatives = 337

Selected specification:
    Candidate C — density reinforcement

Development-set performance:
    ROC-AUC = 0.837006
    Average Precision = 0.755430

Final test:
    Not used for specification selection.
"""


# =============================================================================
# VERSION
# =============================================================================

VERSION = (
    "Weak Supervision V3 Continuous - "
    "Frozen Development Version"
)

METHOD = "weakly_supervised"

MODEL_VERSION = (
    "weak_v3_continuous_frozen"
)


# =============================================================================
# FROZEN SCORE SPECIFICATION
# =============================================================================

SELECTED_CANDIDATE = (
    "score_C_density"
)

PROXIMITY_WINDOW = 50

DENSITY_REINFORCEMENT = 0.35

SCORE_MIN = 0.0
SCORE_MAX = 1.0


# =============================================================================
# DEVELOPMENT-SET RECORD
# =============================================================================

DEVELOPMENT_N = 485
DEVELOPMENT_POSITIVES = 148
DEVELOPMENT_NEGATIVES = 337

DEVELOPMENT_ROC_AUC = 0.837006
DEVELOPMENT_AVERAGE_PRECISION = 0.755430


# =============================================================================
# FINAL-TEST LOCK
# =============================================================================

FINAL_TEST_USED_FOR_SELECTION = False